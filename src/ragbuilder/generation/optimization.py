"""
Optimization module for generation components in RAGBuilder.
Uses Optuna for efficient hyperparameter tuning of generation pipelines.
"""
import logging
import optuna
from typing import Dict, Any, Optional, List
from datetime import datetime

from optuna import Trial, create_study
import numpy as np

from ragbuilder.config import LogConfig, GenerationOptionsConfig, GenerationConfig
from ragbuilder.core import ConfigStore, DBLoggerCallback, setup_rich_logging, console
from ragbuilder.core.exceptions import OptimizationError, DependencyError
from ragbuilder.core.results import GenerationResults
from ragbuilder.generation.evaluation import GenerationEvaluator
from ragbuilder.generation.pipeline import GenerationPipeline
from ragbuilder.generation.prompt_templates import load_prompts

CONFIG_STORE = ConfigStore()

class GenerationOptimizer:
    """
    Optimizer for generation components that uses Optuna for hyperparameter tuning.
    """
    
    def __init__(
        self,
        options_config: GenerationOptionsConfig,
        evaluator: GenerationEvaluator,
        retriever: Optional[Any] = None,
        verbose: bool = False,
        show_progress_bar: bool = True,
        callback=None
    ):
        """
        Initialize the optimizer with configuration options.
        
        Args:
            options_config: Configuration options for generation optimization
            evaluator: Evaluator for generation pipelines
            retriever: Retriever component to use with generation pipelines
            verbose: Whether to log detailed information
            show_progress_bar: Whether to show progress bar during optimization
            callback: Optional callback for custom logging
        """
        self.logger = logging.getLogger("ragbuilder.generation.optimization")
        self.options_config = options_config
        self.evaluator = evaluator
        self.show_progress_bar = show_progress_bar
        self.verbose = verbose
        
        # Load retriever
        self.retriever = retriever
        if self.retriever is None:
            self.logger.warning("No retriever provided, will attempt to get from CONFIG_STORE")
            retriever_pipeline = CONFIG_STORE.get_best_retriever_pipeline()
            if retriever_pipeline is None:
                raise DependencyError("No retriever pipeline found. Run retrieval optimization first.")
            self.retriever = retriever_pipeline
        
        # Load prompt templates
        self.prompt_templates = load_prompts(
            options_config.prompt_template_path, 
            options_config.local_prompt_template_path, 
            options_config.read_local_only
        )
        self.logger.debug(f"Loaded {len(self.prompt_templates)} prompt templates")
        
        # Map LLM configurations and prompt templates for easier access in trials
        self.llm_map = {i: llm for i, llm in enumerate(options_config.llms)}
        self.prompt_map = {i: (key, template) for i, (key, template) in enumerate(self.prompt_templates)}
        
        # Setup DB logging callback if enabled
        self.callbacks = []
        if options_config.database_logging:
            try:
                db_callback = DBLoggerCallback(
                    study_name=options_config.optimization.study_name,
                    config=options_config,
                    module_type='generation'
                )
                self.callbacks.append(db_callback)
                self.logger.debug("Database logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize database logging: {e}")

        # Add any additional callbacks
        if callback:
            self.callbacks.append(callback)
    
    def _generate_generation_config(self, trial: Trial) -> GenerationConfig:
        """
        Generate a generation configuration from trial parameters.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            GenerationConfig for this trial
        """
        # Select LLM
        llm_index = trial.suggest_categorical("llm_index", list(self.llm_map.keys()))
        llm_config = self.llm_map[llm_index]
        
        # Select prompt template
        prompt_index = trial.suggest_categorical("prompt_index", list(self.prompt_map.keys()))
        prompt_key, prompt_template = self.prompt_map[prompt_index]
        
        # Create configuration
        params = {
            "llm": llm_config,
            "prompt_template": prompt_template.template,
            "prompt_key": prompt_key
        }
        
        return GenerationConfig(**params)
    
    def _objective(self, trial: Trial) -> float:
        """
        Optimization objective function.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Evaluation score for this trial
        """
        console.print(f"[heading]Generation Trial {trial.number}/{self.options_config.optimization.n_trials - 1}[/heading]")
        
        try:
            # Generate config for this trial
            config = self._generate_generation_config(trial)
            
            # Check if we've already evaluated this exact config
            trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    self.logger.info(f"Configuration already evaluated with score: {t.value}")
                    return t.value
            
            # Create generation pipeline for this config
            pipeline = GenerationPipeline(config, self.retriever, verbose=self.verbose).pipeline
            
            # Evaluate the pipeline
            config_key = f"{config.prompt_key}_{trial.number}"
            results = self.evaluator.evaluate_generation(pipeline, config_key)
            
            # Extract score
            score = results['score']
            
            # Store results in study's user attributes
            trial.study.set_user_attr(
                f"trial_{trial.number}_results",
                {
                    'score': score,
                    'metrics': results['metrics'],
                    'config': config.model_dump(),
                    'summary': {
                        'prompt_key': config.prompt_key,
                        'prompt': config.prompt_template
                    },
                    'detailed_results': results.get('detailed_results', [])
                }
            )
            
            # Call callbacks
            for callback in self.callbacks:
                try:
                    callback(trial.study, trial)
                except Exception as e:
                    self.logger.warning(f"Callback error: {e}")
            
            self.logger.debug(f"Trial {trial.number} score: {score:.4f}")
            return score
            
        except Exception as e:
            console.print(f"[red]Trial failed: {str(e)}[/red]")
            self.logger.exception("Trial error")
            return float('nan')
    
    def optimize(self) -> GenerationResults:
        """
        Run optimization process.
        
        Returns:
            GenerationResults containing optimization results and best pipeline
        """
        console.rule("[heading]Starting Generation Optimization[/heading]")
        start_time = datetime.now()
        
        if self.options_config.optimization.overwrite_study and \
            self.options_config.optimization.study_name in optuna.study.get_all_study_names(storage=self.options_config.optimization.storage):
            self.logger.info(f"Overwriting existing study: {self.options_config.optimization.study_name}")
            optuna.delete_study(study_name=self.options_config.optimization.study_name, storage=self.options_config.optimization.storage)

        # Create study with appropriate settings
        study = create_study(
            storage=self.options_config.optimization.storage,
            study_name=self.options_config.optimization.study_name,
            load_if_exists=self.options_config.optimization.load_if_exists,
            direction=self.options_config.optimization.optimization_direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self._objective,
            n_trials=self.options_config.optimization.n_trials,
            n_jobs=self.options_config.optimization.n_jobs,
            timeout=self.options_config.optimization.timeout,
            show_progress_bar=self.show_progress_bar
        )
        
        # Get best configuration
        if study.best_trial is None:
            raise OptimizationError("No successful trials completed")
        
        # Get metrics from the best trial
        best_trial_key = f"trial_{study.best_trial.number}_results"
        trial_results = study.user_attrs.get(best_trial_key, {})
        metrics = trial_results.get("metrics", {})
        
        # Create best configuration and pipeline
        best_config = self._generate_generation_config(study.best_trial)
        best_pipeline = GenerationPipeline(best_config, self.retriever, verbose=self.verbose).pipeline
        
        # Store best pipeline in config store for other modules to use
        CONFIG_STORE.store_best_generator_pipeline(best_pipeline)
        
        # Create structured results object
        results = GenerationResults(
            best_config=best_config,
            best_score=study.best_value,
            best_pipeline=best_pipeline,
            best_prompt=best_config.prompt_template,
            n_trials=self.options_config.optimization.n_trials,
            completed_trials=len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            optimization_time=(datetime.now() - start_time).total_seconds(),
            avg_latency=metrics.get("avg_latency"),
            error_rate=metrics.get("error_rate")
        )
        
        # Log results
        console.print("[success]Generation Optimization Complete![/success]")
        console.print(f"[success]Best Score:[/success] [value]{results.best_score:.4f}[/value]")
        console.print("[success]Best Configuration:[/success]")
        console.print(results.get_config_summary(), style="value")
        
        return results


def run_generation_optimization(
    options_config: GenerationOptionsConfig,
    retriever: Optional[Any] = None,
    log_config: Optional[LogConfig] = None
) -> GenerationResults:
    """
    Run generation optimization process.
    
    Args:
        options_config: Generation configuration options
        retriever: Optional retriever to use for generation
        log_config: Optional logging configuration
    
    Returns:
        GenerationResults containing optimization results and best generator pipeline
    """
    # Setup logging
    if log_config:
        setup_rich_logging(log_config.log_level, log_config.log_file)
    
    # Create evaluator
    evaluator = GenerationEvaluator(options_config.evaluation_config)
    
    # Create optimizer
    optimizer = GenerationOptimizer(
        options_config,
        evaluator,
        retriever=retriever,
        verbose=log_config.verbose if log_config else False,
        show_progress_bar=log_config.show_progress_bar if log_config else True
    )
    
    # Run optimization
    return optimizer.optimize()