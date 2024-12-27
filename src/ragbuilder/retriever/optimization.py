import optuna
from typing import List, Dict, Any, Optional
from importlib import import_module
import logging
from optuna import Trial, create_study
import numpy as np

from ragbuilder.config import LogConfig, RetrievalOptionsConfig, RetrievalConfig
from ragbuilder.core import (
    DocumentStore, 
    ConfigStore, 
    DBLoggerCallback, 
    setup_rich_logging, 
    console
)
from ragbuilder.core.exceptions import OptimizationError
from ragbuilder.retriever.pipeline import RetrieverPipeline
from ragbuilder.core.results import RetrievalResults
from .evaluation import Evaluator, RetrieverF1ScoreEvaluator

STORE = DocumentStore()
CONFIG_STORE = ConfigStore()

class RetrieverOptimization:
    def __init__(
        self, 
        options_config: RetrievalOptionsConfig, 
        evaluator: Evaluator, 
        vectorstore: Optional[Any] = None,
        show_progress_bar: bool = True,
        verbose: bool = False,
        callback=None
    ):
        self.options_config = options_config
        self.evaluator = evaluator
        self.show_progress_bar = show_progress_bar
        self.logger = logging.getLogger("ragbuilder.retriever.optimizer")
        self.verbose = verbose
        self.best_config = CONFIG_STORE.get_best_config()
        if not self.best_config:
            raise OptimizationError("No optimized data ingestion configuration found")
        
        # Use provided vectorstore if available, otherwise get from DocumentStore
        self.vectorstore = vectorstore if vectorstore is not None else STORE.get_best_config_vectorstore()
        if self.vectorstore is None:
            raise OptimizationError("No vectorstore found for best configuration")
        
        # Load test dataset
        self.test_queries = self._load_test_queries()
        # Setup component maps
        self.retriever_map = {i: retriever for i, retriever in enumerate(options_config.retrievers)}
        self.reranker_map = {
            i: reranker for i, reranker in enumerate(options_config.rerankers)
        } if options_config.rerankers else {}
        
        # Setup DB logging callback if enabled
        self.callbacks = []
        if options_config.database_logging:
            try:
                db_callback = DBLoggerCallback(
                    study_name=options_config.optimization.study_name,
                    config=options_config,
                    module_type='retriever'
                )
                self.callbacks.append(db_callback)
                self.logger.debug("Database logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize database logging: {e}")

        # Add any additional callbacks
        if callback:
            self.callbacks.append(callback)

    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from the specified dataset."""
        # Implementation depends on test dataset format
        pass

    def _generate_retrieval_config(self, trial: Trial) -> RetrievalConfig:
        """Generate a retrieval configuration from trial parameters."""
        retrievers = []
        
        # If only one retriever option, use it directly
        if len(self.retriever_map) == 1:
            retriever_config = self.retriever_map[0]
            # If only one k value, use it directly
            k = (retriever_config.retriever_k[0] if len(retriever_config.retriever_k) == 1
                 else trial.suggest_categorical("retriever_k", retriever_config.retriever_k))
            
            retrievers.append(
                retriever_config.model_copy(
                    update={"retriever_k": [k], "weight": 1.0}
                )
            )
        else:
            # Decide number of retrievers to combine (1 to 3)
            n_retrievers = trial.suggest_int("n_retrievers", 1, min(3, len(self.retriever_map)))
            
            for i in range(n_retrievers):
                # Select retriever using index
                retriever_index = trial.suggest_categorical(f"retriever_{i}_index", list(self.retriever_map.keys()))
                retriever_config = self.retriever_map[retriever_index]

                if retriever_config in retrievers:
                    continue
                
                # Select retriever k from available options
                k = (retriever_config.retriever_k[0] if len(retriever_config.retriever_k) == 1
                     else trial.suggest_categorical(f"retriever_{i}_k", retriever_config.retriever_k))
                
                # Assign weight for ensemble (except for single retriever)
                weight = 1.0
                if n_retrievers > 1:
                    weight = retriever_config.weight if hasattr(retriever_config, 'weight') else 1.0
                    # weight = (retriever_config.weight if retriever_config.weight != 1.0 
                    #          else trial.suggest_float(f"retriever_{i}_weight", 0.0, 1.0))
                
                retrievers.append(
                    retriever_config.model_copy(
                        update={"retriever_k": [k], "weight": weight}
                    )
                )

        # Handle rerankers if present
        rerankers = []
        if self.reranker_map:
            # Only suggest using rerankers if more than one option exists
            use_rerankers = (True if len(self.reranker_map) == 1
                            else trial.suggest_categorical("use_rerankers", [True, False]))
            
            if use_rerankers:
                # TODO: For now, only support single reranker. Explore multi-reranker setup in the future, depending on performance.
                rerankers = ([self.reranker_map[0]] if len(self.reranker_map) == 1
                             else [self.reranker_map[trial.suggest_categorical("reranker_index", list(self.reranker_map.keys()))]])


                # Multi-reranker setup
                # # Decide number of rerankers to chain (1 to 2)
                # n_rerankers = trial.suggest_int("n_rerankers", 1, min(2, len(self.reranker_map)))
                    
                # for i in range(n_rerankers):
                #     reranker_index = trial.suggest_categorical(f"reranker_{i}_index", list(self.reranker_map.keys()))
                #     reranker_config = self.reranker_map[reranker_index]
                #     rerankers.append(reranker_config)

        # Select final top k
        final_k = (self.options_config.top_k[0] if len(self.options_config.top_k) == 1
                   else trial.suggest_categorical("final_k", self.options_config.top_k))
        
        params = {
            "retrievers": retrievers,
            "rerankers": rerankers,
            "top_k": final_k
        }
        
        self.logger.debug(f"Trial parameters: {params}")

        return RetrievalConfig(**params)

    def _objective(self, trial: Trial) -> float:
        """Optimization objective function."""
        console.print(f"[heading]Trial {trial.number}/{self.options_config.optimization.n_trials - 1}[/heading]")
        try:
            # Generate config for this trial
            config = self._generate_retrieval_config(trial)

            trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    # Use the existing value as trial duplicated the parameters.
                    self.logger.info(f"Config already evaluated with score: {t.value}")
                    return t.value
            
            if self.verbose:
                self.logger.info(f"Running pipeline with config: {config}")
            
            # Create retriever pipeline
            pipeline = RetrieverPipeline(config=config, vectorstore=self.vectorstore, verbose=self.verbose)
            avg_score, question_details = self.evaluator.evaluate(pipeline)
            metrics = self._calculate_aggregate_metrics(question_details)
            
            # Store results in study's user attributes
            trial.study.set_user_attr(
                f"trial_{trial.number}_results",
                {
                    "avg_score": avg_score,
                    "question_details": question_details,
                    "metrics": metrics,
                    "config": config.model_dump()
                }
            )

            for callback in self.callbacks:
                try:
                    callback(trial.study, trial)
                except Exception as e:
                    self.logger.warning(f"Callback error: {e}")
            
            self.logger.debug(f"Trial Score: {avg_score:.4f}")
            return avg_score
            
        except Exception as e:
            console.print(f"[red]Trial failed: {str(e)}[/red]")
            raise OptimizationError(f"Trial failed: {str(e)}") from e

    def optimize(self) -> RetrievalResults:
        """
        Run optimization process.
        
        Returns:
            RetrievalResults containing optimization results and best pipeline
        """
        console.rule("[heading]Starting retriever optimization...[/heading]")
        
        if self.options_config.optimization.overwrite_study and \
            self.options_config.optimization.study_name in optuna.study.get_all_study_names(storage=self.options_config.optimization.storage):
            self.logger.info(f"Overwriting existing study: {self.options_config.optimization.study_name}")
            optuna.delete_study(study_name=self.options_config.optimization.study_name, storage=self.options_config.optimization.storage)

        # Create study with appropriate settings
        self.study = create_study(
            storage=self.options_config.optimization.storage,
            study_name=self.options_config.optimization.study_name,
            load_if_exists=self.options_config.optimization.load_if_exists,
            direction=self.options_config.optimization.optimization_direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        
        self.study.optimize(
            self._objective,
            n_trials=self.options_config.optimization.n_trials,
            n_jobs=self.options_config.optimization.n_jobs,
            timeout=self.options_config.optimization.timeout,
            show_progress_bar=self.show_progress_bar
        )
        
        # Translate indices to actual component names
        # readable_params = {}
        # for param, value in self.study.best_params.items():
        #     if param.startswith("retriever_") and param.endswith("_index"):
        #         retriever_index = int(param.split("_")[1])
        #         readable_params[f"retriever_{retriever_index}"] = self.retriever_map[value].type
        #     elif param == "reranker_index":
        #         readable_params["reranker"] = self.reranker_map[value].type
        #     else:
        #         readable_params[param] = value
        
        # Get metrics from the best trial
        best_trial_key = f"trial_{self.study.best_trial.number}_results"
        trial_results = self.study.user_attrs.get(best_trial_key, {})
        metrics = trial_results.get("metrics", {})
        
        best_config = self._generate_retrieval_config(self.study.best_trial)
        best_pipeline = RetrieverPipeline(config=best_config, vectorstore=self.vectorstore)
        CONFIG_STORE.store_best_retriever_pipeline(best_pipeline)
        
        # Create structured results object
        results = RetrievalResults(
            best_config=best_config,
            best_score=self.study.best_value,
            best_pipeline=best_pipeline,
            n_trials=self.options_config.optimization.n_trials,
            completed_trials=len(self.study.trials),
            optimization_time=(self.study.trials[-1].datetime_complete - self.study.trials[0].datetime_start).total_seconds(),
            avg_latency=metrics.get("avg_latency"),
            error_rate=metrics.get("error_rate")
        )
        
        # Log results
        console.print(f"[heading]✓ Optimization complete![/heading]")
        console.print(f"[success]Best Score:[/success] [value]{results.best_score:.4f}[/value]")
        console.print("[success]Best Configuration:[/success]")
        console.print(results.get_config_summary(), style="value")
        
        return results

    def _calculate_aggregate_metrics(self, question_details: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from detailed results."""
        successful_evals = [d for d in question_details if "error" not in d]
        
        if not successful_evals:
            return {
                "avg_latency": None,
                "total_questions": len(question_details),
                "successful_questions": 0,
                "error_rate": 1.0,
                "avg_context_precision": 0.0,
                "avg_context_recall": 0.0
            }
        
        return {
            "avg_latency": np.mean([d["latency"] for d in successful_evals]),
            "total_questions": len(question_details),
            "successful_questions": len(successful_evals),
            "error_rate": 1 - (len(successful_evals) / len(question_details)),
            "avg_context_precision": np.nanmean([d["metrics"]["context_precision"] for d in successful_evals]),
            "avg_context_recall": np.nanmean([d["metrics"]["context_recall"] for d in successful_evals])
        }

def run_retrieval_optimization(
    options_config: RetrievalOptionsConfig, 
    vectorstore: Optional[Any] = None,
    log_config: Optional[LogConfig] = LogConfig()
) -> RetrievalResults:
    """
    Run retriever optimization process.
    
    Args:
        options_config: Retrieval configuration options
        vectorstore: Optional vectorstore to use instead of loading from DocumentStore
        log_config: Optional logging configuration
    
    Returns:
        RetrievalResults containing optimization results including best_config,
        best_score, best_pipeline, and study statistics
    """
    setup_rich_logging(log_config.log_level, log_config.log_file)
    
    # Create evaluator based on config
    if options_config.evaluation_config.type == "custom":
        module_path, class_name = options_config.evaluation_config.custom_class.rsplit('.', 1)
        module = import_module(module_path)
        evaluator_class = getattr(module, class_name)
        evaluator = evaluator_class(options_config.evaluation_config)
    else:
        evaluator = RetrieverF1ScoreEvaluator(options_config.evaluation_config)

    optimizer = RetrieverOptimization(
        options_config, 
        evaluator, 
        vectorstore=vectorstore,
        show_progress_bar=log_config.show_progress_bar,
        verbose=log_config.verbose
    )
    
    return optimizer.optimize()
