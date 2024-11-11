import optuna
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from .config import DataIngestOptionsConfig, DataIngestConfig, LogConfig, EvaluatorType
from .pipeline import DataIngestPipeline
from .evaluation import Evaluator, SimilarityEvaluator
from .callbacks import DBLoggerCallback
from tqdm.notebook import tqdm    
import numpy as np
from importlib import import_module
from .utils import load_environment
from .components import validate_environment

class Optimizer:
    def __init__(self, options_config: DataIngestOptionsConfig, evaluator: Evaluator, callback=None):
        self.options_config = options_config
        self.evaluator = evaluator
        self.callbacks = []
        self._setup_logging(options_config.log_config)

        # Setup DB logging callback if enabled
        if options_config.database_logging:
            try:
                db_callback = DBLoggerCallback(
                    study_name=options_config.optimization.study_name,
                    config=options_config
                )
                self.callbacks.append(db_callback)
                self.logger.info("Database logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize database logging: {e}")
        
        # Add any additional callbacks
        if callback:
            self.callbacks.append(callback)

        self.document_loader_map = {i: loader for i, loader in enumerate(self.options_config.document_loaders)}
        self.chunking_strategy_map = {i: chunking_strategy for i, chunking_strategy in enumerate(self.options_config.chunking_strategies)}
        self.embedding_model_map = {i: model for i, model in enumerate(self.options_config.embedding_models)}
        self.vector_db_map = {i: db for i, db in enumerate(self.options_config.vector_databases)}
        self._setup_logging(options_config.log_config)

    def _setup_logging(self, log_config: LogConfig):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_config.log_level)

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler with formatter
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_config.log_file:
            file_handler = logging.FileHandler(log_config.log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def optimize(self):
        self.logger.info("Starting optimization process")
        def objective(trial):
            # TODO: Handle duplicate trials
            self.logger.info(f"Starting trial {trial.number + 1}/{self.options_config.optimization.n_trials}")
            config = self._build_trial_config(trial)
            trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    # Use the existing value as trial duplicated the parameters.
                    self.logger.info(f"Config already evaluated with score: {t.value}: {config}")
                    return t.value
            
            self.logger.debug(f"Running pipeline with config: {config}")
            pipeline = DataIngestPipeline(config)
            pipeline.run()
            
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
            
            return avg_score

        if self.options_config.optimization.overwrite_study and \
            self.options_config.optimization.study_name in optuna.study.get_all_study_names(storage=self.options_config.optimization.storage):
            self.logger.info(f"Overwriting existing study: {self.options_config.optimization.study_name}")
            optuna.delete_study(study_name=self.options_config.optimization.study_name, storage=self.options_config.optimization.storage)

        study = optuna.create_study(
            storage=self.options_config.optimization.storage,
            study_name=self.options_config.optimization.study_name,
            load_if_exists=self.options_config.optimization.load_if_exists,
            direction=self.options_config.optimization.optimization_direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(
            objective, 
            n_trials=self.options_config.optimization.n_trials,
            n_jobs=self.options_config.optimization.n_jobs,
            timeout=self.options_config.optimization.timeout,
            show_progress_bar=self.options_config.log_config.show_progress_bar
        )

        self.logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")
        
        best_config = DataIngestConfig(
            input_source=self.options_config.input_source,
            test_dataset=self.options_config.test_dataset,
            chunking_strategy=self.chunking_strategy_map[study.best_params["chunking_strategy_id"]] if "chunking_strategy_id" in study.best_params else self.options_config.chunking_strategies[0],
            chunk_size=study.best_params["chunk_size"],
            chunk_overlap=study.best_params["chunk_overlap"] if "chunk_overlap" in study.best_params else self.options_config.chunk_overlap[0],
            embedding_model=self.embedding_model_map[study.best_params["embedding_model_id"]] if "embedding_model_id" in study.best_params else self.options_config.embedding_models[0],
            vector_database=self.vector_db_map[study.best_params["vector_database_id"]] if "vector_database_id" in study.best_params else self.options_config.vector_databases[0],
            sampling_rate=self.options_config.sampling_rate
        )
        return best_config, study.best_value

    def _build_trial_config(self, trial) -> DataIngestConfig:
        """Build config from trial parameters"""
        if len(self.options_config.document_loaders) == 1:
            document_loader = self.options_config.document_loaders[0]
        else:
            document_loader = self.document_loader_map[trial.suggest_categorical("document_loader_index", list(self.document_loader_map.keys()))]
        
        if len(self.options_config.chunking_strategies) == 1:
            chunking_strategy = self.options_config.chunking_strategies[0]
        else:
            chunking_strategy = self.chunking_strategy_map[trial.suggest_categorical("chunking_strategy_index", list(self.chunking_strategy_map.keys()))]
        
        chunk_size = trial.suggest_int("chunk_size", self.options_config.chunk_size.min, self.options_config.chunk_size.max, step=self.options_config.chunk_size.stepsize)

        if len(self.options_config.chunk_overlap) == 1:
            chunk_overlap = self.options_config.chunk_overlap[0]
        else:
            chunk_overlap = trial.suggest_categorical("chunk_overlap", self.options_config.chunk_overlap)

        if len(self.options_config.embedding_models) == 1:  
            embedding_model = self.options_config.embedding_models[0]
        else:
            embedding_model = self.embedding_model_map[trial.suggest_categorical("embedding_model_index", list(self.embedding_model_map.keys()))]

        if len(self.options_config.vector_databases) == 1:
            vector_database = self.options_config.vector_databases[0]
        else:
            vector_database = self.vector_db_map[trial.suggest_categorical("vector_database_index", list(self.vector_db_map.keys()))]
        
        # Handle Chroma persistence directory for trials
        if persist_directory := vector_database.vectordb_kwargs.get('persist_directory'):
            self.original_persist_directory = persist_directory if not hasattr(self, 'original_persist_directory') else self.original_persist_directory
            vector_database.vectordb_kwargs['persist_directory'] = f"{self.original_persist_directory}/{trial.number}"
        
        params = {
            "input_source": self.options_config.input_source,
            "test_dataset": self.options_config.test_dataset,
            "document_loader": document_loader,
            "chunking_strategy": chunking_strategy,
            "chunk_overlap": chunk_overlap,
            "chunk_size": chunk_size,
            "embedding_model": embedding_model,
            "vector_database": vector_database,
            "sampling_rate": self.options_config.sampling_rate,
        }
        self.logger.info(f"Trial parameters: {params}")
        return DataIngestConfig(**params)

    def _calculate_aggregate_metrics(self, question_details: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics from detailed results"""
        successful_evals = [d for d in question_details if "error" not in d]
        
        if not successful_evals:
            return {
                "avg_latency": None,
                "total_questions": len(question_details),
                "successful_questions": 0,
                "error_rate": 1.0
            }
        
        return {
            "avg_latency": np.mean([d["latency"] for d in successful_evals]),
            "total_questions": len(question_details),
            "successful_questions": len(successful_evals),
            "error_rate": 1 - (len(successful_evals) / len(question_details))
        }

def _run_optimization_core(options_config: DataIngestOptionsConfig):
    # Load environment variables first
    load_environment()
    
    # Validate environment variables for selected components
    missing_vars = []

    # Check graph
    if options_config.graph:
        # TODO: Check graph config and create graph
        pass
    
    # Check document loaders
    for loader in options_config.document_loaders:
        if missing := validate_environment('loader', loader.type):
            missing_vars.extend(missing)
    
    # Check embedding models
    for model in options_config.embedding_models:
        if missing := validate_environment('embedding', model.type):
            missing_vars.extend(missing)
    
    # Check vector databases
    for db in options_config.vector_databases:
        if missing := validate_environment('vectordb', db.type):
            missing_vars.extend(missing)
    
    if missing_vars:
        missing_vars = sorted(set(missing_vars))  # Remove duplicates
        raise ValueError(
            "Missing required environment variables for selected components:\n" + 
            "\n".join(f"- {var}" for var in missing_vars)
        )
    
    # Create evaluator based on config
    if options_config.evaluation_config.type == EvaluatorType.CUSTOM:
        module_path, class_name = options_config.evaluation_config.custom_class.rsplit('.', 1)
        module = import_module(module_path)
        evaluator_class = getattr(module, class_name)
        evaluator = evaluator_class(
            options_config.test_dataset,
            options_config.evaluation_config
        )
    else:
        evaluator = SimilarityEvaluator(options_config.test_dataset, options_config.evaluation_config)
    
    optimizer = Optimizer(options_config, evaluator)
    best_config, best_score = optimizer.optimize()
    
    # TODO: Revisit this to use cache to avoid running the pipeline twice
    best_pipeline = DataIngestPipeline(best_config)
    best_index = best_pipeline.run()
    
    return best_config, best_score, best_index

def run_optimization(options_config_path: str):
    options_config = DataIngestOptionsConfig.from_yaml(options_config_path)
    return _run_optimization_core(options_config)

def run_optimization_from_dict(options_config_dict: dict):
    options_config = DataIngestOptionsConfig(**options_config_dict)
    return _run_optimization_core(options_config)