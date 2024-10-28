import optuna
import logging
from dataclasses import dataclass
from typing import Optional
from .config import DataIngestOptionsConfig, DataIngestConfig, LogConfig
from .pipeline import DataIngestPipeline
from .evaluation import Evaluator, SimilarityEvaluator
from tqdm.notebook import tqdm    

class Optimizer:
    def __init__(self, options_config: DataIngestOptionsConfig, evaluator: Evaluator):
        self.options_config = options_config
        self.evaluator = evaluator
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
            self.logger.info(f"Starting trial {trial.number + 1}/{self.options_config.optimization.n_trials}")

            if len(self.options_config.chunking_strategies) == 1:
                chunking_strategy = self.options_config.chunking_strategies[0]
            else:
                chunking_strategy = trial.suggest_categorical("chunking_strategy", self.options_config.chunking_strategies)
            
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
            
            # Avoid the InvalidDimensionException by persisting to a unique directory for each trial
            if vector_database.persist_directory:
                self.original_persist_directory = vector_database.persist_directory if not hasattr(self, 'original_persist_directory') else self.original_persist_directory
                vector_database.persist_directory = f"{self.original_persist_directory}/{trial.number}"
            
            params = {
                "input_source": self.options_config.input_source,
                "test_dataset": self.options_config.test_dataset,
                "chunking_strategy": chunking_strategy,
                "chunk_overlap": chunk_overlap,
                "chunk_size": chunk_size,
                "embedding_model": embedding_model,
                "vector_database": vector_database,
                "top_k": self.options_config.top_k,
                "sampling_rate": self.options_config.sampling_rate,
                "custom_chunker": self.options_config.custom_chunker if hasattr(self.options_config, 'custom_chunker') else None
            }
            self.logger.info(f"Trial parameters: {params}")

            config = DataIngestConfig(**params)
            self.logger.debug(f"Running pipeline with config: {config}")

            pipeline = DataIngestPipeline(config)
            index = pipeline.run()
            
            score = self.evaluator.evaluate(pipeline)

            return score

        study = optuna.create_study(
            storage=self.options_config.optimization.storage,
            study_name=self.options_config.optimization.study_name,
            load_if_exists=self.options_config.optimization.load_if_exists,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        study.optimize(
            objective, 
            n_trials=self.options_config.optimization.n_trials,
            n_jobs=self.options_config.optimization.n_jobs,
            timeout=self.options_config.optimization.timeout,
            show_progress_bar=True
        )

        self.logger.info(f"Optimization completed. Best score: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {study.best_params}")
        
        best_config = DataIngestConfig(
            input_source=self.options_config.input_source,
            test_dataset=self.options_config.test_dataset,
            chunking_strategy=study.best_params["chunking_strategy"] if "chunking_strategy" in study.best_params else self.options_config.chunking_strategies[0],
            chunk_size=study.best_params["chunk_size"],
            chunk_overlap=study.best_params["chunk_overlap"] if "chunk_overlap" in study.best_params else self.options_config.chunk_overlap[0],
            embedding_model=self.embedding_model_map[study.best_params["embedding_model_index"]] if "embedding_model_index" in study.best_params else self.options_config.embedding_models[0],
            vector_database=self.vector_db_map[study.best_params["vector_database_index"]] if "vector_database_index" in study.best_params else self.options_config.vector_databases[0],
            top_k=self.options_config.top_k,
            sampling_rate=self.options_config.sampling_rate
        )
        return best_config, study.best_value

def _run_optimization_core(options_config: DataIngestOptionsConfig):
    evaluator = SimilarityEvaluator(options_config.test_dataset, options_config.top_k)
    optimizer = Optimizer(options_config, evaluator)
    best_config, best_score = optimizer.optimize()
    
    best_pipeline = DataIngestPipeline(best_config)
    best_index = best_pipeline.run()
    
    return best_config, best_score, best_index

def run_optimization(options_config_path: str):
    options_config = DataIngestOptionsConfig.from_yaml(options_config_path)
    return _run_optimization_core(options_config)

def run_optimization_from_dict(options_config_dict: dict):
    options_config = DataIngestOptionsConfig(**options_config_dict)
    return _run_optimization_core(options_config)