import optuna
import time
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from importlib import import_module
from optuna import Trial
from ragbuilder.config import DataIngestOptionsConfig, DataIngestConfig, LogConfig
from ragbuilder.core import DBLoggerCallback, DocumentStore, ConfigStore, setup_rich_logging, console
from ragbuilder.core.results import DataIngestResults
from .pipeline import DataIngestPipeline
from .evaluation import Evaluator, SimilarityEvaluator
from langchain.docstore.document import Document
from ragbuilder.graph_utils.graph_loader import load_graph 

class DataIngestOptimizer:
    def __init__(
        self, 
        options_config: DataIngestOptionsConfig, 
        evaluator: Evaluator, 
        show_progress_bar: bool, 
        verbose: bool,
        callback=None
    ):
        self.options_config = options_config
        self.evaluator = evaluator
        self.show_progress_bar = show_progress_bar
        self.verbose = verbose
        self.callbacks = []
        self.doc_store = DocumentStore()
        self.config_store = ConfigStore()
        self.logger = logging.getLogger("ragbuilder.data_ingest.optimizer")
        # self.logger = setup_rich_logging(log_level=logging.INFO)

        # Setup DB logging callback if enabled
        if options_config.database_logging:
            try:
                db_callback = DBLoggerCallback(
                    study_name=options_config.optimization.study_name,
                    config=options_config,
                    module_type='data_ingest'
                )
                self.callbacks.append(db_callback)
                self.logger.debug("Database logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize database logging: {e}")
        
        # Add any additional callbacks
        if callback:
            self.callbacks.append(callback)

        self.document_loader_map = {i: loader for i, loader in enumerate(self.options_config.document_loaders)}
        self.chunking_strategy_map = {i: chunking_strategy for i, chunking_strategy in enumerate(self.options_config.chunking_strategies)}
        self.embedding_model_map = {i: model for i, model in enumerate(self.options_config.embedding_models)}
        self.vector_db_map = {i: db for i, db in enumerate(self.options_config.vector_databases)}

    def _build_trial_config(self, trial) -> Tuple[DataIngestConfig, List[Document]]:
        """Build config from trial parameters"""
        # Get loader configuration
        if len(self.options_config.document_loaders) == 1:
            document_loader = self.options_config.document_loaders[0]
        else:
            document_loader = self.document_loader_map[trial.suggest_categorical(
                "document_loader_index", 
                list(self.document_loader_map.keys())
            )]
        
        chunking_strategy = (self.options_config.chunking_strategies[0] if len(self.options_config.chunking_strategies) == 1
                             else self.chunking_strategy_map[trial.suggest_categorical("chunking_strategy_index", list(self.chunking_strategy_map.keys()))])
        
        chunk_size = trial.suggest_int("chunk_size", self.options_config.chunk_size.min, self.options_config.chunk_size.max, step=self.options_config.chunk_size.stepsize)

        chunk_overlap = (self.options_config.chunk_overlap[0] if len(self.options_config.chunk_overlap) == 1
                         else trial.suggest_categorical("chunk_overlap", self.options_config.chunk_overlap))

        embedding_model = (self.options_config.embedding_models[0] if len(self.options_config.embedding_models) == 1
                           else self.embedding_model_map[trial.suggest_categorical("embedding_model_index", list(self.embedding_model_map.keys()))])

        vector_database = (self.options_config.vector_databases[0] if len(self.options_config.vector_databases) == 1
                           else self.vector_db_map[trial.suggest_categorical("vector_database_index", list(self.vector_db_map.keys()))])
        
        # Handle Chroma persistence directory for trials
        if persist_directory := vector_database.vectordb_kwargs.get('persist_directory'):
            self.original_persist_directory = persist_directory if not hasattr(self, 'original_persist_directory') else self.original_persist_directory
            vector_database.vectordb_kwargs['persist_directory'] = f"{self.original_persist_directory}/{trial.number}"
        
        params = {
            "input_source": self.options_config.input_source,
            "document_loader": document_loader,
            "chunking_strategy": chunking_strategy,
            "chunk_overlap": chunk_overlap,
            "chunk_size": chunk_size,
            "embedding_model": embedding_model,
            "vector_database": vector_database,
            "sampling_rate": self.options_config.sampling_rate
        }
        
        self.logger.debug(f"Trial parameters: {params}")        
        return DataIngestConfig(**params)

    def optimize(self):
        console.rule("[heading]Starting Data Ingestion Optimization...[/heading]")
        
        def objective(trial: Trial) -> float:
            console.print(f"[heading]Trial {trial.number}/{self.options_config.optimization.n_trials - 1}[/heading]")
            config = self._build_trial_config(trial)
            trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            for t in reversed(trials_to_consider):
                if trial.params == t.params:
                    # Use the existing value as trial duplicated the parameters.
                    self.logger.info(f"Config already evaluated with score: {t.value}")
                    return t.value
            
            self.logger.debug(f"Running pipeline with config: {config}")
            pipeline = DataIngestPipeline(config, verbose=self.verbose)
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
            
            console.print(f"[success]Trial Score:[/success] [value]{avg_score:.4f}[/value]")
            return avg_score

        if self.options_config.optimization.overwrite_study and \
            self.options_config.optimization.study_name in optuna.study.get_all_study_names(storage=self.options_config.optimization.storage):
            self.logger.info(f"Overwriting existing study: {self.options_config.optimization.study_name}")
            optuna.delete_study(study_name=self.options_config.optimization.study_name, storage=self.options_config.optimization.storage)

        self.study = optuna.create_study(
            storage=self.options_config.optimization.storage,
            study_name=self.options_config.optimization.study_name,
            load_if_exists=self.options_config.optimization.load_if_exists,
            direction=self.options_config.optimization.optimization_direction,
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        self.study.optimize(
            objective, 
            n_trials=self.options_config.optimization.n_trials,
            n_jobs=self.options_config.optimization.n_jobs,
            timeout=self.options_config.optimization.timeout,
            show_progress_bar=self.show_progress_bar
        )

        console.print(f"[heading]Optimization Complete![/heading]")
        # Translate indices to actual component names
        readable_params = {}
        for param, value in self.study.best_params.items():
            if param == "chunking_strategy_index":
                readable_params["chunking_strategy"] = self.chunking_strategy_map[value].type
            elif param == "document_loader_index":
                readable_params["document_loader"] = self.document_loader_map[value].type
            elif param == "embedding_model_index":
                readable_params["embedding_model"] = self.embedding_model_map[value].type
            elif param == "vector_database_index":
                readable_params["vector_database"] = self.vector_db_map[value].type
            else:
                readable_params[param] = value
        console.print(f"[success]Best Score:[/success] [value]{self.study.best_value:.4f}[/value]\n[success]Best Parameters:[/success]\n[value]{readable_params}[/value]")
        
        best_config = DataIngestConfig(
            input_source=self.options_config.input_source,
            document_loader=self.document_loader_map[self.study.best_params["document_loader_index"]] if "document_loader_index" in self.study.best_params else self.options_config.document_loaders[0],
            chunking_strategy=self.chunking_strategy_map[self.study.best_params["chunking_strategy_index"]] if "chunking_strategy_index" in self.study.best_params else self.options_config.chunking_strategies[0],
            chunk_size=self.study.best_params["chunk_size"],
            chunk_overlap=self.study.best_params["chunk_overlap"] if "chunk_overlap" in self.study.best_params else self.options_config.chunk_overlap[0],
            embedding_model=self.embedding_model_map[self.study.best_params["embedding_model_index"]] if "embedding_model_index" in self.study.best_params else self.options_config.embedding_models[0],
            vector_database=self.vector_db_map[self.study.best_params["vector_database_index"]] if "vector_database_index" in self.study.best_params else self.options_config.vector_databases[0],
            sampling_rate=self.options_config.sampling_rate
        )
        
        # Store the best configuration
        self.config_store.store_config(
            key=f"data_ingest_best_{int(time.time())}",
            config=best_config.model_dump(),
            score=self.study.best_value,
            source_module="data_ingest",
            additional_info={
                "n_trials": self.options_config.optimization.n_trials,
                "optimization_direction": self.options_config.optimization.optimization_direction,
                "input_source": self.options_config.input_source
            }
        )
        
        return best_config, self.study.best_value

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

def run_data_ingest_optimization(
    options_config: DataIngestOptionsConfig, 
    log_config: LogConfig = LogConfig()
) -> DataIngestResults:
    """
    Run data ingestion optimization
    
    Returns:
        DataIngestResults containing optimization results including vectorstore
    """
    setup_rich_logging(log_config.log_level, log_config.log_file)

    # Create evaluator based on config
    if options_config.evaluation_config.type == "custom":
        module_path, class_name = options_config.evaluation_config.custom_class.rsplit('.', 1)
        module = import_module(module_path)
        evaluator_class = getattr(module, class_name)
        evaluator = evaluator_class(
            options_config.evaluation_config
        )
    else:
        evaluator = SimilarityEvaluator(options_config.evaluation_config)
    
    optimizer = DataIngestOptimizer(
        options_config, 
        evaluator, 
        show_progress_bar=log_config.show_progress_bar,
        verbose=log_config.verbose
    )
    
    best_config, best_score = optimizer.optimize()
    
    # Create pipeline with best config to ensure vectorstore is cached
    pipeline = DataIngestPipeline(best_config)
    best_index = pipeline.run()
    
    # Get metrics of the best trial
    best_trial_key = f"trial_{optimizer.study.best_trial.number}_results"
    trial_results = optimizer.study.user_attrs.get(best_trial_key, {})
    metrics = trial_results.get("metrics", {})
    
    # Create results object with all fields
    results = DataIngestResults(
        best_config=best_config,
        best_score=best_score,
        best_pipeline=pipeline,
        best_index=best_index, 
        n_trials=options_config.optimization.n_trials,
        completed_trials=len(optimizer.study.trials),
        optimization_time=(optimizer.study.trials[-1].datetime_complete - optimizer.study.trials[0].datetime_start).total_seconds(),
        avg_latency=metrics.get("avg_latency"),
        error_rate=metrics.get("error_rate")
    )
    
    # Store the best pipeline in ConfigStore
    optimizer.config_store.store_best_data_ingest_pipeline(pipeline)
    
    # Set the best config key in DocumentStore
    optimizer.doc_store.set_best_config_key(pipeline.loader_key, pipeline.config_key)

    # Load graph, if enabled
    if options_config.graph:
        # TODO: Check & validate graph config 
        console.print("[status]Loading graph...[/status]")
        doc_loader = options_config.graph.document_loaders if options_config.graph.document_loaders else best_config.document_loader
        chunking_strategy = options_config.graph.chunking_strategy if options_config.graph.chunking_strategy else best_config.chunking_strategy
        chunk_size = options_config.graph.chunk_size if options_config.graph.chunk_size else (best_config.chunk_size or 3000)
        chunk_overlap = options_config.graph.chunk_overlap if options_config.graph.chunk_overlap else (best_config.chunk_overlap or 100)
        embedding_model = options_config.graph.embedding_model if options_config.graph.embedding_model else best_config.embedding_model
        vector_database = best_config.vector_database
        
        config = DataIngestConfig(
            input_source=options_config.input_source,
            document_loader=doc_loader,
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            vector_database=vector_database
        )
        pipeline = DataIngestPipeline(config, verbose=log_config.verbose)
        chunks = pipeline.ingest()
        
        if not options_config.graph.llm:
            options_config.graph.llm = ConfigStore().get_default_llm()

        llm_config = options_config.graph.llm
        llm = llm_config.llm
        
        graph = load_graph(chunks, pipeline.embedder, llm)
        optimizer.doc_store.store_graph(graph)

    console.print("[success]✓ Successfully optimized and cached best configuration[/success]")
    
    return results

def optimize_data_ingest_from_yaml(options_config_path: str):
    options_config = DataIngestOptionsConfig.from_yaml(options_config_path)
    return run_data_ingest_optimization(options_config)

def optimize_data_ingest_from_dict(options_config_dict: dict):
    options_config = DataIngestOptionsConfig(**options_config_dict)
    return run_data_ingest_optimization(options_config)