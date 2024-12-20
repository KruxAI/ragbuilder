from typing import Optional, Any, Dict, Union
from ragbuilder.config.data_ingest import DataIngestOptionsConfig 
from ragbuilder.config.retriever import RetrievalOptionsConfig
from ragbuilder.config.generator import GenerationOptionsConfig

from ragbuilder.config.base import LogConfig
from ragbuilder.data_ingest.optimization import run_data_ingest_optimization
from ragbuilder.retriever.optimization import run_retrieval_optimization
from src.ragbuilder.generation.optimization import run_generation_optimization
from ragbuilder.generate_data import TestDatasetManager
from ragbuilder.core.logging_utils import setup_rich_logging, console
from .exceptions import DependencyError
import logging
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# TODO: Bring generation entry point here
# - Add optimize_generation method
# - Handle retriever as a argument vs using optimized retriever
# - Save the results
# TODO: Return consistent results across optimize methods - ideally dict of whatever relevant
class RAGBuilder:
    def __init__(
            self, 
            data_ingest_config: Optional[DataIngestOptionsConfig] = None,
            retrieval_config: Optional[RetrievalOptionsConfig] = None,
            generation_config: Optional[GenerationOptionsConfig] = None,
            log_config: Optional[LogConfig] = None
        ):
        self._log_config = log_config or LogConfig()
        self._data_ingest_config = data_ingest_config
        self._retrieval_config = retrieval_config
        self._generation_config = generation_config
        self.logger = setup_rich_logging(
            self._log_config.log_level,
            self._log_config.log_file
        )
        self._optimized_store = None
        self._optimized_retriever = None
        self._optimized_generation = None
        self._optimization_results = {
            "data_ingest": None,
            "retrieval": None,
            "generation": None
        }
        self._test_dataset_manager = TestDatasetManager(self._log_config)

    @classmethod
    def from_source_with_defaults(cls, 
                         input_source: str, 
                         test_dataset: Optional[str] = None,
                         log_config: Optional[LogConfig] = None
                         ) -> 'RAGBuilder':
        config = DataIngestOptionsConfig.with_defaults(
            input_source=input_source,
            test_dataset=test_dataset
        )
        return cls(data_ingest_config=config, log_config=log_config)
    
    @classmethod
    def generation_with_defaults(cls, 
                         input_source: str, 
                         eval_data_set_path: str,
                         retriever: Optional[Any] = None,
                         llm: Optional[Any] = None,
                         model_kwargs: Optional[Any] = None,
                         local_prompt_template_path: Optional[Any] = None,
                         read_local_only: Optional[Any] = None,
                         log_config: Optional[LogConfig] = None
                         ) -> 'RAGBuilder':
        config = GenerationOptionsConfig.with_defaults(
            input_source=input_source,
            eval_data_set_path=eval_data_set_path,
            retriever=retriever,
            llm=llm,
            model_kwargs=model_kwargs,
            local_prompt_template_path=local_prompt_template_path,
            read_local_only=read_local_only
        )
        return cls(generation_config=config, log_config=log_config)
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'RAGBuilder':
        """Create RAGBuilder from YAML config"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        builder = cls()
        if not any(key in config_dict for key in ['data_ingest', 'retrieval', 'generation']):
            raise ValueError("YAML must contain at least 'data_ingest', 'retrieval' or 'generation' configuration")
        
        if 'data_ingest' in config_dict:
            builder._data_ingest_config = DataIngestOptionsConfig(**config_dict['data_ingest'])
        
        # TODO: Handle vectorstore provided by user instead of using the one from data_ingest
        if 'retrieval' in config_dict:  
            builder._retrieval_config = RetrievalOptionsConfig(**config_dict['retrieval'])

        if 'generation' in config_dict:
            builder._generation_config = GenerationOptionsConfig(**config_dict['generation'])
            print("generation_config",GenerationOptionsConfig(**config_dict['generation']))
        return builder

    def _ensure_eval_dataset(self, config: Union[DataIngestOptionsConfig, RetrievalOptionsConfig, GenerationOptionsConfig]) -> None:
            """Ensure config has a test dataset, generating one if needed"""
            if (hasattr(config, 'evaluation_config') and config.evaluation_config.test_dataset) or (hasattr(config, 'eval_data_set_path') and config.eval_data_set_path):
            # if config.evaluation_config.test_dataset or hasattr(config, 'eval_data_set_path'):
                # self.logger.info(f"Using provided test dataset: {config.evaluation_config.test_dataset or config.eval_data_set_path}")
                return
            
            # Check if we already have a test dataset from data ingestion
            if (self._data_ingest_config and self._data_ingest_config.evaluation_config.test_dataset):
                if hasattr(config, 'evaluation_config'):
                    config.evaluation_config.test_dataset = self._data_ingest_config.evaluation_config.test_dataset
                else:
                    config.eval_data_set_path = self._data_ingest_config.evaluation_config.test_dataset
                    self.logger.info(f"Reusing test dataset from data ingestion: {config.eval_data_set_path}")
                return
            
            elif (self._retrieval_config and self._retrieval_config.evaluation_config.test_dataset):
                config.eval_data_set_path = self._retrieval_config.evaluation_config.test_dataset
                self.logger.info(f"Reusing test dataset from retrieval: {config.eval_data_set_path}")
                return
            
            if not hasattr(config, 'input_source'):
                raise ValueError("input_source is required when test_dataset is not provided")
            
            source_data = (getattr(config, 'input_source', None) or 
                        (self._data_ingest_config.input_source if self._data_ingest_config else None))
            
            if not source_data:
                raise ValueError("input_source is required when test_dataset is not provided")
                
            with console.status("Generating eval dataset..."):
                test_dataset = self._test_dataset_manager.get_or_generate_dataset(
                    source_data=source_data
                )
            if hasattr(config, 'evaluation_config'):
                config.evaluation_config.test_dataset = test_dataset
            else:
                config.eval_data_set_path = test_dataset
            self.logger.info(f"Eval dataset: {test_dataset}")


    def optimize_data_ingest(self, config: Optional[DataIngestOptionsConfig] = None) -> Dict[str, Any]:
        """
        Run data ingestion optimization
        
        Returns:
            Dict containing optimization results including best_config, best_score,
            best_index, best_pipeline, and study_statistics
        """
        if config:
            self._data_ingest_config = config

        self._ensure_eval_dataset(self._data_ingest_config)
        self.logger.info("Starting data ingestion optimization")
        
        results = run_data_ingest_optimization(
            self._data_ingest_config, 
            log_config=self._log_config
        )
        
        self._optimization_results["data_ingest"] = results
        self._optimized_store = results["best_index"]
        
        return results
    
    def optimize_retrieval(
        self, 
        config: Optional[RetrievalOptionsConfig] = None, 
        vectorstore: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run retrieval optimization
        
        Returns:
            Dict containing optimization results including best_config, best_score,
            best_pipeline, and study_statistics
        """
        if vectorstore:
            self._optimized_store = vectorstore
        elif self._optimized_store is None:
            raise DependencyError("No vectorstore found. Run data ingestion first or provide existing vectorstore.")

        if config:
            self._retrieval_config = config
        elif not self._retrieval_config:
            self._retrieval_config = RetrievalOptionsConfig.with_defaults()
            
        self._ensure_eval_dataset(self._retrieval_config)
            
        results = run_retrieval_optimization(
            self._retrieval_config, 
            vectorstore=self._optimized_store,
            log_config=self._log_config
        )
        
        self._optimization_results["retrieval"] = results
        self._optimized_retriever = results["best_pipeline"].retriever_chain
        
        return results
    
    def optimize_generation(
        self, 
        config: Optional[GenerationOptionsConfig] = None, 
        retriever: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run Generation optimization
        
        Returns:
            Dict containing optimization results including best_config, best_score,
            best_pipeline, and study_statistics
        """
        print("self._generation_config",self._generation_config)
        if retriever:
            self._optimized_retriever = retriever
        elif self._optimized_retriever is None:
            raise DependencyError("No vectorstore found. Run data ingestion first or provide existing vectorstore.")
        if config:
            self._generation_config = config
        elif not self._generation_config:
            self._generation_config = GenerationOptionsConfig.with_defaults()
        
        self._ensure_eval_dataset(self._generation_config)

        results = run_generation_optimization(
            self._generation_config, 
            retriever=self._optimized_retriever
        )
        
        # Store results for later use
        self._optimization_results["generation"] = results
        self._optimized_generation = results["best_pipeline"]
        
        return results
    
    def optimize(self) -> Dict[str, Dict[str, Any]]:
        """
        Run end-to-end optimization for both data ingestion and retrieval
        
        Returns:
            Dict containing results for both data ingestion and retrieval optimizations
        """
        data_ingest_results = self.optimize_data_ingest()
        retrieval_results = self.optimize_retrieval()
        generation_results = self.optimize_generation()
        
        self._optimization_results = {
            "data_ingest": data_ingest_results,
            "retrieval": retrieval_results,
            "generation": generation_results
        }

        return self._optimization_results
    
    @property
    def optimization_results(self) -> Dict[str, Dict[str, Any]]:
        """Access the latest optimization results"""
        return self._optimization_results

    def get_best_pipeline(self, module: str = "retrieval") -> Optional[Any]:
        """
        Get the best pipeline from optimization results
        
        Args:
            module: Either "data_ingest" or "retrieval"
        
        Returns:
            The best pipeline if optimization has been run, None otherwise
        """
        if not self._optimization_results[module]:
            return None
        return self._optimization_results[module]["best_pipeline"]

    def get_configs(self) -> Dict[str, Any]:
        """Get current configurations"""
        configs = {}
        if self._data_ingest_config:
            configs['data_ingest'] = self._data_ingest_config.model_dump()
        if self._retrieval_config:
            configs['retrieval'] = self._retrieval_config.model_dump()
        return configs

    def save_configs(self, file_path: str) -> None:
        """Save current configurations to YAML"""
        configs = self.get_configs()
        with open(file_path, 'w') as f:
            yaml.dump(configs, f)

    def serve(self, host: str = "0.0.0.0", port: int = 8005):
        """
        Launch a FastAPI server to serve RAG queries
        
        Args:
            host: Host address to bind the server to
            port: Port number to listen on
        """
        if not self._optimization_results.get("generation"):
            raise DependencyError("No generation pipeline found. Run generation optimization first.")
            
        app = FastAPI(title="RAGBuilder API")
        
        @app.post("/invoke")
        async def invoke(request: QueryRequest) -> Dict[str, Any]:
            try:
                result = self._optimization_results["generation"]["best_pipeline"].invoke(
                    request.query
                )
                console.print(f"Question:{request.query}")
                console.print(f"Response:{result}")
                return {"response": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        self.logger.info(f"Starting RAG server on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)

class QueryRequest(BaseModel):
    query: str
    question: Optional[str] = None

    def get_query(self) -> str:
        """Return either query or question field"""
        return self.query or self.question or ""
