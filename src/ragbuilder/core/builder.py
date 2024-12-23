from typing import Optional, Any, Dict, Union
from ragbuilder.config.data_ingest import DataIngestOptionsConfig 
from ragbuilder.config.retriever import RetrievalOptionsConfig
from ragbuilder.config.generator import GenerationOptionsConfig

from ragbuilder.config.base import LogConfig
from ragbuilder.data_ingest.optimization import run_data_ingest_optimization
from ragbuilder.retriever.optimization import run_retrieval_optimization
from ragbuilder.generation.optimization import run_generation_optimization
from ragbuilder.generate_data import TestDatasetManager
from ragbuilder.core.logging_utils import setup_rich_logging, console
from ragbuilder.core.telemetry import telemetry
from ragbuilder.core.utils import validate_environment, serialize_config, SimpleConfigEncoder
from .exceptions import DependencyError
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
from pathlib import Path
import json
import shutil
from datetime import datetime, timedelta

DEFAULT_DB_PATH = "eval.db"

# TODO: Return consistent results across optimize methods - ideally create a results object
class RAGBuilder:
    SUPPORTED_VECTORSTORES: Dict[str, str] = {
        "CHROMA": "chroma",
        "FAISS": "faiss"
    }
    
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
        self._test_dataset_manager = TestDatasetManager(
            self._log_config,
            db_path=self._data_ingest_config.database_path if self._data_ingest_config else DEFAULT_DB_PATH
        )

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
                    self.logger.debug(f"Reusing test dataset from data ingestion: {config.eval_data_set_path}")
                return
            
            elif (self._retrieval_config and self._retrieval_config.evaluation_config.test_dataset):
                config.eval_data_set_path = self._retrieval_config.evaluation_config.test_dataset
                self.logger.debug(f"Reusing test dataset from retrieval: {config.eval_data_set_path}")
                return
            
            if not hasattr(config, 'input_source'):
                raise ValueError("input_source is required when test_dataset is not provided")
            
            source_data = (getattr(config, 'input_source', None) or 
                        (self._data_ingest_config.input_source if self._data_ingest_config else None))
            
            if not source_data:
                raise ValueError("input_source is required when test_dataset is not provided")
                
            with console.status("Generating eval dataset..."):
                test_dataset = self._test_dataset_manager.get_or_generate_eval_dataset(
                    source_data=source_data,
                    eval_data_generation_config=config.evaluation_config.eval_data_generation_config if hasattr(config, 'evaluation_config') else None,
                    sampling_rate=config.sampling_rate if hasattr(config, 'sampling_rate') else None
                )
            if hasattr(config, 'evaluation_config'):
                config.evaluation_config.test_dataset = test_dataset
            else:
                config.eval_data_set_path = test_dataset
            self.logger.debug(f"Eval dataset: {test_dataset}")


    def optimize_data_ingest(
        self, 
        config: Optional[DataIngestOptionsConfig] = None,
        validate_env: bool = True
    ) -> Dict[str, Any]:
        """
        Run data ingestion optimization
        
        Returns:
            Dict containing optimization results including best_config, best_score,
            best_index, best_pipeline, and study_statistics
        """
        if config:
            self._data_ingest_config = config
        elif not self._data_ingest_config:
            raise ValueError("No data ingestion configuration provided")
        
        if validate_env:
            validate_environment(self._data_ingest_config)

        self._ensure_eval_dataset(self._data_ingest_config)
        
        with telemetry.optimization_span("data_ingest", self._data_ingest_config.model_dump()) as span:
            try:
                results = run_data_ingest_optimization(
                    self._data_ingest_config, 
                    log_config=self._log_config
                )
                
                # Store results and update telemetry
                self._optimization_results["data_ingest"] = results
                self._optimized_store = results["best_index"]
                telemetry.update_optimization_results(span, results, "data_ingest")                
                return results
                
            except Exception as e:
                telemetry.track_error(
                    "data_ingest",
                    e,
                    context={
                        "config_type": "default" if not config else "custom",
                    }
                )
                raise
            finally:
                telemetry.flush()
        

    def optimize_retrieval(
        self, 
        config: Optional[RetrievalOptionsConfig] = None, 
        vectorstore: Optional[Any] = None,
        validate_env: bool = True
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

        if validate_env:
            validate_environment(self._retrieval_config)
            
        self._ensure_eval_dataset(self._retrieval_config)
        
        with telemetry.optimization_span("retriever", self._retrieval_config.model_dump()) as span:
            try:
                results = run_retrieval_optimization(
                    self._retrieval_config, 
                    vectorstore=self._optimized_store,
                    log_config=self._log_config
                )
                
                # Store results and update telemetry
                self._optimization_results["retrieval"] = results
                self._optimized_retriever = results["best_pipeline"].retriever_chain
                telemetry.update_optimization_results(span, results, "retriever")
                return results
                
            except Exception as e:
                telemetry.track_error(
                    "retriever",
                    e,
                    context={
                        "config_type": "default" if not config else "custom",   
                        "vectorstore_provided": vectorstore is not None                     
                    }
                )
                raise
            finally:
                telemetry.flush()

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
        if retriever:
            self._optimized_retriever = retriever
        elif self._optimized_retriever is None:
            raise DependencyError("No retriever found. Run retrieval optimization first or provide existing retriever.")
        
        if config:
            self._generation_config = config
        elif not self._generation_config:
            self._generation_config = GenerationOptionsConfig.with_defaults()
        
        self._ensure_eval_dataset(self._generation_config)
        
        with telemetry.optimization_span("generator", self._generation_config.model_dump()) as span:
            try:
                results = run_generation_optimization(
                    self._generation_config, 
                    retriever=self._optimized_retriever,
                    log_config=self._log_config
                )
                
                # Store results and update telemetry
                self._optimization_results["generation"] = results
                self._optimized_generation = results["best_pipeline"]
                telemetry.update_optimization_results(span, results, "generator")
                return results
                
            except Exception as e:
                telemetry.track_error(
                    "generator",
                    e,
                    context={
                        "config_type": "default" if not config else "custom",
                        "retriever_provided": retriever is not None
                    }
                )
                raise
            finally:
                telemetry.flush()

    def optimize(self) -> Dict[str, Dict[str, Any]]:
        """
        Run end-to-end optimization for both data ingestion and retrieval
        
        Returns:
            Dict containing results for both data ingestion and retrieval optimizations
        """
        with telemetry.optimization_span("ragbuilder", {"end_to_end": True}) as span:
            try:
                with console.status("[bold green]Validating data ingestion environment...") as status:
                    validate_environment(self._data_ingest_config)
                    status.update("[bold green]Validating retrieval environment...")
                    if not self._retrieval_config:
                        self._retrieval_config = RetrievalOptionsConfig.with_defaults()
                    validate_environment(self._retrieval_config)
                
                data_ingest_results = self.optimize_data_ingest(validate_env=False)
                retrieval_results = self.optimize_retrieval(validate_env=False)
                generation_results = self.optimize_generation()
                
                self._optimization_results = {
                    "data_ingest": data_ingest_results,
                    "retrieval": retrieval_results,
                    "generation": generation_results
                }
                span.set_attribute("data_ingest_score", data_ingest_results.get("best_score", 0))
                span.set_attribute("retrieval_score", retrieval_results.get("best_score", 0))
                span.set_attribute("generation_score", generation_results.get("best_score", 0))

                return self._optimization_results
                
            except Exception as e:
                telemetry.track_error(
                    "ragbuilder",
                    e,
                    context={
                        "completed_modules": [k for k, v in self._optimization_results.items() if v is not None]
                    }
                )
                raise
            finally:
                telemetry.flush()

    def __del__(self):
        if telemetry:
            try:
                telemetry.shutdown()
            except Exception as e:
                self.logger.debug(f"Error shutting down telemetry: {e}")
    
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
        asyncio.run(uvicorn.run(app, host=host, port=port))

    def save(self, path: str, include_vectorstore: bool = True) -> None:
        """Save complete RAG setup including vectorstore if specified."""
        print("WARNING: This is an experimental feature and may not work as expected. ⚠️ Use with caution!")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (save_path / "configs").mkdir(exist_ok=True)
        (save_path / "metadata").mkdir(exist_ok=True)
        if include_vectorstore:
            (save_path / "vectorstore").mkdir(exist_ok=True)

        # Save configurations using existing serialization
        if self._data_ingest_config:
            with open(save_path / "configs" / "data_ingest.yaml", "w") as f:
                yaml.dump(json.loads(serialize_config(self._data_ingest_config)), f)
        
        if self._retrieval_config:
            with open(save_path / "configs" / "retriever.yaml", "w") as f:
                yaml.dump(json.loads(serialize_config(self._retrieval_config)), f)
            
        if self._generation_config:
            with open(save_path / "configs" / "generator.yaml", "w") as f:
                yaml.dump(json.loads(serialize_config(self._generation_config)), f)

        # Save optimization results
        if self._optimization_results:
            # Define non-serializable types to skip or handle specially
            non_serializable_types = (
                'Chroma', 'FAISS',  # Vector stores
                'RetrieverPipeline', 'DataIngestPipeline',  # Pipelines
                'Embeddings', 'BaseRetriever'  # Base classes
            )
            
            serialized_results = {}
            for module, results in self._optimization_results.items():
                if results:
                    serialized_results[module] = {}
                    for key, value in results.items():
                        # Skip pipeline objects and other non-serializable types
                        if key not in ["best_score", "best_config"]:
                            continue
                        
                        # Try to serialize if it's a config object
                        try:
                            if hasattr(value, 'model_dump'):
                                serialized_results[module][key] = json.loads(serialize_config(value))
                            else:
                                serialized_results[module][key] = value
                        except (TypeError, ValueError) as e:
                            self.logger.warning(f"Skipping non-serializable value for {key}: {str(e)}")
                            continue
                else:
                    serialized_results[module] = None

            with open(save_path / "metadata" / "optimization_results.json", "w") as f:
                json.dump(serialized_results, f, indent=2, cls=SimpleConfigEncoder)

        # Save vectorstore if requested
        if include_vectorstore and self._optimized_store:
            vectorstore_type = self._get_vectorstore_type()
            if vectorstore_type:
                vectorstore_path = save_path / "vectorstore" / vectorstore_type
                self._save_vectorstore(vectorstore_path)

        # Create manifest
        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "components": {
                "data_ingest": bool(self._data_ingest_config),
                "retriever": bool(self._retrieval_config),
                "generator": bool(self._generation_config)
            },
            "vectorstore": {
                "included": include_vectorstore,
                "type": self._get_vectorstore_type() if self._optimized_store else None
            }
        }
        
        with open(save_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Successfully saved RAG setup to {path}")

    @classmethod
    def load(cls, path: str) -> 'RAGBuilder':
        """Load complete RAG setup from saved project.
        
        Args:
            path: Directory path containing the saved RAG setup
            
        Returns:
            Initialized RAGBuilder instance
        """
        print("WARNING: This is an experimental feature and may not work as expected. ⚠️ Use with caution!")
        load_path = Path(path)
        if not load_path.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Load manifest
        try:
            with open(load_path / "manifest.json", "r") as f:
                manifest = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Invalid project directory: manifest.json not found in {path}")

        # Initialize configs
        data_ingest_config = None
        retrieval_config = None
        generation_config = None
        
        # Load configurations if they exist
        if manifest["components"]["data_ingest"]:
            with open(load_path / "configs" / "data_ingest.yaml", "r") as f:
                data_ingest_config = DataIngestOptionsConfig(**yaml.safe_load(f))
                
        if manifest["components"]["retriever"]:
            with open(load_path / "configs" / "retriever.yaml", "r") as f:
                retrieval_config = RetrievalOptionsConfig(**yaml.safe_load(f))
                
        if manifest["components"]["generator"]:
            with open(load_path / "configs" / "generator.yaml", "r") as f:
                generation_config = GenerationOptionsConfig(**yaml.safe_load(f))

        # Create builder instance
        builder = cls(
            data_ingest_config=data_ingest_config,
            retrieval_config=retrieval_config,
            generation_config=generation_config
        )

        # Load optimization results if they exist
        if (load_path / "metadata" / "optimization_results.json").exists():
            with open(load_path / "metadata" / "optimization_results.json", "r") as f:
                builder._optimization_results = json.load(f)

        # Load vectorstore if it exists
        if manifest["vectorstore"]["included"] and manifest["vectorstore"]["type"]:
            vectorstore_path = load_path / "vectorstore" / manifest["vectorstore"]["type"]
            builder._optimized_store = builder._load_vectorstore(
                vectorstore_path,
                manifest["vectorstore"]["type"]
            )

        return builder

    def _get_vectorstore_type(self) -> Optional[str]:
        """Determine the type of vectorstore being used."""
        if not self._optimized_store:
            return None
            
        store_class = self._optimized_store.__class__.__name__.lower()
        return next(
            (v for k, v in self.SUPPORTED_VECTORSTORES.items() 
             if k.lower() in store_class),
            None
        )

    def _save_vectorstore(self, path: Path) -> None:
        """Save vectorstore data to specified path."""
        vectorstore_type = self._get_vectorstore_type()
        
        if vectorstore_type == "chroma":
            # For Chroma, copy the persist_directory if it exists
            if hasattr(self._optimized_store, "_client") and hasattr(self._optimized_store._client, "_persist_directory"):
                persist_dir = Path(self._optimized_store._client._persist_directory)
                if persist_dir.exists():
                    shutil.copytree(persist_dir, path, dirs_exist_ok=True)
                    
        elif vectorstore_type == "faiss":
            # For FAISS, save the index file
            if hasattr(self._optimized_store, "save_local"):
                self._optimized_store.save_local(str(path))
                
        else:
            self.logger.warning(f"Saving not implemented for vectorstore type: {vectorstore_type}")

    def _load_vectorstore(self, path: Path, vectorstore_type: str) -> Optional[Any]:
        """Load vectorstore from saved data."""
        if not path.exists():
            self.logger.warning(f"Vectorstore path does not exist: {path}")
            return None
            
        try:
            if vectorstore_type == "chroma":
                from langchain_community.vectorstores import Chroma
                return Chroma(
                    persist_directory=str(path),
                    embedding_function=self._get_embedding_function()
                )
                
            elif vectorstore_type == "faiss":
                from langchain_community.vectorstores import FAISS
                return FAISS.load_local(
                    str(path),
                    embeddings=self._get_embedding_function()
                )
                
            else:
                self.logger.warning(f"Loading not implemented for vectorstore type: {vectorstore_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load vectorstore: {str(e)}")
            return None

    def _get_embedding_function(self) -> Any:
        """Get the embedding function from the configuration."""
        if self._data_ingest_config and self._data_ingest_config.embedding_models:
            embedding_config = self._data_ingest_config.embedding_models[0]
            # Create and return embedding function based on config
            # This would need to mirror the logic in your data ingestion pipeline
            pass
        return None

class QueryRequest(BaseModel):
    query: str
    question: Optional[str] = None

    def get_query(self) -> str:
        """Return either query or question field"""
        return self.query or self.question or ""
