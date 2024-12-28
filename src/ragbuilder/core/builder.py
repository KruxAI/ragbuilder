from typing import Optional, Any, Dict, Union
from ragbuilder.config.data_ingest import DataIngestOptionsConfig 
from ragbuilder.config.retriever import RetrievalOptionsConfig
from ragbuilder.config.generation import GenerationOptionsConfig
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.embeddings import Embeddings
from ragbuilder.config.base import LogConfig, LLMConfig, EmbeddingConfig
from ragbuilder.data_ingest.optimization import run_data_ingest_optimization
from ragbuilder.retriever.optimization import run_retrieval_optimization
from ragbuilder.generation.optimization import run_generation_optimization
from ragbuilder.generate_data import TestDatasetManager
from ragbuilder.core.logging_utils import setup_rich_logging, console
from ragbuilder.core.telemetry import telemetry
from ragbuilder.core.config_store import ConfigStore
from ragbuilder.core.utils import validate_environment, serialize_config, SimpleConfigEncoder
from ragbuilder.core.results import DataIngestResults, RetrievalResults, GenerationResults, OptimizationResults
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
            default_llm: Optional[Union[Dict[str, Any], LLMConfig, BaseChatModel, BaseLLM]] = None,
            default_embeddings: Optional[Union[Dict[str, Any], EmbeddingConfig, Embeddings]] = None,
            n_trials: Optional[int] = None,
            log_config: Optional[LogConfig] = None
        ):
        ConfigStore.set_default_llm(default_llm) if default_llm else None
        ConfigStore.set_default_embeddings(default_embeddings) if default_embeddings else None
        ConfigStore.set_default_n_trials(n_trials) if n_trials else None
        self._log_config = log_config or LogConfig()
        self.data_ingest_config = data_ingest_config
        self.retrieval_config = retrieval_config
        self.generation_config = generation_config
        self.logger = setup_rich_logging(
            self._log_config.log_level,
            self._log_config.log_file
        )
        self._optimized_store = None
        self._optimized_retriever = None
        self._optimized_generation = None
        self._optimization_results = OptimizationResults()
        self._test_dataset_manager = TestDatasetManager(
            self._log_config,
            db_path=(self.data_ingest_config.database_path 
                     if self.data_ingest_config and self.data_ingest_config.database_path 
                     else DEFAULT_DB_PATH)
        )

    @classmethod
    def from_source_with_defaults(cls, 
                         input_source: str, 
                         test_dataset: Optional[str] = None,
                         default_llm: Optional[Union[Dict[str, Any], LLMConfig, BaseChatModel, BaseLLM]] = None,
                         default_embeddings: Optional[Union[Dict[str, Any], EmbeddingConfig, Embeddings]] = None,
                         n_trials: Optional[int] = None,
                         log_config: Optional[LogConfig] = None
                         ) -> 'RAGBuilder':
        """Create RAGBuilder instance with default configuration"""
        ConfigStore.set_default_llm(default_llm) if default_llm else None
        ConfigStore.set_default_embeddings(default_embeddings) if default_embeddings else None
        ConfigStore.set_default_n_trials(n_trials) if n_trials else None
        
        data_ingest_config = DataIngestOptionsConfig.with_defaults(
            input_source=input_source,
            test_dataset=test_dataset
        )
        
        return cls(data_ingest_config=data_ingest_config, log_config=log_config)
    
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
            builder.data_ingest_config = DataIngestOptionsConfig(**config_dict['data_ingest'])
        
        # TODO: Handle vectorstore provided by user instead of using the one from data_ingest
        if 'retrieval' in config_dict:  
            builder.retrieval_config = RetrievalOptionsConfig(**config_dict['retrieval'])

        if 'generation' in config_dict:
            builder.generation_config = GenerationOptionsConfig(**config_dict['generation'])
            print("generation_config",GenerationOptionsConfig(**config_dict['generation']))
        return builder


    def _ensure_eval_dataset(self, config: Union[DataIngestOptionsConfig, RetrievalOptionsConfig, GenerationOptionsConfig]) -> None:
        """Ensure config has a test dataset, generating one if needed"""
        if config.evaluation_config.test_dataset:
            return
        
        # Check if we already have a test dataset from data ingestion
        if (self.data_ingest_config and self.data_ingest_config.evaluation_config.test_dataset):
            config.evaluation_config.test_dataset = self.data_ingest_config.evaluation_config.test_dataset
            self.logger.info(f"Reusing test dataset from data ingestion: {config.evaluation_config.test_dataset}")
            return
        
        if (self.retrieval_config and self.retrieval_config.evaluation_config.test_dataset):
            config.evaluation_config.test_dataset = self.retrieval_config.evaluation_config.test_dataset
            self.logger.info(f"Reusing test dataset from retrieval: {config.evaluation_config.test_dataset}")
            return
        
        if not hasattr(config, 'input_source'):
            raise ValueError("input_source is required when test_dataset is not provided")
        
        source_data = (getattr(config, 'input_source', None) or 
                      (self.data_ingest_config.input_source if self.data_ingest_config else None))
        
        if not source_data:
            raise ValueError("input_source is required when test_dataset is not provided")
            
        with console.status("Generating eval dataset..."):
            test_dataset = self._test_dataset_manager.get_or_generate_eval_dataset(
                source_data=source_data
            )
        config.evaluation_config.test_dataset = test_dataset
        self.logger.info(f"Eval dataset: {test_dataset}")



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
            self.data_ingest_config = config
        elif not self.data_ingest_config:
            raise ValueError("No data ingestion configuration provided")
        
        self.data_ingest_config.apply_defaults()
        
        if validate_env:
            validate_environment(self.data_ingest_config)

        self._ensure_eval_dataset(self.data_ingest_config)
        
        with telemetry.optimization_span("data_ingest", self.data_ingest_config.model_dump()) as span:
            try:
                results = run_data_ingest_optimization(
                    self.data_ingest_config, 
                    log_config=self._log_config
                )
                
                # Store results and update telemetry
                self._optimization_results.data_ingest = results
                self._optimized_store = results.best_index
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
    ) -> RetrievalResults:
        """
        Run retrieval optimization
        
        Args:
            config: Optional retrieval configuration options. If not provided, uses default config
            vectorstore: Optional vectorstore to use. If not provided, uses the one from data_ingest
            validate_env: Whether to validate environment variables
            
        Returns:
            RetrievalResults containing optimization results
        """
        vectorstore = vectorstore or self._optimized_store
        if not vectorstore:
            raise DependencyError("No vectorstore found. Run data ingestion first or provide existing vectorstore.")

        self.retrieval_config = config or RetrievalOptionsConfig.with_defaults()
        if not self.retrieval_config:
            raise ValueError("No retrieval configuration provided")

        self.retrieval_config.apply_defaults()

        if validate_env:
            validate_environment(self.retrieval_config)
            
        self._ensure_eval_dataset(self.retrieval_config)
        
        with telemetry.optimization_span("retriever", self.retrieval_config.model_dump()) as span:
            try:
                results = run_retrieval_optimization(
                    self.retrieval_config, 
                    vectorstore=self._optimized_store,
                    log_config=self._log_config
                )
                
                # Store results and update telemetry
                self._optimization_results.retrieval = results
                self._optimized_retriever = results.best_pipeline.retriever_chain
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
        self._optimized_retriever = retriever or self._optimized_retriever
        if not self._optimized_retriever:
            raise DependencyError("No retriever found. Run retrieval optimization first or provide existing retriever.")
        
        self.generation_config = config or GenerationOptionsConfig.with_defaults()
        if not self.generation_config:
            raise ValueError("No generation configuration provided")
        
        self.generation_config.apply_defaults()

        self._ensure_eval_dataset(self.generation_config)
        
        with telemetry.optimization_span("generation", self.generation_config.model_dump()) as span:
            try:
                results = run_generation_optimization(
                    self.generation_config, 
                    retriever=self._optimized_retriever,
                    log_config=self._log_config
                )
                
                # Store results and update telemetry
                self._optimization_results.generation = results
                self._optimized_generation = results.best_pipeline
                telemetry.update_optimization_results(span, results, "generation")
                return results
                
            except Exception as e:
                telemetry.track_error(
                    "generation",
                    e,
                    context={
                        "config_type": "default" if not config else "custom",
                        "retriever_provided": retriever is not None
                    }
                )
                raise
            finally:
                telemetry.flush()

    def optimize(self) -> OptimizationResults:
        """
        Run end-to-end optimization for data ingestion, retrieval, and generation
        
        Returns:
            OptimizationResults containing results for all optimization stages
        """
        with telemetry.optimization_span("ragbuilder", {"end_to_end": True}) as span:
            try:
                with console.status("[bold green]Validating data ingestion environment...") as status:
                    validate_environment(self.data_ingest_config)
                    status.update("[bold green]Validating retrieval environment...")
                    if not self.retrieval_config:
                        self.retrieval_config = RetrievalOptionsConfig.with_defaults()
                    validate_environment(self.retrieval_config)
                
                # Run optimizations and store results in structured format
                self._optimization_results.data_ingest = self.optimize_data_ingest(validate_env=False)
                self._optimization_results.retrieval = self.optimize_retrieval(validate_env=False)
                self._optimization_results.generation = self.optimize_generation()
                
                # Add telemetry attributes
                if self._optimization_results.data_ingest:
                    span.set_attribute("data_ingest_score", self._optimization_results.data_ingest.best_score)
                if self._optimization_results.retrieval:
                    span.set_attribute("retrieval_score", self._optimization_results.retrieval.best_score)
                if self._optimization_results.generation:
                    span.set_attribute("generation_score", self._optimization_results.generation.best_score)

                return self._optimization_results
                
            except Exception as e:
                telemetry.track_error(
                    "ragbuilder",
                    e,
                    context={
                        "completed_modules": [
                            module for module in ["data_ingest", "retrieval", "generation"]
                            if getattr(self._optimization_results, module) is not None
                        ]
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

    def get_configs(self) -> Dict[str, Any]:
        """Get current configurations"""
        configs = {}
        if self.data_ingest_config:
            configs['data_ingest'] = self.data_ingest_config.model_dump()
        if self.retrieval_config:
            configs['retrieval'] = self.retrieval_config.model_dump()
        if self.generation_config:
            configs['generation'] = self.generation_config.model_dump()
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
        if not self._optimization_results.generation:
            raise DependencyError("No generation pipeline found. Run generation optimization first.")
            
        app = FastAPI(title="RAGBuilder API")
        
        @app.post("/invoke")
        async def invoke(request: QueryRequest) -> Dict[str, Any]:
            try:
                result = self._optimized_generation.query(
                    request.get_query()
                )
                console.print(f"Question:{request.get_query()}")
                console.print(f"Response:{result}")
                return {"response": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
                
        self.logger.info(f"Starting RAG server on http://{host}:{port}")
        asyncio.run(uvicorn.run(app, host=host, port=port))

    def save(self, path: str, include_vectorstore: bool = True) -> None:
        """
        Save complete RAG setup including vectorstore if specified.
        
        Args:
            path: Directory path to save the RAG setup
            include_vectorstore: Whether to save the vectorstore data
        """
        print("WARNING: This is an experimental feature and may not work as expected. ⚠️ Use with caution!")
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (save_path / "configs").mkdir(exist_ok=True)
        (save_path / "metadata").mkdir(exist_ok=True)
        if include_vectorstore:
            (save_path / "vectorstore").mkdir(exist_ok=True)

        # Save configurations using existing serialization
        if self.data_ingest_config:
            with open(save_path / "configs" / "data_ingest.yaml", "w") as f:
                yaml.dump(json.loads(serialize_config(self.data_ingest_config)), f)
        
        if self.retrieval_config:
            with open(save_path / "configs" / "retriever.yaml", "w") as f:
                yaml.dump(json.loads(serialize_config(self.retrieval_config)), f)
            
        if self.generation_config:
            with open(save_path / "configs" / "generation.yaml", "w") as f:
                yaml.dump(json.loads(serialize_config(self.generation_config)), f)

        # Save optimization results
        if self._optimization_results:
            serialized_results = {
                "data_ingest": self._optimization_results.data_ingest.model_dump(exclude={"best_pipeline", "best_index"}) if self._optimization_results.data_ingest else None,
                "retrieval": self._optimization_results.retrieval.model_dump(exclude={"best_pipeline"}) if self._optimization_results.retrieval else None,
                "generation": self._optimization_results.generation.model_dump(exclude={"best_pipeline"}) if self._optimization_results.generation else None
            }
            
            with open(save_path / "metadata" / "optimization_results.json", "w") as f:
                json.dump(serialized_results, f, indent=2, cls=SimpleConfigEncoder)

        # Save vectorstore if requested
        if include_vectorstore and self._optimization_results.data_ingest:
            vectorstore = self._optimization_results.data_ingest.get_vectorstore()
            if vectorstore:
                vectorstore_config = self._get_vectorstore_config()
                if vectorstore_config:
                    vectorstore_path = save_path / "vectorstore" / vectorstore_config["type"]
                    self._save_vectorstore(vectorstore_path)

        # Save defaults
        defaults = {
            "llm": self._serialize_llm_config(ConfigStore().get_default_llm()),
            "embeddings": self._serialize_embedding_config(ConfigStore().get_default_embedding()),
            "n_trials": ConfigStore().get_default_n_trials()
        }
        
        # Create manifest with vectorstore info and defaults
        manifest = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "components": {
                "data_ingest": bool(self._optimization_results.data_ingest),
                "retriever": bool(self._optimization_results.retrieval),
                "generation": bool(self._optimization_results.generation)
            },
            "vectorstore": {
                "included": include_vectorstore,
                "config": self._get_vectorstore_config(),
                "embedding": self._get_embedding_config()
            },
            "defaults": defaults
        }
        
        with open(save_path / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        self.logger.info(f"Successfully saved RAG setup to {path}")

    def _serialize_llm_config(self, llm_config: Optional[LLMConfig]) -> Optional[Dict]:
        """Serialize LLM config for storage"""
        if not llm_config:
            return None
        return {
            "type": llm_config.type.value if llm_config.type else None,
            "model_kwargs": llm_config.model_kwargs
        }

    def _serialize_embedding_config(self, embedding_config: Optional[EmbeddingConfig]) -> Optional[Dict]:
        """Serialize embedding config for storage"""
        if not embedding_config:
            return None
        return {
            "type": embedding_config.type.value if embedding_config.type else None,
            "model_kwargs": embedding_config.model_kwargs
        }

    @classmethod
    def load(cls, path: str) -> 'RAGBuilder':
        """
        Load complete RAG setup from saved project.
        
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

        # Restore defaults if they exist
        if "defaults" in manifest:
            defaults = manifest["defaults"]
            if defaults.get("llm"):
                ConfigStore.set_default_llm(defaults["llm"])
            if defaults.get("embeddings"):
                ConfigStore.set_default_embeddings(defaults["embeddings"])
            if defaults.get("n_trials"):
                ConfigStore.set_default_n_trials(defaults["n_trials"])

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
                
        if manifest["components"]["generation"]:
            with open(load_path / "configs" / "generation.yaml", "r") as f:
                generation_config = GenerationOptionsConfig(**yaml.safe_load(f))

        # Create builder instance
        builder = cls(
            data_ingest_config=data_ingest_config,
            retrieval_config=retrieval_config,
            generation_config=generation_config
        )

        # Initialize OptimizationResults
        builder._optimization_results = OptimizationResults()

        # Load optimization results if they exist
        if (load_path / "metadata" / "optimization_results.json").exists():
            with open(load_path / "metadata" / "optimization_results.json", "r") as f:
                saved_results = json.load(f)
                
                if saved_results.get("data_ingest"):
                    builder._optimization_results.data_ingest = DataIngestResults(**saved_results["data_ingest"])
                
                if saved_results.get("retrieval"):
                    builder._optimization_results.retrieval = RetrievalResults(**saved_results["retrieval"])
                
                if saved_results.get("generation"):
                    builder._optimization_results.generation = GenerationResults(**saved_results["generation"])

        # Load vectorstore if it exists
        if manifest["vectorstore"]["included"] and manifest["vectorstore"]["config"]:
            vectorstore_path = load_path / "vectorstore" / manifest["vectorstore"]["config"]["type"]
            vectorstore = builder._load_vectorstore(vectorstore_path, manifest)
            if vectorstore and builder._optimization_results.data_ingest:
                builder._optimization_results.data_ingest.best_index = vectorstore

        return builder

    def _get_vectorstore_config(self) -> Optional[Dict[str, Any]]:
        """Get vectorstore configuration from data ingestion results"""
        if not self._optimization_results.data_ingest:
            return None
        
        config = self._optimization_results.data_ingest.best_config
        return {
            "type": config.vector_database.type.value.lower(),
            "kwargs": config.vector_database.model_kwargs
        }

    def _get_embedding_config(self) -> Optional[Dict[str, Any]]:
        """Get embedding configuration from data ingestion results"""
        if not self._optimization_results.data_ingest:
            return None
        
        config = self._optimization_results.data_ingest.best_config
        return {
            "type": config.embedding_model.type.value,
            "kwargs": config.embedding_model.model_kwargs
        }

    def _create_embedding_function(self, embedding_config: Dict[str, Any]) -> Any:
        """Create embedding function from saved configuration"""
        from ragbuilder.config.data_ingest import EMBEDDING_MAP
        
        try:
            embedding_class = EMBEDDING_MAP[embedding_config["type"]]
            return embedding_class(**embedding_config["kwargs"])
        except Exception as e:
            self.logger.error(f"Failed to create embedding function: {str(e)}")
            return None

    def _save_vectorstore(self, path: Path) -> None:
        """Save vectorstore data to specified path."""
        vectorstore = self._optimization_results.data_ingest.get_vectorstore()
        vectorstore_config = self._get_vectorstore_config()
        
        if vectorstore_config["type"] == "chroma":
            # For Chroma, copy the persist_directory if it exists
            if hasattr(vectorstore, "_client") and vectorstore._client.get_settings().is_persistent:
                persist_dir = Path(vectorstore._client._persist_directory)
                if persist_dir.exists():
                    shutil.copytree(persist_dir, path, dirs_exist_ok=True)
                    
        elif vectorstore_config["type"] == "faiss":
            # For FAISS, save the index file
            if hasattr(vectorstore, "save_local"):
                vectorstore.save_local(str(path))
                
        else:
            self.logger.warning(f"Saving not implemented for vectorstore type: {vectorstore_config['type']}")

    def _load_vectorstore(self, path: Path, manifest: Dict[str, Any]) -> Optional[Any]:
        """Load vectorstore from saved data using config from manifest."""
        if not path.exists():
            self.logger.warning(f"Vectorstore path does not exist: {path}")
            return None
        
        try:
            vectorstore_config = manifest["vectorstore"]["config"]
            embedding_config = manifest["vectorstore"]["embedding"]
            
            # Create embedding function from saved config
            embedding_function = self._create_embedding_function(embedding_config)
            if not embedding_function:
                raise ValueError("Failed to create embedding function")
            
            if vectorstore_config["type"] == "chroma":
                from langchain_community.vectorstores import Chroma
                return Chroma(
                    persist_directory=str(path),
                    embedding_function=embedding_function,
                    **vectorstore_config.get("kwargs", {})
                )
                
            elif vectorstore_config["type"] == "faiss":
                from langchain_community.vectorstores import FAISS
                return FAISS.load_local(
                    str(path),
                    embeddings=embedding_function,
                    **vectorstore_config.get("kwargs", {})
                )
                
            else:
                self.logger.warning(f"Loading not implemented for vectorstore type: {vectorstore_config['type']}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load vectorstore: {str(e)}")
            return None

class QueryRequest(BaseModel):
    query: str
    question: Optional[str] = None

    def get_query(self) -> str:
        """Return either query or question field"""
        return self.query or self.question or ""
