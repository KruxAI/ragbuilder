from urllib.parse import urlparse
import os
import logging
import time
import random
from importlib import import_module
from copy import deepcopy
from typing import List, Optional, Any
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore

from ragbuilder.config.components import (
    LOADER_MAP, CHUNKER_MAP, EMBEDDING_MAP, VECTORDB_MAP,
    ParserType, ChunkingStrategy, EmbeddingType, VectorDatabase
)
from ragbuilder.config.data_ingest import DataIngestConfig
from ragbuilder.core.logging_utils import setup_rich_logging, console
from ragbuilder.core.exceptions import RAGBuilderError, ConfigurationError, ComponentError, PipelineError
from ragbuilder.core.document_store import DocumentStore
from ragbuilder.core.config_store import ConfigStore
from ragbuilder.sampler import DataSampler

class DataIngestPipeline:
    def __init__(self, config: DataIngestConfig, documents: List[Document] = None, verbose: bool = False):
        self._validate_config(config)
        self.config = config
        self.logger = logging.getLogger("ragbuilder")
        self.verbose = verbose
        self.doc_store = DocumentStore()        
        self.data_source_type = 'url' if urlparse(self.config.input_source).scheme in ['http', 'https'] else (
            'dir' if os.path.isdir(self.config.input_source) else 'file'
        )
        self.data_source_size = None
        self.loader_key = self._get_loader_key()
        self.config_key = self._get_config_key()
        
        # Initialize components
        self.chunker = self._create_chunker()
        self.embedder = self._create_embedder()
        self.indexer = None
        self.enable_sampling = self.config.sampling_rate is not None and self.config.sampling_rate < 1
        self._documents = documents if documents is not None else self._get_or_load_documents()

    def _make_hashable(self, obj: Any) -> Any:
        """Convert a potentially unhashable object into a hashable one"""
        if isinstance(obj, dict):
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            return tuple(sorted(self._make_hashable(item) for item in obj))
        return obj

    def _get_loader_key(self) -> str:
        """Generate a unique key for document loader configuration"""
        loader_kwargs = self.config.document_loader.loader_kwargs or {}
        return f"loader_{self.config.document_loader.type}_{self.config.input_source}_{hash(self._make_hashable(loader_kwargs))}"

    def _get_config_key(self) -> str:
        """Generate a unique key for the complete configuration"""
        if self.config.embedding_model.type:
            embedding_type = self.config.embedding_model.type
        elif hasattr(self.config.embedding_model._initialized_embedding, "__class__"):
            embedding_type = self.config.embedding_model._initialized_embedding.__class__.__name__
        else:
            embedding_type = "unknown"
            
        components = [
            f"loader_{self._get_loader_key()}",
            f"chunker_{self.config.chunking_strategy.type}_{self.config.chunk_size}_{self.config.chunk_overlap}",
            f"embedder_{embedding_type}",
            f"vectordb_{self.config.vector_database.type}"
        ]
        
        # Add kwargs hashes using _make_hashable
        if self.config.chunking_strategy.chunker_kwargs:
            components.append(f"chunker_kwargs_{hash(self._make_hashable(self.config.chunking_strategy.chunker_kwargs))}")

        if self.config.embedding_model.model_kwargs:
            components.append(f"embedder_kwargs_{hash(self._make_hashable(self.config.embedding_model.model_kwargs))}")
        elif hasattr(self.config.embedding_model._initialized_embedding, "model_kwargs"):
            components.append(f"embedder_kwargs_{hash(self._make_hashable(self.config.embedding_model._initialized_embedding.model_kwargs))}")

        if self.config.vector_database.vectordb_kwargs:
            components.append(f"vectordb_kwargs_{hash(self._make_hashable(self.config.vector_database.vectordb_kwargs))}")
            
        return "_".join(components)

    def _get_or_load_documents(self) -> List[Document]:
        """Get documents from cache or load them"""
        if self.doc_store.has_documents(self.loader_key):
            self.logger.debug(f"Using cached documents for loader: {self.config.document_loader.type}")
            self.data_source_size = self.doc_store.get_metadata(self.loader_key).get("data_source_size", 0)
            return self.doc_store.get_documents(self.loader_key)

        self.logger.debug(f"Loading documents with loader: {self.config.document_loader.type}")
        
        input_source = self.config.input_source
        sampling_metadata = {}
    
        
        # Check if we already have sampled data for this input source
        cached_sample = self.doc_store.get_sampled_data(input_source)
        if cached_sample:
            self.logger.debug("Using previously sampled data...")
            input_source = cached_sample["sampled_path"]
            sampling_metadata = cached_sample["metadata"]
            self.data_source_size = sampling_metadata.get("original_size", 0)
        else:
            # TODO: Pass loader spec to sampler to sample using that particular loader
            sampler = DataSampler(
                input_source,
                enable_sampling=self.enable_sampling,
                sample_ratio=self.config.sampling_rate
            )
            
            self.data_source_size = sampler.estimate_data_size()
                
            if self.enable_sampling:
                self.logger.info("Sampling data before loading...")
                sampled_path = sampler.sample_data()
                self.logger.debug(f"Using sampled data from: {sampled_path}")
                
                sampling_metadata = {
                    "sampled": True,
                    "sampling_ratio": self.config.sampling_rate,
                    "original_size": self.data_source_size
                }
                
                # Cache the sampled data path and metadata
                self.doc_store.store_sampled_data(
                    input_source, 
                    sampled_path,
                    sampling_metadata
                )
                
                input_source = sampled_path
        
        documents = self._create_parser(input_source).load()
        
        if not documents:
            raise PipelineError("No documents were loaded from the input source")
        
        # Store documents with metadata
        metadata = {
            "loader_type": self.config.document_loader.type,
            "loader_kwargs": self.config.document_loader.loader_kwargs,
            "input_source": self.config.input_source,
            "data_source_type": self.data_source_type,
            "data_source_size": self.data_source_size,
            "timestamp": time.time(),
            **sampling_metadata
        }
        
        self.doc_store.store_documents(
            self.loader_key, 
            documents,
            metadata=metadata
        )
        
        return documents

    def _validate_config(self, config: DataIngestConfig) -> None:
        """Validate pipeline configuration."""
        if not config.input_source:
            raise ConfigurationError("Input source cannot be empty")
        if not config.document_loader:
            raise ConfigurationError("Document loader configuration is required")
        if not config.chunking_strategy:
            raise ConfigurationError("Chunking strategy configuration is required")
        if not config.embedding_model:
            raise ConfigurationError("Embedding model configuration is required")
        if not config.vector_database:
            raise ConfigurationError("Vector database configuration is required")

    def _create_parser(self, input_source: str):
        try:
            if not input_source:
                raise ConfigurationError("Input source cannot be empty")
            
            if not os.path.exists(input_source) and not urlparse(input_source).scheme:
                raise ConfigurationError(f"Unable to access input source: {input_source}")
            
            
            if self.config.document_loader.type == ParserType.CUSTOM:
                return self._instantiate_custom_class(
                    self.config.document_loader.custom_class,
                    input_source,
                    **(self.config.document_loader.loader_kwargs or {})
                )
            
            loader_kwargs = self.config.document_loader.loader_kwargs or {}
            
            # URL handling
            if self.data_source_type == 'url':
                web_loader_class = LOADER_MAP[ParserType.WEB]()
                return web_loader_class(input_source, **loader_kwargs)
            
            # Directory handling
            elif self.data_source_type == 'dir':
                glob_pattern = loader_kwargs.pop('glob', '*')
                loader_class = LOADER_MAP.get(self.config.document_loader.type)()
                if not loader_class:
                    raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
                directory_loader_class = LOADER_MAP[ParserType.DIRECTORY]()
                return directory_loader_class(
                    input_source,
                    glob=glob_pattern,
                    loader_cls=loader_class,
                    loader_kwargs=loader_kwargs
                )
            
            # Single file handling
            elif self.data_source_type == 'file':
                loader_class = LOADER_MAP.get(self.config.document_loader.type)()
                if not loader_class:
                    raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
                return loader_class(input_source, **loader_kwargs)
            
            raise ValueError(f"Unsupported input source: {input_source}")
            
        except Exception as e:
            raise ComponentError(f"Failed to create document parser: {str(e)}") from e

    def _create_chunker(self):
        if self.config.chunking_strategy.type == ChunkingStrategy.CUSTOM:
            if type(self.config.chunking_strategy.custom_class) == str:
                return self._instantiate_custom_class(
                    self.config.chunking_strategy.custom_class,
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    **(self.config.chunking_strategy.chunker_kwargs or {})
                )
            else:
                return self.config.chunking_strategy.custom_class
        
        chunker_class = CHUNKER_MAP.get(self.config.chunking_strategy.type)()
        if not chunker_class:
            raise ValueError(f"Unsupported chunking strategy: {self.config.chunking_strategy.type}")
        
        return chunker_class(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            **(self.config.chunking_strategy.chunker_kwargs or {})
        )

    def _create_embedder(self) -> Embeddings:
        if not self.config.embedding_model:
            return ConfigStore().get_default_embeddings().embeddings
            
        return self.config.embedding_model.embeddings

    def _create_indexer(self, chunks: List[Document]) -> VectorStore:
        if self.config.vector_database.type == VectorDatabase.CUSTOM:
            custom_class = self._instantiate_custom_class(
                self.config.vector_database.custom_class,
                embedding_function=self.embedder,
                **(self.config.vector_database.vectordb_kwargs or {})
            )
            custom_class.add_documents(chunks)
            return custom_class
        
        vectordb_class = VECTORDB_MAP.get(self.config.vector_database.type)()
        if not vectordb_class:
            raise ValueError(f"Unsupported vector database: {self.config.vector_database.type}")
        
        if self.config.vector_database.type == VectorDatabase.CHROMA:
            vectordb_kwargs = deepcopy(self.config.vector_database.vectordb_kwargs or {})
            if 'collection_name' not in vectordb_kwargs:
                # print(f"Setting collection name to: ragbuilder-{int(time.time())}-{random.randint(1, 1000)}")
                vectordb_kwargs['collection_name'] = f"ragbuilder_{int(time.time())}_{random.randint(1, 1000)}" 
            # TODO: Handle the scenario where user has specified a collection name, and we want to avoid upserts/ dups in the collection
            # elif 'client_settings' not in vectordb_kwargs:
            #     import chromadb
            #     vectordb_kwargs['client_settings'] = chromadb.config.Settings(allow_reset=True, persist_directory=vectordb_kwargs.get('persist_directory', './chroma'))

            return vectordb_class.from_documents(
                chunks,
                self.embedder,
                **vectordb_kwargs
            )    
                
        # print("Vector Database: ", vectordb_class)
        # print("Vector Database Kwargs: ", self.config.vector_database.vectordb_kwargs)
        return vectordb_class.from_documents(
            chunks,
            self.embedder,
            **(self.config.vector_database.vectordb_kwargs or {})
        )

    def _instantiate_custom_class(self, class_path: str, *args, **kwargs):
        print('class_path: ', class_path)
        if class_path.startswith('.'):
            # Relative import
            module_name, class_name = class_path.rsplit('.', 1)
            module = import_module(module_name, package=__package__)
        else:
            # Absolute import
            module_name, class_name = class_path.rsplit('.', 1)
            module = import_module(module_name)
        custom_class = getattr(module, class_name)
        return custom_class(*args, **kwargs)


    def ingest(self, status=None):
        try:
            if status is None:
                with console.status("[status]Running pipeline...[/status]") as status:
                    return self._ingest(status)
            else:
                return self._ingest(status)
            
        except RAGBuilderError as e:
            console.print(f"[error]Pipeline execution failed:[/error] {str(e)}")
            console.print_exception()
            raise

    def _ingest(self, status):
        if self._documents is None:
            raise PipelineError("No documents were loaded from the input source")
        
        if self.verbose:
            status.update("[status]Chunking documents...[/status]")
        chunks = self.chunker.split_documents(self._documents)
        if not chunks:
            raise ValueError("No chunks were generated from the documents")
        
        self.logger.debug("Chunking done", len(chunks))
        # console.print("[success]✓ Pipeline execution complete![/success]")
        return chunks

    
    def run(self):
        try:
            with console.status("[status]Running pipeline...[/status]") as status:
                # Check vectorstore cache first
                if vectorstore := self.doc_store.get_vectorstore(self.config_key):
                    self.logger.debug(f"Using cached vectorstore for config: {self.config_key}")
                    self.indexer = vectorstore
                    return vectorstore
                
                if self.verbose:
                    status.update("[status]Chunking documents...[/status]")
                
                chunks = self.chunker.split_documents(self._documents)
                if not chunks:
                    raise ValueError("No chunks were generated from the documents")
                
                if self.verbose:
                    status.update("[status]Creating vector index...[/status]")
                
                self.indexer = self._create_indexer(chunks)
                
                # Store vectorstore in cache
                self.doc_store.store_vectorstore(self.config_key, self.indexer)
                
                console.print("[success]✓ Pipeline execution complete![/success]")
                return self.indexer
            
            
        except RAGBuilderError as e:
            console.print(f"[error]Pipeline execution failed:[/error] {str(e)}")
            console.print_exception()
            raise