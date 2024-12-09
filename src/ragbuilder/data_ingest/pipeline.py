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
    ParserType, ChunkingStrategy, EmbeddingModel, VectorDatabase
)
from ragbuilder.config.data_ingest import DataIngestConfig
from ragbuilder.core.logging_utils import setup_rich_logging, console
from ragbuilder.core.exceptions import RAGBuilderError, ConfigurationError, ComponentError, PipelineError
from ragbuilder.core.document_store import DocumentStore

class DataIngestPipeline:
    def __init__(self, config: DataIngestConfig, documents: List[Document] = None):
        self._validate_config(config)
        self.config = config
        self.logger = logging.getLogger("ragbuilder")
        self.doc_store = DocumentStore()
        
        # Generate unique keys for caching
        self.loader_key = self._get_loader_key()
        self.config_key = self._get_config_key()
        
        # Initialize components
        self.chunker = self._create_chunker()
        self.embedder = self._create_embedder()
        self.indexer = None
        
        # Get cached documents or store provided ones
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
        components = [
            f"loader_{self._get_loader_key()}",
            f"chunker_{self.config.chunking_strategy.type}_{self.config.chunk_size}_{self.config.chunk_overlap}",
            f"embedder_{self.config.embedding_model.type}",
            f"vectordb_{self.config.vector_database.type}"
        ]
        
        # Add kwargs hashes using _make_hashable
        if self.config.chunking_strategy.chunker_kwargs:
            components.append(f"chunker_kwargs_{hash(self._make_hashable(self.config.chunking_strategy.chunker_kwargs))}")
        if self.config.embedding_model.model_kwargs:
            components.append(f"embedder_kwargs_{hash(self._make_hashable(self.config.embedding_model.model_kwargs))}")
        if self.config.vector_database.vectordb_kwargs:
            components.append(f"vectordb_kwargs_{hash(self._make_hashable(self.config.vector_database.vectordb_kwargs))}")
            
        return "_".join(components)

    def _get_or_load_documents(self) -> List[Document]:
        """Get documents from cache or load them"""
        if self.doc_store.has_documents(self.loader_key):
            self.logger.info(f"Using cached documents for loader: {self.config.document_loader.type}")
            return self.doc_store.get_documents(self.loader_key)

        self.logger.info(f"Loading documents with loader: {self.config.document_loader.type}")
        documents = self._create_parser().load()
        if not documents:
            raise PipelineError("No documents were loaded from the input source")
        
        # Store documents with metadata
        self.doc_store.store_documents(
            self.loader_key, 
            documents,
            metadata={
                "loader_type": self.config.document_loader.type,
                "loader_kwargs": self.config.document_loader.loader_kwargs,
                "input_source": self.config.input_source,
                "timestamp": time.time()
            }
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

    def _create_parser(self):
        try:
            if not self.config.input_source:
                raise ConfigurationError("Input source cannot be empty")
            
            if not os.path.exists(self.config.input_source) and not urlparse(self.config.input_source).scheme:
                raise ConfigurationError(f"Unable to access input source: {self.config.input_source}")
            
            if self.config.document_loader.type == ParserType.CUSTOM:
                return self._instantiate_custom_class(
                    self.config.document_loader.custom_class,
                    self.config.input_source,
                    **(self.config.document_loader.loader_kwargs or {})
                )
            
            loader_kwargs = self.config.document_loader.loader_kwargs or {}
            
            # URL handling
            if urlparse(self.config.input_source).scheme in ['http', 'https']:
                web_loader_class = LOADER_MAP[ParserType.WEB]()
                return web_loader_class(self.config.input_source, **loader_kwargs)
            
            # Directory handling
            elif os.path.isdir(self.config.input_source):
                glob_pattern = loader_kwargs.pop('glob', '*')
                loader_class = LOADER_MAP.get(self.config.document_loader.type)()
                if not loader_class:
                    raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
                directory_loader_class = LOADER_MAP[ParserType.DIRECTORY]()
                return directory_loader_class(
                    self.config.input_source,
                    glob=glob_pattern,
                    loader_cls=loader_class,
                    loader_kwargs=loader_kwargs
                )
            
            # Single file handling
            elif os.path.isfile(self.config.input_source):
                loader_class = LOADER_MAP.get(self.config.document_loader.type)()
                if not loader_class:
                    raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
                return loader_class(self.config.input_source, **loader_kwargs)
            
            raise ValueError(f"Unsupported input source: {self.config.input_source}")
            
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
        if self.config.embedding_model.type == EmbeddingModel.CUSTOM:
            return self._instantiate_custom_class(
                self.config.embedding_model.custom_class,
                **(self.config.embedding_model.model_kwargs or {})
            )
        
        embedder_class = EMBEDDING_MAP.get(self.config.embedding_model.type)()
        if not embedder_class:
            raise ValueError(f"Unsupported embedding model: {self.config.embedding_model.type}")
        
        return embedder_class(**(self.config.embedding_model.model_kwargs or {}))

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


    def ingest(self):
        try:
            with console.status("[status]Running pipeline...[/status]") as status:
                if self._documents is None:
                    raise PipelineError("No documents were loaded from the input source")
                
                status.update("[status]Chunking documents...[/status]")
                chunks = self.chunker.split_documents(self._documents)
                if not chunks:
                    raise ValueError("No chunks were generated from the documents")
                
                print("Chunking done", len(chunks))
                # console.print("[success]✓ Pipeline execution complete![/success]")
                return chunks
            
        except RAGBuilderError as e:
            console.print(f"[error]Pipeline execution failed:[/error] {str(e)}")
            console.print_exception()
            raise
    
    def run(self):
        try:
            with console.status("[status]Running pipeline...[/status]") as status:
                # Check vectorstore cache first
                if vectorstore := self.doc_store.get_vectorstore(self.config_key):
                    self.logger.info(f"Using cached vectorstore for config: {self.config_key}")
                    self.indexer = vectorstore
                    return vectorstore
                
                status.update("[status]Chunking documents...[/status]")
                chunks = self.chunker.split_documents(self._documents)
                if not chunks:
                    raise ValueError("No chunks were generated from the documents")
                
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