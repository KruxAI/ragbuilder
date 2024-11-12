from urllib.parse import urlparse
import os
import logging
import time
import random
from importlib import import_module
from typing import Any, List
from copy import deepcopy

from .components import (
    LOADER_MAP, CHUNKER_MAP, EMBEDDING_MAP, VECTORDB_MAP, DIRECTORY_LOADER, WEB_LOADER,
    ParserType, ChunkingStrategy, EmbeddingModel, VectorDatabase
)
from .config import DataIngestConfig
from .logging_utils import setup_rich_logging, console, get_progress_bar

class DataIngestPipeline:
    def __init__(self, config: DataIngestConfig):
        self.config = config
        self.logger = logging.getLogger("ragbuilder")
        
        # Environment validation is now handled at optimization level
        self.parser = self._create_parser()
        self.chunker = self._create_chunker()
        self.embedder = self._create_embedder()
        self.indexer = None

    def _create_parser(self):
        if not self.config.input_source:
            raise ValueError("Input source cannot be empty")
        
        if not os.path.exists(self.config.input_source) and not urlparse(self.config.input_source).scheme:
            raise FileNotFoundError(f"Unable to access input source: {self.config.input_source}")
        
        if self.config.document_loader.type == ParserType.CUSTOM:
            return self._instantiate_custom_class(
                self.config.document_loader.custom_class,
                self.config.input_source,
                **(self.config.document_loader.loader_kwargs or {})
            )
        
        loader_kwargs = self.config.document_loader.loader_kwargs or {}
        
        # URL handling
        if urlparse(self.config.input_source).scheme in ['http', 'https']:
            return WEB_LOADER(self.config.input_source, **loader_kwargs)
        
        # Directory handling
        elif os.path.isdir(self.config.input_source):
            glob_pattern = loader_kwargs.pop('glob', '*')
            loader_cls = LOADER_MAP.get(self.config.document_loader.type)
            if not loader_cls:
                raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
            return DIRECTORY_LOADER(
                self.config.input_source,
                glob=glob_pattern,
                loader_cls=loader_cls,
                loader_kwargs=loader_kwargs
            )
        
        # Single file handling
        elif os.path.isfile(self.config.input_source):
            loader_cls = LOADER_MAP.get(self.config.document_loader.type)
            if not loader_cls:
                raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
            return loader_cls(self.config.input_source, **loader_kwargs)
        
        raise ValueError(f"Unsupported input source: {self.config.input_source}")

    def _create_chunker(self):
        if self.config.chunking_strategy.type == ChunkingStrategy.CUSTOM:
            return self._instantiate_custom_class(
                self.config.chunking_strategy.custom_class,
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                **(self.config.chunking_strategy.chunker_kwargs or {})
            )
        
        chunker_class = CHUNKER_MAP.get(self.config.chunking_strategy.type)
        if not chunker_class:
            raise ValueError(f"Unsupported chunking strategy: {self.config.chunking_strategy.type}")
        
        return chunker_class(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            **(self.config.chunking_strategy.chunker_kwargs or {})
        )

    def _create_embedder(self):
        if self.config.embedding_model.type == EmbeddingModel.CUSTOM:
            return self._instantiate_custom_class(
                self.config.embedding_model.custom_class,
                **(self.config.embedding_model.model_kwargs or {})
            )
        
        embedder_class = EMBEDDING_MAP.get(self.config.embedding_model.type)
        if not embedder_class:
            raise ValueError(f"Unsupported embedding model: {self.config.embedding_model.type}")
        
        # print("Embedding Model: ", embedder_class)
        # print("Embedding Model Kwargs: ", self.config.embedding_model.model_kwargs)
        return embedder_class(**(self.config.embedding_model.model_kwargs or {}))

    def _create_indexer(self, chunks):
        if self.config.vector_database.type == VectorDatabase.CUSTOM:
            custom_class = self._instantiate_custom_class(
                self.config.vector_database.custom_class,
                embedding_function=self.embedder,
                **(self.config.vector_database.vectordb_kwargs or {})
            )
            custom_class.add_documents(chunks)
            return custom_class
        
        vectordb_class = VECTORDB_MAP.get(self.config.vector_database.type)
        if not vectordb_class:
            raise ValueError(f"Unsupported vector database: {self.config.vector_database.type}")
        
        if self.config.vector_database.type == VectorDatabase.CHROMA:
            vectordb_kwargs = deepcopy(self.config.vector_database.vectordb_kwargs or {})
            if 'collection_name' not in vectordb_kwargs:
                # print(f"Setting collection name to: ragbuilder-{int(time.time())}-{random.randint(1, 1000)}")
                vectordb_kwargs['collection_name'] = f"ragbuilder-{int(time.time())}-{random.randint(1, 1000)}" 
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

    def run(self):
        try:
            with console.status("[status]Loading documents...[/status]") as status:
                documents = self.parser.load()
                if not documents:
                    raise ValueError("No documents were loaded from the input source")
                
                status.update("[status]Chunking documents...[/status]")
                chunks = self.chunker.split_documents(documents)
                if not chunks:
                    raise ValueError("No chunks were generated from the documents")
                
                status.update("[status]Creating vector index...[/status]")
                self.indexer = self._create_indexer(chunks)
                
                console.print("[success]✓ Pipeline execution complete![/success]")
                return self.indexer
            
        except Exception as e:
            console.print(f"[error]Pipeline execution failed:[/error] {str(e)}")
            console.print_exception()
            raise
