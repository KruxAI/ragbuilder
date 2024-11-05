from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader, DirectoryLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS, Chroma
from .config import DataIngestConfig, ParserType, ChunkingStrategy, EmbeddingModel, VectorDatabase
from typing import Any, List
from importlib import import_module
from urllib.parse import urlparse
import os

class DataIngestPipeline:
    def __init__(self, config: DataIngestConfig):
        self.config = config
        self.parser = self._create_parser()
        self.chunker = self._create_chunker()
        self.embedder = self._create_embedder()
        self.indexer = None

    def _create_parser(self):
        if self.config.document_loader.type == ParserType.CUSTOM:
            return self._instantiate_custom_class(self.config.document_loader.custom_class, self.config.input_source, **(self.config.document_loader.loader_kwargs or {}))
        
        loader_kwargs = self.config.document_loader.loader_kwargs or {}
        
        # URL
        if urlparse(self.config.input_source).scheme in ['http', 'https']:
            return WebBaseLoader(self.config.input_source, **loader_kwargs)
        
        # Directory
        elif os.path.isdir(self.config.input_source):
            glob_pattern = loader_kwargs.pop('glob', '*')  # Default to all files if not specified
            if self.config.document_loader.type == ParserType.UNSTRUCTURED:
                loader_cls = UnstructuredFileLoader  
            elif self.config.document_loader.type == ParserType.PYMUPDF:
                loader_cls = PyMuPDFLoader
            else:
                raise ValueError(f"Unsupported document loader type: {self.config.document_loader.type}")
            return DirectoryLoader(self.config.input_source, glob=glob_pattern, loader_cls=loader_cls, loader_kwargs=loader_kwargs)
        
        # Single file handling
        elif os.path.isfile(self.config.input_source):
            if self.config.document_loader.type == ParserType.UNSTRUCTURED:
                return UnstructuredFileLoader(self.config.input_source, **loader_kwargs)
            elif self.config.document_loader.type == ParserType.PYMUPDF:
                return PyMuPDFLoader(self.config.input_source, **loader_kwargs)
        
        # If none of the above conditions are met
        raise ValueError(f"Unsupported input source or parser type: {self.config.input_source}, {self.config.document_loader.type}")

    def _create_chunker(self):
        if self.config.chunking_strategy.type == ChunkingStrategy.CUSTOM:
            print(f"Using custom chunker: {self.config.chunking_strategy.custom_class}")
            return self._instantiate_custom_class(self.config.chunking_strategy.custom_class, chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap, **(self.config.chunking_strategy.chunker_kwargs or {}))
        
        if self.config.chunking_strategy.type == ChunkingStrategy.CHARACTER:
            return CharacterTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        elif self.config.chunking_strategy.type == ChunkingStrategy.RECURSIVE:
            return RecursiveCharacterTextSplitter(chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap)
        else:
            raise ValueError(f"Unsupported chunking strategy: {self.config.chunking_strategy}")

    def _create_embedder(self):
        if self.config.embedding_model.type == EmbeddingModel.CUSTOM:
            return self._instantiate_custom_class(self.config.embedding_model.custom_class, **(self.config.embedding_model.model_kwargs or {}))
        
        model_kwargs = self.config.embedding_model.model_kwargs or {}
        if self.config.embedding_model.type == EmbeddingModel.OPENAI:
            return OpenAIEmbeddings(model=self.config.embedding_model.model, **model_kwargs)
        elif self.config.embedding_model.type == EmbeddingModel.HUGGINGFACE:
            return HuggingFaceEmbeddings(model_name=self.config.embedding_model.model, **model_kwargs)
        elif self.config.embedding_model.type == EmbeddingModel.OLLAMA:
            return OllamaEmbeddings(model=self.config.embedding_model.model, **model_kwargs)
        else:
            raise ValueError(f"Unsupported embedding model: {self.config.embedding_model.type}")

    def _create_indexer(self, chunks):
        if self.config.vector_database.type == VectorDatabase.CUSTOM:
            custom_class = self._instantiate_custom_class(
                self.config.vector_database.custom_class,
                embedding_function=self.embedder,
                **(self.config.vector_database.client_settings or {})
            )
            # Assume the custom class has an add_documents method
            custom_class.add_documents(chunks)
            return custom_class
        elif self.config.vector_database.type == VectorDatabase.FAISS:
            return FAISS.from_documents(chunks, self.embedder)
        elif self.config.vector_database.type == VectorDatabase.CHROMA:
            return Chroma.from_documents(
                chunks,
                self.embedder,
                collection_name=self.config.vector_database.collection_name,
                persist_directory=self.config.vector_database.persist_directory,
                client_settings=self.config.vector_database.client_settings
            )
        else:
            raise ValueError(f"Unsupported vector database: {self.config.vector_database.type}")

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
        documents = self.parser.load()
        chunks = self.chunker.split_documents(documents)
        self.indexer = self._create_indexer(chunks)
        return self.indexer
