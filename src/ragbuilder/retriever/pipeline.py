import os
import logging
from typing import List, Optional, Any, Dict, Union
from importlib import import_module
from langchain.schema import BaseRetriever, Document
from langchain.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever, ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from rerankers import Reranker

from ragbuilder.config.components import RetrieverType, RerankerType, CHUNKER_MAP
from ragbuilder.config.retriever import RetrievalConfig, RerankerConfig
from ragbuilder.core.logging_utils import console, setup_rich_logging
from ragbuilder.core.document_store import DocumentStore
from ragbuilder.core.config_store import ConfigStore
from ragbuilder.core.exceptions import (
    RAGBuilderError,
    ConfigurationError,
    ComponentError,
    PipelineError
)

class RetrieverPipeline:
    def __init__(self, 
                 config: RetrievalConfig,
                 vectorstore: Any):
        """Initialize retriever pipeline with specific configuration.
        
        Args:
            config: Single instance configuration for the pipeline
            vectorstore: Initialized vector store from data ingest
            
        Raises:
            ConfigurationError: If required config fields are missing or invalid
        """
        console.print("[status]Initializing Retriever Pipeline...[/status]")
        self._validate_config(config)
        self.config = config
        self.vectorstore = vectorstore
        self.final_k = config.top_k
        self.logger = logging.getLogger("ragbuilder")
        self.store = DocumentStore()
        self.best_data_ingest_config = ConfigStore().get_best_config()
        
        # Initialize components
        with console.status("[status]Creating retrieval components...[/status]"):
            self.base_retrievers = self._create_base_retrievers()
            self.retriever_chain = self._create_retriever_chain()
        console.print("[green]✓[/green] Pipeline initialized successfully")

    def _validate_config(self, config: RetrievalConfig) -> None:
        """Validate pipeline configuration."""
        if not config.retrievers:
            raise ConfigurationError("At least one retriever must be specified")
        
        for retriever in config.retrievers:
            if retriever.type == RetrieverType.CUSTOM and not retriever.custom_class:
                raise ConfigurationError("Custom retriever class must be specified")
            if retriever.weight < 0:
                raise ConfigurationError("Retriever weights must be non-negative")

    def _create_base_retrievers(self) -> List[BaseRetriever]:
        """Create the base retrievers from configuration."""
        retrievers = []
        weights = []

        for retriever_config in self.config.retrievers:
            try:
                console.print(f"[status]Creating {retriever_config.type} retriever...[/status]")
                
                if retriever_config.type == RetrieverType.SIMILARITY:
                    retriever = self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": retriever_config.retriever_k[0]},
                        **retriever_config.retriever_kwargs
                    )
                    console.print(f"[green]✓[/green] Created vector similarity search retriever")
                
                elif retriever_config.type == RetrieverType.MMR:
                    retriever = self.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": retriever_config.retriever_k[0],
                            "fetch_k": retriever_config.retriever_k[0] * 2
                        },
                        **retriever_config.retriever_kwargs
                    )
                    console.print(f"[green]✓[/green] Created vector MMR search retriever")
                
                elif retriever_config.type == RetrieverType.MULTI_QUERY:
                    retriever = MultiQueryRetriever.from_llm(
                        retriever=self.vectorstore.as_retriever(
                            search_kwargs={"k": retriever_config.retriever_k[0]}
                        ),
                        llm=retriever_config.retriever_kwargs.get("llm"),
                    )
                    console.print("[green]✓[/green] Created multi-query retriever")
                
                elif retriever_config.type in [RetrieverType.PARENT_DOC_RETRIEVER_FULL, RetrieverType.PARENT_DOC_RETRIEVER_LARGE]:
                    if not self.best_data_ingest_config:
                        raise PipelineError("No optimized data ingestion configuration found")
                    
                    console.print("[status]Setting up document splitters for parent document retriever...[/status]")
                    splitter_class = CHUNKER_MAP[self.best_data_ingest_config["chunking_strategy"]["type"]]()
                    child_splitter = splitter_class(
                        chunk_size=self.best_data_ingest_config["chunk_size"],
                        chunk_overlap=self.best_data_ingest_config["chunk_overlap"],
                        **self.best_data_ingest_config["chunking_strategy"]["chunker_kwargs"] or {}
                    )

                    parent_splitter = None
                    if retriever_config.type == RetrieverType.PARENT_DOC_RETRIEVER_LARGE:
                        parent_splitter = splitter_class(
                            chunk_size=self.best_data_ingest_config["chunk_size"] * 3,
                            chunk_overlap=self.best_data_ingest_config["chunk_overlap"] * 3,
                            **self.best_data_ingest_config["chunking_strategy"]["chunker_kwargs"] or {}
                        )

                    store = InMemoryStore()
                    
                    retriever = ParentDocumentRetriever(
                        vectorstore=self.vectorstore,
                        docstore=store,
                        child_splitter=child_splitter,
                        parent_splitter=parent_splitter
                    )
                    
                    # Fetch documents from the DocumentStore based on the best config
                    docs, _ = self.store.get_best_config_docs()
                    if not docs:
                        raise PipelineError("No documents found in DocumentStore")
                    
                    console.print("[status]Adding documents to parent document retriever...[/status]")
                    retriever.add_documents(docs)
                    console.print(f"[green]✓[/green] Created parent document retriever")
                
                # TODO: Utilize the vector DB's BM25 capability rather than creating in-memory BM25Retriever
                elif retriever_config.type == RetrieverType.BM25_RETRIEVER:
                    if not self.best_data_ingest_config:
                        raise PipelineError("No optimized data ingestion configuration found")
                    
                    console.print("[status]Setting up splitter for BM25 retriever...[/status]")
                    splitter_class = CHUNKER_MAP[self.best_data_ingest_config["chunking_strategy"]["type"]]()
                    splitter = splitter_class(
                        chunk_size=self.best_data_ingest_config["chunk_size"],
                        chunk_overlap=self.best_data_ingest_config["chunk_overlap"],
                        **self.best_data_ingest_config["chunking_strategy"]["chunker_kwargs"] or {}
                    )
                    docs, _ = self.store.get_best_config_docs()
                    if not docs:
                        raise PipelineError("No documents found in DocumentStore")
                    
                    console.print("[status]Chunking documents...[/status]")
                    chunks = splitter.split_documents(docs)
                    if not chunks:
                        raise PipelineError("No chunks were generated from the documents")
                    
                    console.print("[status]Creating BM25 retriever...[/status]")
                    retriever = BM25Retriever.from_documents(
                        chunks,
                        **retriever_config.retriever_kwargs
                    )
                    console.print("[green]✓[/green] Created BM25 retriever")
                
                elif retriever_config.type == RetrieverType.CUSTOM:
                    retriever = self._instantiate_custom_class(
                        retriever_config.custom_class,
                        vectorstore=self.vectorstore,
                        **retriever_config.retriever_kwargs
                    )
                else:
                    raise ConfigurationError(f"Unsupported retriever type: {retriever_config.type}")

                retrievers.append(retriever)
                weights.append(retriever_config.weight)
                
            except Exception as e:
                console.print(f"[red]✗ Failed to create {retriever_config.type} retriever: {str(e)}[/red]")
                raise ComponentError(f"Failed to create base retriever: {str(e)}") from e

        if len(retrievers) == 1:
            return retrievers[0]
        
        console.print("[status]Creating ensemble retriever...[/status]")
        if not any(weights):
            console.print("[yellow]No weights specified, using equal weights[/yellow]")
            return EnsembleRetriever(retrievers=retrievers)
        else:
            total = sum(weights)
            weights = [w/total for w in weights]
            console.print(f"[green]✓[/green] Created ensemble retriever")
            return EnsembleRetriever(retrievers=retrievers, weights=weights)

    def _create_reranker(self, config: RerankerConfig):
        """Create a reranker from configuration."""
        try:
            # Handle different reranker types
            if config.type == RerankerType.COHERE:
                ranker = Reranker(
                    "cohere",
                    model_type='APIRanker',
                    lang='en',
                    api_key=os.getenv('COHERE_API_KEY')
                )
                return ranker.as_langchain_compressor(k=self.final_k)
            
            elif config.type == RerankerType.BAAI_BGE_RERANKER_BASE:
                ranker = Reranker(
                    "BAAI/bge-reranker-base",
                    model_type='TransformerRanker'
                )
                return ranker.as_langchain_compressor(k=self.final_k)
            
            elif config.type == RerankerType.CUSTOM:
                if "model_name" in config.reranker_kwargs:
                    # Handle custom HuggingFace models
                    ranker = Reranker(
                        config.reranker_kwargs["model_name"],
                        # model_type=config.reranker_kwargs.get("model_type", "cross-encoder"),
                        model_type=config.reranker_kwargs.get("model_type"),
                        **{k:v for k,v in config.reranker_kwargs.items() 
                           if k not in ["model_name", "model_type"]}
                    )
                    return ranker.as_langchain_compressor(k=self.final_k)
                return self._instantiate_custom_class(
                    config.custom_class,
                    **config.reranker_kwargs
                )
            
            raise ConfigurationError(f"Unsupported reranker type: {config.type}")
            
        except Exception as e:
            raise ComponentError(f"Failed to create reranker: {str(e)}") from e

    def _create_document_compressor(self, reranker_configs: List[RerankerConfig]) -> DocumentCompressorPipeline:
        """Create a document compressor pipeline from multiple rerankers."""
        compressors = []
        
        for config in reranker_configs:
            compressor = self._create_reranker(config)
            compressors.append(compressor)
            
        return DocumentCompressorPipeline(transformers=compressors)

    def _create_retriever_chain(self) -> BaseRetriever:
        """Create the full retrieval chain including rerankers if specified."""
        retriever = self.base_retrievers
        
        if hasattr(self.config, 'rerankers') and self.config.rerankers:
            compressor = self._create_document_compressor(self.config.rerankers)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
        
        return retriever

    def _instantiate_custom_class(self, class_path: str, *args: Any, **kwargs: Any) -> Any:
        """Instantiate a custom class from its path."""
        try:
            if class_path.startswith('.'):
                module_name, class_name = class_path.rsplit('.', 1)
                module = import_module(module_name, package=__package__)
            else:
                module_name, class_name = class_path.rsplit('.', 1)
                module = import_module(module_name)
            custom_class = getattr(module, class_name)
            return custom_class(*args, **kwargs)
        except Exception as e:
            raise ComponentError(f"Failed to instantiate custom class {class_path}: {str(e)}") from e

    async def aretrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents for a given query.
        
        Args:
            query: The query string
            
        Returns:
            List of retrieved documents
            
        Raises:
            PipelineError: If retrieval fails
        """
        try:
            # Get more documents initially if using rerankers
            documents = await self.retriever_chain.ainvoke(query)
            
            # Trim to final_k
            return documents[:self.final_k]
            
        except Exception as e:
            console.print(f"[red]✗ Retrieval failed: {str(e)}[/red]")
            raise PipelineError(f"Retrieval failed: {str(e)}") from e

    def retrieve(self, query: str) -> List[Document]:
        """Synchronous version of retrieve."""
        try:
            documents = self.retriever_chain.invoke(query)
            return documents[:self.final_k]
        except Exception as e:
            console.print(f"[red]✗ Retrieval failed: {str(e)}[/red]")
            raise PipelineError(f"Retrieval failed: {str(e)}") from e