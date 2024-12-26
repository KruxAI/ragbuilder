import os
import logging
from typing import List, Optional, Any, Dict, Union
from importlib import import_module
from langchain.schema import BaseRetriever, Document
from langchain.storage import InMemoryStore
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever, ParentDocumentRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
# from rerankers import Reranker

from ragbuilder.config.components import RetrieverType, RerankerType, RERANKER_MAP, CHUNKER_MAP
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
                 vectorstore: Any,
                 verbose: bool = True):
        """Initialize retriever pipeline with specific configuration.
        
        Args:
            config: Single instance configuration for the pipeline
            vectorstore: Initialized vector store from data ingest
            
        Raises:
            ConfigurationError: If required config fields are missing or invalid
        """
        self._validate_config(config)
        self.config = config
        self.vectorstore = vectorstore
        self.final_k = config.top_k
        self.verbose = verbose
        self.logger = logging.getLogger("ragbuilder.retriever.pipeline")
        self.store = DocumentStore()
        self.best_data_ingest_config = ConfigStore().get_best_config()
        self.best_data_ingest_pipeline = ConfigStore().get_best_data_ingest_pipeline()
        
        # Initialize components
        # with console.status("[status]Creating retrieval components...[/status]"):
        self.logger.debug("Initializing Retriever Pipeline...")
        self.base_retrievers = self._create_base_retrievers()
        self.retriever_chain = self._create_retriever_chain()
        self.logger.debug("Pipeline initialized successfully")

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

        # with console.status("[status]Running pipeline...[/status]") as status:
        console.print("[status]Creating retrievers...[/status]")
        for retriever_config in self.config.retrievers:
            try:
                # if self.verbose:
                #self.logger.debug(f"Creating {retriever_config.type} retriever...")
                # status.update(f"[status]Creating {retriever_config.type} retriever...[/status]")

                if retriever_config.type == RetrieverType.VECTOR_SIMILARITY:
                    retriever = self.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": retriever_config.retriever_k[0]},
                        **retriever_config.retriever_kwargs
                    )
                    # status.update("[status]Created vector similarity search retriever[/status]")
                
                elif retriever_config.type == RetrieverType.VECTOR_MMR:
                    retriever = self.vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": retriever_config.retriever_k[0],
                            "fetch_k": retriever_config.retriever_k[0] * 2
                        },
                        **retriever_config.retriever_kwargs
                    )
                    # status.update("[status]Created vector MMR search retriever[/status]")
                
                elif retriever_config.type == RetrieverType.MULTI_QUERY:
                    retriever = MultiQueryRetriever.from_llm(
                        retriever=self.vectorstore.as_retriever(
                            search_kwargs={"k": retriever_config.retriever_k[0]}
                        ),
                        llm=retriever_config.retriever_kwargs.get("llm"),
                    )
                    # status.update("[status]Created multi-query retriever[/status]")
                
                elif retriever_config.type in [RetrieverType.PARENT_DOC_RETRIEVER_FULL, RetrieverType.PARENT_DOC_RETRIEVER_LARGE]:
                    if not self.best_data_ingest_config:
                        raise PipelineError("No optimized data ingestion configuration found")
                    
                    # status.update("[status]Setting up document splitters for parent document retriever...[/status]")
                    child_splitter = self.best_data_ingest_pipeline.chunker
                    parent_splitter = None
                    if retriever_config.type == RetrieverType.PARENT_DOC_RETRIEVER_LARGE:
                        if self.best_data_ingest_config["chunking_strategy"]["type"] == "custom":
                            from langchain_text_splitters import RecursiveCharacterTextSplitter
                            splitter = RecursiveCharacterTextSplitter
                        else:
                            splitter = CHUNKER_MAP[self.best_data_ingest_config["chunking_strategy"]["type"]]()
                        parent_splitter = splitter(
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
                    retriever.search_kwargs={"k": retriever_config.retriever_k[0]}
                    
                    # Fetch documents from the DocumentStore based on the best config
                    docs, _ = self.store.get_best_config_docs()
                    if not docs:
                        raise PipelineError("No documents found in DocumentStore")
                    
                    # status.update("[status]Adding documents to parent document retriever...[/status]")
                    retriever.add_documents(docs)
                    # status.update("[status]Created parent document retriever[/status]")
                
                # TODO: Utilize the vector DB's BM25 capability rather than creating in-memory BM25Retriever
                elif retriever_config.type == RetrieverType.BM25:
                    # status.update("[status]Getting documents from best data ingestion pipeline for BM25...[/status]")
                    if not self.best_data_ingest_config:
                        raise PipelineError("No optimized data ingestion configuration found")
                    chunks = self.best_data_ingest_pipeline.ingest(status=None)
                    if not chunks:
                        raise PipelineError("No chunks were generated from the documents")
                    
                    # status.update("[status]Creating BM25 retriever...[/status]")
                    retriever = BM25Retriever.from_documents(
                        chunks,
                        **retriever_config.retriever_kwargs
                    )
                    # status.update("[status]Created BM25 retriever[/status]")
                
                elif retriever_config.type == RetrieverType.GRAPH_RETRIEVER:
                    # Get graph from DocumentStore
                    # embeddings = AzureOpenAIEmbeddings(model="text-embedding-3-large")
                    # status.update("[status]Getting graph from DocumentStore...[/status]")
                    embeddings = self.best_data_ingest_pipeline.embedder
                    graph = self.store.get_graph()
                    if not graph:
                        raise PipelineError("No graph found in DocumentStore")
                    
                    # status.update("[status]Creating Neo4j graph retriever...[/status]")
                    from ragbuilder.graph_utils.graph_retriever import Neo4jGraphRetriever
                    retriever = Neo4jGraphRetriever(
                        graph=graph,
                        top_k=retriever_config.retriever_k[0],
                        embeddings=embeddings,
                        **retriever_config.retriever_kwargs
                    )
                    # status.update("[status]Created Neo4j graph retriever[/status]")

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
                self.logger.error(f"Failed to create {retriever_config.type} retriever: {str(e)}")
                raise ComponentError(f"Failed to create base retriever: {str(e)}") from e
            
        
        if len(retrievers) == 1:
            return retrievers[0]
        
        # status.update("[status]Creating ensemble retriever...[/status]")
        # self.logger.debug("Creating ensemble retriever...")
        if not any(weights):
            # self.logger.warning("No weights specified, using equal weights")
            retriever = EnsembleRetriever(retrievers=retrievers)
            # status.update("[status]Created ensemble retriever[/status]")
            return retriever
        else:
            total = sum(weights)
            weights = [w/total for w in weights]
            # self.logger.debug("Created ensemble retriever")
            retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
            # status.update("[status]Created ensemble retriever[/status]")
            return retriever

    def _create_reranker(self, config: RerankerConfig):
        """Create a reranker from configuration."""
        try:
            self.logger.debug("Creating reranker...")
            if config.type == RerankerType.CUSTOM:
                if "model_name" in config.reranker_kwargs:
                    # Handle custom HuggingFace models
                    Reranker = RERANKER_MAP[config.type]['lazy_load']()
                    ranker = Reranker(
                        config.reranker_kwargs["model_name"],
                        model_type=config.reranker_kwargs.get("model_type"),
                        **{k:v for k,v in config.reranker_kwargs.items() 
                           if k not in ["model_name", "model_type"]}
                    )
                    return ranker.as_langchain_compressor(k=self.final_k)
                return self._instantiate_custom_class(
                    config.custom_class,
                    **config.reranker_kwargs
                )
            
            if config.type not in RERANKER_MAP:
                raise ConfigurationError(f"Unsupported reranker type: {config.type}")
            
            if "TOKENIZERS_PARALLELISM" not in os.environ:
                os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
            
            reranker_config = RERANKER_MAP[config.type]
            # Get the Reranker class using lazy loading
            Reranker = reranker_config['lazy_load']()
            
            # Merge configurations, giving precedence to user-provided kwargs
            merged_kwargs = {
                k: v for k, v in reranker_config.items() 
                if k not in config.reranker_kwargs and k != 'lazy_load'
            }
            merged_kwargs.update(config.reranker_kwargs)  # Add/override with user configs
            
            # Create reranker with merged config
            ranker = Reranker(
                config.type.value,  # model name/path
                **merged_kwargs
            )
            self.logger.debug("Created reranker")
            return ranker.as_langchain_compressor(k=self.final_k)
            
        except Exception as e:
            self.logger.error(f"Failed to create reranker: {str(e)}")
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
        console.print("[success]✓ Pipeline execution complete![/success]")
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
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise PipelineError(f"Retrieval failed: {str(e)}") from e

    def retrieve(self, query: str) -> List[Document]:
        """Synchronous version of retrieve."""
        try:
            documents = self.retriever_chain.invoke(query)
            return documents[:self.final_k]
        except Exception as e:
            self.logger.error(f"Retrieval failed: {str(e)}")
            raise PipelineError(f"Retrieval failed: {str(e)}") from e
