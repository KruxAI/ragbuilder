import logging
from typing import List, Optional, Any, Dict, Union
from importlib import import_module
from langchain.schema import BaseRetriever, Document
from langchain.storage import InMemoryStore
from langchain.retrievers import ContextualCompressionRetriever, BM25Retriever, MultiQueryRetriever, ParentDocumentRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from rerankers import Reranker

from ragbuilder.config.components import (
    RETRIEVER_MAP, RERANKER_MAP,
    RetrieverType, RerankerType
)
from ragbuilder.config.retriever import RetrievalConfig, RerankerConfig
from ragbuilder.core.logging_utils import console
from ragbuilder.core.exceptions import (
    RAGBuilderError,
    ConfigurationError,
    ComponentError,
    PipelineError
)

class RetrieverPipeline:
    def __init__(self, 
                 config: RetrievalConfig,
                 vectorstore: Any,  # The vectorstore from data_ingest
                 final_k: int):
        """Initialize retriever pipeline with specific configuration.
        
        Args:
            config: Single instance configuration for the pipeline
            vectorstore: Initialized vector store from data ingest
            final_k: Final number of documents to return
            
        Raises:
            ConfigurationError: If required config fields are missing or invalid
        """
        self._validate_config(config)
        self.config = config
        self.vectorstore = vectorstore
        self.final_k = final_k
        self.logger = logging.getLogger("ragbuilder")
        
        # Initialize components
        self.base_retriever = self._create_base_retriever()
        self.retriever_chain = self._create_retriever_chain()

    def _validate_config(self, config: RetrievalConfig) -> None:
        """Validate pipeline configuration."""
        if not config.retriever.type:
            raise ConfigurationError("Retriever type must be specified")
        if config.retriever.type == RetrieverType.CUSTOM and not config.retriever.custom_class:
            raise ConfigurationError("Custom retriever class must be specified")

    def _create_base_retriever(self) -> BaseRetriever:
        """Create the base retriever from configuration."""
        try:
            if self.config.type == RetrieverType.CUSTOM:
                return self._instantiate_custom_class(
                    self.config.custom_class,
                    vectorstore=self.vectorstore,
                    **self.config.retriever_kwargs
                )
            
            # Handle different retriever types
            if self.config.type == RetrieverType.SIMILARITY:
                return self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": self.config.retriever_k[0]},
                    **self.config.retriever_kwargs
                )
            
            elif self.config.type == RetrieverType.MMR:
                return self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": self.config.retriever_k[0],
                        "fetch_k": self.config.retriever_k[0] * 2
                    },
                    **self.config.retriever_kwargs
                )
            
            elif self.config.type == RetrieverType.MULTI_QUERY:
                return MultiQueryRetriever.from_llm(
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": self.config.retriever_k[0]}
                    ),
                    llm=self.config.retriever_kwargs.get("llm"),
                )
            
            elif self.config.type == RetrieverType.PARENT_DOC_RETRIEVER_FULL:
                store = InMemoryStore()
                # parent_splitter = self.config.retriever_kwargs.get("parent_splitter")
                child_splitter = self.config.retriever_kwargs.get("child_splitter")
                
                retriever = ParentDocumentRetriever(
                    vectorstore=self.vectorstore,
                    docstore=store,
                    child_splitter=child_splitter
                    # parent_splitter=parent_splitter if parent_splitter else child_splitter
                )
                
                # TODO: Figure out a way to add documents from the data ingest pipeline
                if docs := self.config.retriever_kwargs.get("documents"):
                    retriever.add_documents(docs)
                
                return retriever
            
            # TODO: Figure out a way to add documents from the data ingest pipeline
            # Also figure out how to handle the splitters
            elif self.config.type == RetrieverType.PARENT_DOC_RETRIEVER_LARGE:
                store = InMemoryStore()
                parent_splitter = self.config.retriever_kwargs.get("parent_splitter")
                child_splitter = self.config.retriever_kwargs.get("child_splitter")
                retriever = ParentDocumentRetriever(
                    vectorstore=self.vectorstore,
                    docstore=store,
                    child_splitter=child_splitter,
                    parent_splitter=parent_splitter
                )
            
            # TODO: Figure out how to get documents from the data ingest pipeline
            elif self.config.type == RetrieverType.BM25:
                return BM25Retriever.from_documents(
                    self.config.retriever_kwargs.get("documents", []),
                    **{k:v for k,v in self.config.retriever_kwargs.items() if k != "documents"}
                )
            
            raise ConfigurationError(f"Unsupported retriever type: {self.config.type}")
            
        except Exception as e:
            raise ComponentError(f"Failed to create base retriever: {str(e)}") from e

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
            
            elif config.type == RerankerType.BCE:
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
                        model_type=config.reranker_kwargs.get("model_type", "cross-encoder"),
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
        retriever = self.base_retriever
        
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

    async def retrieve(self, query: str) -> List[Document]:
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
            raise PipelineError(f"Retrieval failed: {str(e)}") from e

    def retrieve_sync(self, query: str) -> List[Document]:
        """Synchronous version of retrieve."""
        try:
            documents = self.retriever_chain.invoke(query)
            return documents[:self.final_k]
        except Exception as e:
            raise PipelineError(f"Retrieval failed: {str(e)}") from e