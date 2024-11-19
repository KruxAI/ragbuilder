import logging
from typing import List, Optional, Any, Dict, Union
from importlib import import_module
from langchain.schema import BaseRetriever, Document
from langchain.retrievers import ContextualCompressionRetriever

from ragbuilder.config.components import (
    RETRIEVER_MAP, RERANKER_MAP,
    RetrieverType, RerankerType
)
from ragbuilder.config.retriever import RetrievalConfig
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
            
            retriever_class = RETRIEVER_MAP.get(self.config.type)
            if not retriever_class:
                raise ConfigurationError(f"Unsupported retriever type: {self.config.type}")

            # Special handling for different retriever types
            if self.config.type == RetrieverType.SIMILARITY:
                return self.vectorstore.as_retriever(
                    search_kwargs={"k": self.config.retriever_k[0]},
                    **self.config.retriever_kwargs
                )
            elif self.config.type == RetrieverType.MMR:
                return self.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": self.config.retriever_k[0],
                        "fetch_k": self.config.retriever_k[0] * 2  # Fetch more for MMR
                    },
                    **self.config.retriever_kwargs
                )
            elif self.config.type == RetrieverType.HYBRID:
                # Implement hybrid retriever logic
                pass
            
            return retriever_class(
                vectorstore=self.vectorstore,
                **self.config.retriever_kwargs
            )
            
        except Exception as e:
            raise ComponentError(f"Failed to create base retriever: {str(e)}") from e

    def _create_reranker(self):
        """Create a reranker from configuration."""
        try:
            if self.config.reranker.type == RerankerType.CUSTOM:
                return self._instantiate_custom_class(
                    self.config.reranker.custom_class,
                    **self.config.reranker.reranker_kwargs
                )
            
            reranker_class = RERANKER_MAP.get(self.config.reranker.type)
            if not reranker_class:
                raise ConfigurationError(f"Unsupported reranker type: {self.config.reranker.type}")
            
            return reranker_class(**self.config.reranker.reranker_kwargs)
            
        except Exception as e:
            raise ComponentError(f"Failed to create reranker: {str(e)}") from e

    def _create_retriever_chain(self) -> BaseRetriever:
        """Create the full retrieval chain including rerankers if specified."""
        retriever = self.base_retriever
        
        # Add rerankers if specified
        if hasattr(self.config, 'reranker') and self.config.reranker:
            reranker = self._create_reranker()
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker,
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