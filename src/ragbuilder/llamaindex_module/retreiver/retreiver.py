from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.retrievers.bm25 import BM25Retriever
from langchain_module.common import setup_logging
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def getRetriever(**kwargs):
    if kwargs['retriever_type']=="vector":
        logger.info("VectorRetriever loaded")
        if kwargs['vector_store_query_mode'] == "similarity":
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT
        elif kwargs['vector_store_query_mode'] == "hybrid":
            vector_store_query_mode=VectorStoreQueryMode.HYBRID
        elif kwargs['vector_store_query_mode'] == "linear_regression":
            vector_store_query_mode=VectorStoreQueryMode.LINEAR_REGRESSION
        elif kwargs['vector_store_query_mode'] == "logistic_regression":
            vector_store_query_mode=VectorStoreQueryMode.LOGISTIC_REGRESSION
        elif kwargs['vector_store_query_mode'] == "mmr":
            vector_store_query_mode=VectorStoreQueryMode.MMR
        elif kwargs['vector_store_query_mode'] == "semantic_hybrid":
            vector_store_query_mode=VectorStoreQueryMode.SEMANTIC_HYBRID
        elif kwargs['vector_store_query_mode'] == "sparse":
            vector_store_query_mode=VectorStoreQueryMode.SPARSE
        elif kwargs['vector_store_query_mode'] == "svm":
            vector_store_query_mode=VectorStoreQueryMode.SVM
        elif kwargs['vector_store_query_mode'] == "text_search":
            vector_store_query_mode=VectorStoreQueryMode.TEXT_SEARCH
        else:
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT
    elif kwargs['retriever_type']=="bm25":
        logger.info("BM25Retriever loaded")
        return BM25Retriever.from_defaults(nodes=kwargs['nodes'], similarity_top_k=kwargs['similarity_top_k'])
    return kwargs['index'].as_retriever(similarity_top_k=kwargs['similarity_top_k'], vector_store_query_mode=vector_store_query_mode)
