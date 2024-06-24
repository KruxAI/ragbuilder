import logging
from ragbuilder.langchain_module.common import setup_logging
from ragbuilder.langchain_module.llms.llmConfig import *
from ragbuilder.langchain_module.embedding_model.embedding import *
from ragbuilder.langchain_module.chunkingstrategy.langchain_chunking import *
from langchain_community.document_transformers import *
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainExtractor
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
    ParentDocumentRetriever,
    BM25Retriever
)
from langchain.storage import InMemoryStore
from langchain.retrievers.document_compressors import *

setup_logging()
logger = logging.getLogger("ragbuilder")

def getRetriever(**kwargs):
    """
    Retrieve the appropriate retriever based on the specified retriever_type.

    Args:
    - kwargs: Keyword arguments containing retrieval configuration details.

    Returns:
    - Retriever object based on the specified retriever_type.
    """
    retriever_type = kwargs.get('retriever_type')
    if not retriever_type:
        raise ValueError("retriever_type must be provided in kwargs")

    if retriever_type in ["vectorSimilarity", "vectorMMR"]:
        logger.info("Vector Retriever Invoked")
        return kwargs['vectorstore'].as_retriever(search_type=kwargs['search_type'], search_kwargs=kwargs['search_kwargs'])
    
    elif retriever_type == "contextual_compression":
        arr_transformer = []
        compressor_config = kwargs.get('retriever_type_compressor', [])

        if 'LLMChainExtractor' in compressor_config:
            llm = getLLM(**kwargs)
            arr_transformer.append(LLMChainExtractor.from_llm(llm))

        if 'EmbeddingsFilter' in compressor_config:
            arr_transformer.append(EmbeddingsRedundantFilter(embeddings=getEmbedding(**kwargs)))

        if 'EmbeddingsClusteringFilter' in compressor_config:
            arr_transformer.append(EmbeddingsClusteringFilter(
                embeddings=getEmbedding(**kwargs),
                num_clusters=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['num_clusters'],
                num_closest=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['num_closest'],
                sorted=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['sorted']
            ))

        if 'LLMChainFilter' in compressor_config:
            llm = getLLM(**kwargs)
            arr_transformer.append(LLMChainFilter.from_llm(llm))

        if 'LongContextReorder' in compressor_config:
            arr_transformer.append(LongContextReorder())

        pipeline_compressor = DocumentCompressorPipeline(transformers=arr_transformer)
        logger.info(f"Contextual Compression Retriever Invoked: {compressor_config}")
        
        return ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=kwargs['vectorstore'].as_retriever(search_type=kwargs['search_type'], search_kwargs=kwargs['search_kwargs'])
        )

    elif retriever_type == "multiQuery":
        llm = getLLM(**kwargs)
        logger.info("Multi Query Retriever Invoked")
        return MultiQueryRetriever.from_llm(
            retriever=kwargs['vectorstore'].as_retriever(search_type=kwargs['search_type'], search_kwargs=kwargs['search_kwargs']),
            llm=llm
        )

    elif retriever_type == "parentDocFullDoc":
        store = InMemoryStore()
        logger.info("Parent Document (Full) Retriever Invoked")
        pdr_full = ParentDocumentRetriever(
            vectorstore=kwargs['vectorstore'],
            docstore=store,
            child_splitter=getChunkingStrategy(**kwargs)
        )
        pdr_full.add_documents(kwargs['docs'])
        return pdr_full

    elif retriever_type == "parentDocLargeChunk":
        store = InMemoryStore()
        logger.info("Parent Document (Large Chunk) Retriever Invoked")
        parent_kwargs = kwargs.copy()
        parent_kwargs['chunk_size'] = kwargs["parent_chunk_size"]
        parent_kwargs['chunk_overlap'] = kwargs["parent_chunk_overlap"]
        pdr_large_chunk = ParentDocumentRetriever(
            vectorstore=kwargs['vectorstore'],
            docstore=store,
            child_splitter=getChunkingStrategy(**kwargs),
            parent_splitter=getChunkingStrategy(**parent_kwargs)
        )
        pdr_large_chunk.add_documents(kwargs['docs'])
        return pdr_large_chunk

    elif retriever_type == "bm25Retriever":
        logger.info("BM25Retriever Retriever Invoked")
        return BM25Retriever.from_documents(kwargs['docs'])

    else:
        raise ValueError(f"Unsupported retriever_type: {retriever_type}")

def getCompressors(**kwargs):
    """
    Retrieve document compressors based on the specified kwargs.

    Args:
    - kwargs: Keyword arguments containing compressor configuration details.

    Returns:
    - DocumentCompressorPipeline object based on the specified compressors.
    """
    arr_transformer = []
    compressor_config = kwargs.get('retriever_kwargs', {}).get('document_compressor_pipeline', [])

    if 'LLMChainExtractor' in compressor_config:
        llm = getLLM(**kwargs)
        arr_transformer.append(LLMChainExtractor.from_llm(llm))

    if 'EmbeddingsFilter' in compressor_config:
        arr_transformer.append(EmbeddingsRedundantFilter(embeddings=getEmbedding(**kwargs)))

    if 'EmbeddingsClusteringFilter' in compressor_config:
        arr_transformer.append(EmbeddingsClusteringFilter(
            embeddings=getEmbedding(**kwargs),
            num_clusters=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['num_clusters'],
            num_closest=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['num_closest'],
            sorted=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['sorted']
        ))

    if 'LLMChainFilter' in compressor_config:
        llm = getLLM(**kwargs)
        arr_transformer.append(LLMChainFilter.from_llm(llm))

    if 'LongContextReorder' in compressor_config:
        arr_transformer.append(LongContextReorder())
        
    if 'CrossEncoderReranker' in compressor_config:
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        arr_transformer.append(CrossEncoderReranker(model=model, top_n=5))

    pipeline_compressor = DocumentCompressorPipeline(transformers=arr_transformer)
    logger.info(f"Contextual Compression Retriever Invoked: {compressor_config}")

    return pipeline_compressor
