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
        code_string = f"""retriever=c.as_retriever(search_type='{kwargs['search_type']}', search_kwargs={{'k': {kwargs['search_kwargs']}}})"""
        import_string = f""

        return {'code_string':code_string,'import_string':import_string}
 

    elif retriever_type == "multiQuery":
        logger.info("Multi Query Retriever Invoked")
        code_string= f"""retriever=MultiQueryRetriever.from_llm(c.as_retriever(search_type='{kwargs['search_type']}', search_kwargs={{'k': {kwargs['search_kwargs']}}}),llm=llm)"""
        import_string = f"""from langchain.retrievers.multi_query import MultiQueryRetriever"""
        return {'code_string':code_string,'import_string':import_string}

    elif retriever_type == "parentDocFullDoc":
        logger.info("Parent Document (Full) Retriever Invoked")
        code_string= f"""
store = InMemoryStore()
retriever=ParentDocumentRetriever(vectorstore=c,docstore=store,child_splitter=splitter)
retriever.add_documents(docs)
        """
        import_string = f"""from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore"""
        return {'code_string':code_string,'import_string':import_string}

    elif retriever_type == "parentDocLargeChunk":
        
        logger.info("Parent Document (Large Chunk) Retriever Invoked")
        parent_kwargs = kwargs.copy()
        parent_kwargs['chunk_size'] = kwargs["chunking_kwargs"]["chunk_size"]*3
        parent_kwargs['chunk_overlap'] = kwargs["chunking_kwargs"]["chunk_overlap"]*3
        parent_kwargs['splitter_name']="parent_splitter"
        parent_chunk_strategy=getChunkingStrategy(**parent_kwargs)
        print('parent_chunk_strategy',parent_chunk_strategy)
        code_string=f"""{parent_chunk_strategy['code_string']}"""
        code_string+="""
store = InMemoryStore()
retriever = ParentDocumentRetriever(vectorstore=c,docstore=store,child_splitter=splitter,parent_splitter=parent_splitter)
retriever.add_documents(docs)"""
        import_string = f"""from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore"""
        return {'code_string':code_string,'import_string':import_string}

    elif retriever_type == "bm25Retriever":
        logger.info("BM25Retriever Retriever Invoked")
        code_string= f"""retriever=BM25Retriever.from_documents(docs)"""
        import_string = f"""from langchain.retrievers import  BM25Retriever"""
        return {'code_string':code_string,'import_string':import_string}

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
    compressor_config = kwargs.get('compressor',None)
    print(compressor_config)
    arr_transformer=[]
    if 'LLMChainExtractor' in compressor_config:
        code_string= f"""arr_comp.append(LLMChainExtractor.from_llm(llm))"""
        import_string = f"""from langchain.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainExtractor"""
        return {'code_string':code_string,'import_string':import_string}

    if 'EmbeddingsFilter' in compressor_config:
        code_string= f"""arr_comp.append(EmbeddingsRedundantFilter(embeddings=embedding))"""
        import_string = f"""from langchain_community.document_transformers import EmbeddingsRedundantFilter"""
        return {'code_string':code_string,'import_string':import_string}

    if 'EmbeddingsClusteringFilter' in compressor_config:
        num_clusters=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['num_clusters']
        num_closest=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['num_closest']
        sorted=kwargs['retriever_kwargs']['EmbeddingsClusteringFilter_kwargs']['sorted']
        code_string= f"""arr_comp.append(EmbeddingsClusteringFilter(embeddings=embedding, num_clusters={num_clusters}, num_closest={num_closest}, sorted={sorted}))"""
        import_string = f"""from langchain_community.document_transformers import EmbeddingsClusteringFilter"""
        return {'code_string':code_string,'import_string':import_string}
    
    if 'EmbeddingsRedundantFilter' in compressor_config:
        code_string= f"""arr_comp.append(EmbeddingsRedundantFilter(embeddings=embedding))"""
        import_string = f"""from langchain_community.document_transformers import EmbeddingsRedundantFilter"""
        return {'code_string':code_string,'import_string':import_string}

    if 'LLMChainFilter' in compressor_config:
        code_string= f"""arr_comp.append(LLMChainFilter.from_llm(llm))"""
        import_string = f"""from langchain.retrievers.document_compressors import LLMChainFilter"""
        return {'code_string':code_string,'import_string':import_string}

    if 'LongContextReorder' in compressor_config:
        code_string= f"""arr_comp.append(LongContextReorder())"""
        import_string = f"""from langchain_community.document_transformers import LongContextReorder"""
        return {'code_string':code_string,'import_string':import_string}
        
    if 'CrossEncoderReranker' in compressor_config:
        code_string= f"""
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
arr_comp.append(CrossEncoderReranker(model=model, top_n=5))"""
        import_string = f"""from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder"""
        return {'code_string':code_string,'import_string':import_string}

    pipeline_compressor = DocumentCompressorPipeline(transformers=arr_transformer)
    logger.info(f"Contextual Compression Retriever Invoked: {compressor_config}")

    return pipeline_compressor
