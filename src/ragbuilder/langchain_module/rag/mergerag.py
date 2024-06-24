import os
import dotenv
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from ragbuilder.langchain_module.common import setup_logging
from ragbuilder.langchain_module.retriever.retriever import *
from ragbuilder.langchain_module.loader.loader import *
import logging
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from ragbuilder.langchain_module.llms.llmConfig import *
from ragbuilder.langchain_module.chunkingstrategy.langchain_chunking import *
from ragbuilder.langchain_module.embedding_model.embedding import *
from ragbuilder.langchain_module.vectordb.vectordb import *
from langchain_community.document_transformers import LongContextReorder
from ragbuilder.langchain_module.common import set_params_helper_by_src

setup_logging()
logger = logging.getLogger("ragbuilder")

def mergerag(**kwargs):
    llm = getLLM(**kwargs)
    logger.info("ragbuilder: Merge RAG Invoked")
    
    for src in kwargs['source_ids']:
        kwargs = set_params_helper_by_src(src, **kwargs)
        logger.info(f"MERGE RAG: Source ID: {src}")
        
        docs = ragbuilder_loader(**kwargs)
        kwargs['docs'] = docs
        
        logger.info(f"MERGE RAG: Document Loader Initialized  {src} : {kwargs['source']} {kwargs['input_path']}")
        logger.info(f"MERGE RAG: Chunking Initialized  {src}:{kwargs['chunk_strategy']}:{kwargs['chunk_size']}:{kwargs['chunk_overlap']}")
        
        splits = getChunkingStrategy(**kwargs)
        
        for em in kwargs['embedding_kwargs'][src]:
            merge_retrievers = []
            embedding_model = em['embedding_model']
            kwargs['embedding_model'] = embedding_model
            
            logger.info(f"MERGE RAG: Vector Initialized  {src}:{kwargs['vectorDB']+'_'+kwargs['embedding_model']}")
            
            kwargs['vectorstore'] = getVectorDB(splits, getEmbedding(**kwargs),kwargs['embedding_model'], kwargs['vectorDB'])
            
            for rtr in kwargs['retriever_kwargs'][src][embedding_model]['retrievers']:
                logger.info(f"MERGE RAG: Retrievers Initialized  {src}:{rtr}")
                kwargs['search_type'] = rtr.get('search_type', 'similarity')
                kwargs['search_kwargs'] = rtr.get('search_kwargs', {"k": 5})
                kwargs['retriever_type'] = rtr.get('retriever_type', 'vectorSimilarity')
                merge_retrievers.append(getRetriever(**kwargs))
            
            lotr = MergerRetriever(retrievers=merge_retrievers)
    
    prompt = hub.pull("rlm/rag-prompt")
    
    if kwargs['retriever_kwargs'].get('contextual_compression_retriever', None) == True:
        logger.info("MERGE RAG: Contextual Compression Retriever Initialized")
        lotr = ContextualCompressionRetriever(
            base_retriever=lotr,
            base_compressor=getCompressors(**kwargs)
        )
    
    def format_docs(docs):
        logger.info("MERGE RAG: Formatting Docs")
        return "\n\n".join(doc.page_content for doc in docs)
    
    logger.info("MERGE RAG: RAG Chain Initialized")
    rag_chain = (
        RunnableParallel(context=lotr, question=RunnablePassthrough())
        .assign(context=itemgetter("context") | RunnableLambda(format_docs))
        .assign(answer=prompt | llm | StrOutputParser())
        .pick(["answer", "context"])
    )
    
    if kwargs.get('prompt_text', None) is None:
        return rag_chain
    else:
        logger.debug("MERGE RAG: RAG Chain Invoked")
        res = rag_chain.invoke(kwargs['prompt_text'])
        logger.debug("MERGE RAG: RAG Chain Completed")
        return res