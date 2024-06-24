# from llama_index.core import SummaryIndex
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# from llama_index.readers.web import SimpleWebPageReader
# import os
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.core import Settings
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.retrievers import RecursiveRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core import get_response_synthesizer
# from llama_index.core.response_synthesizers import ResponseMode
# from llama_index.core.postprocessor import LongContextReorder
# from llama_index.core.node_parser import SentenceSplitter,SemanticSplitterNodeParser
# from llama_index.core import StorageContext
# from llama_index.core import StorageContext
# from llama_index.core.vector_stores.types import VectorStoreQueryMode
# from llama_index.vector_stores.chroma import ChromaVectorStore
# import chromadb
# from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.core import Settings
from llamaindex_module.chunking_strategy.splitter import * 
from llamaindex_module.vectordb.vectordb import * 
from llamaindex_module.loader.loader import * 
from llamaindex_module.postprocessor.postprocessor import * 
from llamaindex_module.retreiver.retreiver import * 
from llamaindex_module.synthesizer.synthesizer import * 
from llamaindex_module.rag.ensemble import * 
from llamaindex_module.rag.queryfusion import * 
from llamaindex_module.llms.llm import * 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine

from langchain_module.common import set_params_helper_by_src
import logging
from langchain_module.common import setup_logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def mergerag(**kwargs):
 
    for src in kwargs['source_ids']:
        kwargs= set_params_helper_by_src(src,**kwargs)
        Settings.llm = getLLM(**kwargs)

    # Load the documents
        documents = getLoader(**kwargs)
        kwargs['documents'] = documents
 
        # split the doxuments into nodes
        nodes =  getSplitter(**kwargs)
        kwargs['nodes'] = nodes

# incase if multiple embedding for a source
        for em in kwargs['embedding_kwargs'][src]:
            embedding_model=em['embedding_model']
            kwargs['embedding_model'] = embedding_model
      
            logger.info(f"MERGE RAG: Vector Initialized  {src}:{kwargs['vectorDB']+'_'+kwargs['embedding_model']}")
            kwargs['index'] = getIndex(**kwargs)
            merge_retrievers = []
        # Define the index and load the nodes
            for rtr in kwargs['retriever_kwargs'][src][embedding_model]['retrievers']:
                
                    logger.info(f"MERGE RAG: Retrievers Initialized  {src}:{rtr}")
                    kwargs['similarity_top_k'] = rtr.get('similarity_top_k',5)
                    kwargs['vector_store_query_mode'] = rtr.get('vector_store_query_mode','similarity')
                    kwargs['retriever_type'] = rtr.get('retriever_type','vector')
                    merge_retrievers.append(getRetriever(**kwargs))
   
            if len(merge_retrievers)>1 and kwargs['retriever_kwargs'].get('QueryFusionRetreiver',None) is not None:
                # Define the retriever
                retriever = QueryFusion(merge_retrievers,kwargs)
            elif len(merge_retrievers)>1 and kwargs['retriever_kwargs'].get('EnsembleRetreiver',None) == True:
                # nodes_with_scores = retriever.retrieve(kwargs['prompt_text'])
                retriever = EnsembleRetriever(merge_retrievers,kwargs)
            else:
                retriever = merge_retrievers[0]
    # Response synthesizer
    response_synthesizer = getResponseSynthesizer(**kwargs)

    # Node postprocessor
    postprocessor = getPostProcessors(**kwargs)


    # Query query_engine
    logger.info("RetrieverQueryEngine Initialized")
    query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=postprocessor,
    )

    # Query Engine Tool
    logger.info("QueryEngineTool Initialized")
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="llama_index",
                description="Input provided by user",
            ),
        ),
    ]
    if kwargs['query_engine_args'].get('SubQuestionQueryEngine',None) == True:
        logger.info("SubQuestionQueryEngine Initialized")
        final_query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            use_async=True,
            verbose=False,
        )
        # execute the query
        respose=final_query_engine.query(kwargs['prompt_text'])
        return respose
    # execute the query
    response=query_engine.query(kwargs['prompt_text'])
    # Collect results
    return response