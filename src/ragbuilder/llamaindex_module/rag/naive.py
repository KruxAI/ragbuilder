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
from llamaindex_module.llms.llm import * 
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from langchain_module.common import set_params_helper
def mergerag(**kwargs):
    args= set_params_helper(**kwargs)
    Settings.llm = getLLM(**args)
    Settings.embed_model = getEmbedding(**args)
    args['embed_model'] = Settings.embed_model
# Load the documents
    documents = getLoader(**args)
    args['documents'] = documents

    # split the doxuments into nodes
    nodes =  getSplitter(**args)
    args['nodes'] = nodes

    # Define the index and load the nodes
    index = getIndexer(**args)
    args['index'] = index

    # Define the retriever
    retreiver=getRetreiver(**args)


    # Response synthesizer
    response_synthesizer = getResponseSynthesizer(**args)

    # Node postprocessor
    postprocessor = getPostProcessors(**args)


    # Query query_engine
    query_engine = RetrieverQueryEngine(
    retriever=retreiver,
    response_synthesizer=response_synthesizer,
    node_postprocessors=postprocessor,
    )

    # Query Engine Tool
    query_engine_tools = [
        QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name="llama_index",
                description="Input provided by user",
            ),
        ),
    ]

    final_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        response_synthesizer=response_synthesizer,
        use_async=True,
    )
    # execute the query
    respose=final_query_engine.query(kwargs['prompt_text'])

    # Collect results
    print(Settings)
    return respose