import os
import dotenv
import json
from operator import itemgetter
from langchain_community.document_loaders import WebBaseLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
# from ragbuilder.langchain_module.common import setup_logging
from ragbuilder.langchain_module.retriever.retriever import *
from ragbuilder.langchain_module.loader.loader import *
# import logging
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from ragbuilder.langchain_module.llms.llmConfig import *
from ragbuilder.langchain_module.chunkingstrategy.langchain_chunking import *
from ragbuilder.langchain_module.embedding_model.embedding import *
from ragbuilder.langchain_module.vectordb.vectordb import *
from langchain_community.document_transformers import LongContextReorder
from ragbuilder.langchain_module.common import set_params_helper_by_src
from ragbuilder.langchain_module.common import setup_logging
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
global_imports = """
import os
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
"""
def codeGen(**kwargs):
    logger.info(f"Generating Code for Langchain Rag Pipeline")
    imports = []
    code_strings = []

    llm = getLLM(**kwargs)
    code_strings.append(llm['code_string'])
    imports.append(llm['import_string'])
    kwargs['input_path']=kwargs['loader_kwargs']['input_path']
    docs = ragbuilder_loader(**kwargs)
    code_strings.append(docs['code_string'])
    imports.append(docs['import_string'])
    kwargs['embedding_model']=kwargs['embedding_kwargs']['embedding_model']
    embedding_model_str=kwargs['embedding_model']
    embedding = getEmbedding(**kwargs)
    code_strings.append(embedding['code_string'])
    imports.append(embedding['import_string'])

    kwargs['chunking_kwargs']=kwargs['chunking_kwargs']
    strategy = getChunkingStrategy(**kwargs)
    code_strings.append(strategy['code_string'])
    imports.append(strategy['import_string'])

    kwargs['db_type'] = kwargs['vectorDB_kwargs']['vectorDB']
    vector = getVectorDB(kwargs['db_type'],kwargs['embedding_model'])
    code_strings.append(vector['code_string'])
    imports.append(vector['import_string'])
    logger.info(f"Retriever String initiated")

    if len(kwargs['retriever_kwargs']['retrievers']) > 0:
        logger.info(f"Retreiver String Completed{kwargs['retriever_kwargs']}")
        code_strings.append("retrievers=[]")
        for rtr in kwargs['retriever_kwargs']['retrievers']:
            kwargs['retriever_type'] = rtr['retriever_type']
            kwargs['search_type'] = rtr.get('search_type',None)
            kwargs['search_kwargs'] = rtr['search_kwargs']
            retriever = getRetriever(**kwargs)
            code_strings.append(retriever['code_string'])
            code_strings.append("retrievers.append(retriever)")
            imports.append(retriever['import_string'])
        code_strings.append('retriever=MergerRetriever(retrievers=retrievers)')
    else:
        retriever = getRetriever(**kwargs)
        code_strings.append(retriever['code_string'])
        imports.append(retriever['import_string'])
    logger.info(f"Retriever String completed")
    if kwargs['retriever_kwargs']['contextual_compression_retriever']:
        code_strings.append("arr_comp=[]")
        for cmp in kwargs['retriever_kwargs']['document_compressor_pipeline']:
            kwargs['compressor'] = cmp
            compressor_code = getCompressors(**kwargs)
            print(compressor_code)
            code_strings.append(compressor_code['code_string'])
            imports.append(compressor_code['import_string'])
        code_strings.append("pipeline_compressor = DocumentCompressorPipeline(transformers=arr_comp)")
        code_strings.append("retriever=ContextualCompressionRetriever(base_retriever=retriever,base_compressor=pipeline_compressor)")
        imports.append("from langchain.retrievers.document_compressors import EmbeddingsFilter")
        imports.append("from langchain.retrievers import ContextualCompressionRetriever")

    code_text =  "\n" + "\n".join(code_strings)
    import_text="\n".join(imports)+global_imports
    function_code = import_text+"""
def rag_pipeline():
    try:
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs) 
        {0}
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {{e}}")

""".format(code_text.replace('\n', '\n        '))
    logger.info(f"Code completed{function_code}")
    return function_code

# generated_code = codeGen(
#         framework='langchain',
#         description= 'Hybrid RAG using Langchain with RecursiveCharacterTextSplitter using Contextual Compression with vectorSimilarity and vectorMMR retrievers and LongContextReorder', 
#         retrieval_model= 'gpt-3.5-turbo',
#         chunking_kwargs= {
#                         1 : {'chunk_strategy':'RecursiveCharacterTextSplitter','chunk_size':1000,'chunk_overlap':200}},
#         loader_kwargs= {
#                         1 : {'input_path':'https://ashwinaravind.github.io/'}},
#         vectorDB_kwargs= {1 :{'vectorDB':'chromaDB'}},
#         embedding_kwargs= { 1 : [{'embedding_model':'text-embedding-3-large'}]},
#         retriever_kwargs={ 1 : 
#                        { 'text-embedding-3-large': 
#                             {'retrievers':[
#                                 {'retriever_type':'vectorSimilarity','search_type':'similarity','search_kwargs': {"k": 5}},
#                                 {'retriever_type':'vectorMMR','search_type':'mmr','search_kwargs': {"k": 5}}]},},
#                         "document_compressor_pipeline": ["LongContextReorder"],
#                         "EmbeddingsClusteringFilter_kwargs":{"embeddings":"text-embedding-3-large","num_clusters":4,"num_closest":1,"sorted":True},
#                         "contextual_compression_retriever":True
#                     })
# print(generated_code)
# locals_dict={}
# globals_dict = globals()
# exec(generated_code,globals_dict,locals_dict)
# print(locals_dict)
# rag_chain = locals_dict['rag_pipeline']()
# res = rag_chain.invoke("how many startups are there in india?")
# print(res)
# generated_code = codeGen(
#     framework =   'langchain',
#     description =  'Configuration for a LangChain-based retrieval system',
#     retrieval_model =   'gpt-3.5-turbo',
#     chunking_kwargs = {
#             'chunk_strategy': 'RecursiveCharacterTextSplitter',
#             'chunk_size': 1000,
#             'chunk_overlap': 200
#         },
#     loader_kwargs = {'input_path':'https://ashwinaravind.github.io/'},
#     vectorDB_kwargs = {'vectorDB': 'chromaDB'},
#     embedding_kwargs = { 'embedding_model': 'text-embedding-3-large'},
#     retriever_kwargs= {'retrievers': 
#                         [{'retriever_type': 'vectorSimilarity', 'search_type': 'similarity', 'search_kwargs': {'k': 5}}], 
#                         'contextual_compression_retriever': False, 
#                         'compressors': []})
# print(generated_code)
# locals_dict={}
# globals_dict = globals()
# exec(generated_code,globals_dict,locals_dict)
# print(locals_dict)
# rag_chain = locals_dict['rag_pipeline']()
# res = rag_chain.invoke("how many startups are there in india?")
# print(res)

from ragbuilder.rag_templates.langchain_templates import nuancedCombos
def generate_configs(configs):
    generated_configs = []
    logger.info("Starting Testing")
    for config_id, config in configs.items():
        description = f"Configuration {config_id} for a LangChain-based retrieval system"
        logger.info(f"************Running Test:{config_id}*************")
        generated_code = codeGen(
            framework=config['framework'],
            description=description,
            retrieval_model=config['retrieval_model'],
            chunking_kwargs=config['chunking_kwargs'],
            vectorDB_kwargs=config['vectorDB_kwargs'],
            embedding_kwargs=config['embedding_kwargs'],
            retriever_kwargs=config['retriever_kwargs'],
            # loader_kwargs = {'input_path':'https://ashwinaravind.github.io/'},
            loader_kwargs = {'input_path':'https://geektheoria.wordpress.com/2024/04/24/challenges-of-deploying-naive-retrieval-augmented-generation-rag-models-in-production/'},
        )
        print(generated_code)
        logger.info(f"Generated Code")
        locals_dict={}
        globals_dict = globals()
        exec(generated_code,globals_dict,locals_dict)
        print(locals_dict)
        rag_chain = locals_dict['rag_pipeline']()
        # res = rag_chain.invoke("how many startups are there in india?")
        res = rag_chain.invoke("what are challenges if implementing Retrieval-Augmented Generation in production?")
        logger.info(f"Result of Code Execute :{res}")
        logger.info(f"End of Code Execute{config_id} :{res}")
    return 'Done'
# e=["EmbeddingsRedundantFilter", "EmbeddingsClusteringFilter", "LLMChainFilter", "CrossEncoderReranker",'compareTemplates', 'generateSyntheticData', 'gpt-4o', 'gpt-4-turbo', 'search_k_10', 'search_k_20', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'text-embedding-3-small', 'text-embedding-ada-002', 'chunk2000', 'chunk3000', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'SemanticChunker', 'CharacterTextSplitter']
# configs=nuancedCombos('chromaDB', e)
# print(configs)
# print(generate_configs(configs))
# combs=[
    # ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk3000', 'chunk1000', 'text-embedding-3-large', 'text-embedding-ada-002', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_20', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
#  ['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'CharacterTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk3000', 'text-embedding-3-small', 'text-embedding-3-large', 'vectorSimilarity', 'vectorMMR', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_20', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
#  ['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'SemanticChunker', 'HTMLHeaderTextSplitter', 'chunk3000', 'chunk1000', 'text-embedding-3-large', 'text-embedding-ada-002', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_20', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['HTMLHeaderTextSplitter', 'CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk3000', 'text-embedding-3-small', 'text-embedding-3-large', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'search_k_10', 'search_k_20', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo']
# ]
# for c in combs:
#     configs=nuancedCombos('chromaDB', c)
#     print(generate_configs(configs))

# for c in combs:
#     configs=nuancedCombos('chromaDB', c)
#     print(generate_configs(configs))
#     x=input("Continue? ")
#     if x.lower() != 'y':
#         break

# advanced_combos=[
#     # ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder','gpt-4o', 'gpt-4-turbo'],
# ['CharacterTextSplitter', 'SemanticChunker', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter','gpt-4o', 'gpt-4-turbo'],
# # ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
# # ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
# # ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo']
# ]
# for c in advanced_combos:
#     configs=nuancedCombos('chromaDB', c)
#     for config_id, config in configs.items():
#         json_config = json.dumps(config, indent=4)
#         print(json_config)
#         print("\n#---------------------------------------------------------------------------#\n")
#         # x=input("Continue? ")
#         # if x.lower() != 'y':
#         #     break
#     # print(generate_configs(configs))
#     x=input("Continue? ")
#     if x.lower() != 'y':
#         break
 
 

from ragbuilder.rag_templates.top_n_templates import top_n_templates
configs=top_n_templates
def generate_configs(configs):
    generated_configs = []
    logger.info("Starting Testing")
    for config_id, config in configs.items():
        description = f"Configuration {config_id} for a LangChain-based retrieval system"
        logger.info(f"************Running Test:{config_id}*************")
        generated_code = codeGen(
            framework=config['framework'],
            description=description,
            retrieval_model=config['retrieval_model'],
            chunking_kwargs=config['chunking_kwargs'],
            vectorDB_kwargs=config['vectorDB_kwargs'],
            embedding_kwargs=config['embedding_kwargs'],
            retriever_kwargs=config['retriever_kwargs'],
            loader_kwargs = {'input_path':'https://ashwinaravind.github.io/'},
        )
        print(generated_code)
        logger.info(f"Generated Code :{config_id}\n{generated_code}")
        locals_dict={}
        globals_dict = globals()
        exec(generated_code,globals_dict,locals_dict)
        print(locals_dict)
        rag_chain = locals_dict['rag_pipeline']()
        res = rag_chain.invoke("how many startups are there in india?")
        logger.info(f"Result of Code Execute :{res}")
        logger.info(f"End of Code Execute{config_id} :{res}")
        # break
    return 'Done'
print(generate_configs(configs))