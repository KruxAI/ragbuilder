import os
import dotenv
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
 
global_imports = """
import os
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
"""
def rag(**kwargs):
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

    strategy = getChunkingStrategy(**kwargs)
    code_strings.append(strategy['code_string'])
    imports.append(strategy['import_string'])

    kwargs['db_type'] = kwargs['vectorDB_kwargs']['vectorDB']
    vector = getVectorDB(kwargs['db_type'],kwargs['embedding_model'])
    code_strings.append(vector['code_string'])
    imports.append(vector['import_string'])

    if len(kwargs['retriever_kwargs']['retrievers']) > 0:
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
    function_code = import_text+f"""
def rag_pipeline():
    try:
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs) 
        {code_text.replace('\n', '\n        ')}
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {{e}}")

"""

    return function_code

# Example usage
generated_code = rag(
    retriever_kwargs= {'retrievers':[
                                # {'retriever_type':'bm25Retriever','search_type':'similarity','search_kwargs': {"k": 5}},
                                {'retriever_type':'vectorSimilarity','search_type':'similarity','search_kwargs': {"k": 5}}
                                ],
                        "document_compressor_pipeline": ["LongContextReorder"],
                        "EmbeddingsClusteringFilter_kwargs":{"embeddings":"text-embedding-3-large","num_clusters":2,"num_closest":1,"sorted":True},
                        "contextual_compression_retriever":True
                    },
    embedding_kwargs={'embedding_model':'text-embedding-3-large'},
    vectorDB_kwargs={'vectorDB':'chromaDB'},
    retrieval_model="gpt-3.5-turbo",
    loader_kwargs={'input_path':"https://ashwinaravind.github.io/"},
    source_ids=["wiki"],
    chunking_kwargs ={'chunk_strategy':'RecursiveCharacterTextSplitter','chunk_size':1000,'chunk_overlap':200,'breakpoint_threshold_type':'percentile','breakpoint_threshold_value':0.95},
)
print(generated_code)
locals_dict={}
globals_dict = globals()
exec(generated_code,globals_dict,locals_dict)
print(locals_dict)
rag_chain = locals_dict['rag_pipeline']()
res = rag_chain.invoke("how many startups are there in india?")
print(res)

