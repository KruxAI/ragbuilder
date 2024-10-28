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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from langchain.retrievers import MergerRetriever,EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
"""
def codeGen(**kwargs):
    logger.info(f"Generating code...")
    imports = []
    code_strings = []
    logger.debug(f"Generating Code:getLLM(**kwargs)")
    llm = getLLM(**kwargs)
    code_strings.append(llm['code_string'])
    imports.append(llm['import_string'])
    kwargs['input_path']=kwargs['loader_kwargs']['input_path']
    logger.debug(f"Generating Code: ragbuilder_loader(**kwargs)")
    docs = ragbuilder_loader(**kwargs)
    code_strings.append(docs['code_string'])
    imports.append(docs['import_string'])
    kwargs['embedding_model']=kwargs['embedding_kwargs']['embedding_model']
    logger.debug(f"Generating Code: getEmbedding(**kwargs)")
    embedding = getEmbedding(**kwargs)
    code_strings.append(embedding['code_string'])
    imports.append(embedding['import_string'])

    kwargs['chunking_kwargs']=kwargs['chunking_kwargs']
    logger.debug(f"Generating Code: getChunkingStrategy(**kwargs)")
    strategy = getChunkingStrategy(**kwargs)
    code_strings.append(strategy['code_string'])
    imports.append(strategy['import_string'])

    kwargs['db_type'] = kwargs['vectorDB_kwargs']['vectorDB']
    logger.debug(f"Generating Code: getVectorDB(**kwargs)")
    vector = getVectorDB(kwargs['db_type'],kwargs['embedding_model'])
    code_strings.append(vector['code_string'])
    imports.append(vector['import_string'])
    logger.debug(f"Retriever String initiated")

    if len(kwargs['retriever_kwargs']['retrievers']) > 0:
        logger.debug(f"Retreiver String Completed{kwargs['retriever_kwargs']}")
        code_strings.append("retrievers=[]")
        for rtr in kwargs['retriever_kwargs']['retrievers']:
            kwargs['retriever_type'] = rtr['retriever_type']
            kwargs['search_type'] = rtr.get('search_type',None)
            kwargs['search_kwargs'] = rtr['search_kwargs']
            logger.debug(f"Generating Code: getRetriever(**kwargs)")
            retriever = getRetriever(**kwargs)
            code_strings.append(retriever['code_string'])
            code_strings.append("retrievers.append(retriever)")
            imports.append(retriever['import_string'])
        code_strings.append('retriever=EnsembleRetriever(retrievers=retrievers)')
    else:
        retriever = getRetriever(**kwargs)
        code_strings.append(retriever['code_string'])
        imports.append(retriever['import_string'])
    logger.debug(f"Retriever String completed")
    if kwargs['retriever_kwargs']['contextual_compression_retriever']:
        code_strings.append("arr_comp=[]")
        for cmp in kwargs['retriever_kwargs']['document_compressor_pipeline']:
            kwargs['compressor'] = cmp
            compressor_code = getCompressors(**kwargs)
            # print(compressor_code)
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
        # prompt = hub.pull("rlm/rag-prompt")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", '''You are a helpful assistant. Answer any questions solely based on the context provided below. If the provided context does not have the relevant facts to answer the question, say I don't know. \n\n<context>\n{{context}}\n</context>'''),
                ("user", "{{question}}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ]
        )
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {{e}}")

##To get the answer and context, use the following code
#res=rag_pipeline().invoke("your prompt here")
#print(res["answer"])
#print(res["context"])

""".format(code_text.replace('\n', '\n        '))
    logger.info(f"Codegen completed")
    return function_code

def sota_code_mod(**kwargs):
    kwargs['embedding_model']=kwargs['embedding_kwargs']['embedding_model']
    code=kwargs['code']
    llm = getLLM(**kwargs)
    embedding = getEmbedding(**kwargs)
    docs = ragbuilder_loader(input_path=kwargs['input_path'])
    codmod=code.replace("{loader_class}",docs['code_string'].replace("\n",'\n        '))
    codmod=codmod.replace("{llm_class}",llm['code_string'].replace("\n",'\n        '))
    codmod=codmod.replace("{embedding_class}",embedding['code_string'].replace("\n",'\n        '))
    # codmod=codmod.replace("\n",'\n        ') 
    return codmod