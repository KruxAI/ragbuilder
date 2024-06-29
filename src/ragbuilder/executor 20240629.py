from ragbuilder.rag_templates.top_n_templates import top_n_templates
from ragbuilder.rag_templates.langchain_templates import nuancedCombos
# from ragbuilder.langchain_module.rag import mergerag as rag
from ragbuilder.langchain_module.rag import getCode as rag
# from ragbuilder.router import router 
from ragbuilder.langchain_module.common import setup_logging
import logging
import json
setup_logging()
from ragbuilder import eval
from dotenv import load_dotenv
# Load environment variables from the .env file (if present)
import os
current_working_directory = os.getcwd()
    # Construct the path to the .env file in the current working directory
dotenv_path = os.path.join(current_working_directory, '.env')
load_dotenv(dotenv_path)
logger = logging.getLogger("ragbuilder")
#Load Sythetic Data
import pandas as pd
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from ragas import RunConfig
RUN_CONFIG_TIMEOUT = int(os.getenv('RUN_CONFIG_TIMEOUT', '240'))
RUN_CONFIG_MAX_WORKERS = int(os.getenv('RUN_CONFIG_MAX_WORKERS', '16'))
RUN_CONFIG_MAX_WAIT = int(os.getenv('RUN_CONFIG_MAX_WAIT', '180'))
RUN_CONFIG_MAX_RETRIES = int(os.getenv('RUN_CONFIG_MAX_RETRIES', '10'))
RUN_CONFIG_IS_ASYNC = os.getenv('RUN_CONFIG_IS_ASYNC', 'true').lower() == 'true'
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-0125", 
    temperature=0.2,
    verbose=True
)
#####
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import *
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import *
from langchain_chroma import *

import os
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import *
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

#####


import time
def rag_builder(**kwargs):
    run_id=kwargs['run_id']
    src_data=kwargs['src_data']
    vectorDB=kwargs['vectorDB']
    test_data=kwargs['test_data'] #loader_kwargs = { 1 : {'source':'url','input_path': url1}},
    test_df=pd.read_csv(test_data)
    test_ds = Dataset.from_pandas(test_df)
    disabled_opts=kwargs['disabled_opts']
    result=None
    if kwargs['compare_templates']:
        for key,val in top_n_templates.items():
                logger.info(f" Top N Templates:{key}:{val['description']}:{val['retrieval_model']}")
                val['loader_kwargs']=src_data
                val['run_id']=run_id
                # print(f"val={val}")
                # print(f"val[retrieval_model]={val['retrieval_model']}")
                rag_builder=RagBuilder(val)
                run_config=RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
                logger.info(f"{repr(run_config)}")
                rageval=eval.RagEvaluator(
                    rag_builder, # code for rag function
                    test_ds, 
                    llm = chat_model, 
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-large"),
                    #TODO: Fetch Run Config settings from advanced settings from front-end
                    run_config = run_config,
                    is_async = RUN_CONFIG_IS_ASYNC
                    )
                result=rageval.evaluate()
        # return result
    if kwargs['include_granular_combos']:
        print(f"vectorDB={kwargs['vectorDB']}")
        for key,val in nuancedCombos(vectorDB,disabled_opts).items():
                logger.info(f" Combination Templates:{key}")
                val['loader_kwargs']=src_data
                val['run_id']=run_id
                # print(f"val={val}")
                # print(f"val[retrieval_model]={val['retrieval_model']}")
                rag_builder=RagBuilder(val)
                run_config=RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
                logger.info(f"{repr(run_config)}")
                rageval=eval.RagEvaluator(
                    rag_builder, # rag function
                    test_ds, 
                    llm=chat_model, 
                    embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
                    #TODO: Fetch Run Config settings from advanced settings from front-end
                    run_config = run_config,
                    is_async = RUN_CONFIG_IS_ASYNC
                    )
                result=rageval.evaluate()
    return result



class RagBuilder:
    def __init__(self, val):
        self.config = val
        self.run_id = val['run_id']
        self.framework=val['framework']
        # self.description=val['description']
        self.retrieval_model = val['retrieval_model']
        self.source_ids = [1]
        self.loader_kwargs =   val['loader_kwargs']
        self.chunking_kwargs=val['chunking_kwargs']
        self.vectorDB_kwargs=val['vectorDB_kwargs']
        self.embedding_kwargs=val['embedding_kwargs']
        self.retriever_kwargs=val['retriever_kwargs']
        # self.prompt_text =  val['prompt_text'] 
        print(f"retrieval model: {self.retrieval_model}")

        # self.router(Configs) # Calls appropriate code generator calls codeGen Within returns Code string
        # namespace={}
        # exec(rag_func_str, namespace) # executes code
        # ragchain=namespace['ragchain'] catch the func object
        # self.runCode=ragchain()

        # output of router is genrated code as string
        self.router=rag.codeGen(
            framework=self.framework,
            # description=self.description,
            retrieval_model = self.retrieval_model,
            source_ids = self.source_ids,
            loader_kwargs = self.loader_kwargs,
            chunking_kwargs=self.chunking_kwargs,
            vectorDB_kwargs=self.vectorDB_kwargs,
            embedding_kwargs=self.embedding_kwargs,
            retriever_kwargs=self.retriever_kwargs
        )
        locals_dict={}
        globals_dict = globals()

        #execution os string
        exec(self.router,globals_dict,locals_dict)
        logger.info(f"Generated Code:\n{self.router}")

        #old rag func hooked to eval
        self.rag = locals_dict['rag_pipeline']()

    # def __repr__(self):
    #     return (
    #             "{"
    #             # f"    run_id={self.run_id!r},\n"
    #             # f"    framework={self.framework!r},\n"
    #             # f"    description={self.description!r},\n"
    #             f'"retrieval_model":{self.retrieval_model!r},'
    #             f'"source":{self.loader_kwargs[1]["input_path"]!r},'
    #             f'"chunking_strategy":{self.chunking_kwargs[1]!r},'
    #             f'"vectorDB_kwargs":{self.vectorDB_kwargs[1]!r},'
    #             f'"embedding_kwargs":{self.embedding_kwargs[1][0]["embedding_model"]!r},'
    #             f'"retriever_kwargs":{self.retriever_kwargs!r}'
    #             # f'"retriever_kwargs":{self.retriever_kwargs[1][self.retrieval_model]!r}'
    #             "}")
    
    def __repr__(self):
        return json.dumps(self.config)