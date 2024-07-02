from ragbuilder.rag_templates.top_n_templates import top_n_templates
# from ragbuilder.rag_templates.langchain_templates import nuancedCombos
import ragbuilder.rag_templates.langchain_templates as lc_templates
# from ragbuilder.langchain_module.rag import mergerag as rag
from ragbuilder.langchain_module.rag import getCode as rag
# from ragbuilder.router import router 
from ragbuilder.langchain_module.common import setup_logging
import logging
import json
from skopt import gp_minimize
from skopt.utils import use_named_args
from skopt.callbacks import DeltaXStopper
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
#Import needed for Executing the Generated Code
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda

import dotenv
from langchain_community.document_loaders import *
from langchain_text_splitters import *
from langchain.retrievers import *
from langchain.retrievers.document_compressors import *
from langchain_community.document_transformers import *
from langchain.retrievers.multi_query import *
from langchain_mistralai.chat_models import *
from langchain_openai import *
from langchain_mistralai import *
from langchain_huggingface import *
from langchain_experimental.text_splitter import *
from langchain_community.embeddings import *
from langchain_chroma import Chroma
from langchain_community.vectorstores import *
from langchain_pinecone import PineconeVectorStore
from langchain.storage import InMemoryStore
 
# import local modules
from ragbuilder.langchain_module.retriever.retriever import *
from ragbuilder.langchain_module.loader.loader import *
from ragbuilder.langchain_module.llms.llmConfig import *
from ragbuilder.langchain_module.chunkingstrategy.langchain_chunking import *
from ragbuilder.langchain_module.embedding_model.embedding import *
from ragbuilder.langchain_module.vectordb.vectordb import *
from ragbuilder.langchain_module.common import setup_logging
import logging
#####


def rag_builder_bayes_optmization(**kwargs):
    run_id=kwargs['run_id']
    src_data=kwargs['src_data']
    vectorDB=kwargs['vectorDB']
    min_chunk_size=kwargs.get('min_chunk_size', 1000)
    max_chunk_size=kwargs.get('max_chunk_size', 1000)
    test_data=kwargs['test_data'] #loader_kwargs ={'source':'url','input_path': url1},
    test_df=pd.read_csv(test_data)
    test_ds = Dataset.from_pandas(test_df)
    disabled_opts=kwargs['disabled_opts']
    result=None
    # Define the configuration space
    lc_templates.set_vectorDB(vectorDB)
    lc_templates.set_arr_chunk_size(min_chunk_size, max_chunk_size)
    space = lc_templates.generate_config_space(exclude_elements=disabled_opts)
    logger.info(f"Config space={space}")
    configs_evaluated=dict()
    
    @use_named_args(space)
    def objective(**params):
        config = lc_templates.generate_config_from_params(params)
        str_config=json.dumps(config)
        score = configs_evaluated.get(str_config, None)
        if score:
            logger.info(f"Config already evaluated with score: {score}: {config}")
            return score
        
        config['loader_kwargs'] = src_data
        config['run_id'] = run_id
        logger.info(f"Config raw={config}\n\n")
        # logger.info(f"Config={json.dumps(config, indent=4)}\n\n")
        
        logger.info(f"Initializing RAG object...")
        rag_builder = RagBuilder(config)
        run_config = RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
        logger.info(f"{repr(run_config)}")

        # Dummy run to test config structures
        # scores=[0.5890422913, 0.7656478429, 0.5935820215, 0.6100727287, 0.7904418542, 0.8966577465, 0.6205320374, 0.5581511382, 0.5966923152, 0.6609632653, 0.550011964, 0.5402692061, 0.5755822793, 0.6234577837, 0.5905206211, 0.5864179955, 0.6062351971, 0.570672658, 0.7500015656, 0.7747984829, 0.7993104194, 0.781805689, 0.5710751929, 0.6645166332, 0.622714199, 0.6356301621, 0.6241188896, 0.8153687664, 0.6827077848, 0.6959527751, 0.8423843881, 0.9609655913, 0.6698080329, 0.5912493806, 0.7359742148, 0.7080427047, 0.6899119678, 0.6105474717, 0.7208188469, 0.695968622, 0.6869681458, 0.7269693914, 0.7424575424, 0.7011177759, 0.8697962711, 0.8088942748, 0.9005903531, 0.8688290896, 0.6666808804, 0.666883309, 0.6888392867, 0.7296173512, 0.6497820307, 0.9349375798, 0.6906564857, 0.7924750533, 0.8931411951, 0.9462395027, 0.881902146, 0.6423630407, 0.7474532458, 0.8388990762, 0.6705516029, 0.7747971947, 0.7218534451, 0.8823771379, 0.8505055572, 0.6567467535, 0.7043667001, 0.6939435603, 0.8808846607, 0.9005438973, 0.8691391629, 0.9763763024, 0.6278870244, 0.7355142518, 0.7633544088, 0.5913903849, 0.626892352, 0.6987860021, 0.6456495151, 0.7416265216, 0.6446452076, 0.7546382667, 0.800226133, 0.9454843785, 0.9280627528, 0.6740895569, 0.7741376011, 0.7247380601, 0.6472672733, 0.8251968841, 0.9085414624, 0.8238757897, 0.6880305725, 0.6632702383, 0.8470425157, 0.6590755791, 0.7576560761, 0.7567810953]
        # try:
        #     score = scores[int(time.time())%100]
        # except:
        #      return -1
        # else:
        #      return -score

        rageval = eval.RagEvaluator(
            rag_builder,
            test_ds, 
            llm=chat_model, 
            embeddings=OpenAIEmbeddings(model="text-embedding-3-large"),
            run_config=run_config,
            is_async=RUN_CONFIG_IS_ASYNC
        )
        # x=input("Continue? ")
        # if x.lower() != 'y':
        #      exit()
        score = rageval.evaluate()
        # time.sleep(60)
        logger.info(f"Adding to configs evaluated...")
        configs_evaluated[str_config]=score
        return -score  # We negate the score because gp_minimize minimizes
    
    # Run Bayesian optimization
    logger.info(f"Running Bayesian optimization...")
    # result = gp_minimize(objective, space, n_calls=20, random_state=42)
    result = gp_minimize(objective, space, n_calls=20, random_state=42) #, callback=DeltaXStopper(1e-8))
    logger.info(f"Completed Bayesian optimization...")

    best_params = result.x
    best_score = -result.fun

    logger.info(f"Best Configuration: {best_params}")
    logger.info(f"Best Score: {best_score}")
    return 0


def rag_builder(**kwargs):
    run_id=kwargs['run_id']
    src_data=kwargs['src_data']
    vectorDB=kwargs['vectorDB']
    min_chunk_size=kwargs.get('min_chunk_size', 1000)
    max_chunk_size=kwargs.get('max_chunk_size', 1000)
    test_data=kwargs['test_data'] #loader_kwargs ={'source':'url','input_path': url1},
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
        lc_templates.set_arr_chunk_size(min_chunk_size, max_chunk_size)
        for key,val in lc_templates.nuancedCombos(vectorDB,disabled_opts).items():
                logger.info(f"Combination Templates: {key}")
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
        logger.info(f"Generated Code\n{self.router}")

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
        try:
            json_config=json.dumps(self.config)
        except Exception as e:
            logger.error(f"Error serializing RAG config as JSON: {e}")
            logger.info(f"self.config = {self.config}")
            raw_config=str(self.config).replace("'", '"')
            return json.dumps({"msg": "Failed to serialize RAG config", "raw_config": raw_config})
        return json_config
             
