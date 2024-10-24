from ragbuilder.rag_templates.top_n_templates import top_n_templates
#use below for testing templates
# from ragbuilder.rag_templates.template_testing import top_n_templates
# from ragbuilder.rag_templates.langchain_templates import nuancedCombos
import ragbuilder.rag_templates.langchain_templates as lc_templates
# from ragbuilder.langchain_module.rag import mergerag as rag
from ragbuilder.langchain_module.rag import getCode as rag
# from ragbuilder.router import router 
from ragbuilder.langchain_module.common import setup_logging, progress_state, LOG_LEVEL
import logging
import json
import openai
import optuna
import math
import sqlite3
from optuna.storages import RDBStorage
from optuna.trial import TrialState
from tenacity import retry, stop_after_attempt, wait_random_exponential, before_sleep_log, retry_if_exception_type, retry_if_result
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
# For loading Synthetic Data
import pandas as pd
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from ragas import RunConfig
RUN_CONFIG_TIMEOUT = int(os.getenv('RUN_CONFIG_TIMEOUT', '240'))
RUN_CONFIG_MAX_WORKERS = int(os.getenv('RUN_CONFIG_MAX_WORKERS', '16'))
RUN_CONFIG_MAX_WAIT = int(os.getenv('RUN_CONFIG_MAX_WAIT', '180'))
RUN_CONFIG_MAX_RETRIES = int(os.getenv('RUN_CONFIG_MAX_RETRIES', '10'))
RUN_CONFIG_IS_ASYNC = os.getenv('RUN_CONFIG_IS_ASYNC', 'true').lower() == 'true'
OVERRIDE_BASELINE_RETRIEVERS = os.getenv('OVERRIDE_BASELINE_RETRIEVERS', 'true').lower() == 'true'
DATABASE = 'eval.db' #TODO: Define this in common.py
# Imports needed for Executing the Generated Code
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
import singlestoredb as s2
load_dotenv()

# Get the database URL from the environment variable
SINGLESTOREDB_URL = os.getenv("SINGLESTOREDB_URL")
PGVECTOR_CONNECTION_STRING = os.getenv("PGVECTOR_CONNECTION_STRING")
MILVUS_CONNECTION_STRING = os.getenv("MILVUS_CONNECTION_STRING")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
import dotenv
from langchain_community.document_loaders import *
from langchain_text_splitters import *
from langchain_community.retrievers import AmazonKendraRetriever
from ragatouille import RAGPretrainedModel
from langchain.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
    MergerRetriever,
    MultiQueryRetriever,
    MultiVectorRetriever,
    ParentDocumentRetriever,
    RePhraseQueryRetriever,
    SelfQueryRetriever,
    TimeWeightedVectorStoreRetriever
)
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from rerankers import Reranker
from langchain_core.documents import Document
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
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import LLMChain, HypotheticalDocumentEmbedder
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.document_loaders import *
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from operator import itemgetter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from ragbuilder.graph_utils.graph_loader import load_graph 
import chromadb
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

def get_model_obj(model_type: str, model: str, temperature: Optional[float] = None):
    if model_type == 'embedding':
        code=getEmbedding(embedding_model=model)
    elif model_type == 'llm':
        code=getLLM(retrieval_model=model, temperature=temperature)
    code_str=f"\n{code['import_string']}\n\n{code['code_string']}"
    locals_dict={}
    exec(code_str, None, locals_dict)
    return locals_dict[model_type]

def rag_builder_bayes_optimization_optuna(**kwargs):
    run_id=kwargs['run_id']
    src_data=kwargs['src_data']
    selected_templates = kwargs.get('selected_templates', [])
    vectorDB=kwargs['vectorDB']
    min_chunk_size=kwargs.get('min_chunk_size', 1000)
    max_chunk_size=kwargs.get('max_chunk_size', 1000)
    num_runs=kwargs.get('num_runs')
    n_jobs=kwargs.get('n_jobs')
    other_embedding=kwargs.get('other_embedding')
    other_llm=kwargs.get('other_llm')
    logger.info(f'other_embedding={other_embedding}')
    logger.info(f'other_llm={other_llm}')
    eval_framework=kwargs.get('eval_framework') # TODO: Add this as an argument to RagEvaluator
    eval_embedding=kwargs.get('eval_embedding')
    eval_llm=kwargs.get('eval_llm')
    sota_embedding=kwargs.get('sota_embedding')
    sota_llm=kwargs.get('sota_llm')
    test_data=kwargs['test_data'] #loader_kwargs ={'source':'url','input_path': url1},
    test_df=pd.read_csv(test_data)
    test_ds = Dataset.from_pandas(test_df)
    disabled_opts=kwargs['disabled_opts']
    result=None
    # Define the configuration space
    logger.info(f"Initializing RAG parameter set...")
    lc_templates.init(vectorDB, min_chunk_size, max_chunk_size, other_embedding, other_llm)
    # configs_to_run=dict()
    cnt_templates=0
    llm = get_model_obj('llm', eval_llm) 
    embeddings = get_model_obj('embedding', eval_embedding)
    # configs_to_run= {1:{'ragname':'simple_rag'},2:{'ragname':'semantic_chunker'},3:{'ragname':'hyde'},4:{'ragname':'hybrid_rag'},4:{'ragname':'crag'}} 
    #TODO: Add a check to see if the templates are to be included

    
    if kwargs['compare_templates']:
        # configs_to_run.update(top_n_templates)
        cnt_templates = len(selected_templates)

    if kwargs['include_granular_combos']:
        lc_templates.filter_exclusions(exclude_elements=disabled_opts, override_baseline_retrievers=OVERRIDE_BASELINE_RETRIEVERS)
        # space = lc_templates.generate_config_space_optuna(exclude_elements=disabled_opts)
        # logger.info(f"Config space={space}")
        cnt_combos=lc_templates.count_combos() + cnt_templates
        logger.info(f"Number of RAG combinations : {cnt_combos}")
        configs_evaluated=dict()
    
        if cnt_combos < num_runs:
            total_runs=cnt_combos
        else:
            total_runs = num_runs + cnt_templates
    else:
        logger.info(f"Number of RAG combinations : {cnt_templates}")
        total_runs = cnt_templates
    
    progress_state.set_total_runs(total_runs)

    # Run Templates first if templates have been selected
    # configs_to_run= {1:{'ragname':'simple_rag'}}
    for key in selected_templates:
        val = top_n_templates[key]
        progress_state.increment_progress()
        logger.info(f"Running: {progress_state.get_progress()['current_run']}/{progress_state.get_progress()['total_runs']}")
        logger.info(f"SOTA template: {key}: {val['description']}")
        # logger.info(f"Template:{key}: {val['description']}:{val['retrieval_model']}")
        print(val)
        val['loader_kwargs']=src_data
        val['embedding_kwargs']={'embedding_model': sota_embedding}
        val['llm']=sota_llm
        val['run_id']=run_id
        rag_builder=SOTARAGBuilder(val)
        run_config=RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
        # logger.info(f"{repr(run_config)}")
        # time.sleep(30)
        # result=0
        logger.info(f"Evaluating RAG Config #{progress_state.get_progress()['current_run']}... (this may take a while)")
        rageval=eval.RagEvaluator(
            rag_builder,
            test_ds, 
            llm = llm, 
            embeddings = embeddings, 
            #TODO: Fetch Run Config settings from advanced settings from front-end
            run_config = run_config,
            is_async=RUN_CONFIG_IS_ASYNC
            )
        result=rageval.evaluate()
        logger.debug(f'progress_state={progress_state.get_progress()}')
        rag_manager.cache_rag(rageval.id, rag_builder.rag)
    
    if kwargs['include_granular_combos']:
        # Objective function for Bayesian optimization on the custom RAG configurations
        
        def objective(trial):
            if n_jobs == -1 or n_jobs > 1:
                delay = random.uniform(0, 10)  # Random delay between 0 and 5 seconds
                logger.info(f"Delaying {trial.number} for {delay} seconds to avoid race conditions...")
                time.sleep(delay)
            try:
                config = lc_templates.generate_config_for_trial_optuna(trial)
                states_to_consider = (TrialState.COMPLETE,)
                trials_to_consider = trial.study.get_trials(deepcopy=False, states=states_to_consider)
                for t in reversed(trials_to_consider):
                    if trial.params == t.params:
                        # Use the existing value as trial duplicated the parameters.
                        progress_state.increment_progress()
                        logger.info(f"Config already evaluated with score: {t.value}: {config}")
                        return t.value
                                    
                config['loader_kwargs'] = src_data
                config['run_id'] = run_id
                logger.info(f"Config raw={config}\n\n")
                # logger.info(f"Config={json.dumps(config, indent=4)}\n\n")
                
                progress_state.increment_progress()
                logger.info(f"Running: {progress_state.get_progress()['current_run']}/{progress_state.get_progress()['total_runs']}")
                logger.info(f"Initializing RAG object...")
                rag_builder = RagBuilder(config)
                run_config = RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
                # logger.info(f"{repr(run_config)}")

                # Dummy run to test config structures
                # scores=[0.5890422913, 0.7656478429, 0.5935820215, 0.6100727287, 0.7904418542, 0.8966577465, 0.6205320374, 0.5581511382, 0.5966923152, 0.6609632653, 0.550011964, 0.5402692061, 0.5755822793, 0.6234577837, 0.5905206211, 0.5864179955, 0.6062351971, 0.570672658, 0.7500015656, 0.7747984829, 0.7993104194, 0.781805689, 0.5710751929, 0.6645166332, 0.622714199, 0.6356301621, 0.6241188896, 0.8153687664, 0.6827077848, 0.6959527751, 0.8423843881, 0.9609655913, 0.6698080329, 0.5912493806, 0.7359742148, 0.7080427047, 0.6899119678, 0.6105474717, 0.7208188469, 0.695968622, 0.6869681458, 0.7269693914, 0.7424575424, 0.7011177759, 0.8697962711, 0.8088942748, 0.9005903531, 0.8688290896, 0.6666808804, 0.666883309, 0.6888392867, 0.7296173512, 0.6497820307, 0.9349375798, 0.6906564857, 0.7924750533, 0.8931411951, 0.9462395027, 0.881902146, 0.6423630407, 0.7474532458, 0.8388990762, 0.6705516029, 0.7747971947, 0.7218534451, 0.8823771379, 0.8505055572, 0.6567467535, 0.7043667001, 0.6939435603, 0.8808846607, 0.9005438973, 0.8691391629, 0.9763763024, 0.6278870244, 0.7355142518, 0.7633544088, 0.5913903849, 0.626892352, 0.6987860021, 0.6456495151, 0.7416265216, 0.6446452076, 0.7546382667, 0.800226133, 0.9454843785, 0.9280627528, 0.6740895569, 0.7741376011, 0.7247380601, 0.6472672733, 0.8251968841, 0.9085414624, 0.8238757897, 0.6880305725, 0.6632702383, 0.8470425157, 0.6590755791, 0.7576560761, 0.7567810953]
                # try:
                #     score = scores[int(time.time())%100]
                # except:
                #      score = -1
                # return score
                logger.info(f"Evaluating RAG Config #{trial.number+1}... \n(this may take a while)")
                rageval = eval.RagEvaluator(
                    rag_builder,
                    test_ds, 
                    llm = llm, 
                    embeddings = embeddings, 
                    run_config=run_config,
                    is_async=RUN_CONFIG_IS_ASYNC
                )
                ## x=input("Continue? ")
                ## if x.lower() != 'y':
                ##      exit()
                result = rageval.evaluate()
                logger.info(f"Completed evaluation. result={result}...")
                if 'answer_correctness' in result and result['answer_correctness'] != float('NaN'):
                    logger.debug("Answer_correctness: ", result.scores["answer_correctness"])
                    none_records = len(result.scores.filter(lambda x: math.isnan(x['answer_correctness']) if x['answer_correctness'] is not None else False))
                    percent_none = (none_records * 1.0 / len(result.scores)) 
                    if percent_none > 0.2:
                        logger.warning(f"More than 20% of the records have 'answer_correctness' as None. Skipping this config...")
                        return float('NaN')
                    
                    if not progress_state.get_progress()['first_eval_complete']:
                        progress_state.set_first_eval_complete()
                    
                    rag_manager.cache_rag(rageval.id, rag_builder.rag)
                    return result['answer_correctness'] 
                return float('NaN')
                
            except Exception as e:
                logger.error(f"Error while evaluating config: {config}")
                logger.error(f"Error: {e}")
                return float('NaN')

        
        # Run Bayesian optimization
        logger.info(f"Running Bayesian optimization...")
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        storage = RDBStorage(
            url=f"sqlite:///{DATABASE}",
            engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
            heartbeat_interval=60, 
            grace_period=180,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3),
        )
        study = optuna.create_study(
            study_name=str(run_id),
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            # directions=["maximize", "minimize"],
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=num_runs, n_jobs=n_jobs, catch=(RagBuilderException, eval.RagEvaluatorException))
        
        logger.info(f"Completed Bayesian optimization...")

        best_trial = study.best_trial
        # best_trial = max(study.best_trials, key=lambda t: t.values[1])
        # best_params = study.best_params
        # best_score = -study.best_value

        logger.info(f"Best Configuration: {best_trial.number}: {best_trial.params}:")
        logger.info(f"Best Score: {best_trial.values}")

    return 0

def rag_builder(**kwargs):
    run_id=kwargs['run_id']
    src_data=kwargs['src_data']
    selected_templates = kwargs.get('selected_templates', [])
    vectorDB=kwargs['vectorDB']
    min_chunk_size=kwargs.get('min_chunk_size', 1000)
    max_chunk_size=kwargs.get('max_chunk_size', 1000)
    other_embedding=kwargs.get('other_embedding')
    other_llm=kwargs.get('other_llm')
    eval_framework=kwargs.get('eval_framework') # TODO: Add this as an argument to RagEvaluator
    eval_embedding=kwargs.get('eval_embedding')
    eval_llm=kwargs.get('eval_llm')
    sota_embedding=kwargs.get('sota_embedding')
    sota_llm=kwargs.get('sota_llm')
    test_data=kwargs['test_data'] #loader_kwargs ={'source':'url','input_path': url1},
    test_df=pd.read_csv(test_data)
    test_ds = Dataset.from_pandas(test_df)
    disabled_opts=kwargs['disabled_opts']
    result=None
    configs_to_run = []
    cnt_combos = 0

    if kwargs['compare_templates']:
        # configs_to_run.update(top_n_templates)
        logger.info(f"Gathering SOTA RAG configs...")
        cnt_combos = len(selected_templates)
        configs_to_run.append({'type': 'SOTA','configs': selected_templates})

    if kwargs['include_granular_combos']:
        logger.info(f"Initializing RAG parameter set...")
        lc_templates.init(vectorDB, min_chunk_size, max_chunk_size, other_embedding, other_llm)
        custom_combos=lc_templates.nuancedCombos(disabled_opts)
        configs_to_run.append({'type': 'CUSTOM','configs': custom_combos})
        cnt_combos+=len(custom_combos)


    logger.info(f"Number of RAG combinations : {cnt_combos}")
    progress_state.set_total_runs(cnt_combos)

    for configs in configs_to_run:
        if configs['type'] == 'SOTA':
            for key in configs['configs']:
                val = top_n_templates[key]
                progress_state.increment_progress()
                logger.info(f"Running: {progress_state.get_progress()['current_run']}/{progress_state.get_progress()['total_runs']}")
                logger.info(f"SOTA template: {key}: {val['description']}")
                val['loader_kwargs']=src_data
                val['embedding_kwargs']={'embedding_model': sota_embedding}
                val['llm']=sota_llm
                val['run_id']=run_id
                rag_builder=SOTARAGBuilder(val)
                run_config=RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
                # logger.info(f"{repr(run_config)}")
                # time.sleep(30)
                # result=0
                logger.info(f"Evaluating RAG Config #{progress_state.get_progress()['current_run']}... (this may take a while)")
                rageval=eval.RagEvaluator(
                    rag_builder, # code for rag function
                    test_ds, 
                    llm = get_model_obj('llm', eval_llm), 
                    embeddings = get_model_obj('embedding', eval_embedding), 
                    #TODO: Fetch Run Config settings from advanced settings from front-end
                    run_config = run_config,
                    is_async=RUN_CONFIG_IS_ASYNC
                    )
                result=rageval.evaluate()
                logger.debug(f'progress_state={progress_state.get_progress()}')
                rag_manager.cache_rag(rageval.id, rag_builder.rag)
        if configs['type'] == 'CUSTOM':
            for val in configs['configs'].values():
                progress_state.increment_progress()
                logger.info(f"Running: {progress_state.get_progress()['current_run']}/{progress_state.get_progress()['total_runs']}")
                val['loader_kwargs']=src_data
                val['run_id']=run_id
                rag_builder=RagBuilder(val)
                run_config=RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
                # logger.info(f"{repr(run_config)}")
                # time.sleep(30)
                # result=0
                logger.info(f"Evaluating RAG Config #{progress_state.get_progress()['current_run']}... (this may take a while)")
                rageval=eval.RagEvaluator(
                    rag_builder, # code for rag function
                    test_ds, 
                    llm = get_model_obj('llm', eval_llm), 
                    embeddings = get_model_obj('embedding', eval_embedding), 
                    #TODO: Fetch Run Config settings from advanced settings from front-end
                    run_config = run_config,
                    is_async=RUN_CONFIG_IS_ASYNC
                    )
                result=rageval.evaluate()
                rag_manager.cache_rag(rageval.id, rag_builder.rag)
    return result

import importlib
class SOTARAGBuilder:
    def __init__(self, val):
        self.config = val
        self.run_id = val['run_id']
        self.loader_kwargs = val['loader_kwargs']
        self.retrieval_model = val['llm']
        self.embedding_kwargs = val['embedding_kwargs']
        logger.debug(f"SOTA template RAGbuilder Invoked {val}")
        sota_module = importlib.import_module('ragbuilder.rag_templates.sota.'+val['module'])
        logger.debug(f"{val['name']} initiated")
        self.router=rag.sota_code_mod(
            code=sota_module.code, 
            input_path=self.loader_kwargs['input_path'],
            retrieval_model=self.retrieval_model,
            embedding_kwargs=self.embedding_kwargs
        )
        locals_dict={}
        globals_dict = globals()

        logger.info("Creating RAG object from generated code...(this may take a while in some cases)")
        try:
            logger.debug(f"Generated Code\n{self.router}")
            exec(self.router,globals_dict,locals_dict)
            self.rag = locals_dict['rag_pipeline']()
        except Exception as e:
            logger.error(f"Error invoking RAG. ERROR: {e}")
    
    def __repr__(self):
        try:
            json_config=json.dumps(self.config)
        except Exception as e:
            logger.error(f"Error serializing RAG config as JSON: {e}")
            logger.debug(f"self.config = {self.config}")
            raw_config=str(self.config).replace("'", '"')
            return json.dumps({"msg": "Failed to serialize RAG config", "raw_config": raw_config})
        return json_config

class RagBuilderException(Exception):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, min=4, max=60),
        retry=(retry_if_result(lambda result: result is None)
                | retry_if_exception_type(openai.APITimeoutError)
                | retry_if_exception_type(openai.APIError)
                | retry_if_exception_type(openai.APIConnectionError)
                | retry_if_exception_type(openai.RateLimitError)),
        before_sleep=before_sleep_log(logger, LOG_LEVEL)) 
def _exec(code):
    locals_dict={}
    exec(code, None, locals_dict)
    ragchain=locals_dict['rag_pipeline']()
    return ragchain

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
        
        logger.info("Creating RAG object from generated code...(this may take a while in some cases)")
        try:
        #execution os string
            logger.debug(f"Generated Code\n{self.router}")
            self.rag = _exec(self.router)
            # print(f"self.rag = {self.rag}")
        except Exception as e:
            logger.error(f"Error creating RAG object from generated code. ERROR: {e}")
            raise RagBuilderException(f"Error creating RAG object from generated code. ERROR: {e}")

    def __repr__(self):
        try:
            json_config=json.dumps(self.config)
        except Exception as e:
            logger.error(f"Error serializing RAG config as JSON: {e}")
            logger.debug(f"self.config = {self.config}")
            raw_config=str(self.config).replace("'", '"')
            return json.dumps({"msg": "Failed to serialize RAG config", "raw_config": raw_config})
        return json_config

def byor_ragbuilder(test_ds,eval_llm,eval_embedding):
    current_directory = os.getcwd()
    print(f"Current Working Directory: {current_directory}")
    folder_path = current_directory+'/byor'

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Only consider .py files
        print('filename',filename)
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Strip .py extension
            module_path = os.path.join(folder_path, filename)
            
            # Dynamically load the module
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the module has 'rag_pipline' function and execute it
            if hasattr(module, 'rag_pipeline'):
                print(f"found rag_pipeline() from {module_name}")
                logger.info("BYOR Ragbuilder Initiated")
                progress_state.increment_progress()
                logger.info(f"Running: {progress_state.get_progress()['current_run']}/{progress_state.get_progress()['total_runs']}")
                # logger.info(f"Template:{key}: {val['description']}:{val['retrieval_model']}")
                logger.info("BYOR Ragbuilder Class Initiated")
                run_config=RunConfig(timeout=RUN_CONFIG_TIMEOUT, max_workers=RUN_CONFIG_MAX_WORKERS, max_wait=RUN_CONFIG_MAX_WAIT, max_retries=RUN_CONFIG_MAX_RETRIES)
                logger.info(f"Evaluating RAG Config #{progress_state.get_progress()['current_run']}... (this may take a while)")
                rag_builder=module.rag_pipline()
                rageval=eval.RagEvaluator(
                    rag_builder, # code for rag function
                    test_ds, 
                    llm = get_model_obj('llm', eval_llm), 
                    embeddings = get_model_obj('embedding', eval_embedding), 
                    #TODO: Fetch Run Config settings from advanced settings from front-end
                    run_config = run_config,
                    is_async=RUN_CONFIG_IS_ASYNC
                    )
                result=rageval.evaluate()
                logger.debug(f'progress_state={progress_state.get_progress()}')
                rag_manager.cache_rag(rageval.id, rag_builder.rag)

class RagManager:
    def __init__(self):
        self.cache: Dict[int, Any] = {}
        self.last_accessed: Dict[int, float] = {}

    def get_rag(self, eval_id: int, db: sqlite3.Connection):
        if eval_id in self.cache:
            self.last_accessed[eval_id] = time.time()
            return self.cache[eval_id]

        # Fetch configuration from database
        cur = db.execute("SELECT code_snippet FROM rag_eval_summary WHERE eval_id = ?", (eval_id,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"No configuration found for eval_id {eval_id}")

        code_snippet = result['code_snippet']
        
        # Reconstruct RAG pipeline
        rag = _exec(code_snippet)

        self.cache[eval_id] = rag
        self.last_accessed[eval_id] = time.time()
        return rag

    def cache_rag(self, eval_id: int, rag_builder: Any):
        self.cache[eval_id] = rag_builder
        self.last_accessed[eval_id] = time.time()

    def clear_cache(self, max_age: int = 86400):
        current_time = time.time()
        to_remove = [eval_id for eval_id, last_access in self.last_accessed.items() 
                     if current_time - last_access > max_age]
        for eval_id in to_remove:
            del self.cache[eval_id]
            del self.last_accessed[eval_id]

rag_manager = RagManager()