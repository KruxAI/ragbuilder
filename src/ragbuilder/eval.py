###############################################################################
#     ______            __            __                                      #
#    / ____/   ______ _/ /_  ______ _/ /_____  _____                          #
#   / __/ | | / / __ `/ / / / / __ `/ __/ __ \/ ___/                          #
#  / /___ | |/ / /_/ / / /_/ / /_/ / /_/ /_/ / /                              #
# /_____/ |___/\__,_/_/\__,_/\__,_/\__/\____/_/                               #
#                                                                             #
# Description:                                                                #
# Evaluation module for any RAG function.                                     #
# - Takes as input the RAG function name, eval dataset (type Dataset) with    #
#   the fields "question" and "ground_truth"                                  #
# - Performs the following steps:                                             #
#   1/ Evaluates the RAG against the eval dataset and calculates precision/   #
#      recall/accuracy metrics.                                               #
#   2/ Measures latency                                                       #
#   3/ Calculates cost                                                        #
#   4/ Writes these metrics to a DB                                           #
#   5/ Returns the result metrics set (answer_correctness, faithfulness,      #
#      answer_relevancy, context_precision, context_recall, latency, cost)    #
###############################################################################

# imports
import pandas as pd
import time
import sqlite3
from datetime import datetime, timezone
from datasets import Dataset
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.runnables import graph
from langchain_community.callbacks import get_openai_callback
from ragas import evaluate, RunConfig
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
import logging
import contextlib
from ragbuilder.langchain_module.common import setup_logging

setup_logging()
logger = logging.getLogger("ragbuilder")


DATABASE = 'eval.db'
OPENAI_PRICING = {
    # Per 1K tokens
    "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
    "gpt-3.5-turbo-instruct": {"input": 0.0015, "output": 0.0020},
    "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}, # gpt-3.5-turbo still defaults to gpt-3.5-turbo-0613
    "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4-32k": {"input": 0.06, "output": 0.12},
    "embedding": {
        "hugging_face": 0, 
        "text-embedding-ada-002": 0.0001, 
        "text-embedding-3-small": 0.00002, 
        "text-embedding-3-large": 0.00013
    },
}


class LogStream(object):
    def write(self, data):
        logger.info(data)

class RagEvaluator:
    def __init__(
            self, 
            rag, #Code for RAG function 
            test_dataset, 
            context_fn=None, 
            llm=None, 
            embeddings=None, 
            run_config=RunConfig(timeout=240, max_workers=1, max_wait=180, max_retries=10), 
            is_async=False,
            model_name=None
    ):
        self.id = int(time.time())
        self.rag = rag
        self.rag_fn = rag.rag
        self.code_snippet = rag.router
        # TODO: Change Rag config assignment to RAG config Json that we'll get either via kwargs or object properties of the RAG object
        self.rag_config = repr(rag) 
        self.test_dataset = test_dataset
        self.context_fn = context_fn
        self.llm = llm
        self.embeddings = embeddings
        self.run_config = run_config
        self.is_async = is_async
        self.eval_dataset = None   # TODO: We can invoke prepare_eval_dataset here itself. Currently going ahead with lazy call option.
        self.result_df = None
        if model_name is None:
            self.model_name=self._get_model_name()
        else:
            self.model_name=model_name
        
    
    def _get_model_name(self):
        _g=self.rag_fn.get_graph()
        for node in _g.nodes.values():
            node_name=graph.node_data_str(node)
            if node_name == 'AzureChatOpenAI' or node_name == 'ChatOpenAI':
                # TODO: Add support for other LLMs
                return node.data.model_name
        return None

    def prepare_eval_dataset(self, loops=1):
        #TODO: Validate that loops>=1. Raise exception if not.
        eval_ds=[]
        for row in self.test_dataset:
            # print(f'Invoking retrieval for query: {row["question"]}')
            latency_results=[]
            for _i in range(loops):
                start = time.perf_counter()
                with get_openai_callback() as cb:
                    try:
                        response = dict(self.rag_fn.invoke(row["question"]))
                        tokens=cb.total_tokens
                        cost=cb.total_cost
                    except Exception as e:
                        logger.error(f"Error invoking RAG for question: {row['question']}. \ERROR: {e}")
                        response = {"answer":None, "context":None}
                        tokens=0
                latency_results.append(1000000000*(time.perf_counter() - start))
            eval_ds.append(
                {
                    "eval_id" : self.id,
                    "run_id" : self.rag.run_id,  
                    "eval_ts" : datetime.now(timezone.utc).timestamp(),
                    "question" : row["question"],
                    "answer" : response["answer"],
                    "contexts" : [response["context"]],
                    "ground_truth" : row["ground_truth"],
                    # TODO: Qs - Should we measure latency >1 times to estimate some confidence intervals?
                    "latency": latency_results if loops>1 else latency_results[0],
                    "tokens": tokens,
                    "cost": cost
                }
            )
        eval_df = pd.DataFrame(eval_ds)
        self.eval_dataset = Dataset.from_pandas(eval_df) # TODO: Use from_dict instead on eval_ds directly?
        return self.eval_dataset
    
    def evaluate(self):
        # Prepare eval dataset
        self.prepare_eval_dataset()

        with contextlib.redirect_stdout(LogStream()):
        # Evaluate RAG retrieval precision/recall & generation accuracy metrics
            result = evaluate(
                self.eval_dataset,
                metrics=[
                    # TODO: Add flexibility to choose metrics. Current set is not exhaustive
                    answer_correctness,
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
                raise_exceptions=False, 
                llm=self.llm,
                embeddings=self.embeddings,
                is_async=self.is_async,
                run_config=self.run_config
            )
            
            self.result_df = result.to_pandas()
            logger.info(f"Eval: Evaluation complete for {self.id}")

            # Transform "contexts" array to string to save to DB properly
            self.result_df['contexts'] = self.result_df['contexts'].apply(lambda x: x[0])
            
            # Save everything to DB
            self._db_write()
            # return result
            return result['answer_correctness']
            # return self.result_df['answer_correctness'].mean() # TODO: OR result['answer_correctness'] maybe?
    
    def _db_write(self):
        db = sqlite3.connect(DATABASE)
        self.result_df.to_sql(f"rag_eval_details_{self.id}", db, index_label='question_id')
        insert_query=f"""
            INSERT INTO rag_eval_details(
                question_id,
                eval_id,
                run_id,
                eval_ts,
                question,
                answer,
                contexts,
                ground_truth,
                answer_correctness,
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                latency,
                tokens,
                cost
            ) 
            SELECT 
                question_id,
                eval_id,
                run_id,
                eval_ts,
                question,
                answer,
                contexts,
                ground_truth,
                answer_correctness,
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                latency,
                tokens,
                cost
            FROM rag_eval_details_{self.id}
        """
        db.execute(insert_query)
        db.commit() 
        
        summary_query=f"""
        INSERT INTO rag_eval_summary (
            run_id,
            eval_id,
            rag_config,
            code_snippet,
            avg_answer_correctness,
            avg_faithfulness,
            avg_answer_relevancy,
            avg_context_precision,
            avg_context_recall,
            avg_tokens,
            avg_cost_per_query,
            avg_latency,
            eval_ts
        )
        SELECT 
            MAX(run_id) as run_id,
            eval_id,
            ? rag_config,
            ? code_snippet,
            avg(answer_correctness) as avg_answer_correctness,
            avg(faithfulness) as avg_faithfulness,
            avg(answer_relevancy) as avg_answer_relevancy,
            avg(context_precision) as avg_context_precision,
            avg(context_recall) as avg_context_recall,
            avg(tokens) as avg_tokens,
            avg(cost) as avg_cost_per_query,
            avg(latency) as avg_latency,
            MAX(eval_ts) as eval_ts
        FROM rag_eval_details
        WHERE eval_id = {self.id}
        GROUP BY eval_id
        """
        db.execute(summary_query, (self.rag_config, self.code_snippet))
        db.commit()
        db.close()

        logger.info(f"Eval: Writing to DB completed for {self.id}")