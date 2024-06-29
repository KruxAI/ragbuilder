###############################################################################
# Synthetic test data generator                                               #
#                                                                             #
# Description:                                                                #
# Synthetic test data generator module for any source data.                   #
# - Takes as input the source dataset (type files/ directory) from which to   #
#   generate "question" and "ground_truth"                                    #
# - Performs the following steps:                                             #
#      #
###############################################################################

# imports
import os
import pandas as pd
from datetime import datetime, timezone
from ragas import RunConfig
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
import logging
import contextlib
from ragbuilder.langchain_module.loader import loader as l
from ragbuilder.langchain_module.common import setup_logging


setup_logging()
logger = logging.getLogger("ragbuilder")

class LogStream(object):
    def write(self, data):
        logger.info(data)

# Parameters
# 1. Source data
# 2. [Optional] Test size
# 3. [Optional] Distribution
# 4. [Optional] Generator Model
# 5. [Optional] Critic Model
# 6. [Optional] Embedding Model
# 7. [Optional] Run Config

def load_src(src_data):
    logger.info("Loading docs...")
    # loader = DirectoryLoader(src_data, glob="*.md", show_progress=True)
    # docs = loader.load()
    try:
        docs = l.ragbuilder_loader(input_path=src_data,return_code=False)
        if docs:
            logger.info("Completed loading docs")
            # Add the filename attribute in metadata for Ragas
            for doc in docs:
                doc.metadata['filename'] = doc.metadata['source']
            return docs
        else:
            logger.error(f"Error loading docs for synthetic test data generation.")    
    except Exception as e:
        logger.error(f"Error loading docs for synthetic test data generation: {e}")
        raise
    

def generate_data(
        src_data,
        test_size = 5,
        distribution = {'simple': 0.5, 'reasoning': 0.1, 'multi_context': 0.4},
        generator_model = "gpt-4o",
        critic_model = "gpt-4o",
        embedding_model = "text-embedding-3-large",
        run_config = RunConfig(timeout=1000, max_workers=1, max_wait=900, max_retries=5)
):
    dist=dict()
    for k, v in distribution.items():
        if k == 'simple':
            dist[simple]=v
        elif k == 'reasoning':
            dist[reasoning]=v
        elif k == 'multi_context':
            dist[multi_context]=v
    
    # Load source data as docs
    docs=load_src(src_data) 
    if docs:
        generator_llm=ChatOpenAI(model=generator_model, temperature=0.2, max_tokens=800)
        critic_llm = ChatOpenAI(model=critic_model, temperature=0.2)
        embeddings = OpenAIEmbeddings(model=embedding_model)
        generator = TestsetGenerator.from_langchain(
            generator_llm,
            critic_llm,
            embeddings
        )

        logger.info(f"Initiating synthetic data generation...")
        # generate testset
        testset = generator.generate_with_langchain_docs(
            documents=docs, 
            test_size=test_size, 
            distributions=dist, 
            is_async=True, 
            raise_exceptions=True, 
            run_config=run_config,
        )
        logger.info("Completed synthetic data generation")

        ts=datetime.now(timezone.utc).timestamp()
        test_df = testset.to_pandas()
        f_name=f'rag_test_data_{critic_llm.model_name}_{ts}.csv'
        logger.info(f"Writing to csv file: {f_name}")
        test_df.to_csv(f_name)
        return f_name

