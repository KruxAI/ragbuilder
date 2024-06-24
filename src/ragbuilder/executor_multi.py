import asyncio
import logging
import os
import time
import pandas as pd
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from rag_templates.top_n_templates import top_n_templates
from rag_templates.langchain_templates import nuancedCombos
from langchain_module.rag import mergerag as rag
from langchain_module.common import setup_logging
import eval
from dotenv import load_dotenv
import os
# Load environment variables from the .env file (if present)
load_dotenv()

# # Setup logging
setup_logging()
logger = logging.getLogger("ragbuilder")

# # Set the OpenAI API key

# Load test data
test_df = pd.read_csv('/Users/ashwinaravind/Desktop/working_ragbuilder/ragbuilder/arxiv.783857.csv')
test_ds = Dataset.from_pandas(test_df.head(5))

# Initialize the chat model
chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.2,
    verbose=True
)

async def async_evaluate_rag_template(key,rag_builder_instance, test_ds, chat_model):
    """Run the RagEvaluator's evaluate method asynchronously."""
    logger.info(f" async_evaluate_rag_template started {key}")
    nrageval = eval.RagEvaluator(
        rag_builder_instance,
        test_ds,
        llm=chat_model,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
    )
    logger.info(f" asyncio.to_thread(nrageval.evaluate)  {key}")
    result = await asyncio.to_thread(nrageval.evaluate)
    return result

# Define the async rag_builder function
async def rag_builder(**kwargs):
    src_data = kwargs['src_data']
    test_data = kwargs['test_data']
    test_df = pd.read_csv(test_data)
    test_ds = Dataset.from_pandas(test_df)

    tasks = []

    if kwargs.get('compare_templates'):
        for key, val in top_n_templates.items():
            logger.info(f" Top N Templates: {key}: {val['description']}: {val['retrieval_model']}")
            val['loader_kwargs'] = src_data
            print(f"val={val}")
            print(f"val[retrieval_model]={val['retrieval_model']}")
            rag_builder_instance = RagBuilder(val)
            logger.info(f" rag_builder_instance {key}")
            task = async_evaluate_rag_template(key,rag_builder_instance, test_ds, chat_model)
            tasks.append(task)

    if kwargs.get('include_granular_combos'):
        for key, val in nuancedCombos().items():
            logger.info(f" Combination Templates: {key}")
            val['loader_kwargs'] = src_data
            print(f"val={val}")
            print(f"val[retrieval_model]={val['retrieval_model']}")
            rag_builder_instance = RagBuilder(val)
            task = async_evaluate_rag_template(rag_builder_instance, test_ds, chat_model)
            tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results

class RagBuilder:
    def __init__(self, val):
        self.run_id = int(time.time())
        self.framework = val['framework']
        self.description = val['description']
        self.retrieval_model = val['retrieval_model']
        self.source_ids = [1]
        self.loader_kwargs = val['loader_kwargs']
        self.chunking_kwargs = val['chunking_kwargs']
        self.vectorDB_kwargs = val['vectorDB_kwargs']
        self.embedding_kwargs = val['embedding_kwargs']
        self.retriever_kwargs = val['retriever_kwargs']
        print(f"retrieval model: {self.retrieval_model}")
        self.rag = rag.mergerag(
            framework=self.framework,
            description=self.description,
            retrieval_model=self.retrieval_model,
            source_ids=self.source_ids,
            loader_kwargs=self.loader_kwargs,
            chunking_kwargs=self.chunking_kwargs,
            vectorDB_kwargs=self.vectorDB_kwargs,
            embedding_kwargs=self.embedding_kwargs,
            retriever_kwargs=self.retriever_kwargs
        )

    def __repr__(self):
        return f"{self.framework!r}"

# Example usage of the async function
async def main():
    result = await rag_builder(
        src_data={1:{'source': 'url', 'input_path': '/Users/ashwinaravind/DEsktop/working_ragbuilder/ragbuilder/InputFiles/arxiv.pdf'}},
        test_data='arxiv.783857.csv',
        compare_templates=True,
        include_granular_combos=False
    )
    print(result)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
