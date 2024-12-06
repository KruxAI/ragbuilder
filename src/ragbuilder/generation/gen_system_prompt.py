# gen_prompt_template.py.py
### Generates System Prompt Template for Retrieval-Augmented Generation
# Runs the retriever and get the result
# Get the eval dataset
# Reads prompt template list
# Loops through each prompt template and for each qs in eval dataset runs the prompt and get the answer and evals the answers
# Return the top performing prompt template

#############
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ragas import evaluate, RunConfig

#############

import pandas as pd
from datasets import Dataset
from ragbuilder.generation.prompt_templates import load_prompts

def sample_retriever():
    print("rag_get_retriever initiated")
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)

        # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")

        # Document loader
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        docs = loader.load()

        # Embedding model
        embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large")

        # Text splitting and embedding storage
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        # Initialize Chroma database
        c = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name="testindex-ragbuilder-retreiver",
            client_settings=chromadb.config.Settings(allow_reset=True),
        )

        # Retriever setup
        retriever = c.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever])
        print("rag_get_retriever completed")
        return ensemble_retriever
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None
def run_generation(prompt_template,retriever):
    """
    Initializes a Retrieval-Augmented Generation pipeline using LangChain with a customizable prompt template.

    Args:
        prompt_template (str): The system prompt template to test.

    Returns:
        RunnableParallel: The RAG pipeline ready to process queries.
    """
    try:
        print("rag_pipeline initiated")
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
                # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")
        # Prompt setup
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("user", "{question}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ]
        )

        # RAG Chain setup
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context") | RunnableLambda(format_docs))
            .assign(answer=prompt | llm | StrOutputParser())
            .pick(["answer", "context"])
        )
        print("rag_pipeline completed")
        return rag_chain
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None

def eval_prompt_template(eval_dataset):
    print("test_prompt initiated")
    results = {}
    prompt_templates = load_prompts()  # Load the prompt templates

    for prompt in prompt_templates:
        print(f"Testing Prompt: {prompt.name}...")
        pipeline = run_generation(prompt.template, sample_retriever())  # Initialize your RAG pipeline with the current prompt template
        
        if pipeline:
            # Iterate through each question in eval_dataset
            for entry in eval_dataset:
                question = entry.get("question", "")  # Extract the question from the dataset
                ground_truth = entry.get("ground_truth", "")  # Extract the ground truth answer from the dataset
                if not question:
                    continue  # Skip entries without a valid question

                # Invoke pipeline for the question
                result = pipeline.invoke(question)

                # Store results
                if prompt.name not in results:
                    results[prompt.name] = []
                results[prompt.name].append({
                    "prompt_key": prompt.name,
                    "question": question,
                    "answer": result.get("answer", "Error"),
                    "context": result.get("context", "Error"),
                    "ground_truth": ground_truth,
                })
                # break  # Remove this `break` if you want to test all questions in the dataset
        break  # Remove this `break` if you want to test all prompts

    # Convert the results to a list of dictionaries
    output_data = []
    for prompt_key, prompt_results in results.items():
        output_data.extend(prompt_results)

    # Convert to a Dataset directly
    from datasets import Dataset
    results_dataset = Dataset.from_list(output_data)

    # Optionally clean up or format the dataset
    if "context" in results_dataset.column_names:
        results_dataset = results_dataset.map(
            lambda x: {
                **x,
                "contexts": eval(x["context"]) if isinstance(x["context"], str) and x["context"].startswith("[") else [x["context"]],
            }
        )

    print("test_prompt completed")
    return results_dataset
gtdataset=get_eval_dataset(csv_file_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_test_data_lilianweng_gpt-4o_1721032414.736622_SEMI.csv")
results=eval_prompt_template(gtdataset)
dataset_list = results.to_dict()

# Iterate through each row
for idx, row in enumerate(dataset_list["prompt_key"]):
    print(f"Row {idx}:")
    print("Prompt Key:", dataset_list["prompt_key"][idx])
    print("Question:", dataset_list["question"][idx])
    # print("Answer:", dataset_list["answer"][idx])
    # print("Context:", dataset_list["contexts"][idx])
    # print("Ground Truth:", dataset_list["ground_truth"][idx])
    print("---")

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
# def evaluate_prompts(eval_dataset):
#     print("evaluate_prompts initiated")
#     print(eval_dataset)
#     result = evaluate(
#             eval_dataset,
#             metrics=[
#                 answer_correctness,
#                 faithfulness,
#                 answer_relevancy,
#                 context_precision,
#                 context_recall,
#             ],
#             raise_exceptions=False, 
#             llm = AzureChatOpenAI(model="gpt-4o-mini"),
#             embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large"),
#             is_async=True,
#             run_config=RunConfig(timeout=240, max_workers=1, max_wait=180, max_retries=10)
#         )
#     result_df = result.to_pandas()
#     output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_eval_results.csv"
#     selected_columns = ["prompt_key","question","answer","ground_truth","answer_correctness","faithfulness","answer_relevancy","context_precision","context_recall"]
#     result_df[selected_columns].to_csv(output_csv_path, index=False)
#     print("evaluate_prompts completed")
#     return Dataset.from_pandas(result_df[selected_columns])
from ragbuilder.generation.evaluation import RAGASEvaluator
ragasEvaluator=RAGASEvaluator(eval_dataset_path="/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_test_data_lilianweng_gpt-4o_1721032414.736622_SEMI.csv")
final_result=ragasEvaluator.evaluate(results)
for idx, row in enumerate(final_result):
    print(f"Row {idx}:")
    print(row)
    print("---")



## Evaluator
# get the eval dataset
# evalauate dataset

