from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from ragas import evaluate, RunConfig
import pandas as pd
from datasets import Dataset
# Assuming the following functions are defined elsewhere
def read_csv(csv_file_path,i):
    print("reading csv")
    csv_file_path = csv_file_path+str(i)+".csv"
    results_df = pd.read_csv(csv_file_path)
    if "context" in results_df.columns:
                    results_df["contexts"] = results_df["context"].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
                    )
    eval_dataset = Dataset.from_pandas(results_df)  # Convert to Dataset for Ragas compatibility
    return eval_dataset

def evaluate_prompts(eval_dataset,iterno):
    print("evaluate_prompts initiated")
    print(eval_dataset)
    result = evaluate(
            eval_dataset,
            metrics=[
                answer_correctness,
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            raise_exceptions=False, 
            llm = AzureChatOpenAI(model="gpt-4o-mini"),
            embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large"),
            is_async=True,
            run_config=RunConfig(timeout=240, max_workers=1, max_wait=180, max_retries=10)
        )
    result_df = result.to_pandas()
    output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_eval_results_mutli"+iterno+".csv"
    selected_columns = ["prompt_key","question","answer","ground_truth","answer_correctness","faithfulness","answer_relevancy","context_precision","context_recall"]
    result_df[selected_columns].to_csv(output_csv_path, index=False)
    print("evaluate_prompts completed")
    return Dataset.from_pandas(result_df[selected_columns])

def run_test(iteration_no):
    file_path = '/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_results'
    prompt_test_dataset = read_csv(file_path, iteration_no)
    prompt_eval_results = evaluate_prompts(prompt_test_dataset, iteration_no)
    print(f"Iteration {iteration_no} completed with results: {prompt_eval_results}")

if __name__ == "__main__":
    # Number of iterations
    iteration_numbers = range(10)

    # Using ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        executor.map(run_test, iteration_numbers)
    
    print("All functions completed")
