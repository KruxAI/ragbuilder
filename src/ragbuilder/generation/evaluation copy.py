
from abc import ABC, abstractmethod
from datasets import Dataset
import pandas as pd
from ragbuilder.generation.config import EvalDataset
from ragbuilder.generation.utils import get_eval_dataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from ragas import evaluate, RunConfig
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from datetime import datetime

class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, eval_dataset: Dataset) -> Dataset:
        """
        Evaluate the prompt generation Phase and returns detailed results.
        
        Returns:
        Dataset: A dataset containing the evaluation results.
        """
        pass
    def __init__(self) -> None:
        super().__init__()
        self.eval_dataset = None

class RAGASEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()
        print("RAGASEvaluator initiated")
    
    def evaluate(self, eval_dataset: Dataset,llm= AzureChatOpenAI(model="gpt-4o-mini"), embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large"))-> Dataset:
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
                is_async=True,
                run_config=RunConfig(timeout=240, max_workers=1, max_wait=180, max_retries=10)
            )
        result_df = result.to_pandas()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_path = 'rag_eval_results_'+timestamp+'.csv'
        selected_columns = ["prompt_key","prompt","question","answer","ground_truth","answer_correctness","faithfulness","answer_relevancy","context_precision","context_recall"]
        result_df[selected_columns].to_csv(output_csv_path, index=False)
        print("evaluate_prompts completed")
        print(Dataset.from_pandas(result_df[selected_columns]))
        return Dataset.from_pandas(result_df[selected_columns])