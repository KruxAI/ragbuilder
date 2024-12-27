from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from datasets import Dataset
from ragas.metrics import (
    context_precision,
    context_recall
)
from ragas import evaluate
from ragas import RunConfig
import time
from datetime import datetime
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from ragbuilder.retriever.pipeline import RetrieverPipeline
from ragbuilder.core.config_store import ConfigStore
from ragbuilder.core.exceptions import EvaluationError
from ragbuilder.core.logging_utils import console
from ragbuilder.config.base import EvaluationConfig

class Evaluator(ABC):
    """Abstract base class for retrieval evaluation."""
    
    @abstractmethod
    def evaluate(self, pipeline: RetrieverPipeline) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate the pipeline and return the average score and per-question details.
        
        Args:
            pipeline: The retriever pipeline to evaluate
            
        Returns:
            Tuple of (average_score, list of per-question evaluation details)
        """
        pass

class RetrieverF1ScoreEvaluator(Evaluator):
    """Evaluator that uses RAGAS metrics for retrieval evaluation."""
    
    def __init__(self, eval_config: EvaluationConfig):
        """
        Initialize the RAGAS-based evaluator.
        
        Args:
            test_dataset_path: Path to the test dataset JSON file
            eval_config: Evaluation configuration containing LLM and other settings
        """
        # TODO: Add metrics to evaluation config

        self.eval_config = eval_config
        self.test_data = Dataset.from_pandas(pd.read_csv(self.eval_config.test_dataset))

        llm_config = self.eval_config.llm if self.eval_config.llm else ConfigStore().get_default_llm()
        self.llm = llm_config.llm

        embedding_config = self.eval_config.embeddings if self.eval_config.embeddings else ConfigStore().get_default_embeddings()
        self.embeddings = embedding_config.embeddings
        
        # Default RAGAS run configuration values
        default_config = {
            "timeout": 240,
            "max_workers": 16,
            "max_wait": 180,
            "max_retries": 10
        }

        # Use provided run_config values or defaults
        config = eval_config.evaluator_kwargs.get("run_config", {})
        self.run_config = RunConfig(
            timeout=config.get("timeout", default_config["timeout"]),
            max_workers=config.get("max_workers", default_config["max_workers"]),
            max_wait=config.get("max_wait", default_config["max_wait"]),
            max_retries=config.get("max_retries", default_config["max_retries"])
        )

    def evaluate(
        self, 
        pipeline: RetrieverPipeline
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate the retriever pipeline using RAGAS metrics.
        
        Args:
            pipeline: The retriever pipeline to evaluate
            
        Returns:
            Tuple of (average_score, list of per-question evaluation details)
        """
        eval_data = []
        eval_timestamp = int(time.time()*1000)
        
        for row in self.test_data:
            try:
                # Record start time for latency calculation
                start_time = datetime.now()
                
                # Get retrieval results
                retrieved_docs = pipeline.retrieve(row["question"])
                
                # Calculate latency in milliseconds
                latency = (datetime.now() - start_time).total_seconds() * 1000
                
                eval_data.append({
                    "question": row["question"],
                    "ground_truth": row["ground_truth"],
                    "contexts": [doc.page_content for doc in retrieved_docs],
                    "latency": latency,
                    "eval_timestamp": eval_timestamp
                })
            except Exception as e:
                console.print(f"[red]Error retrieving for query '{row['question'][:50]}...': {str(e)}[/red]")
                eval_data.append({
                    "question": row["question"],
                    "error": str(e),
                    "eval_timestamp": eval_timestamp
                })
                continue
        
        # Convert to Dataset format
        eval_dataset = Dataset.from_list(eval_data)
        
        # Run RAGAS evaluation
        try:
            results = evaluate(
                eval_dataset,
                metrics=[
                    context_precision,
                    context_recall
                ],
                llm=self.llm,
                embeddings=self.embeddings,
                is_async=True,
                run_config=self.run_config
            )
            
            # Convert results to detailed format
            result_df = results.to_pandas()
            
            # Calculate average score (using context precision and recall)
            # Calculate F1 score properly using harmonic mean
            precision = np.nanmean(result_df['context_precision'])
            recall = np.nanmean(result_df['context_recall'])
            
            # Handle edge cases where precision + recall might be 0
            if precision + recall > 0:
                avg_score = 2 * (precision * recall) / (precision + recall)
            else:
                avg_score = 0.0
            
            # Prepare detailed results
            question_details = []
            for idx, row in result_df.iterrows():
                # Calculate F1 score, handling case where precision + recall = 0
                precision = row['context_precision']
                recall = row['context_recall']
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                detail = {
                    "question": eval_data[idx]["question"],
                    "metrics": {
                        "context_precision": precision,
                        "context_recall": recall,
                        "f1_score": f1_score
                    },
                    "contexts": eval_data[idx]["contexts"],
                    "ground_truth": eval_data[idx]["ground_truth"],
                    "latency": eval_data[idx]["latency"],
                    "eval_timestamp": eval_data[idx]["eval_timestamp"]
                }
                question_details.append(detail)
            
            console.print(f"\n[green]Average Score: {avg_score:.3f}[/green]\nContext Precision: {np.nanmean(result_df['context_precision']):.3f}\nContext Recall: {np.nanmean(result_df['context_recall']):.3f}")
            
            return avg_score, question_details
            
        except Exception as e:
            raise EvaluationError(f"RAGAS evaluation failed: {str(e)}")

