from abc import ABC, abstractmethod
from .pipeline import DataIngestPipeline
from ragbuilder.config.base import EvaluationConfig
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime
import time


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, pipeline: DataIngestPipeline) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate the pipeline and return both average score and detailed results.
        
        Returns:
            Tuple containing:
                - float: Primary evaluation score (used for optimization)
                - List[Dict]: Detailed results including additional metrics
        """
        pass


class SimilarityEvaluator(Evaluator):
    def __init__(self, evaluation_config: EvaluationConfig):
        self.evaluation_config = evaluation_config
        if not evaluation_config.test_dataset:
            raise ValueError("test_dataset must be provided in evaluation_config")
        kwargs = evaluation_config.evaluator_kwargs or {}
        self.top_k = kwargs.get("top_k", 5)
        self.position_weights = kwargs.get("position_weights")
        self.relevance_threshold = kwargs.get("relevance_threshold", 0.25)
        
        # Default position weights if none provided
        if self.position_weights is None:
            self.position_weights = [1.0 - (i / (2 * self.top_k)) for i in range(self.top_k)]
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.position_weights)
        self.position_weights = [w / weight_sum for w in self.position_weights]
        
        # Load test questions
        # TODO: Integrate synthetic test data generation
        # TODO: Make this more robust to skip 1st line ONLY if it's a header
        with open(evaluation_config.test_dataset, 'r') as f:
            self.test_questions = [q.strip() for q in f.readlines()[1:] if q.strip()]

    def _calculate_weighted_score(self, relevance_scores: List[float]) -> float:
        """Calculate position-weighted average of relevance scores."""
        # Pad scores if fewer than top_k results
        padded_scores = relevance_scores + [0.0] * (self.top_k - len(relevance_scores))
        
        # If no score meets threshold, return 0
        if max(padded_scores) < self.relevance_threshold:
            return 0.0
        
        # Calculate weighted score
        return sum(score * weight 
                  for score, weight in zip(padded_scores[:self.top_k], self.position_weights))

    def evaluate(self, pipeline: DataIngestPipeline) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate the pipeline using position-weighted relevance scores.
        Additional metrics like latency are included in the detailed results.
        """
        if not self.test_questions:
            raise ValueError("No test questions found in the test dataset")

        total_score = 0.0
        question_details = []
        eval_timestamp = int(time.time()*1000)

        for question in self.test_questions:
            try:
                # Record start time for latency calculation
                start_time = datetime.now()
                
                # Get results from pipeline
                results = pipeline.indexer.similarity_search_with_relevance_scores(
                    question, k=self.top_k
                )
                
                # Calculate latency
                latency = (datetime.now() - start_time).total_seconds()*1000.0
                
                # Extract scores and chunks
                relevance_scores = [score for _, score in results]
                retrieved_chunks = [
                    {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata,
                        "score": score
                    } 
                    for chunk, score in results
                ]
                
                # Calculate question score
                question_score = self._calculate_weighted_score(relevance_scores)
                total_score += question_score
                
                # Collect detailed results
                question_details.append({
                    "question": question,
                    "retrieved_chunks": retrieved_chunks,
                    "relevance_scores": relevance_scores,
                    "weighted_score": question_score,
                    "latency": latency,
                    "eval_timestamp": eval_timestamp
                })
                
            except Exception as e:
                print(f"Error evaluating question '{question[:50]}...': {str(e)}")
                question_details.append({
                    "question": question,
                    "error": str(e),
                    "eval_timestamp": eval_timestamp
                })
                continue

        avg_score = total_score / len(self.test_questions)
        return avg_score, question_details


