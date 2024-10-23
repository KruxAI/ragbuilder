from abc import ABC, abstractmethod
from .pipeline import DataIngestPipeline
from typing import List
import numpy as np


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, pipeline: DataIngestPipeline) -> float:
        pass

class SimilarityEvaluator(Evaluator):
    def __init__(self, test_dataset: str, top_k=5):
        self.top_k = top_k
        with open(test_dataset, 'r') as f:
            self.test_questions = f.readlines()

    def evaluate(self, pipeline: DataIngestPipeline) -> float:
        total_score = 0
        for question in self.test_questions:
            results = pipeline.indexer.similarity_search_with_relevance_scores(question, k=self.top_k)
            relevance_scores = [score for _, score in results]
            total_score += np.mean(relevance_scores)
        return total_score / len(self.test_questions)
