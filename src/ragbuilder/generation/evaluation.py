"""
Evaluation module for generation components in RAGBuilder.
Uses RAGAS metrics to evaluate generation quality.
"""
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

from datasets import Dataset
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragbuilder.config.base import EvaluationConfig
from ragbuilder.core.config_store import ConfigStore
from ragbuilder.core.exceptions import EvaluationError


class GenerationEvaluator:
    """
    Evaluator for generation components that uses RAGAS metrics.
    Evaluates the quality of generated answers against reference answers.
    """
    
    def __init__(self, eval_config: EvaluationConfig):
        """
        Initialize the evaluator with evaluation configuration.
        
        Args:
            eval_config: Configuration for evaluation including test dataset,
                         LLM settings, and metrics.
        """
        self.logger = logging.getLogger("ragbuilder.evaluator.generation")
        self.eval_config = eval_config
        
        # Get LLM and embeddings from config or defaults
        llm_config = self.eval_config.llm if self.eval_config.llm else ConfigStore().get_default_llm()
        self.llm = llm_config.llm
        
        embedding_config = self.eval_config.embeddings if self.eval_config.embeddings else ConfigStore().get_default_embeddings()
        self.embeddings = embedding_config.embeddings
        
        # Wrap models for RAGAS compatibility if needed
        if not hasattr(self.llm, '_ragasllm'):
            self.llm = LangchainLLMWrapper(self.llm)
            
        if not hasattr(self.embeddings, '_ragasembeddings'):
            self.embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        # Load test dataset
        self.test_dataset_path = self.eval_config.test_dataset
        self.logger.debug(f"Using test dataset: {self.test_dataset_path}")
        
        # Configure RAGAS run settings
        self.run_config = RunConfig(
            timeout=self.eval_config.evaluator_kwargs.get("timeout", 240),
            max_workers=self.eval_config.evaluator_kwargs.get("max_workers", 1),
            max_wait=self.eval_config.evaluator_kwargs.get("max_wait", 180),
            max_retries=self.eval_config.evaluator_kwargs.get("max_retries", 10)
        )
        
        # Initialize metrics
        self.metrics = [
            AnswerCorrectness(llm=self.llm, embeddings=self.embeddings)
        ]
        
    def load_test_dataset(self) -> pd.DataFrame:
        """
        Load the test dataset from the specified path.
        
        Returns:
            DataFrame containing the test dataset
        """
        try:
            df = pd.read_csv(self.test_dataset_path)
            required_columns = ['user_input', 'reference']
            
            # Validate required columns
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Test dataset missing required columns: {', '.join(missing)}")
            
            # Remove rows with missing reference answers
            df = df.dropna(subset=['reference'])
            
            return df
        except Exception as e:
            raise EvaluationError(f"Failed to load test dataset: {str(e)}")
    
    def evaluate_generation(self, pipeline: Any, config_key: str) -> Dict[str, Any]:
        """
        Evaluate a generation pipeline using the test dataset.
        
        Args:
            pipeline: Generation pipeline to evaluate
            config_key: Identifier for the configuration being evaluated
            
        Returns:
            Dictionary with evaluation results including scores and metrics
        """
        try:
            # Load test dataset
            test_df = self.load_test_dataset()
            self.logger.info(f"Evaluating generation pipeline with {len(test_df)} test queries")
            
            # Generate responses for each test query
            response_data = []
            
            for _, row in test_df.iterrows():
                try:
                    # Generate response using the pipeline
                    start_time = datetime.now()
                    result = pipeline.invoke(row['user_input'])
                    latency = (datetime.now() - start_time).total_seconds() * 1000
                    
                    # Extract answer and contexts
                    answer = result.get('answer', '')
                    contexts = result.get('context', '')
                    if isinstance(contexts, list):
                        contexts = "\n".join(contexts)
                    
                    # Add to evaluation dataset
                    response_data.append({
                        'user_input': row['user_input'],
                        'reference': row['reference'],
                        'response': answer,
                        'contexts': contexts,
                        'config_key': config_key,
                        'latency': latency
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error generating response for query: {str(e)}")
                    # Include failed query with error information
                    response_data.append({
                        'user_input': row['user_input'],
                        'reference': row['reference'],
                        'response': f"ERROR: {str(e)}",
                        'contexts': "",
                        'config_key': config_key,
                        'latency': 0,
                        'error': str(e)
                    })
            
            # Convert to RAGAS evaluation dataset
            eval_dataset = EvaluationDataset.from_list(response_data)
            
            # Run RAGAS evaluation
            results = evaluate(
                dataset=eval_dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
                run_config=self.run_config
            )
            
            # Convert to pandas DataFrame for processing
            result_df = results.to_pandas()
            
            # Add config_key and latency back to results
            for i, item in enumerate(response_data):
                if i < len(result_df):
                    result_df.at[i, 'config_key'] = item['config_key']
                    result_df.at[i, 'latency'] = item['latency']
            
            # Calculate aggregate metrics
            metrics = self._calculate_metrics(result_df, response_data)
            
            # Save detailed results for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_df.to_csv(f"generation_eval_{config_key}_{timestamp}.csv", index=False)
            
            return {
                'score': metrics['score'],
                'metrics': metrics,
                'detailed_results': result_df.to_dict('records')
            }
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise EvaluationError(f"Generation evaluation failed: {str(e)}")
    
    def _calculate_metrics(self, result_df: pd.DataFrame, response_data: List[Dict]) -> Dict[str, Any]:
        """
        Calculate aggregate metrics from evaluation results.
        
        Args:
            result_df: DataFrame with RAGAS evaluation results
            response_data: Original response data with additional metadata
            
        Returns:
            Dictionary of calculated metrics
        """
        # Count errors
        errors = sum(1 for item in response_data if 'error' in item)
        total = len(response_data)
        
        # Extract metrics from RAGAS results
        metrics = {}
        for column in result_df.columns:
            if column in ['user_input', 'response', 'reference', 'contexts', 'config_key', 'latency']:
                continue
            if column in result_df.columns:
                metrics[column] = result_df[column].mean()
        
        # Add operational metrics
        metrics.update({
            'error_rate': errors / total if total > 0 else 0,
            'avg_latency': result_df['latency'].mean() if 'latency' in result_df else 0,
            'success_rate': 1 - (errors / total) if total > 0 else 0
        })
        
        # Use answer_correctness directly as the score if available
        if 'answer_correctness' in result_df.columns:
            metrics['score'] = result_df['answer_correctness'].mean()
        else:
            metrics['score'] = 0.0
            
        return metrics