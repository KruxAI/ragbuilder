from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Union
import yaml
from .base import OptimizationConfig, EvaluationConfig, LogConfig
from .components import RetrieverType, RerankerType, EvaluatorType

class BaseRetrieverConfig(BaseModel):
    """Configuration for a specific retriever instance"""
    type: RetrieverType
    retriever_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Retriever-specific parameters")
    custom_class: Optional[str] = None
    retriever_k: List[int] = Field(default=[100], description="Number of documents to retrieve")
    weight: float = Field(
        default=1.0,
        description="Weight for this retriever in ensemble combinations"
    )

class RerankerConfig(BaseModel):
    """Configuration for a specific reranker instance"""
    type: RerankerType
    reranker_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Re-ranker-specific parameters")
    custom_class: Optional[str] = None

class RetrievalOptionsConfig(BaseModel):
    """Configuration for retriever optimization options"""
    retrievers: List[BaseRetrieverConfig] = Field(
        default_factory=lambda: [BaseRetrieverConfig(type=RetrieverType.SIMILARITY)],
        description="List of retrievers to try"
    )
    rerankers: Optional[List[RerankerConfig]] = Field(
        default_factory=list,
        description="List of rerankers to try"
    )
    top_k: List[int] = Field(
        default=[3, 5, 10],
        description="Final number of documents to return after all processing"
    )
    log_config: Optional[LogConfig] = Field(default_factory=LogConfig, description="Logging configuration")
    database_logging: Optional[bool] = Field(default=True, description="Whether to log results to the DB")
    database_path: Optional[str] = Field(default="eval.db", description="Path to the SQLite database file")
    optimization: Optional[OptimizationConfig] = Field(
        default_factory=OptimizationConfig,
        description="Optimization configuration"
    )
    evaluation_config: Optional[EvaluationConfig] = Field(
        default_factory=lambda: EvaluationConfig(
            type=EvaluatorType.RAGAS,
            evaluator_kwargs={"metrics": ["precision", "recall", "f1_score"]}
        ),
        description="Evaluation configuration"
    )

class RetrievalConfig(BaseModel):
    retrievers: List[BaseRetrieverConfig] = Field(
        default_factory=lambda: [BaseRetrieverConfig(type=RetrieverType.SIMILARITY)],
        description="List of retrievers to try"
    )
    rerankers: Optional[List[RerankerConfig]] = Field(
        default_factory=list,
        description="List of rerankers to try"
    )
    top_k: int = Field(default=5, description="Number of top results to consider for similarity scoring")

def load_config(file_path: str) -> Union[RetrievalOptionsConfig, BaseRetrieverConfig]:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Check for required fields
    if 'retriever' not in config_dict or 'reranker' not in config_dict:
        raise ValueError("Configuration must include 'retriever' and 'reranker'")
    
    # TODO: Re-think and redo this logic to see if there's a better way
    try:
        return RetrievalOptionsConfig(**config_dict)
    except ValidationError:
        # If it fails, try RetrieverConfig
        try:
            return BaseRetrieverConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

def save_config(config: Union[RetrievalOptionsConfig, BaseRetrieverConfig], file_path: str) -> None:
    """
    Save configuration to a YAML file.
    """
    config.to_yaml(file_path)