from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Dict, Any, Union
import yaml
from .base import OptimizationConfig, EvaluationConfig, ConfigMetadata
from .components import RetrieverType, RerankerType, EvaluatorType
from ragbuilder.core.config_store import ConfigStore

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
    retrievers: List[BaseRetrieverConfig] = Field(default=None, description="List of retrievers to try")
    rerankers: Optional[List[RerankerConfig]] = Field(default=None, description="List of rerankers to try")
    top_k: List[int] = Field(default=None, description="Final number of documents to return after all processing")
    database_logging: Optional[bool] = Field(default=None, description="Whether to log results to the DB")
    database_path: Optional[str] = Field(default=None, description="Path to the SQLite database file")
    optimization: Optional[OptimizationConfig] = Field(default=None, description="Optimization configuration")
    evaluation_config: Optional[EvaluationConfig] = Field(default=None, description="Evaluation configuration")
    metadata: Optional[ConfigMetadata] = None

    def apply_defaults(self) -> None:
        """Apply default values from ConfigStore and set standard defaults"""
        if self.optimization is None:
            self.optimization = OptimizationConfig()

        if self.optimization.n_trials is None:
            self.optimization.n_trials = ConfigStore().get_default_n_trials()

        if self.retrievers is None:
            self.retrievers = [
                BaseRetrieverConfig(type=RetrieverType.VECTOR_SIMILARITY, retriever_k=[20]),
                BaseRetrieverConfig(type="bm25", retriever_k=[20])
            ]

        if self.rerankers is None:
            self.rerankers = [RerankerConfig(type=RerankerType.BGE_BASE)]
        
        if self.top_k is None:
            self.top_k = [3, 5, 10]
        
        if self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig(type=EvaluatorType.RAGAS)
        
        if self.metadata is None:
            self.metadata = ConfigMetadata()

    @classmethod
    def with_defaults(cls) -> 'RetrievalOptionsConfig':
        """Create a RetrievalOptionsConfig with default values optimized for quick start
        
        Args:
            vectorstore: Optional vectorstore to use for retrieval. If provided, will be
                       passed to the retriever configuration.
        
        Returns:
            RetrievalOptionsConfig with default values optimized for quick start
        """
        
        return cls(
            retrievers=[
                BaseRetrieverConfig(
                    type=RetrieverType.VECTOR_SIMILARITY,
                    retriever_k=[20],
                    weight=1
                ),
                BaseRetrieverConfig(
                    type="bm25",
                    retriever_k=[20],
                    weight=1
                )
            ],
            rerankers=[RerankerConfig(type=RerankerType.BGE_BASE)],
            top_k=[3, 5],
            optimization=OptimizationConfig(
                n_trials=ConfigStore().get_default_n_trials(),
                n_jobs=1,
                optimization_direction="maximize"
            ),
            evaluation_config=EvaluationConfig(type=EvaluatorType.RAGAS),
            metadata=ConfigMetadata(is_default=True)
        )

class RetrievalConfig(BaseModel):
    retrievers: List[BaseRetrieverConfig]
    rerankers: Optional[List[RerankerConfig]] = None
    top_k: int

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