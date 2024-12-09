import yaml
import time
import random
import logging
from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any, ForwardRef
from dataclasses import dataclass
from pathlib import Path
# from .data_ingest import DataIngestOptionsConfig
# from .retriever import RetrievalOptionsConfig
from .components import EvaluatorType
# from langchain.llms.base import BaseLLM
# from langchain.embeddings.base import Embeddings

DataIngestOptionsConfig = ForwardRef('DataIngestOptionsConfig')
RetrievalOptionsConfig = ForwardRef('RetrievalOptionsConfig')

# class BaseConfig(BaseModel):
#     """Base configuration shared across all RAG modules"""
#     input_source: Union[str, List[str]] = Field(..., description="File path, directory path, or URL for input data")
#     test_dataset: str = Field(..., description="Path to CSV file containing test questions")
    
#     @classmethod
#     def from_yaml(cls, file_path: str) -> 'BaseConfig':
#         """Load configuration from a YAML file."""
#         with open(file_path, 'r') as file:
#             config_dict = yaml.safe_load(file)
#         return cls(**config_dict)

#     def to_yaml(self, file_path: str) -> None:
#         """Save configuration to a YAML file."""
#         with open(file_path, 'w') as file:
#             yaml.dump(self.model_dump(), file)

class OptimizationConfig(BaseModel):
    """Optimization settings"""
    type: Optional[str] = "Optuna"
    n_trials: Optional[int] = Field(default=10, description="Number of trials for optimization")
    n_jobs: Optional[int] = Field(default=1, description="Number of jobs for optimization")
    timeout: Optional[int] = Field(default=None, description="Timeout for optimization")
    storage: Optional[str] = Field(default=None, description="Storage URL for Optuna (e.g., 'sqlite:///optuna.db')")
    study_name: Optional[str] = Field(default=f"data_ingest_{int(time.time()*1000+random.randint(1, 1000))}", description="Name of the Optuna study")
    load_if_exists: Optional[bool] = Field(default=False, description="Load existing study if it exists")
    overwrite_study: Optional[bool] = Field(default=False, description="Overwrite existing study if it exists")
    optimization_direction: Optional[str] = Field(default="maximize", description="Whether to maximize or minimize the optimization metric")

class EvaluationConfig(BaseModel):
    type: EvaluatorType = Field(default=EvaluatorType.SIMILARITY, description="Type of evaluator to use")
    custom_class: Optional[str] = Field(default=None, description="Path to custom evaluator class")
    test_dataset: Optional[str] = Field(default=None, description="Path to test dataset")
    # llm: Optional[BaseLLM] = Field(default=None, description="LLM configuration")
    # embeddings: Optional[Embeddings] = Field(default=None, description="Embedding configuration")
    evaluator_kwargs: Optional[Dict[str, Any]] = Field(
        default = {},
        description="Additional parameters for evaluator initialization"
    )

@dataclass
class LogConfig:
    """Configuration for logging"""
    log_level: int = logging.INFO
    log_file: Optional[str] = None
    show_progress_bar: bool = True
    verbose: bool = False

class RAGConfig(BaseModel):
    """Complete RAG configuration"""
    input_source: Union[str, List[str]] = Field(..., description="File path, directory path, or URL for input data")
    test_dataset: str = Field(..., description="Path to CSV file containing test questions")
    data_ingest: DataIngestOptionsConfig = Field(
        default_factory=DataIngestOptionsConfig,
        description="Data ingestion configuration"
    )
    retrieval: RetrievalOptionsConfig = Field(
        default_factory=RetrievalOptionsConfig,
        description="Retrieval configuration"
    )
    # generator: Optional['GeneratorOptionsConfig'] = None
    log_config: LogConfig = Field(
        default_factory=LogConfig,
        description="Logging settings"
    )

    @classmethod
    def from_yaml(cls, file_path: str) -> 'RAGConfig':
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)
