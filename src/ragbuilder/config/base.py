import inspect
import time
import random
import logging
from pydantic import BaseModel, Field, PrivateAttr
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from .components import EvaluatorType, LLMType, LLM_MAP, EmbeddingType, EMBEDDING_MAP
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.embeddings import Embeddings

class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": (), "arbitrary_types_allowed": True}
    
    type: Optional[LLMType] = None  # Enum to specify the LLM
    model_kwargs: Optional[Dict[str, Any]] = Field(default=None, description="Model-specific parameters like model name/type")
    _initialized_llm: Optional[Union[BaseChatModel, BaseLLM]] = PrivateAttr(default=None)

    @property
    def llm(self) -> Optional[Union[BaseChatModel, BaseLLM]]:
        """Get initialized LLM instance"""
        if self._initialized_llm:
            return self._initialized_llm
        
        if not self.type:
            raise ValueError("Cannot initialize LLM without type")
            
        llm_class = LLM_MAP[self.type]()
        return llm_class(**(self.model_kwargs or {}))

    @classmethod
    def from_llm(cls, llm: Union[BaseChatModel, BaseLLM]) -> 'LLMConfig':
        """Create LLMConfig from initialized LLM instance"""
        return cls.model_construct(_initialized_llm=llm)

class EmbeddingConfig(BaseModel):
    model_config = {"protected_namespaces": (), "arbitrary_types_allowed": True}
    
    type: Optional[EmbeddingType] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    custom_class: Optional[str] = None
    _initialized_embedding: Optional[Embeddings] = PrivateAttr(default=None)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Get initialized Embedding instance"""
        if self._initialized_embedding:
            return self._initialized_embedding
        
        if not self.type:
            raise ValueError("Cannot initialize Embedding without type")
            
        embedding_class = EMBEDDING_MAP[self.type]()
        return embedding_class(**(self.model_kwargs or {}))

    @classmethod
    def from_embedding(cls, embedding: Embeddings) -> 'EmbeddingConfig':
        """Create EmbeddingConfig from initialized Embedding instance"""
        return cls.model_construct(_initialized_embedding=embedding)

class EvalDataGenerationConfig(BaseModel):
    generator_model: Optional[LLMConfig] = Field(default=None, description="Generator model")
    critic_model: Optional[LLMConfig] = Field(default=None, description="Critic model")
    embedding_model: Optional[Any] = Field(default=None, description="Embedding model")
    test_size: Optional[int] = Field(default=5, description="Test size")
    distribution: Optional[Dict[str, float]] = Field(default=None, description="Distribution")
    run_config: Optional[Any] = Field(default=None, description="Run configuration")

class OptimizationConfig(BaseModel):
    """Optimization settings"""
    type: Optional[str] = "Optuna"
    n_trials: Optional[int] = Field(default=None, description="Number of trials for optimization")
    n_jobs: Optional[int] = Field(default=1, description="Number of jobs for optimization")
    timeout: Optional[int] = Field(default=None, description="Timeout for optimization")
    storage: Optional[str] = Field(default="sqlite:///eval.db", description="Storage URL for Optuna (e.g., 'sqlite:///optuna.db')")
    study_name: Optional[str] = Field(default=None, description="Name of the Optuna study")
    load_if_exists: Optional[bool] = Field(default=False, description="Load existing study if it exists")
    overwrite_study: Optional[bool] = Field(default=True, description="Overwrite existing study if it exists")
    optimization_direction: Optional[str] = Field(default="maximize", description="Whether to maximize or minimize the optimization metric")

    def model_post_init(self, *args, **kwargs):
        if self.study_name is None:
            # Get the caller module name (data_ingest or retriever)
            frame = inspect.currentframe()
            while frame:
                module_name = inspect.getmodule(frame).__name__
                if 'data_ingest' in module_name:
                    caller_module = 'data_ingest'
                    break
                elif 'retriever' in module_name:
                    caller_module = 'retriever'
                    break
                elif 'generation' in module_name:
                    caller_module = 'generation'
                    break
                frame = frame.f_back
            else:
                caller_module = 'unknown'
                
            timestamp = int(time.time()*1000 + random.randint(1, 1000))
            self.study_name = f"{caller_module}_{timestamp}"

class EvaluationConfig(BaseModel):
    type: EvaluatorType = Field(default=EvaluatorType.SIMILARITY, description="Type of evaluator to use")
    custom_class: Optional[str] = Field(default=None, description="Path to custom evaluator class")
    test_dataset: Optional[str] = Field(default=None, description="Path to test dataset")
    llm: Optional[LLMConfig] = Field(default=None, description="LLM configuration")
    embeddings: Optional[Any] = Field(default=None, description="Embedding configuration")
    eval_data_generation_config: Optional[EvalDataGenerationConfig] = Field(default=None, description="Evaluation data generation configuration")
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

class ConfigMetadata(BaseModel):
    is_default: bool = False
    version: str = "1.0"