import inspect
import time
import random
import logging
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .components import EvaluatorType

class EvalDataGenerationConfig(BaseModel):
    generator_model: Optional[Any] = Field(default=None, description="Generator model")
    critic_model: Optional[Any] = Field(default=None, description="Critic model")
    embedding_model: Optional[Any] = Field(default=None, description="Embedding model")
    test_size: Optional[int] = Field(default=5, description="Test size")
    distribution: Optional[Dict[str, float]] = Field(default=None, description="Distribution")
    run_config: Optional[Any] = Field(default=None, description="Run configuration")

class OptimizationConfig(BaseModel):
    """Optimization settings"""
    type: Optional[str] = "Optuna"
    n_trials: Optional[int] = Field(default=10, description="Number of trials for optimization")
    n_jobs: Optional[int] = Field(default=1, description="Number of jobs for optimization")
    timeout: Optional[int] = Field(default=None, description="Timeout for optimization")
    storage: Optional[str] = Field(default="sqlite:///eval.db", description="Storage URL for Optuna (e.g., 'sqlite:///optuna.db')")
    study_name: Optional[str] = Field(default=None, description="Name of the Optuna study")
    load_if_exists: Optional[bool] = Field(default=False, description="Load existing study if it exists")
    overwrite_study: Optional[bool] = Field(default=False, description="Overwrite existing study if it exists")
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
                frame = frame.f_back
            else:
                caller_module = 'unknown'
                
            timestamp = int(time.time()*1000 + random.randint(1, 1000))
            self.study_name = f"{caller_module}_{timestamp}"

class EvaluationConfig(BaseModel):
    type: EvaluatorType = Field(default=EvaluatorType.SIMILARITY, description="Type of evaluator to use")
    custom_class: Optional[str] = Field(default=None, description="Path to custom evaluator class")
    test_dataset: Optional[str] = Field(default=None, description="Path to test dataset")
    # llm: Optional[BaseLLM] = Field(default=None, description="LLM configuration")
    # embeddings: Optional[Embeddings] = Field(default=None, description="Embedding configuration")
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