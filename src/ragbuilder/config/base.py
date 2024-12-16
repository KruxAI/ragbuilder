import inspect
import time
import random
import logging
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from dataclasses import dataclass
from .components import EvaluatorType

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
            caller_module = inspect.getmodule(frame.f_back).__name__.split('.')[-1]
            timestamp = int(time.time()*1000 + random.randint(1, 1000))
            self.study_name = f"{caller_module}_{timestamp}"

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