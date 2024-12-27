from typing import Optional, List, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd
from ragbuilder.config.components import lazy_load, LLMType
from ragbuilder.config.base import ConfigMetadata, LLMConfig
from ragbuilder.core.config_store import ConfigStore
from .base import OptimizationConfig, EvaluationConfig, ConfigMetadata
from .components import EvaluatorType
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import yaml

# Define Pydantic Model for the Prompt Template
class PromptTemplate(BaseModel):
    name: str
    template: str


# Define the Execution Model for Each Question
class QuestionContext(BaseModel):
    question: str
    ground_truth: str



# Define the Result Model
class ExecutionResult(BaseModel):
    prompt_name: str
    question: str
    context: str
    generated_response: str

# # Define the Pydantic Model for dataset rows
# class EvalDatasetEntry(BaseModel):
#     question: str
#     ground_truth: Optional[str] = None  # Ground truth is optional

# # Define a Wrapper Model for the entire dataset
# class EvalDataset(BaseModel):
#     entries: List[EvalDatasetEntry]


class EvalDatasetItem(BaseModel):
    question: str
    ground_truth: str
    contexts: Optional[str] = None  # Optional field
    evolution_type: Optional[str] = None
    metadata: Optional[str] = None
    episode_done: Optional[bool] = None

    @field_validator('question', mode='before')
    def check_question(cls, v):
        if not v.strip():
            raise ValueError('Question is required and cannot be empty.')
        return v
    
    @field_validator('ground_truth', mode='before')
    def check_ground_truth(cls, v):
        if not v.strip():
            raise ValueError('Ground truth is required and cannot be empty.')
        return v

class EvalDataset(BaseModel):
    items: list[EvalDatasetItem]

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame):
        # Convert the dataframe to a list of EvalDatasetItem instances
        items = [EvalDatasetItem(**row) for row in df.to_dict(orient="records")]
        return cls(items=items)

class BaseConfig(BaseModel):
    """Base configuration shared across all RAG modules"""
    # input_source: Union[str, List[str]] = Field(..., description="File path, directory path, or URL for input data")
    # test_dataset: str = Field(..., description="Path to CSV file containing test questions")
    
    @classmethod
    def from_yaml(cls, file_path: str) -> "GenerationOptionsConfig":
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
        return cls(**config["generation"])

    def to_yaml(self, file_path: str) -> None:
        """Save configuration to a YAML file."""
        with open(file_path, 'w') as file:
            yaml.dump(self.model_dump(), file)

class GenerationConfig(BaseConfig):
    llm: LLMConfig
    prompt_template: Optional[str] = None
    prompt_key: Optional[str] = None

class GenerationOptionsConfig(BaseConfig):
    llms: List[LLMConfig]  # List of LLM configurations
    prompt_template_path: Optional[str] = None
    eval_data_set_path: Optional[str] = None
    local_prompt_template_path: Optional[str] = None
    read_local_only: Optional[bool] = False
    retriever: Optional[Any]=None
    database_logging: Optional[bool] = Field(default=True, description="Whether to log results to the DB")
    database_path: Optional[str] = Field(default="eval.db", description="Path to the SQLite database file")
    optimization: Optional[OptimizationConfig] = Field(
        default_factory=OptimizationConfig,
        description="Optimization configuration"
    )
    evaluation_config: Optional[EvaluationConfig] = Field(
        default_factory=lambda: EvaluationConfig(type=EvaluatorType.RAGAS),
        description="Evaluation configuration"
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.llms:
            self.llms = [ConfigStore().get_default_llm()]

    @classmethod
    def with_defaults(cls) -> 'GenerationOptionsConfig':
        """Create a GenerationOptionsConfig with default values"""        
        return cls(
            llms=[],
            optimization=OptimizationConfig(
                n_trials=ConfigStore().get_default_n_trials(),
                n_jobs=1,
                optimization_direction="maximize"
            ),
            metadata=ConfigMetadata(is_default=True)
        )