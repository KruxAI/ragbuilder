from typing import Optional, List
from pydantic import BaseModel, Field, validator, ConfigDict
import pandas as pd
from ragbuilder.config.components import lazy_load
from ragbuilder.config.base import ConfigMetadata
from .base import OptimizationConfig, EvaluationConfig, ConfigMetadata
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

    @validator('question')
    def check_question(cls, v):
        if not v.strip():
            raise ValueError('Question is required and cannot be empty.')
        return v
    
    @validator('ground_truth')
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
    

import yaml
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum
import importlib

# Step 1: Lazy Loading Helper Function
# def lazy_load2(module_name: str, class_name: str):
#     # print("lazy_load2",module_name,class_name)
#     try:
#         # Dynamically import the module
#         module = importlib.import_module(module_name)
#         # Get the class from the module
#         return getattr(module, class_name)
#     except Exception as e:
#         raise ValueError(f"Error loading {class_name} from module {module_name}: {e}")


# Step 2: Enum Class for LLM Types
class LLM(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    COHERE = "cohere"
    VERTEXAI = "vertexai"
    BEDROCK = "bedrock"
    JINA = "jina"
    CUSTOM = "custom"

# Step 3: Map LLM Types to Lazy-loaded Embedding Classes
LLM_MAP = {
    LLM.OPENAI: lazy_load("langchain_openai", "ChatOpenAI"),
    LLM.AZURE_OPENAI: lazy_load("langchain_openai", "AzureChatOpenAI"),
}

# Step 4: Define the LLM Configuration Model
class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    type: LLM  # Enum to specify the LLM
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific parameters like model name/type")
    custom_class: Optional[str] = None  # Optional: If using a custom class



from typing import List, Union
from pydantic import Field, BaseModel


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


# Step 4: Define the LLM Configuration Model
class LLMConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    type: LLM  # Enum to specify the LLM
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific parameters like model name/type")
    custom_class: Optional[str] = None  # Optional: If using a custom class

# Step 2: Define Pydantic Model for Individual LLM Configuration
class GenerationConfig(BaseConfig):
    model_config  = ConfigDict(protected_namespaces=())
    type: LLM  # Specifies the LLM type
    model_name: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific parameters")
    prompt_template: Optional[str] = None
    prompt_key: Optional[str] = None
    eval_data_set_path: Optional[str] = None
    local_prompt_template_path: Optional[str] = None
    read_local_only: Optional[bool] = False

# Step 3: Define Pydantic Model for Overall Generation Configuration
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

    @classmethod
    def with_defaults(cls) -> 'GenerationOptionsConfig':
            """Create a DataIngestOptionsConfig with default values

            Args:
                input_source: File path, directory path, or URL for input data
                test_dataset: Optional path to test dataset. If None, synthetic test data will be generated
            
            Returns:
                DataIngestOptionsConfig with default values optimized for quick start
            """
            return cls(
                llms=[
                    LLMConfig(type=LLM.AZURE_OPENAI, model_kwargs={"model": "gpt-4o-mini", "temperature": 0.2}),  
                ],
                optimization=OptimizationConfig(
                    n_trials=1,
                    n_jobs=1,
                optimization_direction="maximize"),
                metadata=ConfigMetadata(is_default=True)
            )