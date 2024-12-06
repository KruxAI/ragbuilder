from typing import Optional, List
from pydantic import BaseModel, Field, validator
import pandas as pd
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
def lazy_load(module_name: str, class_name: str):
    try:
        # Dynamically import the module
        module = importlib.import_module(module_name)
        # Get the class from the module
        return getattr(module, class_name)
    except Exception as e:
        raise ValueError(f"Error loading {class_name} from module {module_name}: {e}")

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



# Step 2: Define Pydantic Model for Individual LLM Configuration
class GenerationConfig(BaseModel):
    type: LLM  # Specifies the LLM type
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model-specific parameters")

# Step 3: Define Pydantic Model for Overall Generation Configuration
class GenerationOptionsConfig(BaseModel):
    llms: List[GenerationConfig]  # List of LLM configurations

# Step 5: Load YAML File and Parse Configurations
def load_config_from_yaml(file_path: str) -> GenerationConfig:
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)  # Load YAML into a dictionary
    return GenerationConfig(**yaml_data["generation"]) 