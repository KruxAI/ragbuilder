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