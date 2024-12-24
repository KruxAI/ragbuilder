import logging
import os
import re
from typing import Optional, List, Union, Any, Dict, Tuple
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    DirectoryLoader, 
    WebBaseLoader, 
    UnstructuredFileLoader
)
from langchain_core.documents import Document
from ragbuilder.config.components import COMPONENT_ENV_REQUIREMENTS
from ragbuilder.config.data_ingest import DataIngestOptionsConfig
from ragbuilder.config.retriever import RetrievalOptionsConfig
import json
import importlib

logger = logging.getLogger(__name__)
os.environ['USER_AGENT'] = "ragbuilder"

def load_documents(input_path: str) -> List[Document]:
    """
    Load documents from a file, directory, or URL.
    
    Args:
        input_path: Path to file/directory or URL
        
    Returns:
        List of loaded documents
        
    Raises:
        ValueError: If input path is invalid or documents cannot be loaded
    """
    logger.info(f"Loading documents from: {input_path}")
    
    try:
        # URL pattern check
        if re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_path):
            loader = WebBaseLoader(input_path)
        # Directory check
        elif os.path.isdir(input_path):
            loader = DirectoryLoader(input_path)
        # File check
        elif os.path.isfile(input_path):
            loader = UnstructuredFileLoader(input_path)
        else:
            raise ValueError(f"Invalid input path: {input_path}")
            
        docs = loader.load()
        if not docs:
            raise ValueError(f"No documents loaded from {input_path}")
            
        logger.info(f"Successfully loaded {len(docs)} documents")
        return docs
        
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise ValueError(f"Failed to load documents: {str(e)}")

def load_environment(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file."""
    try:
        loaded = load_dotenv(dotenv_path=env_path, override=True)
        if loaded:
            logger.debug(f"Loaded environment variables from {env_path or '.env'}")
        else:
            logger.debug(f"No environment variables loaded from {env_path or '.env'}")
    except Exception as e:
        logger.warning(f"Error loading environment variables: {str(e)}") 

def validate_environment(config: Union[DataIngestOptionsConfig, RetrievalOptionsConfig]) -> List[str]:
    """Validate environment variables for all components in a config"""
    cwd = os.getcwd()
    dotenv_path = os.path.join(cwd, '.env')
    load_environment(dotenv_path)
    missing_env = []
    missing_packages = []
    
    if hasattr(config, 'document_loaders'):
        for loader in config.document_loaders:
            _missing_env, _missing_packages = validate_component_env(loader.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)
    
    if hasattr(config, 'embedding_models'):
        for model in config.embedding_models:
            _missing_env, _missing_packages = validate_component_env(model.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)
    
    if hasattr(config, 'vector_databases'):
        for db in config.vector_databases:
            _missing_env, _missing_packages = validate_component_env(db.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)

    if hasattr(config, 'retrievers'):
        for retriever in config.retrievers:
            _missing_env, _missing_packages = validate_component_env(retriever.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)

    if hasattr(config, 'rerankers'):
        for reranker in config.rerankers:
            _missing_env, _missing_packages = validate_component_env(reranker.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)
    
    if sorted(set(missing_env)):
        raise ValueError(
            "Missing required environment variables for selected components:\n" + 
            "\n".join(f"- {var}" for var in missing_env)
        )
    
    if sorted(set(missing_packages)):
        raise ValueError(
            "Missing required packages for selected components:\n" + 
            "\n".join(f"- {pkg}" for pkg in missing_packages) + 
            "\n\nPlease install the missing packages and try again."
        )

def validate_component_env(component_value: str) -> Tuple[List[str], List[str]]:
    """Validate required environment variables and packages for a component.
    Args:
        component_value: The specific component value (Eg: 'openai', 'chroma', etc.)
        
    Returns:
        List of missing required environment variables and packages
    """
    requirements = COMPONENT_ENV_REQUIREMENTS.get(component_value, {"required": [], "optional": [], "packages": []})
    missing_env = []
    missing_packages = []
    missing_env.extend([var for var in requirements["required"] if not os.getenv(var)])
    missing_packages.extend([pkg_name for pkg in requirements.get("packages", []) 
                           if (pkg_name := pkg.validate())])
    return missing_env, missing_packages

def simplify_model_config(obj: Any) -> Dict[str, Any]:
    """Extract essential info from LLM/embeddings model."""
    config = {
        "class": obj.__class__.__name__,
        "model": getattr(obj, "model_name", None) or getattr(obj, "model", None)
    }
    
    # Add temperature only for LLMs
    if hasattr(obj, "temperature"):
        config["temperature"] = obj.temperature
        
    return config

class SimpleConfigEncoder(json.JSONEncoder):
    """Simple JSON encoder for config objects."""
    def default(self, obj: Any) -> Any:
        if hasattr(obj, "model_name") or hasattr(obj, "model"):
            return simplify_model_config(obj)
        return super().default(obj)

def serialize_config(config: Any) -> str:
    """Serialize config object to JSON string."""
    try:
        return json.dumps(
            config.model_dump() if hasattr(config, 'model_dump') else config,
            cls=SimpleConfigEncoder
        )
    except Exception as e:
        logger.error(f"Failed to serialize config: {str(e)}")
        return str(config)  # Fallback to string representation