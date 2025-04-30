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
# from langchain_unstructured import UnstructuredLoader
from langchain_core.documents import Document
from ragbuilder.config.components import COMPONENT_ENV_REQUIREMENTS, ParserType
from ragbuilder.config.data_ingest import DataIngestOptionsConfig
from ragbuilder.config.retriever import RetrievalOptionsConfig
import json
import importlib
import requests
from urllib.parse import urlparse

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
    
    # Validate input_source for DataIngestOptionsConfig
    if isinstance(config, DataIngestOptionsConfig):
        input_sources = [config.input_source] if isinstance(config.input_source, str) else config.input_source
        for source in input_sources:
            if not _is_valid_input_source(source):
                raise ValueError(f"Invalid input source: {source}")
    
    # Validate test_dataset if provided in evaluation_config
    if hasattr(config, 'evaluation_config') and config.evaluation_config:
        if config.evaluation_config.test_dataset:
            if not os.path.isfile(config.evaluation_config.test_dataset):
                raise ValueError(f"Invalid test dataset path: {config.evaluation_config.test_dataset}")
        
        # Validate LLM and embeddings in evaluation_config
        if config.evaluation_config.llm and hasattr(config.evaluation_config.llm, 'type'):
            _missing_env, _missing_packages = validate_component_env(config.evaluation_config.llm.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)
            
        if config.evaluation_config.embeddings and hasattr(config.evaluation_config.embeddings, 'type'):
            _missing_env, _missing_packages = validate_component_env(config.evaluation_config.embeddings.type)
            missing_env.extend(_missing_env)
            missing_packages.extend(_missing_packages)
            
        # Validate eval data generation config if present
        if config.evaluation_config.eval_data_generation_config:
            gen_config = config.evaluation_config.eval_data_generation_config
            
            if gen_config.generator_model and hasattr(gen_config.generator_model, 'type'):
                _missing_env, _missing_packages = validate_component_env(gen_config.generator_model.type)
                missing_env.extend(_missing_env)
                missing_packages.extend(_missing_packages)
                
            if gen_config.critic_model and hasattr(gen_config.critic_model, 'type'):
                _missing_env, _missing_packages = validate_component_env(gen_config.critic_model.type)
                missing_env.extend(_missing_env)
                missing_packages.extend(_missing_packages)
                
            if gen_config.embedding_model and hasattr(gen_config.embedding_model, 'type'):
                _missing_env, _missing_packages = validate_component_env(gen_config.embedding_model.type)
                missing_env.extend(_missing_env)
                missing_packages.extend(_missing_packages)

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
    
    if component_value == ParserType.UNSTRUCTURED:
        try:
            import nltk
            nltk_resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger']
            for resource in nltk_resources:
                try:
                    # Determine the correct path prefix based on resource type
                    path_prefix = "taggers" if resource == "averaged_perceptron_tagger" else "tokenizers"
                    nltk.data.find(f'{path_prefix}/{resource}')
                except LookupError:
                    try:
                        logger.info(f"Downloading required NLTK data '{resource}' for unstructured parser...")
                        nltk.download(resource, quiet=True)
                    except Exception as e:
                        logger.warning(f"Failed to download NLTK data '{resource}': {str(e)}")
                        missing_packages.append(f"nltk[{resource}]")

        except ImportError:
            missing_packages.append("nltk")
        except Exception as e:
            logger.warning(f"Failed to validate/download NLTK data: {str(e)}")
            missing_packages.extend([f"nltk[{res}]" for res in nltk_resources])
    
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

def _is_valid_input_source(input_path: str) -> bool:
    """
    Validate if input source is a valid file, directory, or URL.
    
    Args:
        input_path: Path to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    # Check if it's a URL
    parsed = urlparse(input_path)
    if parsed.scheme in ['http', 'https'] and parsed.netloc:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
            response = requests.head(input_path, headers=headers, allow_redirects=True)

            if response.status_code == 405:
                response = requests.get(input_path, headers=headers, stream=True)

            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Failed to validate URL {input_path}: {str(e)}")
            return False
            
    # Check if it's a file or directory
    return os.path.isfile(input_path) or os.path.isdir(input_path)