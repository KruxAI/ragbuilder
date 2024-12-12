import logging
import os
import re
from pathlib import Path
from typing import Optional, List, Union
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
            logger.info(f"Loaded environment variables from {env_path or '.env'}")
        else:
            logger.debug(f"No environment variables loaded from {env_path or '.env'}")
    except Exception as e:
        logger.warning(f"Error loading environment variables: {str(e)}") 

def validate_environment(config: Union[DataIngestOptionsConfig, RetrievalOptionsConfig]) -> List[str]:
    """Validate environment variables for all components in a config"""
    missing_vars = []
    
    if hasattr(config, 'document_loaders'):
        for loader in config.document_loaders:
            if missing := validate_component_environment(loader.type):
                missing_vars.extend(missing)
    
    if hasattr(config, 'embedding_models'):
        for model in config.embedding_models:
            if missing := validate_component_environment(model.type):
                missing_vars.extend(missing)
    
    if hasattr(config, 'vector_databases'):
        for db in config.vector_databases:
            if missing := validate_component_environment(db.type):
                missing_vars.extend(missing)
    
    return sorted(set(missing_vars))

def validate_component_environment(component_value: str) -> List[str]:
    """Validate required environment variables for a component.
    
    Args:
        component_value: The specific component value (Eg: 'openai', 'chroma', etc.)
        
    Returns:
        List of missing required environment variables
    """
    requirements = COMPONENT_ENV_REQUIREMENTS.get(component_value, {"required": [], "optional": []})
    missing = [var for var in requirements["required"] if not os.getenv(var)]
    return missing