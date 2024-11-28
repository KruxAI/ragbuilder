import logging
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Optional, List, Union
from ragbuilder.config.components import COMPONENT_ENV_REQUIREMENTS
from ragbuilder.config.data_ingest import DataIngestConfig
from ragbuilder.config.retriever import BaseRetrieverConfig

logger = logging.getLogger(__name__)

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

def validate_environment(config: Union[DataIngestConfig, BaseRetrieverConfig]) -> List[str]:
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