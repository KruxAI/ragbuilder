from typing import Optional
from pathlib import Path
import logging
from dotenv import load_dotenv

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