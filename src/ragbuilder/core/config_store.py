from typing import Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import json
from pathlib import Path

class ConfigMetadata(BaseModel):
    timestamp: datetime
    score: float
    source_module: str
    additional_info: Optional[Dict[str, Any]] = None

class ConfigStore:
    _instance = None
    _configs: Dict[str, Dict[str, Any]] = {}
    _metadata: Dict[str, ConfigMetadata] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigStore, cls).__new__(cls)
        return cls._instance

    @classmethod
    def store_config(cls, key: str, config: Dict[str, Any], score: float, source_module: str, additional_info: Optional[Dict] = None):
        """Store a configuration with metadata"""
        cls._configs[key] = config
        cls._metadata[key] = ConfigMetadata(
            timestamp=datetime.now(),
            score=score,
            source_module=source_module,
            additional_info=additional_info
        )

    @classmethod
    def get_config(cls, key: str) -> Optional[Dict[str, Any]]:
        """Get a stored configuration"""
        return cls._configs.get(key)

    @classmethod
    def get_best_config(cls) -> Optional[Dict[str, Any]]:
        """Get the configuration with the highest score"""
        if not cls._metadata:
            return None
        best_key = max(cls._metadata.keys(), key=lambda k: cls._metadata[k].score)
        return cls._configs[best_key]

    @classmethod
    def save_to_file(cls, filepath: str):
        """Save all configurations to a file"""
        data = {
            "configs": cls._configs,
            "metadata": {k: v.model_dump() for k, v in cls._metadata.items()}
        }
        Path(filepath).write_text(json.dumps(data, default=str))

    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configurations from a file"""
        data = json.loads(Path(filepath).read_text())
        cls._configs = data["configs"]
        cls._metadata = {k: ConfigMetadata(**v) for k, v in data["metadata"].items()} 