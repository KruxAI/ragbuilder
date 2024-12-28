from typing import Dict, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel
import json
from pathlib import Path
from ragbuilder.config.components import LLMType, EmbeddingType
from ragbuilder.config.base import LLMConfig, EmbeddingConfig
from langchain_core.language_models import BaseChatModel, BaseLLM
from langchain_core.embeddings import Embeddings

class ConfigMetadata(BaseModel):
    timestamp: datetime
    score: float
    source_module: str
    additional_info: Optional[Dict[str, Any]] = None

class ConfigStore:
    _instance = None
    _configs: Dict[str, Dict[str, Any]] = {}
    _metadata: Dict[str, ConfigMetadata] = {}
    _best_data_ingest_pipeline = None
    _best_retriever_pipeline = None
    _default_llm: Optional[LLMConfig] = None
    _default_embeddings: Optional[EmbeddingConfig] = None
    _default_n_trials: Optional[int] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigStore, cls).__new__(cls)
        return cls._instance

    @classmethod
    def set_default_llm(cls, llm_config: Optional[Union[Dict[str, Any], LLMConfig, BaseChatModel, BaseLLM]]) -> None:
        """Store default LLM configuration or instance"""
        if isinstance(llm_config, dict):
            cls._default_llm = LLMConfig(
                type=LLMType.OPENAI,
                model_kwargs=llm_config
            )
        elif isinstance(llm_config, (BaseChatModel, BaseLLM)):
            cls._default_llm = LLMConfig.from_llm(llm_config)
        else:
            cls._default_llm = llm_config

    @classmethod
    def get_default_llm(cls) -> LLMConfig:
        """Get default LLM configuration"""
        return cls._default_llm or LLMConfig(
            type=LLMType.OPENAI,
            model_kwargs={"model": "gpt-4o-mini", "temperature": 0.0}
        )

    @classmethod
    def set_default_embeddings(cls, embedding_config: Optional[Union[Dict[str, Any], EmbeddingConfig, Embeddings]]) -> None:
        """Store default Embedding configuration or instance"""
        if isinstance(embedding_config, dict):
            cls._default_embeddings = EmbeddingConfig(
                type=EmbeddingType.OPENAI,
                model_kwargs=embedding_config
            )
        elif isinstance(embedding_config, Embeddings):
            cls._default_embeddings = EmbeddingConfig.from_embedding(embedding_config)
        else:
            cls._default_embeddings = embedding_config

    @classmethod
    def get_default_embeddings(cls) -> EmbeddingConfig:
        """Get default Embedding configuration"""
        return cls._default_embeddings or EmbeddingConfig(
            type=EmbeddingType.OPENAI,
            model_kwargs={"model": "text-embedding-3-large"}
        )

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

    @classmethod
    def store_best_data_ingest_pipeline(cls, pipeline):
        """Store the best performing pipeline"""
        cls._best_data_ingest_pipeline = pipeline

    @classmethod
    def get_best_data_ingest_pipeline(cls):
        """Get the stored best pipeline"""
        return cls._best_data_ingest_pipeline

    @classmethod
    def store_best_retriever_pipeline(cls, pipeline):
        """Store the best performing retriever pipeline"""
        cls._best_retriever_pipeline = pipeline
    
    @classmethod
    def get_best_retriever_pipeline(cls):
        """Get the stored best retriever pipeline"""
        return cls._best_retriever_pipeline
    
    @classmethod
    def get_best_retriever_config(cls) -> Optional[Dict[str, Any]]:
        """Get the best retriever configuration"""
        if cls._best_retriever_pipeline:
            return cls._best_retriever_pipeline.retriever_chain
        return None

    @classmethod
    def set_default_n_trials(cls, n_trials: Optional[int]) -> None:
        """Store default number of trials for optimization"""
        cls._default_n_trials = n_trials

    @classmethod
    def get_default_n_trials(cls) -> int:
        """Get default number of trials"""
        return cls._default_n_trials or 10