from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Dict, Any
import yaml
import time
import random
from enum import Enum

class ParserType(str, Enum):
    UNSTRUCTURED = "unstructured"
    PYMUPDF = "pymupdf"
    CUSTOM = "custom"

class ChunkingStrategy(str, Enum):
    CHARACTER = "CharacterTextSplitter"
    RECURSIVE = "RecursiveCharacterTextSplitter"
    # SEMANTIC = "SemanticChunker"
    CUSTOM = "custom"

class EmbeddingModel(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"

class VectorDatabase(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    CUSTOM = "custom"

class LoaderConfig(BaseModel):
    type: ParserType
    loader_kwargs: Optional[Dict[str, Any]] = None
    custom_class: Optional[str] = None

class ChunkSizeConfig(BaseModel):
    min: int = Field(default=100, description="Minimum chunk size")
    max: int = Field(default=500, description="Maximum chunk size")
    stepsize: int = Field(default=100, description="Step size for chunk size")

class VectorDBConfig(BaseModel):
    type: VectorDatabase
    collection_name: Optional[str] = "__DEFAULT_COLLECTION__"
    persist_directory: Optional[str] = None
    client_settings: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    custom_class: Optional[str] = None

class EmbeddingConfig(BaseModel):
    type: EmbeddingModel
    model: Optional[str] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    custom_class: Optional[str] = None

class OptimizationConfig(BaseModel):
    type: Optional[str] = "Optuna"
    n_trials: Optional[int] = Field(default=10, description="Number of trials for optimization")
    n_jobs: Optional[int] = Field(default=1, description="Number of jobs for optimization")
    timeout: Optional[int] = Field(default=None, description="Timeout for optimization")
    storage: Optional[str] = Field(default=None, description="Storage URL for Optuna (e.g., 'sqlite:///optuna.db')")
    study_name: Optional[str] = Field(default=f"data_ingest_{int(time.time()*1000+random.randint(1, 1000))}", description="Name of the Optuna study")
    load_if_exists: Optional[bool] = Field(default=False, description="Load existing study if it exists")

class BaseConfig(BaseModel):
    input_source: Union[str, List[str]] = Field(..., description="File path, directory path, or URL for input data")
    test_dataset: str = Field(..., description="Path to CSV file containing test questions")

    @classmethod
    def from_yaml(cls, file_path: str) -> 'DataIngestConfig':
        """
        Load configuration from a YAML file.
        """
        with open(file_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return cls(**config_dict)

    def to_yaml(self, file_path: str) -> None:
        """
        Save configuration to a YAML file.
        """
        with open(file_path, 'w') as file:
            yaml.dump(self.model_dump(), file)

class DataIngestOptionsConfig(BaseConfig):
    document_loaders: Optional[List[LoaderConfig]] = Field(
        default_factory=lambda: [LoaderConfig(type=ParserType.UNSTRUCTURED)], 
        description="Document loader configurations"
    )
    chunking_strategies: Optional[List[str]] = Field(
        default_factory=lambda: [strategy for strategy in ChunkingStrategy if strategy != ChunkingStrategy.CUSTOM], 
        description="Chunking strategies to try"
    )
    custom_chunker: Optional[str] = Field(default=None, description="Custom chunker class. E.g., 'my_module.MyCustomChunker'")
    chunk_size: Optional[ChunkSizeConfig] = Field(default_factory=ChunkSizeConfig, description="Chunk size configuration")
    chunk_overlap: Optional[List[int]] = Field(default=[100], description="List of chunk overlap values to try")
    embedding_models: Optional[List[EmbeddingConfig]] = Field(
        default_factory=lambda: [EmbeddingConfig(type=EmbeddingModel.HUGGINGFACE, model="sentence-transformers/all-MiniLM-L6-v2")],
        description="List of embedding models"
    )
    vector_databases: Optional[List[VectorDBConfig]] = Field(
        default_factory=lambda: [VectorDBConfig(type=VectorDatabase.FAISS, collection_name=None)], 
        description="List of vector databases"
    )
    top_k: Optional[int] = Field(default=5, description="Number of top results to consider for similarity scoring")
    sampling_rate: Optional[float] = Field(default=None, description="Sampling rate for documents (0.0 to 1.0). None or 1.0 means no sampling.")
    optimization: Optional[OptimizationConfig] = Field(default_factory=OptimizationConfig, description="Optimization configuration")

class DataIngestConfig(BaseConfig):
    document_loader: LoaderConfig = Field(
        default_factory=lambda: LoaderConfig(type=ParserType.UNSTRUCTURED), 
        description="Document loader configuration"
    )
    chunking_strategy: str = Field(default=ChunkingStrategy.RECURSIVE, description="Chunking strategy")
    custom_chunker: Optional[str] = Field(default=None, description="Custom chunker class. E.g., 'my_module.MyCustomChunker'")
    chunk_size: int = Field(default=1000, description="Chunk size")
    chunk_overlap: int = Field(default=100, description="Chunk overlap")
    embedding_model: EmbeddingConfig = Field(
        default_factory=lambda: EmbeddingConfig(type=EmbeddingModel.HUGGINGFACE), 
        description="Embedding model configuration"
    )
    vector_database: VectorDBConfig = Field(
        default_factory=lambda: VectorDBConfig(type=VectorDatabase.FAISS), 
        description="Vector store configuration"
    )
    top_k: int = Field(default=5, description="Number of top results to consider for similarity scoring")
    sampling_rate: Optional[float] = Field(default=None, description="Sampling rate for documents (0.0 to 1.0). None or 1.0 means no sampling.")

def load_config(file_path: str) -> Union[DataIngestOptionsConfig, DataIngestConfig]:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Check for required fields
    if 'input_source' not in config_dict or 'test_dataset' not in config_dict:
        raise ValueError("Configuration must include 'input_source' and 'test_data'")
    
    # TODO: Re-think and redo this logic to see if there's a better way
    try:
        return DataIngestOptionsConfig(**config_dict)
    except ValidationError:
        # If it fails, try DataIngestConfig
        try:
            return DataIngestConfig(**config_dict)
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {str(e)}")

def save_config(config: Union[DataIngestOptionsConfig, DataIngestConfig], file_path: str) -> None:
    """
    Save configuration to a YAML file.
    """
    config.to_yaml(file_path)