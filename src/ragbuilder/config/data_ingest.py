from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Union, Dict, Any
import yaml
from ragbuilder.config.components import (
    ParserType, 
    ChunkingStrategy, 
    EmbeddingType, 
    VectorDatabase, 
    EvaluatorType, 
    GraphType
)
from .base import (
    OptimizationConfig, 
    EvaluationConfig, 
    ConfigMetadata, 
    EvalDataGenerationConfig, 
    LLMConfig, 
    EmbeddingConfig
)
from ragbuilder.core.config_store import ConfigStore

class LoaderConfig(BaseModel):
    type: ParserType
    loader_kwargs: Optional[Dict[str, Any]] = None
    custom_class: Optional[str] = None

class ChunkingStrategyConfig(BaseModel):
    type: ChunkingStrategy
    chunker_kwargs: Optional[Dict[str, Any]] = None
    custom_class: Union[str, Any] = None

class ChunkSizeConfig(BaseModel):
    min: int = Field(default=500, description="Minimum chunk size")
    max: int = Field(default=3000, description="Maximum chunk size")
    stepsize: int = Field(default=500, description="Step size for chunk size")

class ChunkSizeStatic(BaseModel):
    val: int = Field(default=500, description="chunk size")
    

class VectorDBConfig(BaseModel):
    type: VectorDatabase
    vectordb_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Vector database specific configuration parameters")
    custom_class: Optional[str] = None

class GraphConfig(BaseModel):
    type: GraphType #neo4j
    document_loaders: Optional[LoaderConfig] = Field(default=None, description="Loader strategy")
    chunking_strategy: Optional[ChunkingStrategyConfig] = Field(default=None, description="Chunking strategy")
    chunk_size: Optional[int] = Field(default=3000, description="Chunk size")
    chunk_overlap: Optional[int] = Field(default=100, description="Chunk overlap")
    embedding_model: Optional[EmbeddingConfig] = Field(default=None, description="Embedding model")
    llm: Optional[LLMConfig] = Field(default=None, description="LLM to use for graph construction")
    graph_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Graph specific configuration parameters")
    custom_class: Optional[str] = None

class DataIngestOptionsConfig(BaseModel):
    """Configuration for data ingestion optimization options.
    
    This config specifies the search space for optimization:
    - What document loaders to try
    - What chunking strategies to evaluate
    - Range of chunk sizes to test
    - etc.
    """
    input_source: Union[str, List[str]] = Field(..., description="File path, directory path, or URL for input data")
    document_loaders: Optional[List[LoaderConfig]] = Field(default=None, description="Document loader configurations")
    chunking_strategies: Optional[List[ChunkingStrategyConfig]] = Field(default=None, description="Chunking strategies to try")
    chunk_size: Optional[ChunkSizeConfig] = Field(default=None, description="Chunk size configuration")
    chunk_overlap: Optional[List[int]] = Field(default=None, description="List of chunk overlap values to try")
    embedding_models: Optional[List[EmbeddingConfig]] = Field(default=None, description="List of embedding models")
    vector_databases: Optional[List[VectorDBConfig]] = Field(default=None, description="List of vector databases")
    sampling_rate: Optional[float] = Field(default=None, description="Sampling rate for documents (0.0 to 1.0). None or 1.0 means no sampling.")
    optimization: Optional[OptimizationConfig] = Field(default=None, description="Optimization configuration")
    database_logging: Optional[bool] = Field(default=None, description="Whether to log results to the DB")
    database_path: Optional[str] = Field(default=None, description="Path to the SQLite database file")
    evaluation_config: Optional[EvaluationConfig] = Field(default=None, description="Evaluation configuration")
    graph: Optional[GraphConfig] = Field(default=None, description="Graph configuration")
    metadata: Optional[ConfigMetadata] = Field(default=None, description="Metadata about the configuration")

    def apply_defaults(self) -> None:
        """Apply default values from ConfigStore and set standard defaults"""
        if self.optimization is None:
            self.optimization = OptimizationConfig()

        if self.optimization.n_trials is None:
            self.optimization.n_trials = ConfigStore().get_default_n_trials()

        if not self.embedding_models:
            self.embedding_models = [ConfigStore().get_default_embeddings()]

        if self.document_loaders is None:
            self.document_loaders = [LoaderConfig(type=ParserType.UNSTRUCTURED)]

        if self.chunking_strategies is None:
            self.chunking_strategies = [ChunkingStrategyConfig(type=ChunkingStrategy.RECURSIVE)]

        if self.chunk_size is None:
            self.chunk_size = ChunkSizeConfig()

        if self.chunk_overlap is None:
            self.chunk_overlap = [100]

        if self.vector_databases is None:
            self.vector_databases = [VectorDBConfig(
                type=VectorDatabase.CHROMA, 
                vectordb_kwargs={'collection_metadata': {'hnsw:space': 'cosine'}, 'persist_directory': './chroma'}
            )]

        if self.database_logging is None:
            self.database_logging = True

        if self.database_path is None:
            self.database_path = "eval.db"

        if self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig(
                type=EvaluatorType.SIMILARITY,
                evaluator_kwargs={
                    "top_k": 5,
                    "position_weights": None,
                    "relevance_threshold": 0.75
                }
            )

        if self.metadata is None:
            self.metadata = ConfigMetadata()

    @classmethod
    def with_defaults(cls, input_source: str, test_dataset: Optional[str] = None) -> 'DataIngestOptionsConfig':
        """Create a DataIngestOptionsConfig with default values

        Args:
            input_source: File path, directory path, or URL for input data
            test_dataset: Optional path to test dataset. If None, synthetic test data will be generated
        
        Returns:
            DataIngestOptionsConfig with default values optimized for quick start
        """
        return cls(
            input_source=input_source,
            document_loaders=[LoaderConfig(type=ParserType.UNSTRUCTURED)],
            chunking_strategies=[ChunkingStrategyConfig(type=ChunkingStrategy.RECURSIVE)],
            chunk_size=ChunkSizeConfig(
                min=500,
                max=3000,
                stepsize=500
            ),
            chunk_overlap=[100],
            embedding_models=[
                EmbeddingConfig(
                    type=EmbeddingType.HUGGINGFACE,
                    model_kwargs={"model_name": "mixedbread-ai/mxbai-embed-large-v1"}
                )
            ],
            vector_databases=[VectorDBConfig(type=VectorDatabase.CHROMA, vectordb_kwargs={'collection_metadata': {'hnsw:space': 'cosine'}})],
            optimization=OptimizationConfig(
                n_trials=ConfigStore().get_default_n_trials(),
                n_jobs=1,
                optimization_direction="maximize"
            ),
            evaluation_config=EvaluationConfig(
                type=EvaluatorType.SIMILARITY,
                test_dataset=test_dataset,
                eval_data_generation_config=EvalDataGenerationConfig(),
                evaluator_kwargs={
                    "top_k": 5,
                    "position_weights": None,
                    "relevance_threshold": 0.75
                }
            ),
            metadata=ConfigMetadata(is_default=True)
        )

class DataIngestConfig(BaseModel):
    """Single instance configuration for data ingestion pipeline.
    
    This config represents a specific combination of parameters:
    - One document loader
    - One chunking strategy
    - One specific chunk size
    - etc.
    """
    input_source: Union[str, List[str]] = Field(..., description="File path, directory path, or URL for input data")
    document_loader: LoaderConfig
    chunking_strategy: ChunkingStrategyConfig
    chunk_size: int
    chunk_overlap: int
    embedding_model: EmbeddingConfig
    vector_database: VectorDBConfig
    sampling_rate: Optional[float] = None

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