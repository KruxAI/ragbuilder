from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import List, Optional, Union, Dict, Any
import yaml
from ragbuilder.config.components import ParserType, ChunkingStrategy, EmbeddingModel, VectorDatabase, EvaluatorType, GraphType, LLMType
from .base import OptimizationConfig, EvaluationConfig, LogConfig

class LLMConfig(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    type: LLMType
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model specific parameters including model name/type")

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

class EmbeddingConfig(BaseModel):
    model_config  = ConfigDict(protected_namespaces=())
    
    type: EmbeddingModel
    model_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Model specific parameters including model name/type")
    custom_class: Optional[str] = None

class GraphConfig(BaseModel):
    type: GraphType #neo4j
    document_loaders: Optional[LoaderConfig] = Field(default=None, description="Loader strategy")
    chunking_strategy: Optional[ChunkingStrategyConfig] = Field(default=None, description="Chunking strategy")
    chunk_size: Optional[int] = Field(default=3000, description="Chunk size")
    chunk_overlap: Optional[int] = Field(default=100, description="Chunk overlap")
    embedding_model: Optional[EmbeddingConfig] = Field(default=None, description="Embedding model")
    llm: Optional[LLMConfig] = Field(default_factory=lambda: LLMConfig(type=LLMType.AZURE_OPENAI, model_kwargs={"model": "gpt-4o-mini", "temperature": 0.2}), description="LLM to use for graph construction")
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
    test_dataset: str = Field(..., description="Path to CSV file containing test questions")
    document_loaders: Optional[List[LoaderConfig]] = Field(
        default_factory=lambda: [LoaderConfig(type=ParserType.UNSTRUCTURED)], 
        description="Document loader configurations"
    )
    chunking_strategies: Optional[List[ChunkingStrategyConfig]] = Field(
        default_factory=lambda: [ChunkingStrategyConfig(type=ChunkingStrategy.RECURSIVE)],
        description="Chunking strategies to try"
    )
    chunk_size: Optional[ChunkSizeConfig] = Field(default_factory=ChunkSizeConfig, description="Chunk size configuration")
    chunk_overlap: Optional[List[int]] = Field(default=[100], description="List of chunk overlap values to try")
    embedding_models: Optional[List[EmbeddingConfig]] = Field(
        default_factory=lambda: [EmbeddingConfig(type=EmbeddingModel.HUGGINGFACE, model_kwargs={"model_name": "mixedbread-ai/mxbai-embed-large-v1"})], #model_kwargs={"model_name": "sentence-transformers/all-MiniLM-L6-v2"})],
        # default_factory=lambda: [EmbeddingConfig(type=EmbeddingModel.AZURE_OPENAI, model_kwargs={"model": "text-embedding-3-large"})], #model_kwargs={"model_name": "sentence-transformers/all-MiniLM-L6-v2"})], 
        description="List of embedding models"
    )
    vector_databases: Optional[List[VectorDBConfig]] = Field(
        # default_factory=lambda: [VectorDBConfig(type=VectorDatabase.FAISS, vectordb_kwargs={})], 
        default_factory=lambda: [VectorDBConfig(type=VectorDatabase.CHROMA, vectordb_kwargs={'collection_metadata': {'hnsw:space': 'cosine'}})],
        description="List of vector databases"
    )
    sampling_rate: Optional[float] = Field(default=None, description="Sampling rate for documents (0.0 to 1.0). None or 1.0 means no sampling.")
    optimization: Optional[OptimizationConfig] = Field(default_factory=OptimizationConfig, description="Optimization configuration")
    # log_config: Optional[LogConfig] = Field(default_factory=LogConfig, description="Logging configuration")
    database_logging: Optional[bool] = Field(default=True, description="Whether to log results to the DB")
    database_path: Optional[str] = Field(default="eval.db", description="Path to the SQLite database file")
    evaluation_config: EvaluationConfig = Field(
        default_factory=lambda: EvaluationConfig(
            type=EvaluatorType.SIMILARITY,
            evaluator_kwargs={
                "top_k": 5,
                "position_weights": None,
                "relevance_threshold": 0.75
            }
        ),
        description="Evaluation configuration"
    )
    graph: Optional[GraphConfig] = Field(default=None, description="Graph configuration")

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
            test_dataset=test_dataset,
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
                    type=EmbeddingModel.HUGGINGFACE,
                    model_kwargs={"model_name": "mixedbread-ai/mxbai-embed-large-v1"}
                )
            ],
            vector_databases=[VectorDBConfig(type=VectorDatabase.CHROMA, vectordb_kwargs={'collection_metadata': {'hnsw:space': 'cosine'}})],
            optimization=OptimizationConfig(
                n_trials=10,
                n_jobs=1,
                optimization_direction="maximize"
            )
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
    test_dataset: str = Field(..., description="Path to CSV file containing test questions")
    document_loader: LoaderConfig = Field(
        default_factory=lambda: LoaderConfig(type=ParserType.UNSTRUCTURED), 
        description="Document loader configuration"
    )
    chunking_strategy: ChunkingStrategyConfig = Field(default_factory=lambda: ChunkingStrategyConfig(type=ChunkingStrategy.RECURSIVE), description="Chunking strategy")
    chunk_size: int = Field(default=1000, description="Chunk size")
    chunk_overlap: int = Field(default=100, description="Chunk overlap")
    embedding_model: EmbeddingConfig = Field(
        # default_factory=lambda: EmbeddingConfig(type=EmbeddingModel.HUGGINGFACE, model_kwargs={"model_name": "mixedbread-ai/mxbai-embed-large-v1"}), #model_kwargs={"model_name": "sentence-transformers/all-MiniLM-L6-v2"}), 
        default_factory=lambda: EmbeddingConfig(type=EmbeddingModel.AZURE_OPENAI, model_kwargs={"model_name": "text-embedding-3-large"}), #model_kwargs={"model_name": "sentence-transformers/all-MiniLM-L6-v2"}), 
        description="Embedding model configuration"
    )
    vector_database: VectorDBConfig = Field(
        default_factory=lambda: VectorDBConfig(type=VectorDatabase.FAISS, vectordb_kwargs={}), 
        description="Vector store configuration"
    )

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