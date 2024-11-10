import os
from typing import List
from enum import Enum
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter, 
    TokenTextSplitter,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_community.embeddings import (
    OllamaEmbeddings,
    CohereEmbeddings,
    VertexAIEmbeddings,
    BedrockEmbeddings,
    JinaEmbeddings
)
from langchain.vectorstores import (
    FAISS,
    Chroma,
    Pinecone,
    Weaviate,
    Qdrant,
    Milvus,
    PGVector,
    ElasticsearchStore
)
from langchain_community.document_loaders import (
    UnstructuredFileLoader,
    PyMuPDFLoader,
    DirectoryLoader,
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
    AzureBlobStorageContainerLoader,
    S3DirectoryLoader
)

# Component type definitions
class ParserType(str, Enum):
    UNSTRUCTURED = "unstructured"
    PYMUPDF = "pymupdf"
    PYPDF = "pypdf"
    DOCX = "docx"
    AZURE_BLOB = "azure_blob"
    S3 = "s3"
    CUSTOM = "custom"

class ChunkingStrategy(str, Enum):
    CHARACTER = "CharacterTextSplitter"
    RECURSIVE = "RecursiveCharacterTextSplitter"
    TOKEN = "TokenTextSplitter"
    MARKDOWN = "MarkdownHeaderTextSplitter"
    HTML = "HTMLHeaderTextSplitter"
    SEMANTIC = "SemanticChunker"
    CUSTOM = "custom"

class EmbeddingModel(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    COHERE = "cohere"
    VERTEXAI = "vertexai"
    BEDROCK = "bedrock"
    JINA = "jina"
    CUSTOM = "custom"

class VectorDatabase(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    PGVECTOR = "pgvector"
    ELASTICSEARCH = "elasticsearch"
    CUSTOM = "custom"

class EvaluatorType(str, Enum):
    SIMILARITY = "similarity"
    CUSTOM = "custom"

# Component mappings
LOADER_MAP = {
    ParserType.UNSTRUCTURED: UnstructuredFileLoader,
    ParserType.PYMUPDF: PyMuPDFLoader,
    ParserType.PYPDF: PyPDFLoader,
    ParserType.DOCX: Docx2txtLoader,
    ParserType.AZURE_BLOB: AzureBlobStorageContainerLoader,
    ParserType.S3: S3DirectoryLoader,
}

DIRECTORY_LOADER = DirectoryLoader
WEB_LOADER = WebBaseLoader

CHUNKER_MAP = {
    ChunkingStrategy.CHARACTER: CharacterTextSplitter,
    ChunkingStrategy.RECURSIVE: RecursiveCharacterTextSplitter,
    ChunkingStrategy.TOKEN: TokenTextSplitter,
    ChunkingStrategy.MARKDOWN: MarkdownHeaderTextSplitter,
    ChunkingStrategy.HTML: HTMLHeaderTextSplitter,
    ChunkingStrategy.SEMANTIC: SemanticChunker,
}

EMBEDDING_MAP = {
    EmbeddingModel.OPENAI: OpenAIEmbeddings,
    EmbeddingModel.AZURE_OPENAI: AzureOpenAIEmbeddings,
    EmbeddingModel.HUGGINGFACE: HuggingFaceEmbeddings,
    EmbeddingModel.OLLAMA: OllamaEmbeddings,
    EmbeddingModel.COHERE: CohereEmbeddings,
    EmbeddingModel.VERTEXAI: VertexAIEmbeddings,
    EmbeddingModel.BEDROCK: BedrockEmbeddings,
    EmbeddingModel.JINA: JinaEmbeddings,
}

VECTORDB_MAP = {
    VectorDatabase.FAISS: FAISS,
    VectorDatabase.CHROMA: Chroma,
    VectorDatabase.PINECONE: Pinecone,
    VectorDatabase.WEAVIATE: Weaviate,
    VectorDatabase.QDRANT: Qdrant,
    VectorDatabase.MILVUS: Milvus,
    VectorDatabase.PGVECTOR: PGVector,
    VectorDatabase.ELASTICSEARCH: ElasticsearchStore,
}

# Environment variable requirements for components
COMPONENT_ENV_REQUIREMENTS = {
    # Embedding Models
    EmbeddingModel.AZURE_OPENAI: {
        "required": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        "optional": ["AZURE_DEPLOYMENT_NAME"]
    },
    EmbeddingModel.COHERE: {
        "required": ["COHERE_API_KEY"]
    },
    EmbeddingModel.VERTEXAI: {
        "required": ["GOOGLE_APPLICATION_CREDENTIALS"]
    },
    EmbeddingModel.BEDROCK: {
        "required": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
    },
    EmbeddingModel.JINA: {
        "required": ["JINA_API_KEY"]
    },
    
    # Vector Databases
    VectorDatabase.PINECONE: {
        "required": ["PINECONE_API_KEY", "PINECONE_ENVIRONMENT"]
    },
    VectorDatabase.WEAVIATE: {
        "required": ["WEAVIATE_URL", "WEAVIATE_API_KEY"]
    },
    VectorDatabase.QDRANT: {
        "required": ["QDRANT_URL"],
        "optional": ["QDRANT_API_KEY"]
    },
    VectorDatabase.MILVUS: {
        "required": ["MILVUS_HOST", "MILVUS_PORT"]
    },
    VectorDatabase.PGVECTOR: {
        "required": ["PGVECTOR_CONNECTION_STRING"]
    },
    VectorDatabase.ELASTICSEARCH: {
        "required": ["ELASTICSEARCH_URL"],
        "optional": ["ELASTICSEARCH_API_KEY"]
    },
    
    # Document Loaders
    ParserType.AZURE_BLOB: {
        "required": ["AZURE_STORAGE_CONNECTION_STRING"]
    },
    ParserType.S3: {
        "required": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"]
    }
}

def validate_environment(component_type: str, component_value: str) -> List[str]:
    """Validate required environment variables for a component.
    
    Args:
        component_type: The component type (e.g., EmbeddingModel, VectorDatabase)
        component_value: The specific component value
        
    Returns:
        List of missing required environment variables
    """
    requirements = COMPONENT_ENV_REQUIREMENTS.get(component_value, {"required": [], "optional": []})
    missing = [var for var in requirements["required"] if not os.getenv(var)]
    return missing