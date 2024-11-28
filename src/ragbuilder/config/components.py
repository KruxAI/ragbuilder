from typing import Callable
from enum import Enum
from importlib import import_module

# Component type definitions
class GraphType(str, Enum):
    NEO4J = "neo4j"
    
class LLMType(str, Enum):
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"

class ParserType(str, Enum):
    UNSTRUCTURED = "unstructured"
    PYMUPDF = "pymupdf"
    PYPDF = "pypdf"
    DOCX = "docx"
    AZURE_BLOB = "azure_blob"
    S3 = "s3"
    DIRECTORY = "directory"
    WEB = "web"
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

class RetrieverType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    MULTI_QUERY = "multi_query"
    BM25_RETRIEVER = "bm25"
    PARENT_DOC_RETRIEVER_FULL = "parent_doc_full"
    PARENT_DOC_RETRIEVER_LARGE = "parent_doc_large"
    # COLBERT_RETRIEVER = "colbert"
    CUSTOM = "custom"

class RerankerType(str, Enum):
    FLASH_RANK = "FlaskRank"
    RANK_GPT = "rankGPT"
    COHERE = "cohere"
    BAAI_BGE_RERANKER_BASE = "BAAI/bge-reranker-base"
    CUSTOM = "custom"

class EvaluatorType(str, Enum):
    SIMILARITY = "similarity"
    RAGAS = "ragas"
    CUSTOM = "custom"

def lazy_load(module_path: str, class_name: str) -> Callable:
    """Lazy loader function that imports a class only when needed."""
    def get_class():
        module = import_module(module_path)
        return getattr(module, class_name)
    return get_class

# Component mappings with lazy loading
LLM_MAP = {
    LLMType.AZURE_OPENAI: lazy_load("langchain_openai", "AzureChatOpenAI"),
}

LOADER_MAP = {
    ParserType.UNSTRUCTURED: lazy_load("langchain_community.document_loaders", "UnstructuredFileLoader"),
    ParserType.PYMUPDF: lazy_load("langchain_community.document_loaders", "PyMuPDFLoader"),
    ParserType.PYPDF: lazy_load("langchain_community.document_loaders", "PyPDFLoader"),
    ParserType.DOCX: lazy_load("langchain_community.document_loaders", "Docx2txtLoader"),
    ParserType.AZURE_BLOB: lazy_load("langchain_community.document_loaders", "AzureBlobStorageContainerLoader"),
    ParserType.S3: lazy_load("langchain_community.document_loaders", "S3DirectoryLoader"),
    ParserType.DIRECTORY: lazy_load("langchain_community.document_loaders", "DirectoryLoader"),
    ParserType.WEB: lazy_load("langchain_community.document_loaders", "WebBaseLoader"),
}

CHUNKER_MAP = {
    ChunkingStrategy.CHARACTER: lazy_load("langchain_text_splitters", "CharacterTextSplitter"),
    ChunkingStrategy.RECURSIVE: lazy_load("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
    ChunkingStrategy.TOKEN: lazy_load("langchain_text_splitters", "TokenTextSplitter"),
    ChunkingStrategy.MARKDOWN: lazy_load("langchain_text_splitters", "MarkdownHeaderTextSplitter"),
    ChunkingStrategy.HTML: lazy_load("langchain_text_splitters", "HTMLHeaderTextSplitter"),
    ChunkingStrategy.SEMANTIC: lazy_load("langchain_experimental.text_splitter", "SemanticChunker"),
}

EMBEDDING_MAP = {
    EmbeddingModel.OPENAI: lazy_load("langchain_openai", "OpenAIEmbeddings"),
    EmbeddingModel.AZURE_OPENAI: lazy_load("langchain_openai", "AzureOpenAIEmbeddings"),
    EmbeddingModel.HUGGINGFACE: lazy_load("langchain_huggingface", "HuggingFaceEmbeddings"),
    EmbeddingModel.OLLAMA: lazy_load("langchain_community.embeddings", "OllamaEmbeddings"),
    EmbeddingModel.COHERE: lazy_load("langchain_community.embeddings", "CohereEmbeddings"),
    EmbeddingModel.VERTEXAI: lazy_load("langchain_community.embeddings", "VertexAIEmbeddings"),
    EmbeddingModel.BEDROCK: lazy_load("langchain_community.embeddings", "BedrockEmbeddings"),
    EmbeddingModel.JINA: lazy_load("langchain_community.embeddings", "JinaEmbeddings"),
}

VECTORDB_MAP = {
    VectorDatabase.FAISS: lazy_load("langchain.vectorstores", "FAISS"),
    VectorDatabase.CHROMA: lazy_load("langchain.vectorstores", "Chroma"),
    VectorDatabase.PINECONE: lazy_load("langchain.vectorstores", "Pinecone"),
    VectorDatabase.WEAVIATE: lazy_load("langchain.vectorstores", "Weaviate"),
    VectorDatabase.QDRANT: lazy_load("langchain.vectorstores", "Qdrant"),
    VectorDatabase.MILVUS: lazy_load("langchain.vectorstores", "Milvus"),
    VectorDatabase.PGVECTOR: lazy_load("langchain.vectorstores", "PGVector"),
    VectorDatabase.ELASTICSEARCH: lazy_load("langchain.vectorstores", "ElasticsearchStore"),
}

RETRIEVER_MAP = {
    RetrieverType.BM25_RETRIEVER: lazy_load("langchain.retrievers", "BM25Retriever"),
}

# RERANKER_MAP = {
#     RerankerType.flashrank: lazy_load("rerankers", "Reranker"),
#     RerankerType.rankGPT: lazy_load("rerankers", "Reranker"),
# }

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
