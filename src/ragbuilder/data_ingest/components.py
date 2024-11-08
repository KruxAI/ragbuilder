from enum import Enum
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain_community.document_loaders import UnstructuredFileLoader, PyMuPDFLoader, DirectoryLoader, WebBaseLoader

# Component type definitions
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
    OLLAMA = "ollama"
    CUSTOM = "custom"

class VectorDatabase(str, Enum):
    FAISS = "faiss"
    CHROMA = "chroma"
    CUSTOM = "custom"

# Component mappings
LOADER_MAP = {
    ParserType.UNSTRUCTURED: UnstructuredFileLoader,
    ParserType.PYMUPDF: PyMuPDFLoader,
}

DIRECTORY_LOADER = DirectoryLoader
WEB_LOADER = WebBaseLoader

CHUNKER_MAP = {
    ChunkingStrategy.CHARACTER: CharacterTextSplitter,
    ChunkingStrategy.RECURSIVE: RecursiveCharacterTextSplitter,
}

EMBEDDING_MAP = {
    EmbeddingModel.OPENAI: OpenAIEmbeddings,
    EmbeddingModel.HUGGINGFACE: HuggingFaceEmbeddings,
    EmbeddingModel.OLLAMA: OllamaEmbeddings,
}

VECTORDB_MAP = {
    VectorDatabase.FAISS: FAISS,
    VectorDatabase.CHROMA: Chroma,
} 