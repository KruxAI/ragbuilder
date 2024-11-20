from typing import Dict, List, Optional
from langchain.docstore.document import Document
import logging

logger = logging.getLogger(__name__)

class DocumentStore:
    """Singleton class to manage document storage across modules"""
    _instance = None
    _documents: Dict[str, List[Document]] = {}
    _metadata: Dict[str, Dict] = {}  # Store metadata about each document set

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentStore, cls).__new__(cls)
        return cls._instance

    @classmethod
    def store_documents(cls, key: str, documents: List[Document], metadata: Optional[Dict] = None):
        """Store documents with optional metadata"""
        cls._documents[key] = documents
        if metadata:
            cls._metadata[key] = metadata
        logger.info(f"Stored {len(documents)} documents with key: {key}")

    @classmethod
    def get_documents(cls, key: str) -> Optional[List[Document]]:
        """Retrieve documents by key"""
        documents = cls._documents.get(key)
        if documents is None:
            logger.info(f"No documents found for key: {key}")
        return documents

    @classmethod
    def get_metadata(cls, key: str) -> Optional[Dict]:
        """Retrieve metadata for a document set"""
        return cls._metadata.get(key)

    @classmethod
    def has_documents(cls, key: str) -> bool:
        """Check if documents exist for a given key"""
        return key in cls._documents

    @classmethod
    def clear(cls):
        """Clear all stored documents and metadata"""
        cls._documents.clear()
        cls._metadata.clear()
        logger.info("Document store cleared")

    @classmethod
    def get_storage_info(cls) -> Dict:
        """Get information about stored documents"""
        return {
            key: {
                "num_documents": len(docs),
                "metadata": cls._metadata.get(key, {})
            }
            for key, docs in cls._documents.items()
        } 