from typing import Dict, List, Optional, Tuple, Any
from langchain.docstore.document import Document
import logging
import time

logger = logging.getLogger(__name__)

class DocumentStore:
    """Singleton class to manage document storage across modules"""
    _instance = None
    _documents: Dict[str, List[Document]] = {}
    _metadata: Dict[str, Dict] = {}  # Store metadata about each document set
    _vectorstores: Dict[str, Any] = {}  # Store vectorstores
    _sampled_paths: Dict[str, Dict[str, Any]] = {}  # Store sampled data paths and metadata
    _best_config_key: Optional[str] = None  # New property to store best config key
    _graph: Optional[Any] = None

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
        logger.debug(f"Stored {len(documents)} documents with key: {key}")

    @classmethod
    def get_documents(cls, key: str) -> Optional[List[Document]]:
        """Retrieve documents by key"""
        documents = cls._documents.get(key)
        if documents is None:
            logger.debug(f"No documents found for key: {key}")
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
        """Clear all stored documents, metadata, vectorstores, and sampled paths"""
        cls._documents.clear()
        cls._metadata.clear()
        cls._vectorstores.clear()
        cls._sampled_paths.clear()
        cls._graph = None
        cls._best_config_key = None
        cls._best_loader_key = None
        logger.info("Document store cleared")

    @classmethod
    def get_storage_info(cls) -> Dict:
        """Get information about stored documents and graphs"""
        return {
            key: {
                "num_documents": len(docs),
                "metadata": cls._metadata.get(key, {}),
                "has_graph": cls._graph is not None
            }
            for key, docs in cls._documents.items()
        } 

    @classmethod
    def set_best_config_key(cls, loader_key: str, config_key: str):
        """Set the key for the best configuration's documents"""
        if loader_key not in cls._documents:
            logger.warning(f"Setting best config key to {loader_key} but no documents exist for this key")
        cls._best_loader_key = loader_key
        logger.debug(f"Set best loader key to: {loader_key}")

        if config_key not in cls._vectorstores:
            logger.warning(f"Setting best config key to {config_key} but no vectorstore exists for this key")
        cls._best_config_key = config_key
        logger.debug(f"Set best config key to: {config_key}")

    @classmethod
    def get_best_config_docs(cls) -> Optional[Tuple[List[Document], Dict]]:
        """Get documents and metadata for the best configuration."""
        if not cls._best_loader_key:
            logger.warning("No best configuration key set")
            return None
            
        documents = cls.get_documents(cls._best_loader_key)
        metadata = cls.get_metadata(cls._best_loader_key)
        
        if documents is None:
            logger.warning(f"No documents found for best config (key: {cls._best_loader_key})")
            return None
            
        return documents, metadata 

    @classmethod
    def store_vectorstore(cls, key: str, vectorstore: Any):
        """Store vectorstore for a given configuration"""
        cls._vectorstores[key] = vectorstore
        logger.debug(f"Stored vectorstore with key: {key}")

    @classmethod
    def get_vectorstore(cls, key: str) -> Optional[Any]:
        """Retrieve vectorstore by key"""
        vectorstore = cls._vectorstores.get(key)
        if vectorstore is None:
            logger.debug(f"No vectorstore found for key: {key}")
        return vectorstore

    @classmethod
    def get_best_config_vectorstore(cls) -> Optional[Any]:
        """Get vectorstore for the best configuration."""
        if not cls._best_config_key:
            logger.warning("No best configuration key set")
            return None
            
        return cls.get_vectorstore(cls._best_config_key)

    @classmethod
    def store_graph(cls, graph: Any):
        """Store a knowledge graph"""
        cls._graph = graph
        logger.debug(f"Stored knowledge graph")

    @classmethod
    def get_graph(cls) -> Optional[Any]:
        """Retrieve the stored knowledge graph"""
        if cls._graph is None:
            logger.debug("No graph found")
        return cls._graph

    @classmethod
    def store_sampled_data(cls, original_path: str, sampled_path: str, metadata: Dict[str, Any]):
        """Store sampled data path and its metadata"""
        cls._sampled_paths[original_path] = {
            "sampled_path": sampled_path,
            "metadata": metadata,
            "timestamp": time.time()
        }
        logger.debug(f"Stored sampled data path for {original_path}")

    @classmethod
    def get_sampled_data(cls, original_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve sampled data path and metadata for an original path"""
        if original_path in cls._sampled_paths:
            logger.debug(f"Found cached sampled data for {original_path}")
            return cls._sampled_paths[original_path]
        return None