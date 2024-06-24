from ragbuilder.langchain_module.common import setup_logging
import logging
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, SingleStoreDB
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import uuid
import time
from getpass import getpass
setup_logging()
logger = logging.getLogger("ragbuilder")

def getVectorDB(splits, embedding, embedding_model, db_type):
    """
    Initialize and return a vector database object based on the specified db_type.

    Args:
    - splits (list): List of documents or splits to be indexed.
    - embedding: Embedding model instance or configuration needed for initialization.
    - db_type (str): Type of vector database to initialize. Supported values: "CHROMA", "FAISS".

    Returns:
    - Vector database object (Chroma or FAISS).

    Raises:
    - ValueError: If db_type is not supported.
    """
    timestamp = str(int(time.time()))
    index_name = "testindex-ragbuilder-" + timestamp
    if db_type == "chromaDB":
        logger.info("Chroma DB Loaded")
        logger.info(f"Chroma DB Index Created {index_name}")
        return Chroma.from_documents(documents=splits, embedding=embedding, collection_name=index_name,)
    elif db_type == "faissDB":
        logger.info("FAISS DB Loaded")
        return FAISS.from_documents(documents=splits, embedding=embedding)
    elif db_type == "singleStoreDB":
        index_name = "testindex_ragbuilder_" + timestamp
        logger.info("Singlestore DB Loaded")
        logger.info(f"Singlestore DB Index Created {index_name}")
        return SingleStoreDB.from_documents(documents=splits, embedding=embedding,table_name=index_name)
    elif db_type == "pineconeDB":
        logger.info("PineCone DB Loaded")
        if embedding_model in ["text-embedding-3-small","text-embedding-ada-002"]:
            embedding_dimension=1536
        elif embedding_model in ["text-embedding-3-large"]:
            embedding_dimension=3072
        elif embedding_model in ['mistral-embed']:
             embedding_dimension=1024
        elif embedding_model in ['all-MiniLM-l6-v2']:
             embedding_dimension=384
        else:
            raise
        pc = Pinecone()
        pc.create_index(
            name=index_name,
            dimension=embedding_dimension,
            metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),)
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
        logger.info(f"Pinecone DB Index Created {index_name}")
        return PineconeVectorStore.from_documents(splits, embedding, index_name=index_name)
    else:
        raise ValueError(f"Unsupported db_type: {db_type}. Supported types are singleStoreDB, 'chromaDB', 'pineconeDB' and 'faissDB'.")

