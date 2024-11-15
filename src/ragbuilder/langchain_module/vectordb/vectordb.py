from ragbuilder.langchain_module.common import setup_logging
import logging
import chromadb
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, SingleStoreDB
from langchain_postgres.vectorstores import PGVector
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_milvus import Milvus
import uuid
import time
import random
from getpass import getpass
setup_logging()
logger = logging.getLogger("ragbuilder")

def getVectorDB(db_type,embedding_model):
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
    logger.debug(f"getVectorDB:{db_type}{embedding_model}")
    timestamp = str(int(time.time()*1000+random.randint(1, 1000)))
    index_name = "testindex-ragbuilder-" + timestamp
    if db_type == "chromaDB":
        logger.info("Chroma DB Loaded")
        logger.info(f"Chroma DB Index Created {index_name}")
        code_string= f"""c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='{index_name}', client_settings=chromadb.config.Settings(allow_reset=True))"""
        import_string = f"""from langchain_chroma import Chroma
import chromadb"""
        print({'code_string':code_string,'import_string':import_string})
    elif db_type == "faissDB":
        logger.info("FAISS DB Loaded")
        code_string= f"""c=FAISS.from_documents(documents=splits, embedding=embedding)"""
        import_string = f"""from langchain_community.vectorstores import FAISS"""
    elif db_type == "milvusDB":
        index_name = "testindex_ragbuilder_" + timestamp
        logger.info("Milvus DB Loaded")
        code_string= f"""c = Milvus.from_documents(splits,embedding,collection_name='{index_name}',connection_args={{"uri": MILVUS_CONNECTION_STRING}},)"""
        print(code_string)
        import_string = f"""from langchain_milvus import Milvus"""
    elif db_type == "qdrantDB":
        logger.info("qdrant DB Loaded")
        code_string= f"""sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
c = QdrantVectorStore.from_documents(splits,embedding,location=":memory:",collection_name='{index_name}',retrieval_mode=RetrievalMode.HYBRID,sparse_embedding=sparse_embeddings)"""
        import_string = f"""from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient"""
    elif db_type == "weaviateDB":
        logger.info("weaviate DB Loaded")
        code_string= f"""weaviate_client = weaviate.connect_to_local()
c = WeaviateVectorStore.from_documents(docs, embedding, client=weaviate_client)"""
        import_string = f"""import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore"""
    elif db_type == "singleStoreDB":
        index_name = "testindex_ragbuilder"
        logger.info("Singlestore DB Loaded")
        logger.info(f"Singlestore DB Index Created {index_name}")
        code_string= f"""
index_name='{index_name}'
import singlestoredb as s2
conn = s2.connect(SINGLESTOREDB_URL)
with conn:
    conn.autocommit(True)
    with conn.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS {index_name}')
c=SingleStoreDB.from_documents(documents=splits, embedding=embedding,table_name='{index_name}')""" 
        import_string= f"""from langchain_community.vectorstores import SingleStoreDB"""
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
        logger.info(f"Pinecone DB Index Created {index_name}")
        code_string= f"""
index_name='{index_name}'
pc = Pinecone()
pc.create_index(
name='{index_name}',
dimension={embedding_dimension},
metric="cosine",
spec=ServerlessSpec(cloud="aws", region="us-east-1"),)
while not pc.describe_index(index_name).status["ready"]:
    time.sleep(1)
c=PineconeVectorStore.from_documents(splits, embedding, index_name='{index_name}')"""
        import_string = f"""from langchain_pinecone import PineconeVectorStore"""
    elif db_type == "pgvector":
        code_string= f"""
connection = PGVECTOR_CONNECTION_STRING
collection_name = '{index_name}'
c = PGVector(embeddings=embedding,collection_name=collection_name,connection=connection,use_jsonb=True)"""
        import_string = f"""from langchain_postgres.vectorstores import PGVector"""
    else:
        raise ValueError(f"Unsupported db_type: {db_type}. Supported types are singleStoreDB, 'chromaDB', 'pineconeDB' and 'faissDB'.")
    return {'code_string':code_string,'import_string':import_string}

