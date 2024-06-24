
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llamaindex_module.embedding_model.embedding_model import *
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss    
from llama_index.embeddings.mistralai import MistralAIEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
def getIndex(**kwargs):
    if kwargs['vectorDB'] == "CHROMA":
        chroma_client = chromadb.EphemeralClient()
        chroma_collection = chroma_client.create_collection("quickstart")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        storage_context.docstore.add_documents(kwargs['nodes'])
        index = VectorStoreIndex.from_documents(kwargs['documents'], storage_context=storage_context,embeddings= getEmbedding(**kwargs))
        print("calling embedding function")
    elif kwargs['vectorDB'] == "FAISS":
        d = 1536
        faiss_index = faiss.IndexFlatL2(d)
        from llama_index.core import ServiceContext, set_global_service_context        
        # service_context = ServiceContext.from_defaults(embed_model=OpenAIEmbedding())
        service_context = ServiceContext.from_defaults(embed_model=getEmbedding(**kwargs))
        set_global_service_context(service_context)
        
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(kwargs['documents'], storage_context=storage_context)  
    return index

