from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import torch

# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
import os
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever,EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from byaldi import RAGMultiModalModel

def rag_pipeline():
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs) 
        
        llm= AzureChatOpenAI(model='gpt-4o-mini')
        
        loader = WebBaseLoader('https://ashwinaravind.github.io/')
        docs = loader.load()
        RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)
        RAG.index(
        input_path="https://ashwinaravind.github.io/",
        index_name="attention",
        store_collection_with_index=True, # set this to false if you don't want to store the base64 representation
        overwrite=True,
        )
        retriever = RAG.as_langchain_retriever(k=3)
        # retriever=EnsembleRetriever(retrievers=retriever)
        arr_comp=[]
        pipeline_compressor = DocumentCompressorPipeline(transformers=arr_comp)
        retriever=ContextualCompressionRetriever(base_retriever=retriever,base_compressor=pipeline_compressor)
        # prompt = hub.pull("rlm/rag-prompt")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", '''You are a helpful assistant. Answer any questions solely based on the context provided below. If the provided context does not have the relevant facts to answer the question, say I don't know. 

<context>
{context}
</context>'''),
                ("user", "{question}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ]
        )
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        res=rag_chain.invoke("what is RAG")
        print(res["answer"])
        print(res["context"])
    except Exception as e:
        print(f"An error occurred: {e}")

#To get the answer and context, use the following code
# rag_pipeline()

from byaldi import RAGMultiModalModel

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")


RAG.index(
    input_path="https://ashwinaravind.github.io/", # The path to your documents
    index_name='index_name', # The name you want to give to your index. It'll be saved at `index_root/index_name/`.
    store_collection_with_index=False, # Whether the index should store the base64 encoded documents.
    doc_ids=[0, 1, 2], # Optionally, you can specify a list of document IDs. They must be integers and match the number of documents you're passing. Otherwise, doc_ids will be automatically created.
    metadata=[{"author": "John Doe", "date": "2021-01-01"}], # Optionally, you can specify a list of metadata for each document. They must be a list of dictionaries, with the same length as the number of documents you're passing.
    overwrite=True # Whether to overwrite an index if it already exists. If False, it'll return None and do nothing if `index_root/index_name` exists.
)


# from diffusers import DiffusionPipeline
# import torch

# pipeline = DiffusionPipeline.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("mps")
# pipeline.enable_attention_slicing()

# import torch

# # Check that MPS is available
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
#     else:
#         print("MPS not available because the current MacOS version is not 12.3+ "
#               "and/or you do not have an MPS-enabled device on this machine.")

# else:
#     print("all good")