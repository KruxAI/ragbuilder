from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb

from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ragbuilder.generation.evaluation import RAGASEvaluator
from langchain_community.vectorstores import FAISS
def sample_retriever(url):
    print("rag_get_retriever initiated")
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)

        # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")

        # Document loader
        loader = WebBaseLoader(url)
        docs = loader.load()

        # Embedding model
        embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large")

        # Text splitting and embedding storage
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        # Initialize Chroma database
        c=FAISS.from_documents(documents=splits, embedding=embedding)
        # c = Chroma.from_documents(
        #     documents=splits,
        #     embedding=embedding,
        #     collection_name="testindex-ragbuilder-retreiver_33333",
        #     client_settings=chromadb.config.Settings(allow_reset=True),
        # )

        # Retriever setup
        retriever = c.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever])
        print("rag_get_retriever completed")
        return ensemble_retriever
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None
# sample_retriever()