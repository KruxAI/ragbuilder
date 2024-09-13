code="""from langchain_community.llms import Ollama
from langchain_community.document_loaders import *
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
def rag_pipeline():
    try:
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs) 
        
        {llm_class}
        
        {loader_class}
        
        {embedding_class}

        splitter = SemanticChunker(embedding, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90)
        splits=splitter.create_documents(docs[0].page_content)
        c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='testindex-ragbuilder',)
        retrievers=[]
        retriever=c.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        retrievers.append(retriever)
        retriever=MergerRetriever(retrievers=retrievers)
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {e}")"""