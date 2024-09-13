code="""from langchain_community.llms import Ollama
from langchain_community.document_loaders import *
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.prompts import ChatPromptTemplate
from langchain.load import dumps, loads
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
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
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits=splitter.split_documents(docs)
        c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='testindex-ragbuilder-1724418081',)   
        retriever = c.as_retriever()

        rewrite_prompt = '''You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
        Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
        Original query: {question}'''
        query_rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt)

        query_rewrite_chain =query_rewrite_prompt|llm|StrOutputParser() 

        generate_queries = RunnableParallel({"new_query": query_rewrite_chain , "question": RunnablePassthrough()}).pick('new_query')

        retreive_format = generate_queries | retriever | RunnableLambda(format_docs)

        prompt = hub.pull("rlm/rag-prompt")

        answer = prompt | llm | StrOutputParser()

        rag_new_query_chain = (
            RunnableParallel(context=retreive_format, question=RunnablePassthrough())
        ).assign(answer=answer)
        return rag_new_query_chain
    except Exception as e:
        print(f"An error occurred: {e}")
""" 
   