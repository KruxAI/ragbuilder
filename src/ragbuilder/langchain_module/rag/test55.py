from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import os
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

def rag_pipeline():
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs) 
        
        llm=ChatOpenAI(model='gpt-3.5-turbo')
        
        loader = WebBaseLoader('https://ashwinaravind.github.io/')
        docs = loader.load()
        
        embedding=OpenAIEmbeddings(model='text-embedding-3-large')
        
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits=splitter.split_documents(docs)
        c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='testindex-ragbuilder-1719591605',)
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
        print(f"An error occurred: {e}")