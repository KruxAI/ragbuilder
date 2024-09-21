code="""from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import  BM25Retriever
import os
from operator import itemgetter
from langchain_community.embeddings import OllamaEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever,EnsembleRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_ollama import ChatOllama
def rag_pipeline():
    try:
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs) 
        {llm_class}
        
        {loader_class}
        
        {embedding_class}
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits=splitter.split_documents(docs)
        document_array=[]
        for i in range(len(splits)):
        # Get 10 adjacent chunks around the current chunk (including the current chunk)
            chunk_group = splits[max(0, i - 5):min(len(splits), i + 5)]
            chunk_content = format_docs([splits[i]])  # The current chunk

        # Combine adjacent chunks for context
            chunk_group_content = format_docs(chunk_group)

        # Construct the prompt with chunk group as the document context
            prompt = f'''
            <document> 
             {chunk_group_content}
            </document> 
            Here is the chunk we want to situate within the chunk group 
            <chunk> 
            {chunk_content}
            </chunk>
            {{question}}
            '''
            prompt=prompt.replace('{', '(').replace('}', ')')
            cr_prompt = ChatPromptTemplate.from_template(prompt)
            cr_prompt_chain =cr_prompt|llm|StrOutputParser() 
            r=cr_prompt_chain.invoke({'question':'Please give a short succinct context to situate this chunk within the overall document (the group of 10 chunks) for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.'})
            document= Document(
                page_content=r+' '+chunk_content,
                id=i)
            document_array.append(document)
        timestamp = str(int(time.time()))
        index_name = "testindex-ragbuilder-" + timestamp
        c=Chroma.from_documents(documents=document_array, embedding=embedding, collection_name=index_name, client_settings=chromadb.config.Settings(allow_reset=True))
        retrievers=[]
        retriever=c.as_retriever(search_type='similarity', search_kwargs={'k': 100})
        retrievers.append(retriever)
        retriever=BM25Retriever.from_documents(docs)
        retrievers.append(retriever)
        retriever=EnsembleRetriever(retrievers=retrievers)
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                    .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                    .assign(answer=prompt | llm | StrOutputParser())
                    .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {e}")"""
