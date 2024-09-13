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
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
def rag_pipeline():
    question=RunnablePassthrough()
    try:
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs) 
        
        def rrf(results):
            fused_scores = {}
            k=60
            for docs in results:
                for rank, doc in enumerate(docs):
                    doc_str = dumps(doc)
                    # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                    # print('\\n')
                    if doc_str not in fused_scores:
                        fused_scores[doc_str] = 0
                        # Retrieve the current score of the document, if any
                        previous_score = fused_scores[doc_str]
                        # Update the score of the document using the RRF formula: 1 / (rank + k)
                        fused_scores[doc_str] += 1 / (rank + k)

                # final reranked result
            reranked_results = [
                (loads(doc), score)
                    for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
            return reranked_results

        
        {llm_class}
        
        {loader_class}
        
        {embedding_class}
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits=splitter.split_documents(docs)
        c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='testindex-ragbuilder-1724418081',)   
        retriever = c.as_retriever()

        template = '''You are a helpful assistant that generates multiple search queries based on a single input query. \\n
        Generate multiple search queries related to: {question} \\n
        Output (4 queries):'''
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)

        generate_queries = (
            prompt_rag_fusion 
            | llm
            | StrOutputParser() 
            | (lambda x: x.split("\\n"))
        )

        ## Chain for extracting relevant documents
        retrieval_chain_rag_fusion = generate_queries | retriever.map()

        results = retrieval_chain_rag_fusion.invoke({"question": question})

        lst=[]
        for ddxs in results:
            for ddx in ddxs:
                if ddx.page_content not in lst:
                    lst.append(ddx.page_content)

        reranked_results=rrf(results)
        template = '''Answer the following question based on this context:
        {context}

        Question: {question}
        '''

        prompt = ChatPromptTemplate.from_template(template)
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {e}")
"""