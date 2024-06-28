def rag_pipeline():
        from langchain.retrievers.document_compressors import DocumentCompressorPipeline
        from operator import itemgetter
        from langchain import hub
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
        from langchain.retrievers import ContextualCompressionRetriever, MergerRetriever
        from langchain_community.document_transformers import LongContextReorder
        def format_docs(docs):
            return ".".join(doc.page_content for doc in docs)



        from langchain_openai import ChatOpenAI


        from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader, UnstructuredFileLoader


        from langchain.text_splitter import RecursiveCharacterTextSplitter

        from langchain_openai import OpenAIEmbeddings

        from langchain_chroma import Chroma



        from langchain.retrievers import  BM25Retriever

        from langchain.retrievers import ContextualCompressionRetriever


        from langchain_openai import ChatOpenAI
        llm=ChatOpenAI(model='gpt-3.5-turbo')



        loader = WebBaseLoader('https://ashwinaravind.github.io/')
        docs = loader.load()



        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits=splitter.split_documents(docs)


        embedding=OpenAIEmbeddings(model='text-embedding-3-large')

        c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='index_name',)
        retrievers=[]
        retriever=c.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        retrievers.append(retriever)
        retriever=BM25Retriever.from_documents(docs)
        retrievers.append(retriever)
        
        retriever=MergerRetriever(retrievers=retrievers)
        from langchain.retrievers.document_compressors import EmbeddingsFilter
        embeddings_filter =EmbeddingsFilter(embeddings=embedding, similarity_threshold=0.76)
        arr_comp=[]
        # arr_comp.append(embeddings_filter)
        arr_comp.append(LongContextReorder())
        pipeline_compressor = DocumentCompressorPipeline(transformers=arr_comp)
        retriever=ContextualCompressionRetriever(base_retriever=retriever,base_compressor=pipeline_compressor)
        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
                RunnableParallel(context=retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"]))
        return rag_chain

r=rag_pipeline()
res=r.invoke("How many startups are there in India?")   
print(res)