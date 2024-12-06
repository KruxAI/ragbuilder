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
from ragas import evaluate, RunConfig

def rag_get_retriever():
    print("rag_get_retriever initiated")
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)

        # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")

        # Document loader
        loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
        docs = loader.load()

        # Embedding model
        embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large")

        # Text splitting and embedding storage
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        splits = splitter.split_documents(docs)

        # Initialize Chroma database
        c = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name="testindex-ragbuilder-retreiver",
            client_settings=chromadb.config.Settings(allow_reset=True),
        )

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

def rag_pipeline(prompt_template,retriever):
    """
    Initializes a Retrieval-Augmented Generation pipeline using LangChain with a customizable prompt template.

    Args:
        prompt_template (str): The system prompt template to test.

    Returns:
        RunnableParallel: The RAG pipeline ready to process queries.
    """
    try:
        print("rag_pipeline initiated")
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
                # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")
        # Prompt setup
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_template),
                ("user", "{question}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ]
        )

        # RAG Chain setup
        rag_chain = (
            RunnableParallel(context=retriever, question=RunnablePassthrough())
            .assign(context=itemgetter("context") | RunnableLambda(format_docs))
            .assign(answer=prompt | llm | StrOutputParser())
            .pick(["answer", "context"])
        )
        print("rag_pipeline completed")
        return rag_chain

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None


# Define a dictionary of prompts with descriptive keys
prompts = {
    ""
    "default_informative": """
    You are a helpful assistant. Answer any questions solely based on the context provided below. 
    If the provided context does not have the relevant facts to answer the question, say "I don't know."

    <context>
    {context}
    </context>
    """,
    "new": """ You are a highly accurate assistant. Analyze the provided context thoroughly to extract relevant information that directly answers the question. Ensure that your response is concise, summarized, and supported by evidence from the context. If the context lacks sufficient information to formulate an answer, state "I don't know." Maintain an informative tone while providing clear and coherent insights.
    <context>
    {context}
    </context>"""
    }
questions=[ 
    "What is the inspiration behind the HNSW (Hierarchical Navigable Small World) algorithm and how does it function?",
    "Can you provide a detailed overview of the inspiration and theoretical foundations behind the HNSW (Hierarchical Navigable Small World) algorithm? Additionally, please explain its operational mechanisms, including how it structures data for efficient nearest neighbor search and the specific advantages it offers compared to other algorithms in high-dimensional spaces.",
    "1. What inspired the development of the HNSW (Hierarchical Navigable Small World) algorithm? 2. How does the HNSW algorithm function in terms of its structure and computational process? "
]
questions=[ 
    "What is the inspiration behind the HNSW (Hierarchical Navigable Small World) algorithm and how does it function?"]
retriever=rag_get_retriever()
for prompt_key, prompt_template in prompts.items():
    pipeline = rag_pipeline(prompt_template,retriever)
    for q in questions:
        result = pipeline.invoke(q)
        print(result)
 