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


def rag_pipeline():
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)
                # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")
        # Prompt setup
        rewrite_prompt = '''You are a highly accurate assistant. Analyze the provided context thoroughly to extract relevant information that directly answers the question. Ensure that your response is concise, summarized, and supported by evidence from the context. If the context lacks sufficient information to formulate an answer, state "I don't know." Maintain an informative tone while providing clear and coherent insights.
        {question}'''
        query_rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt)

        query_rewrite_chain =query_rewrite_prompt|llm|StrOutputParser() 
        return query_rewrite_chain

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None
source="""
Question: What is the inspiration behind the HNSW (Hierarchical Navigable Small World) algorithm and how does it function?
Ground_truth: Hey, HNSW (Hierarchical Navigable Small World) leverages small-world network principles, enabling efficient search via hierarchical layers. Upper layers provide shortcuts for broad navigation, while lower layers house data points for precise refinement. Searches start at the top, progressing downward, combining fast traversal with high accuracy inspired by "six degrees of separation.
Context: HNSW (Hierarchical Navigable Small World): It is inspired by the idea of small world networks where most nodes can be reached by any other nodes within a small number of steps; e.g. “six degrees of separation” feature of social networks. HNSW builds hierarchical layers of these small-world graphs, where the bottom layers contain the actual data points. The layers in the middle create shortcuts to speed up search. When performing a search, HNSW starts from a random node in the top layer and
where the bottom layers contain the actual data points. The layers in the middle create shortcuts to speed up search. When performing a search, HNSW starts from a random node in the top layer and navigates towards the target. When it can’t get any closer, it moves down to the next layer, until it reaches the bottom layer. Each move in the upper layers can potentially cover a large distance in the data space, and each move in the lower layers refines the search quality.
some extent, it mimics a hashing function. ANNOY search happens in all the trees to iteratively search through the half that is closest to the query and then aggregates the results. The idea is quite related to KD tree but a lot more scalable.
Fig. 13. The generative agent architecture. (Image source: Park et al. 2023)
This fun simulation results in emergent social behavior, such as information diffusion, relationship memory (e.g. two agents continuing the conversation topic) and coordination of social events (e.g. host a party and invite many others).
Proof-of-Concept Examples#
LSH (Locality-Sensitive Hashing): It introduces a hashing function such that similar input items are mapped to the same buckets with high probability, where the number of buckets is much smaller than the number of inputs.
"""
print(rag_pipeline().invoke(source))

#Method 1: 
#Generate Prompt formats for RAG
#Pass the questions to the different RAG prompt formats
#Pick the one with highest answer correctness

#Method 2: 
#Send the Prompt formats, question , answer and ground truth. AsK LLM to come up with the best prompt format based on the prompt formats 