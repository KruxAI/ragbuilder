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
        rewrite_prompt = '''Here are the different RAG prompt formats.
PROMPT_FORMATS:

1. **Informative**  
   You are a helpful assistant. Answer any questions solely based on the context provided below.  
   If the provided context does not have the relevant facts to answer the question, say "I don't know."

2. **Concise Minimal**  
   Answer the question only using the provided context.  
   If the context lacks the information required, respond with "I don't know."

3. **Factual**  
   You are a highly accurate assistant. Respond only with the facts found in the provided context.  
   If there is insufficient information in the context, say "I don't know."

4. **Strict Contextual**  
   Do not use any external knowledge. Answer the question solely based on the context below.  
   If the context does not contain the answer, say "I don't know."

5. **Step-by-Step**  
   Think step-by-step explanation based only on the information in the context.  
   If the context is insufficient to answer the question, respond with "I don't know."

7. **Conversational**  
   You are a friendly assistant engaging in a natural conversation.  
   Respond based only on the provided context, using a tone that is casual and approachable.  
   If the context lacks information, politely indicate that you don’t know.

8. **Exploratory**  
   You are an assistant who explores ideas critically and presents potential interpretations or analyses based on the provided context.  
   If the context does not contain enough information, explain why further details are necessary.

9. **Summarization**  
   Summarize the main points from the provided context in a concise and coherent manner.  
   Focus only on what is directly supported by the context.

10. **Comparative**  
    Based on the provided context, compare and contrast the given topics.  
    Highlight similarities and differences, avoiding any external information.  
    If the context is insufficient, respond with "I don't know."

11. **Persuasive**  
    Formulate a compelling argument based solely on the context provided.  
    Present supporting evidence clearly, and conclude your argument.  
    If the context lacks sufficient information, say "I don't know."

12. **Creative**  
    Based on the provided context, craft a creative response such as a story, analogy, or metaphor that helps explain the content.  
    Avoid adding any information not in the context.

13. **Multi-Turn**  
    You are a conversational assistant. Respond based on the provided context, anticipating potential follow-up questions and preparing to clarify details iteratively.  
    If information is missing, say, "The context does not provide enough information."

14. **Critical Thinking**  
    Critically evaluate the information in the provided context.  
    Identify any assumptions, implications, or inconsistencies.  
    If the context lacks sufficient detail, say "I don't know."

15. **Instructional**  
    Use the provided context to create a step-by-step guide or tutorial.  
    Ensure clarity and coherence in the instructions.  
    If the context does not provide enough detail, state, "The context does not include sufficient information for a tutorial."

16. **Hypothetical**  
    Based on the provided context, explore hypothetical scenarios or outcomes.  
    If the context does not support any hypothetical reasoning, respond with 'The context does not allow for such exploration.'

17. **Role-Specific**  
    You are a [doctor/scientist/lawyer/teacher].  
    Answer questions using the tone and knowledge expected from your role, but only based on the provided context.  
    If the context lacks information, respond with 'I don't know.'

18. **Decision-Making**  
    Using the provided context, suggest a course of action or make a recommendation.  
    Clearly state the reasoning behind your decision.  
    If the context does not provide enough information, say "I don't know."

19. **Data-Driven**  
    Analyze the data provided in the context and summarize any trends, insights, or anomalies.  
    Avoid introducing external knowledge or assumptions.  
    If the context lacks sufficient detail, respond with 'I don't know.'

Here are the Questions and Ground Truth Answers. From the ground truth answer find the Tone, personality, factualness ,conciseness, and style from the above prompt techniques. It could be a mis and make a prompt that best suits the 
QUESTION: {question}
GROUND_TRUTH: Retrieval-Augmented Generation enhances text generation by incorporating externally retrieved information into the generative process. It combines retrieval-based models and generative models to fetch contextually relevant and accurate data from vast knowledge bases, which is then used to inform the text generation process. This ensures that the output is not only relevant but also factually precise and tailored to the input it responds to.
'''
        query_rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt)

        query_rewrite_chain =query_rewrite_prompt|llm|StrOutputParser() 
        return query_rewrite_chain

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None

print(rag_pipeline().invoke("How does Retrieval-Augmented Generation enhance text generation?"))

#Method 1: 
#Generate Prompt formats for RAG
#Pass the questions to the different RAG prompt formats
#Pick the one with highest answer correctness

#Method 2: 
#Send the Prompt formats, question , answer and ground truth. AsK LLM to come up with the best prompt format based on the prompt formats 