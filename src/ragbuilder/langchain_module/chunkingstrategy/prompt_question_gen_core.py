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
        rewrite_prompt = '''
You are a highly intelligent and detail-oriented assistant. Your task is to refine the given question(s) so that they become clear, specific, and contextually accurate. The question may contain one or multiple subquestions, or require a structured, step-by-step breakdown. Follow the process below for refinement:

1. **Detect Multiple Subquestions**: 
   - If the input contains conjunctions like "and", "or", or punctuation that separates distinct queries, identify and segment these into separate questions.
   - Example: "Who won the match, and what was the score?" → Segment into: "Who won the match?" and "What was the score of the match?"

2. **Segment Subquestions**: 
   - If multiple subquestions are detected, break them down into clear, logically independent questions.
   - If the subquestions are tightly related, combine them into one, cohesive question for clarity.
   
3. **Step-by-Step Analysis**: 
   - If the question requires a step-by-step process to answer, break it down into the appropriate steps and provide a plan to reach the final answer. 
   - Example: "How do I solve this equation?" → Subquestions: 
     - Step 1: "What is the formula for solving this type of equation?" 
     - Step 2: "What values do we need to plug into the formula?"
     - Final Answer: "What is the result after performing the steps?"

4. **Special Considerations for Ambiguity and Scope**: 
   - If the input question is unclear or has multiple interpretations, use heuristics to clarify. Consider the following:
     - Resolve ambiguous terms (e.g., "big" → "large in size" or "significant increase").
     - Add missing context to make the question more specific (e.g., specifying "in 2021" if the question is about a year-specific event).
     - Ensure temporal clarity when needed (e.g., “in the past year” vs. “currently”).
     - Narrow the scope when the question is too broad (e.g., instead of asking “What are the benefits of exercise?”, specify “What are the cardiovascular benefits of regular aerobic exercise?”).
     - Avoid overgeneralization and replace vague qualifiers (e.g., “What is the best city?” → “What is the most affordable city to live in based on median rent?”).
     - Include keywords relevant to the topic to ensure focus (e.g., specify the topic like "climate change" or "technology sector").
   
5. **Grammar and Syntax Correction**: 
   - Correct any grammatical or syntactic errors to improve readability and comprehension.

6. **Remove Irrelevant Details**: 
   - Discard any information that doesn’t contribute directly to answering the query.

7. **Clarify Pronouns or References**: 
   - Ensure that pronouns and vague references (e.g., "it", "they") are clarified to avoid ambiguity.

8. **Handle Single Questions**: 
   - If only a single question is provided, apply the standard refinement heuristics to clarify and improve it.

**Note**: Provide only the **refined questions** after applying the necessary refinements, ensuring that they are clear, specific, and contextually accurate.

**Input Question**: 
{question}

---

### **Additional Considerations**:
- **Clarification of Intent**: If the question seems to involve multiple potential interpretations (e.g., "What’s the best way to learn AI?"), try to identify the intent of the user by suggesting the most common or plausible interpretation based on context.
- **Prioritization of Relevance**: Ensure that refinements don’t introduce unnecessary complexity. Focus on keeping the question as simple and relevant as possible, especially in cases of overloaded or multiple parts.
'''
        query_rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt)
        query_rewrite_chain =query_rewrite_prompt|llm|StrOutputParser() 
        return query_rewrite_chain

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None
# source="""Quesion: What is the inspiration behind the HNSW (Hierarchical Navigable Small World) algorithm and how does it function?"""
source="""Given the information about Lyft financials for year 2021 and Uber financials for year 2021.
Compare and contrast the customer segments and geographies that grew the fastest"""
print(rag_pipeline().invoke(source))

#Method 1: 
#Generate Prompt formats for RAG
#Pass the questions to the different RAG prompt formats
#Pick the one with highest answer correctness

#Method 2: 
#Send the Prompt formats, question , answer and ground truth. AsK LLM to come up with the best prompt format based on the prompt formats 


# !mkdir -p 'data/10k/'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf' -O 'data/10k/uber_2021.pdf'
# !wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/lyft_2021.pdf' -O 'data/10k/lyft_2021.pdf'