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
    "concise_minimal": """
    Answer the question only using the provided context. If the context lacks the information required, respond with "I don't know."

    <context>
    {context}
    </context>
    """,
    "factual_accurate": """
    You are a highly accurate assistant. Respond only with the facts found in the provided context. 
    If there is insufficient information in the context, say "I don't know."

    <context>
    {context}
    </context>
    """,
    "strict_contextual": """
    Do not use any external knowledge. Answer the question solely based on the context below. If the context does not contain the answer, say "I don't know."

    <context>
    {context}
    </context>
    """,
    "step_by_step": """
    Provide a step-by-step explanation based only on the information in the context. If the context is insufficient to answer the question, respond with "I don't know."

    <context>
    {context}
    </context>
    """,
    "custom": """You are an expert assistant. Use the provided context to answer the question accurately and concisely. Ensure your response strictly aligns with the context and matches the expected structure and detail level. If the context does not contain sufficient information, state, 'The context does not provide enough information to answer.' Avoid adding any information not present in the context
    
    <context>
    {context}
    </context>""",
    "current": """You are a helpful assistant. Answer any questions solely based on the context provided below. If the provided context does not have the relevant facts to answer the question, say I don't know
    <context>
    {context}
    </context>""",
    "custom_modifed": """You are an expert assistant. Use the provided context to answer the question in a detailed and structured manner. Ensure your response aligns strictly with the context while elaborating on key concepts sequentially. Begin by introducing the overarching idea, then explain the process step-by-step, and conclude with its impact. Avoid adding any information not present in the context. If the context does not contain sufficient information, state, "The context does not provide enough information to answer".
    
    <context>
    {context}
    </context>""",
    "custom_modifed_fewshot": """You are an expert assistant. Use the provided context to answer the question in a detailed and structured manner. Ensure your response aligns strictly with the context while elaborating on key concepts sequentially. Begin by introducing the overarching idea, then explain the process step-by-step, and conclude with its impact. Avoid adding any information not present in the context. If the context does not contain sufficient information, state, "The context does not provide enough information to answer."

Few-shot Examples:

Example 1:
Q: How does Transfer Learning improve model performance?
Context: Transfer Learning allows models to leverage knowledge from pre-trained tasks and apply it to new, related tasks. By reusing features or embeddings learned during pretraining, models can adapt more quickly to smaller datasets and achieve better performance with less labeled data.
A: Transfer Learning improves model performance by enabling models to reuse knowledge from pre-trained tasks. It leverages features or embeddings learned during pretraining to adapt efficiently to new, related tasks. This reduces the need for large labeled datasets and improves performance by transferring relevant insights, particularly when data for the new task is scarce.

Example 2:
Q: What are the advantages of using convolutional layers in neural networks?
Context: Convolutional layers are a type of neural network layer designed to process spatial information. They use filters to detect patterns like edges, textures, and shapes, making them ideal for image and video data. By leveraging shared weights and local connectivity, convolutional layers reduce the number of parameters and improve computational efficiency.
A: Convolutional layers offer several advantages in neural networks. They are specifically designed to process spatial information using filters that detect patterns such as edges, textures, and shapes. This makes them highly effective for image and video data. Additionally, shared weights and local connectivity reduce the number of parameters, enhancing computational efficiency and scalability.
    <context>
    {context}
    </context>""",
    "current": """You are a helpful assistant. Answer any questions solely based on the context provided below. If the provided context does not have the relevant facts to answer the question, say I don't know
    <context>
    {context}
    </context>""",
    "claude_base": """ You are a highly precise AI assistant. Your task is to answer questions based solely on the provided context. Follow these guidelines:
Role: You are an expert at analyzing information and providing accurate, focused answers.

Context Guidelines:
- Only use information explicitly stated in the provided context
- If the context is insufficient, state so clearly
- Do not introduce external knowledge
- Highlight key evidence that supports your answer

Answer Format:
1. Direct Answer: Provide a clear, concise answer

Response Structure:
Answer: [Concise answer]
 <context>
    {context}
    </context>""",
    "claude_enhanced_w_reasoning": """ You are a highly precise AI assistant. Analyze the question and context carefully before providing an answer.

1. Question Analysis:
   - Identify key components of the question
   - List specific information needed to answer completely

2. Context Evaluation:
   - Review provided context thoroughly
   - Map available information to question requirements
   - Identify any information gaps

3. Reasoning Process:
   - Break down your thinking step-by-step
   - Connect evidence to conclusions explicitly
   - Acknowledge uncertainties

4. Answer Formulation:
   - State your confident findings first
   - Support with direct quotes from context
   - Address any limitations or assumptions
   - Suggest what additional context would help (if needed)

Response Format:
Answer: [Clear, supported conclusion]
 <context>
    {context}
    </context>
""",
"claude_fact_checking": """You are a precise AI assistant focused on factual accuracy. Your task is to:

1. Initial Assessment:
   - Compare the provided answer against the ground truth
   - Identify any discrepancies or misalignments

2. Context Analysis:
   - Evaluate if the context supports the answer
   - Note any missing or contradictory information

3. Accuracy Evaluation:
   - Score accuracy on a scale of 1-5
   - Identify specific improvements needed

4. Improvement Suggestions:
   - Propose specific rewording or corrections
   - Highlight additional context needed

Response Format:
Accuracy Score: [1-5 with explanation]
Alignment with Ground Truth: [Analysis]
Context Support: [Evidence assessment]
Recommended Improvements: [Specific suggestions]
Additional Context Needed: [If any]
 <context>
    {context}
    </context>
"""
# "new":""" 
# You are an expert assistant. Use the provided context to answer the question in a detailed and structured manner. Ensure your response aligns strictly with the context while elaborating on key concepts sequentially. Begin by introducing the overarching idea, then explain the process step-by-step, and conclude with its impact. Avoid adding any information not present in the context. If the context does not contain sufficient information, state, "The context does not provide enough information to answer."  
# Example:  
# Q: How does Retrieval-Augmented Generation enhance text generation?  
# A: Retrieval-Augmented Generation enhances text generation by incorporating externally retrieved information into the generative process. It combines retrieval-based models and generative models to fetch contextually relevant and accurate data from vast knowledge bases, which is then used to inform the text generation process. This ensures that the output is not only relevant but also factually precise and tailored to the input it responds to.
#  <context>
#     {context}
#     </context>
# """
    
    }
prompts ={
"test_prompt":
"""Based on the ground truth answer provided, the following attributes can be identified:

- **Tone**: Informative and technical
- **Personality**: Professional and analytical
- **Factualness**: Highly factual, relying solely on the provided information
- **Conciseness**: Reasonably concise, presenting the information clearly without unnecessary elaboration
- **Style**: Explanatory and descriptive, focusing on the process and benefits of Retrieval-Augmented Generation

Given these attributes, the most suitable prompt format for the provided question and answer would be **Factual**. Here’s the crafted prompt:

---

**Prompt**: You are a highly accurate assistant. Respond only with the facts found in the provided context. If there is insufficient information in the context, say "I don't know."

---

This format ensures that the answer is focused on delivering factual information in a clear and precise manner, aligning with the tone and style of the ground truth answer.
"""}
 
# print(get_eval_dataset())
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
import pandas as pd
from datasets import Dataset


def evaluate_prompts(eval_dataset,iterno):
    print("evaluate_prompts initiated")
    print(eval_dataset)
    result = evaluate(
            eval_dataset,
            metrics=[
                answer_correctness,
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            raise_exceptions=False, 
            llm = AzureChatOpenAI(model="gpt-4o-mini"),
            embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large"),
            is_async=True,
            run_config=RunConfig(timeout=240, max_workers=1, max_wait=180, max_retries=10)
        )
    result_df = result.to_pandas()
    output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_eval_results"+iterno+".csv"
    selected_columns = ["prompt_key","question","answer","ground_truth","answer_correctness","faithfulness","answer_relevancy","context_precision","context_recall"]
    result_df[selected_columns].to_csv(output_csv_path, index=False)
    print("evaluate_prompts completed")
    return Dataset.from_pandas(result_df[selected_columns])

def get_eval_dataset(csv_file_path):
    print("get_eval_dataset initiated")
    # Specify the path to your CSV file
    csv_file_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_test_data_lilianweng_gpt-4o_1721032414.736622_SEMI.csv"
    # Read the CSV file into a pandas DataFrame
    # df = pd.read_csv(csv_file_path,on_bad_lines='skip')
    df = pd.read_csv(csv_file_path)
    eval_dataset = Dataset.from_pandas(df)  # Convert to Dataset for Ragas compatibility
    print("get_eval_dataset completed",df)
    return eval_dataset, df  # Returning both for flexibility


# Iterate through each prompt
def test_prompt(eval_dataset,iterno):
    print("test_prompt initiated")
    results = {}
    for prompt_key, prompt_template in prompts.items():
        print(f"Testing Prompt: {prompt_key}...")
        pipeline = rag_pipeline(prompt_template,retriever)  # Initialize your RAG pipeline with the current prompt template
        if pipeline:
            # Iterate thrazzough each question in eval_dataset
            for entry in eval_dataset:
                question = entry.get("question", "")  # Extract the question from the dataset
                ground_truth = entry.get("ground_truth", "")  # Extract the ground truth answer from the dataset
                if not question:
                    continue  # Skip entries without a valid question

                # Invoke pipeline for the question
                result = pipeline.invoke(question)

                # Store results
                if prompt_key not in results:
                    results[prompt_key] = []
                results[prompt_key].append({
                    "prompt_key": prompt_key,
                    "question": question,
                    "answer": result.get("answer", "Error"),
                    "context": result.get("context", "Error"),
                    "ground_truth": ground_truth,
                })
                print(f"Question: {question}")
                print(f"Ground_truth: {ground_truth}")
                print(f"Context: {result.get("context", "Error")}")
                break
        break
    import pandas as pd
    output_data = []
    for prompt_key, prompt_results in results.items():
        output_data.extend(prompt_results)

    results_df = pd.DataFrame(output_data)
    print("check1")
    if "context" in results_df.columns:
                    results_df["contexts"] = results_df["context"].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
                    )
    # Save the DataFrame to a CSV file
    output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_results"+iterno+".csv"
    results_df.to_csv(output_csv_path, index=False)
    prompt_test_dataset = Dataset.from_pandas(results_df)
    print("test_prompt completed")
    return prompt_test_dataset

def agg_eval_prompts(prompt_test_dataset,iterno):
    print("eval_prompts initiated")
    # Group by 'prompt_key' and calculate the mean of 'answer_correctness'
    # output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_results.csv"
    # results_df = pd.read_csv(output_csv_path, delimiter=',', quotechar='"', on_bad_lines='skip')
    results_df = prompt_test_dataset.to_pandas()
    average_correctness = results_df.groupby('prompt_key')['answer_correctness'].mean().reset_index()
    # Rename columns for clarity (optional)
    average_correctness.columns = ['prompt_key', 'average_correctness']
    # Save the result to a CSV file
    average_correctness.to_csv('rag_average_correctness'+iterno+'.csv', index=False)
    print("The results have been saved to 'average_correctness.csv'")
    # eval_dataset = Dataset.from_pandas(results_df)
    print("eval_prompts completed")

# Load the evaluation dataset
# Initialize results dictionary
retriever=rag_get_retriever()
def read_csv(csv_file_path,i):
    print("reading csv")
    csv_file_path = csv_file_path+str(i)+".csv"
    results_df = pd.read_csv(csv_file_path)
    if "context" in results_df.columns:
                    results_df["contexts"] = results_df["context"].apply(
                        lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
                    )
    eval_dataset = Dataset.from_pandas(results_df)  # Convert to Dataset for Ragas compatibility
    return eval_dataset
def run_test(iteration_no):
    print("Iteration initiated",iteration_no)
    eval_dataset, df = get_eval_dataset("/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_test_data_lilianweng_gpt-4o_1721032414.736622_SEMI.csv")
    prompt_test_dataset=test_prompt(eval_dataset,iteration_no)
    prompt_test_dataset=read_csv('/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_results',iteration_no)
    prompt_eval_results=evaluate_prompts(prompt_test_dataset,iteration_no)
    agg_eval_prompts(prompt_eval_results,iteration_no)
    print("Iteration completed",iteration_no)
print("all functions completed")
# from ragbuilder.langchain_module.chunkingstrategy.testprompt import run_test