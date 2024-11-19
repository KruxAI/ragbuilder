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
    try:
        def format_docs(docs):
            return "\n".join(doc.page_content for doc in docs)

        # LLM setup
        llm = AzureChatOpenAI(model="gpt-4o-mini")

        # Document loader
        loader = WebBaseLoader("https://ashwinaravind.github.io/")
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
            collection_name="testindex-ragbuilder-1ee",
            client_settings=chromadb.config.Settings(allow_reset=True),
        )

        # Retriever setup
        retriever = c.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(retrievers=[retriever])
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

        return rag_chain

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return None


# Define a dictionary of prompts with descriptive keys
prompts = {
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
    """
}
def get_eval_dataset():
    import pandas as pd
    from datasets import Dataset
    # Specify the path to your CSV file
    csv_file_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/india.csv"
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    eval_dataset = Dataset.from_pandas(df) # TODO: Use from_dict instead on eval_ds directly?
    return eval_dataset

# print(get_eval_dataset())
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
def evaluate1(eval_dataset):
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
    output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_eval_results.csv"
    selected_columns = ["promt_key","question","answer","ground_truth","answer_correctness","faithfulness","answer_relevancy","context_precision","context_recall"]
    result_df[selected_columns].to_csv(output_csv_path, index=False)
    print(result_df)
import pandas as pd
from datasets import Dataset
def get_eval_dataset():
    # Specify the path to your CSV file
    csv_file_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/india.csv"
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    eval_dataset = Dataset.from_pandas(df)  # Convert to Dataset for Ragas compatibility
    return eval_dataset, df  # Returning both for flexibility

# Load the evaluation dataset
eval_dataset, df = get_eval_dataset()

# Initialize results dictionary
results = {}
retriever=rag_get_retriever()
# Iterate through each prompt
for prompt_key, prompt_template in prompts.items():
    print(f"Testing Prompt: {prompt_key}...")
    pipeline = rag_pipeline(prompt_template,retriever)  # Initialize your RAG pipeline with the current prompt template
    if pipeline:
        # Iterate through each question in eval_dataset
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
                "promt_key": prompt_key,
                "question": question,
                "answer": result.get("answer", "Error"),
                "context": result.get("context", "Error"),
                "ground_truth": ground_truth,
            })
import pandas as pd
output_data = []
for prompt_key, prompt_results in results.items():
    output_data.extend(prompt_results)

results_df = pd.DataFrame(output_data)
if "context" in results_df.columns:
                results_df["contexts"] = results_df["context"].apply(
                    lambda x: eval(x) if isinstance(x, str) and x.startswith("[") else [x]
                )
# Save the DataFrame to a CSV file
output_csv_path = "/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/rag_results.csv"
results_df.to_csv(output_csv_path, index=False)

print(f"Results written to {output_csv_path}")

eval_dataset = Dataset.from_pandas(results_df)

evaluate1(eval_dataset)