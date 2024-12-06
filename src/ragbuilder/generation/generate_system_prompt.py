# gen_prompt_template.py.py
### Generates System Prompt Template for Retrieval-Augmented Generation
# Runs the retriever and get the result
# Get the eval dataset
# Reads prompt template list
# Loops through each prompt template and for each qs in eval dataset runs the prompt and get the answer and evals the answers
# Return the top performing prompt template

#############
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
from ragbuilder.generation.evaluation import RAGASEvaluator
############################################################################################################
# def sample_retriever():
#     print("rag_get_retriever initiated")
#     try:
#         def format_docs(docs):
#             return "\n".join(doc.page_content for doc in docs)

#         # LLM setup
#         llm = AzureChatOpenAI(model="gpt-4o-mini")

#         # Document loader
#         loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#         docs = loader.load()

#         # Embedding model
#         embedding = AzureOpenAIEmbeddings(model="text-embedding-3-large")

#         # Text splitting and embedding storage
#         splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
#         splits = splitter.split_documents(docs)

#         # Initialize Chroma database
#         c = Chroma.from_documents(
#             documents=splits,
#             embedding=embedding,
#             collection_name="testindex-ragbuilder-retreiver",
#             client_settings=chromadb.config.Settings(allow_reset=True),
#         )

#         # Retriever setup
#         retriever = c.as_retriever(search_type="similarity", search_kwargs={"k": 5})
#         ensemble_retriever = EnsembleRetriever(retrievers=[retriever])
#         print("rag_get_retriever completed")
#         return ensemble_retriever
#     except Exception as e:
#         import traceback
#         print(f"An error occurred: {e}")
#         traceback.print_exc()
#         return None
############################################################################################################
import pandas as pd
from datasets import Dataset
from ragbuilder.generation.prompt_templates import load_prompts
from ragbuilder.generation.utils import get_eval_dataset
class Generator:
    def __init__(self,eval_dataset_path=None,retriever=None, evaluator=None, prompt_template_path=None,read_local=False,llm=None):
        """
        Initialize the SystemPromptGenerator instance.

        Args:
        - eval_dataset_path (str): Path to the evaluation dataset CSV file.
        - retriever: The retriever object.
        - evaluator: The evaluator object.
        - prompt_template_path (str): Optional file name. If provided, the prompt templates will be read from this file also in addition to the default prompt templates from github.
        - read_local (bool): Flag to determine whether to read locally. Default is False. If True, the prompt templates will be read from the local file.
        """
        self.prompt_templates = load_prompts(prompt_template_path)
        self.eval_dataset = get_eval_dataset(eval_dataset_path)
        self.retriever = retriever
        self.evaluator=evaluator
        self.llm=llm
    def setup_retrieval_qa(self,prompt_template,retriever,llm):
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

    def run_prompt_templates(self):
        print("test_prompt initiated")
        results = {}

        for prompt in  self.prompt_templates:
            print(f"Testing Prompt: {prompt.name}...")
            pipeline = self.setup_retrieval_qa(prompt.template, self.retriever(),self.llm)  # Initialize your RAG pipeline with the current prompt template
            
            if pipeline:
                # Iterate through each question in eval_dataset
                for entry in self.eval_dataset:
                    question = entry.get("question", "")  # Extract the question from the dataset
                    ground_truth = entry.get("ground_truth", "")  # Extract the ground truth answer from the dataset
                    if not question:
                        continue  # Skip entries without a valid question

                    # Invoke pipeline for the question
                    result = pipeline.invoke(question)

                    # Store results
                    if prompt.name not in results:
                        results[prompt.name] = []
                    results[prompt.name].append({
                        "prompt_key": prompt.name,
                        "prompt": prompt.template,
                        "question": question,
                        "answer": result.get("answer", "Error"),
                        "context": result.get("context", "Error"),
                        "ground_truth": ground_truth,
                    })
                    break  # Remove this `break` if you want to test all questions in the dataset
            break  # Remove this `break` if you want to test all prompts

        # Convert the results to a list of dictionaries
        output_data = []
        for prompt_key, prompt_results in results.items():
            output_data.extend(prompt_results)

        # Convert to a Dataset directly
        from datasets import Dataset
        results_dataset = Dataset.from_list(output_data)

        # Optionally clean up or format the dataset
        if "context" in results_dataset.column_names:
            results_dataset = results_dataset.map(
                lambda x: {
                    **x,
                    "contexts": eval(x["context"]) if isinstance(x["context"], str) and x["context"].startswith("[") else [x["context"]],
                }
            )

        print("test_prompt completed")
        return results_dataset
    def calculate_metrics(self, result):
        # Convert the results to a pandas DataFrame
        results_df = result.to_pandas()
        
        # Calculate average correctness per prompt key
        average_correctness = results_df.groupby(['prompt_key','prompt'])['answer_correctness'].mean().reset_index()
        average_correctness.columns = ['prompt_key', "prompt", 'average_correctness']
        
        # Save the average correctness to a CSV file
        average_correctness.to_csv('rag_average_correctness.csv', index=False)
        print("The average correctness results have been saved to 'rag_average_correctness.csv'")
        
        # Find the row with the highest average correctness
        best_prompt_row = average_correctness.loc[average_correctness['average_correctness'].idxmax()]
        
        # Extract prompt_key, prompt, and average_correctness
        prompt_key = best_prompt_row['prompt_key']
        prompt = best_prompt_row['prompt']
        max_average_correctness = best_prompt_row['average_correctness']
        
        return prompt_key, prompt, max_average_correctness
    
    def optimize(self):
        prompt_results=self.run_prompt_templates()
        results=self.evaluator.evaluate(prompt_results, llm= AzureChatOpenAI(model="gpt-4o-mini"), embeddings=AzureOpenAIEmbeddings(model="text-embedding-3-large"))
        best_prompt_key, best_prompt, best_average_correctness=self.calculate_metrics(results)
        return best_prompt, best_average_correctness