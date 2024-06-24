# # import os
# # from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# # import generate_data


# # f_name = generate_data.generate_data(
# #     src_data='/Users/aravind/KruxAI/ragbuilder/langchain_for_ragas/'
# # )
# # print(f"Data generated and saved to {f_name}")


# import asyncio

# # Define an asynchronous function
# async def say_hello():
#     print("Hello...")
#     # Simulate an I/O-bound operation using asyncio.sleep
#     await asyncio.sleep(20)
#     print("...world!")

# # Define another asynchronous function
# async def say_goodbye():
#     print("Goodbye...")
#     # Simulate an I/O-bound operation using asyncio.sleep
#     await asyncio.sleep(10)
#     print("...everyone!")

# # Define a main function to run the asynchronous functions
# async def main():
#     # Create tasks for the asynchronous functions
#     task1 = asyncio.create_task(say_hello())
#     task2 = asyncio.create_task(say_goodbye())
    
#     # Wait for both tasks to complete
#     await task1
#     await task2

# # Run the main function
# asyncio.run(main())
import os
from dotenv import load_dotenv
# Load environment variables from the .env file (if present)
load_dotenv()
# from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
# embeddings = HuggingFaceEndpointEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
text = "This is a test document."
# query_result = embeddings.embed_query(text)
# print(query_result[:3])

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
      model_name="sentence-transformers/all-MiniLM-l6-v2"
)
query_result = embeddings.embed_query(text)
print('##################')
print('HF embedding',query_result[:3])

import os
from dotenv import load_dotenv
# Load environment variables from the .env file (if present)
load_dotenv()
# Set up the Hugging Face Hub API token
 

from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Define the repository ID for the Gemma 2b model
repo_id = "google/gemma-7b"

# Set up a Hugging Face Endpoint for Gemma 2b model
llm = HuggingFaceEndpoint(
    repo_id=repo_id, max_length=1024, temperature=0.1
)

question = "Who won the FIFA World Cup in the year 1994?"

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(prompt=prompt, llm=llm)
print(llm_chain.invoke(question))