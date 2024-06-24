from dotenv import load_dotenv
import os
load_dotenv()  # take environment variables from .env

def test_invoke():
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    print(OPENAI_API_KEY)
    assert(OPENAI_API_KEY=="sk-kruxai-service-account-q8k4AHfwNjmrDBAaxLgLT3BlbkFJg2SHm2K2UTqydtZPGmip")

# def test_invoke_ashwin_github_fail():
#     url=["https://ashwinaravind.github.io/"]
#     assert(rag.naiverag(
#         input_path=url,
#         chunk_strategy=getLangchainRecursiveCharacterTextSplitter(1000, 200),
#         embedding_model=getOpenAIEmbedding(),
#         vectorDB="FAISS",
#         retreival_model=getOpenaiLLM(),
#         prompt_text="How many startups are there in India?",)['answer']=="There are 1.8 million startups in India.")

# def test_invoke_ashwin_github_pass():
#     url=["https://ashwinaravind.github.io/"]
#     assert(rag.naiverag(
#         input_path=url,
#         chunk_strategy=getLangchainRecursiveCharacterTextSplitter(1000, 200),
#         embedding_model=getOpenAIEmbedding(),
#         vectorDB="CHROMA",
#         retreival_model=getOpenaiLLM(),
#         prompt_text="How many startups are there in India?")['answer']=="There are 1.8 million startups in India.")
   
 
        