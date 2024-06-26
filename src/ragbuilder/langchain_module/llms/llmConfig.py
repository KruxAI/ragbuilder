from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from ragbuilder.langchain_module.common import setup_logging
import logging
setup_logging()
import os
logger = logging.getLogger("ragbuilder")
def getLLM(**kwargs):
    logger.info("LLM Invoked")
    return_code= kwargs.get('return_code', False)
    if kwargs['retrieval_model'] in ["gpt-3.5-turbo","gpt-4o"]:
        return getOpenaiLLM(kwargs['retrieval_model'],return_code)
    elif kwargs['retrieval_model'] in ["mistral-7b", "mistral-small-latest","mistral-large-latest"] :
        logger.info("LLM Codgen Invoked")
        return getMistralLLM(kwargs['retrieval_model'],return_code)
    else:
        raise ValueError(f"Invalid LLM: {kwargs['retrieval_model']}")
    
def getOpenaiLLM(retrieval_model, return_code):
    logger.info(f"model={retrieval_model} Invoked")
    if not return_code:
        llm = ChatOpenAI(model=retrieval_model)
    else:
        logger.info("getOpenaiLLM Codgen Invoked")
        llm = f"""llm=ChatOpenAI(model='{retrieval_model}')"""  # Return the code as a string
    return llm

def getMistralLLM(retrieval_model,return_code):
    logger.info(f"model={retrieval_model} Invoked")
    if return_code is None:
        llm = ChatMistralAI(
            api_key=os.environ.get("MISTRAL_API_KEY"),
            model=retrieval_model)
    else:
        logger.info("getMistralLLM Codgen Invoked")
        llm = f"""
    llm=ChatMistralAI(api_key=os.environ.get('MISTRAL_API_KEY'),model=retrieval_model)"""
    return llm
def getHuggingFaceLLM(retrieval_model,return_code):
    logger.info(f"model={retrieval_model} Invoked")
    if return_code is None:
        llm = HuggingFaceEndpoint(
            repo_id=retrieval_model,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),)
    else:
        llm = "HuggingFaceEndpoint(repo_id=retrieval_model,huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'))"
    return llm
