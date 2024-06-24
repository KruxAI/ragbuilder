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
    if kwargs['retrieval_model'] in ["gpt-3.5-turbo","gpt-4o"]:
        return getOpenaiLLM(kwargs['retrieval_model'])
    elif kwargs['retrieval_model'] in ["mistral-7b", "mistral-small-latest","mistral-large-latest"] :
        return getMistralLLM(kwargs['retrieval_model'])
    else:
        raise ValueError(f"Invalid LLM: {kwargs['retrieval_model']}")
def getOpenaiLLM(retrieval_model):
    logger.info("model={retrieval_model} Invoked")
    llm = ChatOpenAI(model=retrieval_model)
    return llm
def getMistralLLM(retrieval_model):
     logger.info("model={retrieval_model} Invoked")
     return ChatMistralAI(
        api_key=os.environ.get("MISTRAL_API_KEY"),
        model=retrieval_model)
def getHuggingFaceLLM(retrieval_model):
    logger.info("model={retrieval_model} Invoked")
    return HuggingFaceEndpoint(
    repo_id=retrieval_model,
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),)
