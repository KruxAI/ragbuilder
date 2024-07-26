from langchain_mistralai.chat_models import ChatMistralAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from ragbuilder.langchain_module.common import setup_logging,codeGen
import logging
setup_logging()
import os
logger = logging.getLogger("ragbuilder")
def getLLM(**kwargs):
    logger.info("LLM Invoked")
    retrieval_model=kwargs['retrieval_model']

    if kwargs['retrieval_model'] in ["gpt-3.5-turbo","gpt-4o"]:
        logger.info(f"LLM Code Gen Invoked: {retrieval_model}")
        import_string = f"""from langchain_openai import ChatOpenAI""" 
        code_string = f"""llm=ChatOpenAI(model='{retrieval_model}')"""
    elif kwargs['retrieval_model'] in ["mistral-small-latest","mistral-large-latest"] :
        logger.info(f"LLM Code Gen Invoked: {retrieval_model}")
        import_string = f"""from langchain_mistralai.chat_models import ChatMistralAI"""
        code_string = f"""llm=ChatMistralAI(api_key=os.environ.get('MISTRAL_API_KEY'),model='{retrieval_model}')""" 
    elif kwargs['retrieval_model'] in ["meta-llama/Meta-Llama-3.1-70B","meta-llama/Meta-Llama-3.1-8B","meta-llama/Meta-Llama-3.1-405B"]:
        logger.info(f"LLM Code Gen Invoked: {retrieval_model}")
        import_string = f"""from langchain_huggingface import HuggingFaceEndpoint"""
        code_string = f"""llm=HuggingFaceEndpoint(repo_id='{retrieval_model}',huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN'))"""
    else:
        raise ValueError(f"Invalid LLM: {kwargs['retrieval_model']}")
    return {'code_string':code_string,'import_string':import_string}

 