from langchain_module.common import setup_logging
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.mistralai import MistralAIEmbedding
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def getEmbedding(**kwargs):
    logger.info("LLM Invoked")
    if kwargs['embedding_model'] == "openai":
        return OpenAIEmbedding()
    elif kwargs['embedding_model'] == "mistral-embed":
        return  MistralAIEmbedding(model_name="mistral-embed")
    else:
        raise ValueError(f"Invalid LLM: {kwargs['embedding_model']}")