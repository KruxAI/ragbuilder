
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import  SimpleDirectoryReader
from langchain_module.common import setup_logging
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def getLoader(**kwargs):
    if kwargs["source"] == "directory":
        logger.info("SimpleDirectoryReader Invoked")
        return SimpleDirectoryReader("InputFiles").load_data()
    elif kwargs["source"] == "url":
        logger.info("SimpleWebPageReader Invoked")
        return SimpleWebPageReader(html_to_text=True).load_data(kwargs["input_path"])
    else:
        logger.info("Invalid Input")
        return None