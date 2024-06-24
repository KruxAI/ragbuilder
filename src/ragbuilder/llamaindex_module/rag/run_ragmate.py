from llamaindex_module.rag.mergerag import mergerag
import logging
from langchain_module.common import setup_logging
setup_logging()
logger = logging.getLogger("ragbuilder")
logger.debug('This is a debug message from main.py')
logger.info('This is an info message from main.py')
def module_imported():
    return "print_module_imported"
# naiverag()