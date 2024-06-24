import logging
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, MarkdownNodeParser
from langchain_module.common import setup_logging

setup_logging()
logger = logging.getLogger("ragbuilder")

def getSplitter(**kwargs):
    chunk_strategy = kwargs.get('chunk_strategy')
    documents = kwargs.get('documents')

    if chunk_strategy == 'semantic':
        logger.info("SemanticSplitterNodeParser Invoked")
        splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=kwargs.get('breakpoint_percentile_threshold'),
            embed_model=kwargs.get('embed_model')
        )
    elif chunk_strategy == 'sentence':
        logger.info("SentenceSplitter Invoked")
        splitter = SentenceSplitter(
            chunk_size=kwargs.get('chunk_size'),
            chunk_overlap=kwargs.get('chunk_overlap')
        )
    elif chunk_strategy == 'markdown':
        logger.info("MarkdownNodeParser Invoked")
        splitter = MarkdownNodeParser(
            include_metadata=kwargs.get('include_metadata')
        )
    else:
        logger.error("Invalid chunk strategy provided")
        return None

    return splitter.get_nodes_from_documents(documents)
