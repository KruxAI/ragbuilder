from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter
from ragbuilder.langchain_module.common import setup_logging
import logging

setup_logging()
logger = logging.getLogger("ragbuilder")

def getChunkingStrategy(**kwargs):
    try:
        strategy = kwargs.get('chunk_strategy')
        if not strategy:
            raise ValueError("Missing chunking strategy in kwargs")

        if strategy == "RecursiveCharacterTextSplitter":
            return getLangchainRecursiveCharacterTextSplitter(**kwargs)
        elif strategy == "CharacterTextSplitter":
            return getLangchainCharacterTextSplitter(**kwargs)
        elif strategy == "SemanticChunker":
            return getLangchainSemanticChunker(**kwargs)
        elif strategy == "MarkdownHeaderTextSplitter":
            return getMarkdownHeaderTextSplitter(**kwargs)
        elif strategy == "HTMLHeaderTextSplitter":
            return getHTMLHeaderTextSplitter(**kwargs)
        else:
            raise ValueError(f"Invalid chunking strategy: {strategy}")
    except Exception as e:
        logger.error(f"Error in getChunkingStrategy: {e}")
        raise

def getLangchainRecursiveCharacterTextSplitter(**kwargs):
    try:
        logger.info("RecursiveCharacterTextSplitter Invoked")
        splitter = RecursiveCharacterTextSplitter(chunk_size=kwargs['chunk_size'], chunk_overlap=kwargs['chunk_overlap'])
        retriever_type = kwargs.get('retriever_type')
        if retriever_type in ["parentDocFullDoc", "parentDocLargeChunk" ]:
            return splitter
        else:
            return splitter.split_documents(kwargs['docs'])
    except KeyError as e:
        logger.error(f"Missing key in kwargs for RecursiveCharacterTextSplitter: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in RecursiveCharacterTextSplitter: {e}")
        raise

def getLangchainCharacterTextSplitter(**kwargs):
    try:
        logger.info("CharacterTextSplitter Invoked")
        splitter = CharacterTextSplitter(chunk_size=kwargs['chunk_size'], chunk_overlap=kwargs['chunk_overlap'])
        retriever_type = kwargs.get('retriever_type')
        if retriever_type in ["parentDocFullDoc", "parentDocLargeChunk" ]:
            return splitter
        else:
            return splitter.split_documents(kwargs['docs'])
    except KeyError as e:
        logger.error(f"Missing key in kwargs for CharacterTextSplitter: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in CharacterTextSplitter: {e}")
        raise

def getLangchainSemanticChunker(**kwargs):
    try:
        logger.info("SemanticChunker Invoked")
        splitter = SemanticChunker(kwargs['embedding_model'], breakpoint_threshold_type=kwargs['breakpoint_threshold_type'])
        retriever_type = kwargs.get('retriever_type')
        if retriever_type in ["parentDocFullDoc", "parentDocLargeChunk" ]:
            return splitter
        else:
            return splitter.create_documents(kwargs['docs'][0].page_content)
    except KeyError as e:
        logger.error(f"Missing key in kwargs for SemanticChunker: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in SemanticChunker: {e}")
        raise

def getMarkdownHeaderTextSplitter(**kwargs):
    try:
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
        logger.info("MarkdownHeaderTextSplitter Invoked")
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        retriever_type = kwargs.get('retriever_type')
        if retriever_type in ["parentDocFullDoc", "parentDocLargeChunk" ]:
            return splitter
        else:
            return splitter.split_text(kwargs['docs'][0].page_content)
    except KeyError as e:
        logger.error(f"Missing key in kwargs for MarkdownHeaderTextSplitter: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in MarkdownHeaderTextSplitter: {e}")
        raise

def getHTMLHeaderTextSplitter(**kwargs):
    try:
        logger.info("HTMLHeaderTextSplitter Invoked")
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),]
        splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        retriever_type = kwargs.get('retriever_type')
        if retriever_type in ["parentDocFullDoc", "parentDocLargeChunk" ]:
            return splitter
        else:
            return splitter.split_text(kwargs['docs'][0].page_content)
    except KeyError as e:
        logger.error(f"Missing key in kwargs for HTMLHeaderTextSplitter: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in HTMLHeaderTextSplitter: {e}")
        raise
