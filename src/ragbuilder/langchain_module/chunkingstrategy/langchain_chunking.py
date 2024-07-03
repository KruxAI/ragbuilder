# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
# from langchain.text_splitter import MarkdownHeaderTextSplitter
# from langchain_text_splitters import HTMLHeaderTextSplitter
from ragbuilder.langchain_module.common import setup_logging,codeGen
import logging

setup_logging()
logger = logging.getLogger("ragbuilder")

def getChunkingStrategy(**kwargs):
    try:
        strategy = kwargs.get('chunking_kwargs').get('chunk_strategy')
        kwargs['chunk_size'] = kwargs.get('chunking_kwargs').get('chunk_size')
        kwargs['chunk_overlap'] = kwargs.get('chunking_kwargs').get('chunk_overlap')
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
        splitter_name=kwargs.get('splitter_name','splitter')
        code_string = f"""
{splitter_name} = RecursiveCharacterTextSplitter(chunk_size={kwargs['chunk_size']}, chunk_overlap={kwargs['chunk_overlap']})
splits={splitter_name}.split_documents(docs)"""
        import_string = f"""from langchain.text_splitter import RecursiveCharacterTextSplitter"""
        return {'code_string':code_string,'import_string':import_string}
    except KeyError as e:
        logger.error(f"Missing key in kwargs for RecursiveCharacterTextSplitter: {e}")
        raise

def getLangchainCharacterTextSplitter(**kwargs):
    try:
        logger.info("CharacterTextSplitter Invoked")
        splitter_name=kwargs.get('splitter_name','splitter')
        code_string = f"""
{splitter_name} = CharacterTextSplitter(chunk_size={kwargs['chunk_size']}, chunk_overlap={kwargs['chunk_overlap']})
splits={splitter_name}.split_documents(docs)"""
        import_string = f"""from langchain.text_splitter import CharacterTextSplitter"""
        return {'code_string':code_string,'import_string':import_string}
    except KeyError as e:
        logger.error(f"Missing key in kwargs for CharacterTextSplitter: {e}")
        raise

def getLangchainSemanticChunker(**kwargs):
    try:
        logger.info("SemanticChunker Invoked")
        splitter_name=kwargs.get('splitter_name','splitter')
        code_string = f"""
{splitter_name} = SemanticChunker(embedding, breakpoint_threshold_type='{kwargs.get('chunking_kwargs').get('breakpoint_threshold_type','percentile')}')
splits={splitter_name}.create_documents(docs[0].page_content)
"""
        import_string = f"""from langchain_experimental.text_splitter import SemanticChunker"""
        return {'code_string':code_string,'import_string':import_string}
    except KeyError as e:
        logger.error(f"Missing key in kwargs for SemanticChunker: {e}")
        raise

def getMarkdownHeaderTextSplitter(**kwargs):
    try:
        splitter_name=kwargs.get('splitter_name','splitter')
        logger.info("MarkdownHeaderTextSplitter Invoked")
        code_string = f"""
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")]
{splitter_name} = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
splits={splitter_name}.split_text(docs[0].page_content)"""
        import_string = f"""from langchain.text_splitter import MarkdownHeaderTextSplitter"""
        return {'code_string':code_string,'import_string':import_string}
    except KeyError as e:
        logger.error(f"Missing key in kwargs for MarkdownHeaderTextSplitter: {e}")
        raise
def getHTMLHeaderTextSplitter(**kwargs):
    try:
        logger.info("HTMLHeaderTextSplitter Invoked")
        splitter_name=kwargs.get('splitter_name','splitter')
        code_string = f"""
headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),]
from langchain_text_splitters import HTMLHeaderTextSplitter
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits={splitter_name}.split_text(docs[0].page_content)
"""
        import_string = f"""from langchain_text_splitters import HTMLHeaderTextSplitter"""
        return {'code_string':code_string,'import_string':import_string}
    except KeyError as e:
        logger.error(f"Missing key in kwargs for HTMLHeaderTextSplitter: {e}")
        raise
    
