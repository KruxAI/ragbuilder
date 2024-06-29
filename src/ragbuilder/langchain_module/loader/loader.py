import re
import os
import logging
from ragbuilder.langchain_module.common import setup_logging,codeGen
from langchain_community.document_loaders import DirectoryLoader, WebBaseLoader, UnstructuredFileLoader

# Setup logging
setup_logging()
logger = logging.getLogger("ragbuilder")

def classify_path(input_str):
    logger.info("classify_path Invoked")
    
    # Check if the input is a URL
    if re.match(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', input_str):
        return "url"
    
    # Check if the path is a directory
    elif os.path.isdir(input_str):
        return "directory"
    
    # Check if the path is a file
    elif os.path.isfile(input_str):
        return "file"
    
    # If the path exists but is neither a regular file nor a directory
    elif os.path.exists(input_str):
        return "Exists but is neither a regular file nor a directory"
    
    # If the path does not exist
    else:
        return "Neither"

def ragbuilder_loader(**kwargs):
    logger.info(f"ragbuilder_loader Invoked:{kwargs}")
    
    try:
        input_path = kwargs.get("input_path")
        return_code = kwargs.get("return_code",True)
        if not input_path:
            raise ValueError("Input path is missing or empty.")
        
        source = classify_path(input_path)
        logger.info(f"Source type identified: {source}")
        
        if source == "directory" and return_code:
            return ragbuilder_directory_loader(input_path)
        elif source == "url" and return_code:
            return ragbuilder_url_loader(input_path)
        elif source == "file" and return_code:
            return ragbuilder_file_loader(input_path)
        if source == "directory" and not return_code:
            return ragbuilder_directory_loader_exec(input_path)
        elif source == "url" and not return_code:
            return ragbuilder_url_loader_exec(input_path)
        elif source == "file" and not return_code:
            return ragbuilder_file_loader_exec(input_path)
        else:
            logger.error("Invalid input path type.")
            return None
        
    except Exception as e:
        logger.error(f"Error in ragbuilder_loader: {e}")
        return None

def ragbuilder_directory_loader(input_path):
    logger.info("ragbuilder_directory_loader Invoked")
    import_string = f"""
from langchain_community.document_loaders import DirectoryLoader"""

    code_string = f"""
loader = DirectoryLoader('{input_path}')
docs = loader.load()
"""
    return {'code_string':code_string,'import_string':import_string}

def ragbuilder_url_loader(input_path):
    logger.info("ragbuilder_url_loader Invoked")
    import_string = f"""
from langchain_community.document_loaders import WebBaseLoader"""

    code_string = f"""
loader = WebBaseLoader('{input_path}')
docs = loader.load()
"""
    return {'code_string':code_string,'import_string':import_string}
    
def ragbuilder_file_loader(input_path):
    logger.info("ragbuilder_file_loader Invoked")
    import_string = f"""
from langchain_community.document_loaders import UnstructuredFileLoader"""

    code_string = f"""
loader = UnstructuredFileLoader('{input_path}')
docs = loader.load()
"""
    return {'code_string':code_string,'import_string':import_string}


def ragbuilder_directory_loader_exec(input_path):
    logger.info("ragbuilder_directory_loader_exec Invoked")
    
    try:
        loader = DirectoryLoader(input_path)
        docs = loader.load()
        return docs
    
    except Exception as e:
        logger.error(f"Error in ragbuilder_directory_loader_exec: {e}")
        return None

def ragbuilder_url_loader_exec(input_path):
    logger.info("ragbuilder_url_loader_exec Invoked")
    
    try:
        loader = WebBaseLoader(input_path)
        docs = loader.load()
        return docs
    
    except Exception as e:
        logger.error(f"Error in ragbuilder_url_loader_exec: {e}")
        return None

def ragbuilder_file_loader_exec(input_path):
    logger.info("ragbuilder_file_loader_exec Invoked")
    
    try:
        loader = UnstructuredFileLoader(input_path)
        docs = loader.load()
        return docs
    
    except Exception as e:
        logger.error(f"Error in ragbuilder_file_loader_exec: {e}")
        return None


