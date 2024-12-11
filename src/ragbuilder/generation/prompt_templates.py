import yaml
import os
import requests
from ragbuilder.generation.config import PromptTemplate
import pandas as pd
def load_prompts(file_name: str = "rag_prompts.yaml", url: str= os.getenv("RAG_PROMPT_URL"),read_local: bool = False):
    """
    Load YAML prompts either from a local file or an online source.

    Args:
        file_name (str): Name of the YAML file. Defaults to "rag_prompts.yaml".
        read_local (bool): If True, read from a local file. Otherwise, fetch from an online URL.

    Returns:
        List[PromptTemplate]: A list of PromptTemplate objects.
    """
    yaml_content = None

    if read_local:
        # Attempt to read from the local file
        if os.path.exists(file_name):
            print(f"Loading prompts from local file: {file_name}")
            with open(file_name, 'r') as f:
                yaml_content = f.read()
        else:
            raise FileNotFoundError(f"Local file not found: {file_name}")
    else:
        # Attempt to fetch from an online source
        print(f"Fetching prompts from online file: {url}")
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an HTTP error for bad responses
            yaml_content = response.text
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to load prompts from URL {url}: {e}")

    # Parse the YAML content
    try:
        prompts_data = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML content: {e}")

    # Convert YAML entries into PromptTemplate objects
    prompts = [
        PromptTemplate(name=entry['name'], template=entry['template'])
        for entry in prompts_data
    ]
    return prompts