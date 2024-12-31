from ragbuilder.config.generation import PromptTemplate
import os
import logging
import yaml
import requests
from typing import List, Dict
from ragbuilder.core import DBLoggerCallback, DocumentStore, ConfigStore, setup_rich_logging, console
logger = logging.getLogger("ragbuilder.prompt_templates")

class PromptTemplate:
    def __init__(self, name: str, template: str):
        self.name = name
        self.template = template

def load_prompts(
    prompt_template_path: str = None,
    local_prompt_template_path: str = None,
    read_local_only: bool = False
) -> List[PromptTemplate]:
    """
    Load YAML prompts from a local file and/or an online source.

    Args:
        prompt_template_path (str): URL of the YAML file.
        local_prompt_template_path (str): Path to the local YAML file.
        read_local_only (bool): If True, only attempt to load from the local file.

    Returns:
        List[PromptTemplate]: A list of PromptTemplate objects.

    Raises:
        FileNotFoundError: If the local file is required but not found.
        RuntimeError: If the online file cannot be fetched.
        ValueError: If the YAML content is malformed.
    """
    yaml_content_local = None
    yaml_content_online = None
    prompt_template_path=os.getenv('RAG_PROMPT_URL', 'https://raw.githubusercontent.com/KruxAI/ragbuilder/refs/heads/main/rag_prompts.yml')
    # Handle local-only behavior
    if read_local_only:
        if not local_prompt_template_path:
            raise ValueError("`local_prompt_template_path` must be provided when `read_local_only` is True.")
        if not os.path.exists(local_prompt_template_path):
            raise FileNotFoundError(f"Local file not found: {local_prompt_template_path}")
        logger.debug(f"Loading prompts from local file: {local_prompt_template_path}")
        with open(local_prompt_template_path, 'r') as f:
            yaml_content_local = f.read()
        logger.debug(f"Loading Prompts from {local_prompt_template_path}")
    # Handle loading from the local file if specified
    elif local_prompt_template_path and os.path.exists(local_prompt_template_path):
        logger.debug(f"Loading prompts from local file: {local_prompt_template_path}")
        with open(local_prompt_template_path, 'r') as f:
            yaml_content_local = f.read()

    # Handle loading from the online source if specified
    if not read_local_only and prompt_template_path:
        logger.debug(f"Fetching prompts from online file: {prompt_template_path}")
        try:
            response = requests.get(prompt_template_path)
            response.raise_for_status()
            yaml_content_online = response.text
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to load prompts from URL {prompt_template_path}: {e}")

    # Parse YAML content
    prompts_local = []
    prompts_online = []

    if yaml_content_local:
        try:
            prompts_local = yaml.safe_load(yaml_content_local)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse local YAML content: {e}")

    if yaml_content_online:
        try:
            prompts_online = yaml.safe_load(yaml_content_online)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse online YAML content: {e}")

    # Combine prompts and validate
    all_prompts = {}
    for entry in prompts_local + prompts_online:
        if 'name' not in entry or 'template' not in entry:
            raise ValueError(f"Invalid prompt entry found: {entry}")
        all_prompts[entry['name']] = PromptTemplate(name=entry['name'], template=entry['template'])
    return list(all_prompts.items())

