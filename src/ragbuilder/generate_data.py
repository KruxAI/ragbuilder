###############################################################################
# Synthetic test data generator                                               #
#                                                                             #
# Description:                                                                #
# Synthetic test data generator module for any source data.                   #
# - Takes as input the source dataset (type files/ directory) from which to   #
#   generate "question" and "ground_truth"                                    #
# - Performs the following steps:                                             #
#      1. Loads source documents using appropriate document loader            #
#      2. Initializes test generator with specified LLM and embedding models  #
#      3. Generates test cases based on configured distribution:              #
#         - Simple questions (direct answers from text)                       #
#         - Reasoning questions (require inference)                           #
#         - Multi-context questions (need multiple document sections)         #
#      4. Validates generated test cases using critic model                   #
#      5. Exports validated test dataset to CSV format                        #
###############################################################################

# imports
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from ragas import RunConfig
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from ragbuilder.langchain_module.common import setup_logging
from ragbuilder.config.base import LogConfig, EvalDataGenerationConfig
from langchain_core.documents import Document
from ragbuilder.core.utils import load_documents
from ragbuilder.core.telemetry import telemetry
from ragbuilder.core.config_store import ConfigStore
import sqlite3
import hashlib
from urllib.parse import urlparse
logger = logging.getLogger("ragbuilder.generate_data")

class SQLiteCache:
    """SQLite-based cache for synthetic test datasets with in-memory lookup"""
    
    def __init__(self, db_path: str = "eval.db"):
        self.db_path = db_path
        self._cache = {}  # In-memory cache
        self._init_db()
        self._load_cache()
    
    def _init_db(self) -> None:
        """Initialize SQLite database and create table if needed"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS synthetic_data_hashmap (
                    hash TEXT PRIMARY KEY,
                    test_data_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def _load_cache(self) -> None:
        """Load entire cache table into memory"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('SELECT hash, test_data_path FROM synthetic_data_hashmap')
            self._cache = {row[0]: row[1] for row in cursor.fetchall()}
            logger.debug(f"Loaded {len(self._cache)} entries from cache")
    
    def get(self, key: str) -> Optional[str]:
        """Get test dataset path from cache"""
        return self._cache.get(key)
    
    def set(self, key: str, value: str) -> None:
        """Set test dataset path in both memory and SQLite"""
        self._cache[key] = value
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO synthetic_data_hashmap (hash, test_data_path) VALUES (?, ?)',
                (key, value)
            )
            conn.commit()

def get_hash(source_data: str, use_sampling: bool = False) -> str:
    """Generate hash for source data considering sampling"""    
    src_type = 'url' if urlparse(source_data).scheme in ['http', 'https'] else (
            'dir' if os.path.isdir(source_data) else (
                'file' if os.path.isfile(source_data) else 'unknown'
            )
        )
    prefix = "sampled_" if use_sampling else ""
    
    if src_type == "dir":
        return _get_hash_dir(source_data, prefix)
    elif src_type == "url":
        return _get_hash_url(source_data, prefix)
    elif src_type == "file":
        return _get_hash_file(source_data, prefix)
    else:
        logger.error(f"Invalid input path type {source_data}")
        return None

def _get_hash_file(source_data: str, prefix: str = "") -> str:
    """Generate hash for a file"""
    md5_hash = hashlib.md5(prefix.encode())
    with open(source_data, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def _get_hash_url(url: str, prefix: str = "") -> str:
    """Generate hash for a URL's content"""
    try:
        import requests
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, allow_redirects=True)
        response.raise_for_status()
        md5_hash = hashlib.md5(prefix.encode())
        md5_hash.update(response.content)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error fetching URL: {e}")
        return None

def _get_hash_dir(dir_path: str, prefix: str = "") -> str:
    """Generate hash for a directory's contents"""
    if not os.path.isdir(dir_path):
        logger.error(f"{dir_path} is not a valid directory.")
        return None
    md5_hash = hashlib.md5(prefix.encode())
    try:
        for root, _, files in os.walk(dir_path):
            for file_name in sorted(files):
                file_path = os.path.join(root, file_name)
                with open(file_path, 'rb') as file:
                    while chunk := file.read(8192):
                        md5_hash.update(chunk)
        return md5_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error hashing directory content: {e}")
        return None

class TestDatasetManager:
    """Manages test dataset generation and caching"""
    
    def __init__(self, log_config: Optional[LogConfig] = None, db_path: str = "eval.db"):
        self._log_config = log_config
        self.cache = SQLiteCache(db_path)
        
    def get_or_generate_eval_dataset(
        self, 
        source_data: str,
        test_dataset: Optional[str] = None,
        eval_data_generation_config: Optional[EvalDataGenerationConfig] = None,
        sampling_rate: Optional[float] = None,
        **generation_kwargs
    ) -> str:
        """
        Get existing test dataset or generate synthetic one
        
        Args:
            source_data: Source data path for synthetic generation if needed
            test_dataset: Optional path to existing test dataset
            eval_data_generation_config: Eval data generation configuration
            sampling_rate: Optional sampling rate for source data
            generation_kwargs: Optional kwargs for generate_data function
        
        Returns:
            Path to test dataset file
        """
        if test_dataset:
            if not Path(test_dataset).exists():
                raise FileNotFoundError(f"Test dataset not found: {test_dataset}")
            return test_dataset
            
        try:
            # Generate cache key based on source and sampling
            cache_key = get_hash(source_data, use_sampling=sampling_rate is not None and sampling_rate < 1)
            if not cache_key:
                logger.warning("Failed to generate hash, proceeding without caching")
            else:
                # Check cache
                if cached_path := self.cache.get(cache_key):
                    logger.info(f"Found cached test dataset: {cached_path}")
                    if os.path.exists(cached_path):
                        return cached_path
                    logger.warning(f"Cached path {cached_path} not found, regenerating...")
            
            # Use default models if not provided
            generator_model = (eval_data_generation_config.generator_model if eval_data_generation_config and eval_data_generation_config.generator_model
                            else ConfigStore.get_default_llm().llm)
            critic_model = (eval_data_generation_config.critic_model if eval_data_generation_config and eval_data_generation_config.critic_model
                          else ConfigStore.get_default_llm().llm)
            embedding_model = (eval_data_generation_config.embedding_model if eval_data_generation_config and eval_data_generation_config.embedding_model
                            else ConfigStore.get_default_embeddings().embeddings)
            
            # Extract model info for telemetry
            generator_model_name = getattr(generator_model, 'model', None) or getattr(generator_model, 'model_name', '')
            generator_model_info = f"{generator_model.__class__.__name__}:{generator_model_name}" if generator_model_name else str(generator_model.__class__.__name__)
            
            critic_model_name = getattr(critic_model, 'model', None) or getattr(critic_model, 'model_name', '')
            critic_model_info = f"{critic_model.__class__.__name__}:{critic_model_name}" if critic_model_name else str(critic_model.__class__.__name__)
            
            embedding_model_name = getattr(embedding_model, 'model', None) or getattr(embedding_model, 'model_name', '')
            embedding_model_info = f"{embedding_model.__class__.__name__}:{embedding_model_name}" if embedding_model_name else str(embedding_model.__class__.__name__)

            with telemetry.eval_datagen_span(
                test_size=eval_data_generation_config.test_size if eval_data_generation_config else 5,
                distribution=eval_data_generation_config.distribution if eval_data_generation_config else None,
                generator_model=generator_model_info,
                critic_model=critic_model_info,
                embedding_model=embedding_model_info
            ) as _:
                dataset_path = generate_data(
                    src_data=source_data,
                    generator_model=generator_model,
                    critic_model=critic_model,
                    embedding_model=embedding_model,
                    test_size=eval_data_generation_config.test_size if eval_data_generation_config else 5,
                    distribution=eval_data_generation_config.distribution if eval_data_generation_config else None,
                    run_config=eval_data_generation_config.run_config if eval_data_generation_config else None,
                    **generation_kwargs
                )
                
                # Cache the result if we have a valid cache key
                if cache_key and dataset_path:
                    self.cache.set(cache_key, dataset_path)
                    logger.info(f"Cached new test dataset: {dataset_path}")
                
                return dataset_path
            
        except Exception as e:
            logger.error(f"Error in test dataset management: {str(e)}")
            telemetry.track_error("eval_data_generation", e, context={
                "sampling_rate": sampling_rate
            })
            raise

def load_src(src_data: str) -> List[Document]:
    """Load source documents for test generation"""
    logger.info("Loading docs...")
    try:
        docs = load_documents(src_data)
        # Add the filename attribute in metadata for Ragas
        for doc in docs:
            doc.metadata['filename'] = doc.metadata.get('source', '')
        return docs
        
    except Exception as e:
        logger.error(f"Error loading docs for synthetic test data generation: {e}")
        raise

def generate_data(
        src_data: str,
        generator_model: Any,
        critic_model: Any,
        embedding_model: Any,
        test_size: int = 5,
        distribution: Dict[str, float] = None,
        run_config: Optional[RunConfig] = None
) -> str:
    """Generate synthetic test dataset
    
    Args:
        src_data: Path to source data
        generator_model: LLM for question generation
        critic_model: LLM for answer validation
        embedding_model: Model for embeddings
        test_size: Number of test cases to generate
        distribution: Distribution of question types
        run_config: RAGAS run configuration
        
    Returns:
        Path to generated test dataset
    """
    # Default distribution if none provided
    if distribution is None:
        distribution = {'simple': 0.5, 'reasoning': 0.1, 'multi_context': 0.4}
    
    # Convert string keys to evolution types
    dist = {
        simple: distribution.get('simple', 0.5),
        reasoning: distribution.get('reasoning', 0.1),
        multi_context: distribution.get('multi_context', 0.4)
    }
    
    # Use default run config if none provided
    if run_config is None:
        run_config = RunConfig(
            timeout=1000,
            max_workers=1,
            max_wait=900,
            max_retries=5
        )
    
    docs = load_src(src_data)
    if not docs:
        raise ValueError("No documents loaded for test generation")

    generator = TestsetGenerator.from_langchain(
        generator_model,
        critic_model,
        embedding_model
    )

    logger.info(f"Generating {test_size} test cases...")
    testset = generator.generate_with_langchain_docs(
        documents=docs, 
        test_size=test_size, 
        distributions=dist, 
        is_async=True, 
        raise_exceptions=True, 
        run_config=run_config,
    )
    
    logger.info("Test generation complete")
    ts = datetime.now(timezone.utc).timestamp()
    f_name = f'rag_test_data_{ts}.csv'
    
    logger.info(f"Writing to {f_name}")
    test_df = testset.to_pandas()
    test_df.to_csv(f_name)
    
    return f_name

