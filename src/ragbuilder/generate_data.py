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
from typing import Optional, Dict, Any, List
from ragas import RunConfig
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
# from ragbuilder.langchain_module.common import setup_logging
from ragbuilder.config.base import LogConfig
from langchain_core.documents import Document
from ragbuilder.core.utils import load_documents

logger = logging.getLogger("ragbuilder.generate_data")

class TestDatasetManager:
    """Manages test dataset generation and caching"""
    
    def __init__(self, log_config: Optional[LogConfig] = None):
        self._log_config = log_config
        self._cached_dataset: Optional[str] = None
        self._source_data: Optional[str] = None
        
    def get_or_generate_dataset(
        self, 
        source_data: str,
        test_dataset: Optional[str] = None,
        force_regenerate: bool = False,
        **generation_kwargs
    ) -> str:
        """
        Get existing test dataset or generate synthetic one
        
        Args:
            source_data: Source data path for synthetic generation if needed
            test_dataset: Optional path to existing test dataset
            force_regenerate: If True, regenerate synthetic data even if cached
            generation_kwargs: Optional kwargs for generate_data function
        
        Returns:
            Path to test dataset file
        """
        if test_dataset:
            if not Path(test_dataset).exists():
                raise FileNotFoundError(f"Test dataset not found: {test_dataset}")
            return test_dataset
            
        if not force_regenerate and self._cached_dataset and self._source_data == source_data:
            logger.info("Using cached synthetic test dataset")
            return self._cached_dataset
            
        logger.info("Generating synthetic test dataset...")
        try:
            # Use default models if not provided in kwargs
            generator_model = generation_kwargs.get('generator_model', 
                AzureChatOpenAI(model="gpt-4o", temperature=0.0))
            critic_model = generation_kwargs.get('critic_model',
                AzureChatOpenAI(model="gpt-4o", temperature=0.0))
            embedding_model = generation_kwargs.get('embedding_model',
                AzureOpenAIEmbeddings(model="text-embedding-3-large"))
            
            dataset_path = generate_data(
                src_data=source_data,
                generator_model=generator_model,
                critic_model=critic_model,
                embedding_model=embedding_model,
                test_size=generation_kwargs.get('test_size', 5)
            )
            
            self._cached_dataset = dataset_path
            self._source_data = source_data
            
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic test dataset: {e}")
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

