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
#      3. Generates test cases based on configured distribution using         #
#         knowledge graph approach:                                           #
#         - Creates knowledge graph from documents                            #
#         - Applies transforms to enrich knowledge graph                      #
#         - Generates questions using configured synthesizers                 #
#      4. Exports generated test dataset to CSV format                        #
###############################################################################

# imports
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from ragas import RunConfig
from ragas.testset import TestsetGenerator, Testset
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
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
        self._kg_cache = {}  # In-memory cache for knowledge graphs
        self._init_db()
        self._load_cache()
    
    def _init_db(self) -> None:
        """Initialize SQLite database and create table if needed"""
        with sqlite3.connect(self.db_path) as conn:
            # Create test dataset table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS synthetic_data_hashmap (
                    hash TEXT PRIMARY KEY,
                    test_data_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create knowledge graph table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS knowledge_graph_hashmap (
                    hash TEXT PRIMARY KEY,
                    kg_path TEXT NOT NULL,
                    node_count INTEGER,
                    relationship_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    
    def _load_cache(self) -> None:
        """Load entire cache table into memory"""
        with sqlite3.connect(self.db_path) as conn:
            # Load test dataset cache
            cursor = conn.execute('SELECT hash, test_data_path FROM synthetic_data_hashmap')
            self._cache = {row[0]: row[1] for row in cursor.fetchall()}
            logger.debug(f"Loaded {len(self._cache)} test dataset entries from cache")
            
            # Load knowledge graph cache
            cursor = conn.execute('SELECT hash, kg_path FROM knowledge_graph_hashmap')
            self._kg_cache = {row[0]: row[1] for row in cursor.fetchall()}
            logger.debug(f"Loaded {len(self._kg_cache)} knowledge graph entries from cache")
    
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
    
    def get_kg(self, key: str) -> Optional[str]:
        """Get knowledge graph path from cache"""
        return self._kg_cache.get(key)
    
    def set_kg(self, key: str, kg_path: str, node_count: int, relationship_count: int) -> None:
        """Set knowledge graph path in both memory and SQLite"""
        self._kg_cache[key] = kg_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO knowledge_graph_hashmap (hash, kg_path, node_count, relationship_count) VALUES (?, ?, ?, ?)',
                (key, kg_path, node_count, relationship_count)
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
                          else None)  # Critic model is no longer used in the new approach
            embedding_model = (eval_data_generation_config.embedding_model if eval_data_generation_config and eval_data_generation_config.embedding_model
                            else ConfigStore.get_default_embeddings().embeddings)
            
            # Extract model info for telemetry
            generator_model_name = getattr(generator_model, 'model', None) or getattr(generator_model, 'model_name', '')
            generator_model_info = f"{generator_model.__class__.__name__}:{generator_model_name}" if generator_model_name else str(generator_model.__class__.__name__)
            
            embedding_model_name = getattr(embedding_model, 'model', None) or getattr(embedding_model, 'model_name', '')
            embedding_model_info = f"{embedding_model.__class__.__name__}:{embedding_model_name}" if embedding_model_name else str(embedding_model.__class__.__name__)

            with telemetry.eval_datagen_span(
                test_size=eval_data_generation_config.test_size if eval_data_generation_config else 5,
                distribution=eval_data_generation_config.distribution if eval_data_generation_config else None,
                generator_model=generator_model_info,
                critic_model="n/a",  # Critic model is no longer used
                embedding_model=embedding_model_info
            ) as _:
                # Check if we have a cached knowledge graph for this source data
                cached_kg_path = self.cache.get_kg(cache_key) if cache_key else None
                
                dataset_path = generate_data(
                    src_data=source_data,
                    generator_model=generator_model,
                    embedding_model=embedding_model,
                    test_size=eval_data_generation_config.test_size if eval_data_generation_config else 5,
                    distribution=eval_data_generation_config.distribution if eval_data_generation_config else None,
                    run_config=eval_data_generation_config.run_config if eval_data_generation_config else None,
                    cache=self.cache,
                    cache_key=cache_key,
                    cached_kg_path=cached_kg_path,
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

def default_transforms(
    documents: List[Document],
    llm: Any,
    embedding_model: Any,
) -> Any:
    """
    Creates and returns a default set of transforms for processing a knowledge graph.

    This function defines a series of transformation steps to be applied to a
    knowledge graph, including extracting summaries, keyphrases, titles,
    headlines, and embeddings, as well as building similarity relationships
    between nodes.



    Returns
    -------
    Transforms
        A list of transformation steps to be applied to the knowledge graph.

    """
    from ragas.testset.transforms.extractors import (
        EmbeddingExtractor,
        HeadlinesExtractor,
        SummaryExtractor,
    )
    from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
    from ragas.testset.transforms.filters import CustomNodeFilter
    from ragas.testset.transforms.relationship_builders import (
        CosineSimilarityBuilder,
        OverlapScoreBuilder,
    )
    from ragas.testset.transforms.splitters import HeadlineSplitter
    from ragas.utils import num_tokens_from_string
    from ragas.testset.transforms.engine import Parallel

    def count_doc_length_bins(documents, bin_ranges):
        data = [num_tokens_from_string(doc.page_content) for doc in documents]
        bins = {f"{start}-{end}": 0 for start, end in bin_ranges}

        for num in data:
            for start, end in bin_ranges:
                if start <= num <= end:
                    bins[f"{start}-{end}"] += 1
                    break  # Move to the next number once it’s placed in a bin

        return bins

    def filter_doc_with_num_tokens(node, min_num_tokens=500):
        return (
            node.type == NodeType.DOCUMENT
            and num_tokens_from_string(node.properties["page_content"]) > min_num_tokens
        )

    def filter_docs(node):
        return node.type == NodeType.DOCUMENT

    def filter_chunks(node):
        return node.type == NodeType.CHUNK

    bin_ranges = [(0, 100), (101, 500), (501, 10000000)]
    result = count_doc_length_bins(documents, bin_ranges)
    result = {k: v / len(documents) for k, v in result.items()}

    transforms = []

    if result["501-10000000"] >= 0.25:
        headline_extractor = HeadlinesExtractor(
            llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node)
        )
        splitter = HeadlineSplitter(min_tokens=500)
        summary_extractor = SummaryExtractor(
            llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node)
        )

        theme_extractor = ThemesExtractor(
            llm=llm, filter_nodes=lambda node: filter_chunks(node)
        )
        ner_extractor = NERExtractor(
            llm=llm, filter_nodes=lambda node: filter_chunks(node)
        )

        summary_emb_extractor = EmbeddingExtractor(
            embedding_model=embedding_model,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=lambda node: filter_doc_with_num_tokens(node),
        )

        cosine_sim_builder = CosineSimilarityBuilder(
            property_name="summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.7,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node),
        )

        ner_overlap_sim = OverlapScoreBuilder(
            threshold=0.01, filter_nodes=lambda node: filter_chunks(node)
        )

        node_filter = CustomNodeFilter(
            llm=llm, filter_nodes=lambda node: filter_chunks(node)
        )
        transforms = [
            headline_extractor,
            splitter,
            summary_extractor,
            node_filter,
            Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
            Parallel(cosine_sim_builder, ner_overlap_sim),
        ]
    elif result["101-500"] >= 0.25:
        summary_extractor = SummaryExtractor(
            llm=llm, filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100)
        )
        summary_emb_extractor = EmbeddingExtractor(
            embedding_model=embedding_model,
            property_name="summary_embedding",
            embed_property_name="summary",
            filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
        )

        cosine_sim_builder = CosineSimilarityBuilder(
            property_name="summary_embedding",
            new_property_name="summary_similarity",
            threshold=0.5,
            filter_nodes=lambda node: filter_doc_with_num_tokens(node, 100),
        )

        ner_extractor = NERExtractor(llm=llm)
        ner_overlap_sim = OverlapScoreBuilder(threshold=0.01)
        theme_extractor = ThemesExtractor(
            llm=llm, filter_nodes=lambda node: filter_docs(node)
        )
        node_filter = CustomNodeFilter(llm=llm)

        transforms = [
            summary_extractor,
            node_filter,
            Parallel(summary_emb_extractor, theme_extractor, ner_extractor),
            Parallel(cosine_sim_builder, ner_overlap_sim),
        ]
    else:
        raise ValueError(
            "Documents appears to be too short (ie 100 tokens or less). Please provide longer documents."
        )

    return transforms

def create_or_load_knowledge_graph(
    docs: List[Document],
    generator_llm: Any,
    generator_embeddings: Any,
    run_config: RunConfig,
    cache: Optional[SQLiteCache] = None,
    cache_key: Optional[str] = None,
    cached_kg_path: Optional[str] = None
) -> Tuple[KnowledgeGraph, bool]:
    """Create a new knowledge graph or load from cache if available
    
    Args:
        docs: List of documents to create KG from
        generator_llm: LLM for generating KG content
        generator_embeddings: Embedding model for KG
        run_config: RAGAS run configuration
        cache: Optional cache instance
        cache_key: Optional cache key for storing/retrieving KG
        cached_kg_path: Optional path to cached KG file
    
    Returns:
        Tuple of (knowledge_graph, was_loaded_from_cache)
    """
    # Check if we can load from cache
    if cached_kg_path and os.path.exists(cached_kg_path):
        try:
            logger.info(f"Loading knowledge graph from cache: {cached_kg_path}")
            kg = KnowledgeGraph.load(cached_kg_path)
            logger.info(f"Loaded knowledge graph with {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
            return kg, True
        except Exception as e:
            logger.warning(f"Failed to load cached knowledge graph: {e}. Creating new one.")
    
    # Create a new knowledge graph
    logger.info("Creating knowledge graph from documents...")
    kg = KnowledgeGraph()
    
    # Add documents as nodes
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content, 
                    "document_metadata": doc.metadata
                }
            )
        )
    
    # Create and apply transforms
    logger.info("Applying transforms to enrich knowledge graph...")
    
    # Use modified parameters for transforms to improve relationship building
    # This uses more relaxed thresholds to create more connections between nodes
    try:
        # from ragas.testset.transforms.relationship_builders.traditional import OverlapScoreBuilder
        # from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
        # from ragas.testset.transforms.extractors import EmbeddingExtractor
        # from ragas.testset.transforms.extractors import HeadlinesExtractor
        # from ragas.testset.transforms.extractors.llm_based import (
        #     NERExtractor,
        #     SummaryExtractor,
        #     ThemesExtractor,
        #     KeyphrasesExtractor
        # )
        # from ragas.testset.transforms.splitters import HeadlineSplitter
        # from ragas.testset.transforms import Parallel

        # # Define filters for different node types
        # filter_docs = lambda node: node.type == NodeType.DOCUMENT
        # filter_chunks = lambda node: node.type == NodeType.CHUNK

        # # Custom transforms with more relaxed thresholds
        # transforms = [
        #     # 1. Extract headlines from original documents
        #     HeadlinesExtractor(llm=generator_llm, filter_nodes=filter_docs),
        #     # 2. Split documents into chunks based on headlines
        #     HeadlineSplitter(filter_nodes=filter_docs, min_tokens=300, max_tokens=1000), # Adjust token limits as needed

        #     # Extract entities, summaries, and themes
        #     # 3. Apply extractors to the *new* CHUNK nodes
        #     Parallel(
        #         NERExtractor(llm=generator_llm, max_num_entities=25, filter_nodes=filter_chunks),
        #         SummaryExtractor(llm=generator_llm, filter_nodes=filter_chunks),
        #         ThemesExtractor(llm=generator_llm, filter_nodes=filter_chunks),
        #         KeyphrasesExtractor(llm=generator_llm, max_num=20, filter_nodes=filter_chunks)
        #     ),

        #     # 4. Generate embeddings for CHUNK node summaries (or page_content if preferred)
        #     EmbeddingExtractor(
        #         embed_property_name="summary",
        #         property_name="summary_embedding",
        #         embedding_model=generator_embeddings,
        #         filter_nodes=filter_chunks # Apply to chunks
        #     ),

        #     # 5. Build relationships between CHUNK nodes
        #     Parallel(
        #         OverlapScoreBuilder(
        #             property_name="entities",
        #             distance_threshold=0.5,
        #             threshold=0.01,
        #             filter_nodes=filter_chunks # Apply to chunks
        #         ),
        #         OverlapScoreBuilder(
        #             property_name="keywords",
        #             distance_threshold=0.5,
        #             threshold=0.01,
        #             filter_nodes=filter_chunks # Apply to chunks
        #         ),
        #         CosineSimilarityBuilder(
        #             property_name="summary_embedding",
        #             new_property_name="summary_similarity",
        #             threshold=0.1,  # Lower threshold (was 0.2)
        #             filter_nodes=filter_chunks # Apply to chunks
        #         )
        #     )
        # ]
        transforms = default_transforms(
            documents=docs, llm=generator_llm, embedding_model=generator_embeddings
        )
        
        apply_transforms(kg, transforms, run_config=run_config)
    except ImportError:
        # Fall back to default transforms if specific imports fail
        logger.warning("Using default transforms; custom transforms failed to import")
        transforms = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
        apply_transforms(kg, transforms, run_config=run_config)
    
    # Save knowledge graph if caching is enabled
    if cache and cache_key:
        ts = datetime.now(timezone.utc).timestamp()
        kg_path = f'knowledge_graph_{ts}.json'
        kg.save(kg_path)
        logger.info(f"Saved knowledge graph to {kg_path}")
        
        # Update cache
        cache.set_kg(cache_key, kg_path, len(kg.nodes), len(kg.relationships))
    
    return kg, False

def generate_data(
        src_data: str,
        generator_model: Any,
        embedding_model: Any,
        test_size: int = 5,
        distribution: Dict[str, float] = None,
        run_config: Optional[RunConfig] = None,
        cache: Optional[SQLiteCache] = None,
        cache_key: Optional[str] = None,
        cached_kg_path: Optional[str] = None
) -> str:
    """Generate synthetic test dataset using the knowledge graph approach
    
    Args:
        src_data: Path to source data
        generator_model: LLM for question generation
        embedding_model: Model for embeddings
        test_size: Number of test cases to generate
        distribution: Distribution of question types
        run_config: RAGAS run configuration
        cache: Optional cache instance
        cache_key: Optional cache key for storing KG
        cached_kg_path: Optional path to cached KG file
        
    Returns:
        Path to generated test dataset
    """
    # Use default run config if none provided
    if run_config is None:
        run_config = RunConfig(
            timeout=1000,
            max_workers=1,
            max_wait=900,
            max_retries=5
        )
    
    # Load source documents
    docs = load_src(src_data)
    if not docs:
        raise ValueError("No documents loaded for test generation")

    # Wrap models if needed
    if not hasattr(generator_model, '_ragasllm'):
        # Wrap LangChain models with RAGAS wrappers
        generator_llm = LangchainLLMWrapper(generator_model)
    else:
        generator_llm = generator_model
        
    if not hasattr(embedding_model, '_ragasembeddings'):
        # Wrap LangChain models with RAGAS wrappers
        generator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
    else:
        generator_embeddings = embedding_model

    # Create or load knowledge graph
    kg, loaded_from_cache = create_or_load_knowledge_graph(
        docs=docs,
        generator_llm=generator_llm,
        generator_embeddings=generator_embeddings,
        run_config=run_config,
        cache=cache,
        cache_key=cache_key,
        cached_kg_path=cached_kg_path
    )
    
    # Configure generator
    logger.info(f"Initializing TestsetGenerator with {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
    generator = TestsetGenerator(
        llm=generator_llm, 
        embedding_model=generator_embeddings,
        knowledge_graph=kg
    )

    # Import synthesizers
    from ragas.testset.synthesizers import (
        SingleHopSpecificQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        MultiHopAbstractQuerySynthesizer
    )

    # Configure query distribution - try in multiple configurations if needed
    try:
        if distribution is None:
            query_distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.7), 
                (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.15),
                (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.15)
            ]
        else:
            # Map old distribution format to new synthesizers
            simple_weight = distribution.get('simple', 0.5)
            reasoning_weight = distribution.get('reasoning', 0.1)
            multi_context_weight = distribution.get('multi_context', 0.4)
            
            # Create a combined distribution using the provided weights
            query_distribution = [
                (SingleHopSpecificQuerySynthesizer(llm=generator_llm), simple_weight),
                (MultiHopSpecificQuerySynthesizer(llm=generator_llm), reasoning_weight),
                (MultiHopAbstractQuerySynthesizer(llm=generator_llm), multi_context_weight)
            ]

        # Generate testset
        logger.info(f"Generating {test_size} test cases with mixed query types...")
        testset = None
        testset = generator.generate(
            testset_size=test_size,
            query_distribution=query_distribution
        )
        
    except Exception as e:
        error_message = str(e)
        logger.warning(f"Error generating questions with mixed types: {error_message}")
        
        # if "No clusters found in the knowledge graph" in error_message:
        logger.info("Falling back to single-hop questions only...")
        
        # Second attempt: Use only single-hop questions
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)
        ]
        
        try:
            testset = generator.generate(
                testset_size=test_size,
                query_distribution=query_distribution
            )
            logger.info("Test generation complete")
        except Exception as fallback_error:
            # If even single-hop generation fails, provide a more helpful error message
            logger.error(f"Error generating single-hop questions: {str(fallback_error)}")
            raise ValueError(
                "Failed to generate test questions with this dataset. The knowledge graph may not have "
                "enough meaningful content. Try with a larger or more structured dataset, or consider "
                "manually creating test questions."
            ) from fallback_error
    
    if testset:
        ts = datetime.now(timezone.utc).timestamp()
        f_name = f'rag_test_data_{ts}.csv'
        logger.info(f"Writing to {f_name}")
        test_df = testset.to_pandas()
        test_df.to_csv(f_name)
    else:
        logger.error("Testset generation failed")
        raise ValueError("Testset generation failed")
    
    return f_name

