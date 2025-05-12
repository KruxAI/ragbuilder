"""
Test script for RAGBuilder v2 to verify advanced functionality with custom configurations
"""

import os
import tempfile
from pathlib import Path
import unittest
from dotenv import load_dotenv
from ragbuilder import RAGBuilder
from ragbuilder.config.base import LogConfig, LLMConfig, EmbeddingConfig, OptimizationConfig, EvaluationConfig
from ragbuilder.config.components import (
    LLMType, EmbeddingType, ParserType, ChunkingStrategy, 
    VectorDatabase, RetrieverType, RerankerType, EvaluatorType
)
from ragbuilder.config.data_ingest import DataIngestOptionsConfig, LoaderConfig, ChunkingStrategyConfig, ChunkSizeConfig, VectorDBConfig
from ragbuilder.config.retriever import RetrievalOptionsConfig, BaseRetrieverConfig, RerankerConfig
from ragbuilder.config.generation import GenerationOptionsConfig

# Load environment variables from .env file
load_dotenv()

class TestRAGBuilderAdvanced(unittest.TestCase):
    """Advanced tests for RAGBuilder functionality with custom configurations"""
    
    def setUp(self):
        """Set up test environment with sample files from SampleInputFiles directory"""
        # Create a temporary directory for test output
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Use LangChain documentation directory from SampleInputFiles as the test source
        self.docs_dir_path = Path("../SampleInputFiles/md")
        
        # Check if the documentation directory exists
        if not self.docs_dir_path.exists() or not any(self.docs_dir_path.iterdir()):
            # Create a test file if directory doesn't exist
            self.docs_dir_path = Path(self.temp_dir.name)
            test_file = self.docs_dir_path / "sample_doc.md"
            with open(test_file, "w") as f:
                f.write("""
                # RAGBuilder Test Document
                
                This is a test document for RAGBuilder advanced tests.
                """)
        
        # Use the directory or a single file depending on what's available
        self.test_file_path = self.docs_dir_path
        
        # Use the PDF file for PDF-specific tests
        self.pdf_file_path = Path("../SampleInputFiles/pdf/uber_10k.pdf")
        self.pdf_available = self.pdf_file_path.exists()
        
        # Look for test dataset or create fallback
        self.test_dataset_path = Path("../SampleInputFiles/rag_test_data_gpt4o_shortlisted_small.csv")
        if not self.test_dataset_path.exists():
            # Create a fallback test dataset
            self.test_dataset_path = Path(self.temp_dir.name) / "test_dataset.csv"
            with open(self.test_dataset_path, "w") as f:
                f.write("""user_input,reference
"What is an agent in LangChain?","Agents use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning a response to the user."
"How do I create a basic agent?","You can create a basic agent by importing the necessary tools and agent types, then using agent_executor.create() with your tools and an LLM."
"What are the differences between different agent types?","Different agent types use different prompting strategies. For example, ReAct agents break down reasoning into steps, while OpenAI Assistants use OpenAI's API directly."
"What are tools in the context of agents?","Tools are functions that agents can use to interact with the world. They require a name, description and the actual function to run."
"How can I create custom tools for my agent?","You can create custom tools by defining a function and using the @tool decorator with a description of what the tool does."
""")
        
        # PDF test dataset
        self.pdf_test_dataset_path = Path("../SampleInputFiles/pdf/uber10k_shortlist.csv")
        if not self.pdf_test_dataset_path.exists():
            # Create a fallback PDF test dataset
            self.pdf_test_dataset_path = Path(self.temp_dir.name) / "pdf_test_dataset.csv"
            with open(self.pdf_test_dataset_path, "w") as f:
                f.write("""user_input,reference
"What is the company's business model?","The 10K report describes the company's business model including its primary revenue streams and operational structure."
"What are the main risk factors?","The risk factors include market competition, regulatory challenges, and economic uncertainties as detailed in the report."
"What was the revenue in the past fiscal year?","The company reported specific revenue figures for the past fiscal year as documented in the financial statements section."
""")
        
        # Set up default configurations for testing
        self.log_config = LogConfig(log_level="INFO", log_file=None)
        
        # Use Azure OpenAI for embeddings
        self.default_embeddings = EmbeddingConfig(
            type=EmbeddingType.AZURE_OPENAI,
            model_kwargs={
                "model": "text-embedding-3-large"
            }
        )
        
        # Use Azure OpenAI for LLM
        self.default_llm = LLMConfig(
            type=LLMType.AZURE_OPENAI,
            model_kwargs={
                "model": "gpt-4o",
                "temperature": 0.1
            }
        )
        
        # Fallback to HuggingFace if Azure keys are not available
        if not os.getenv("AZURE_OPENAI_API_KEY"):
            self.default_embeddings = EmbeddingConfig(
                type=EmbeddingType.HUGGINGFACE,
                model_kwargs={"model_name": "all-MiniLM-L6-v2"}
            )
            
            self.default_llm = LLMConfig(
                type=LLMType.HUGGINGFACE,
                model_kwargs={
                    "repo_id": "google/flan-t5-small",
                    "temperature": 0.1
                }
            )
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_custom_data_ingest_config(self):
        """Test data ingestion with custom configuration"""
        # Skip if using HuggingFace models for faster testing
        if self.default_llm.type == LLMType.HUGGINGFACE:
            self.skipTest("Skipping test with HuggingFace models for speed")
            
        # Create custom data ingest configuration
        data_ingest_config = DataIngestOptionsConfig(
            input_source=str(self.test_file_path),
            document_loaders=[
                LoaderConfig(type=ParserType.UNSTRUCTURED)
            ],
            chunking_strategies=[
                ChunkingStrategyConfig(type=ChunkingStrategy.RECURSIVE)
            ],
            chunk_size=ChunkSizeConfig(
                min=500,
                max=1000,
                stepsize=500
            ),
            chunk_overlap=[50],
            embedding_models=[self.default_embeddings],
            vector_databases=[
                VectorDBConfig(type=VectorDatabase.CHROMA)
            ],
            optimization=OptimizationConfig(
                n_trials=2,
                n_jobs=1,
                optimization_direction="maximize"
            ),
            evaluation_config=EvaluationConfig(
                type=EvaluatorType.SIMILARITY,
                test_dataset=str(self.test_dataset_path)
            )
        )
        
        # Initialize RAGBuilder with custom configuration
        builder = RAGBuilder(
            data_ingest_config=data_ingest_config,
            default_llm=self.default_llm,
            default_embeddings=self.default_embeddings,
            log_config=self.log_config
        )
        
        # Run data ingest optimization
        results = builder.optimize_data_ingest()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.best_index)
        self.assertIsNotNone(results.best_config)
        self.assertIsNotNone(results.best_score)
        
        # Check that config was applied
        self.assertEqual(builder.data_ingest_config.input_source, str(self.test_file_path))
        self.assertEqual(len(builder.data_ingest_config.chunking_strategies), 1)
    
    def test_pdf_data_ingest(self):
        """Test data ingestion with PDF file"""
        # Skip if PDF file is not available
        if not self.pdf_available:
            self.skipTest("Skipping PDF test since no PDF file available")
            
        # Skip if using HuggingFace models for faster testing
        if self.default_llm.type == LLMType.HUGGINGFACE:
            self.skipTest("Skipping PDF test with HuggingFace models for speed")
            
        # Create custom data ingest configuration for PDF
        data_ingest_config = DataIngestOptionsConfig(
            input_source=str(self.pdf_file_path),
            document_loaders=[
                LoaderConfig(type=ParserType.PYPDF),
                LoaderConfig(type=ParserType.UNSTRUCTURED)
            ],
            chunking_strategies=[
                ChunkingStrategyConfig(type=ChunkingStrategy.RECURSIVE)
            ],
            embedding_models=[self.default_embeddings],
            optimization=OptimizationConfig(
                n_trials=1,
                n_jobs=1,
                optimization_direction="maximize"
            ),
            evaluation_config=EvaluationConfig(
                type=EvaluatorType.SIMILARITY,
                test_dataset=str(self.pdf_test_dataset_path)
            )
        )
        
        # Initialize RAGBuilder with custom configuration
        builder = RAGBuilder(
            data_ingest_config=data_ingest_config,
            default_llm=self.default_llm,
            default_embeddings=self.default_embeddings,
            log_config=self.log_config
        )
        
        # Run data ingest optimization
        results = builder.optimize_data_ingest()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.best_index)
        self.assertTrue(len(results.best_index._collection.peek()['metadatas']) > 0)
    
    def test_custom_retrieval_config(self):
        """Test retrieval with custom configuration"""
        # Skip if using HuggingFace models for faster testing
        if self.default_llm.type == LLMType.HUGGINGFACE:
            self.skipTest("Skipping test with HuggingFace models for speed")
            
        # First create and optimize data ingest
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.test_dataset_path),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,
            log_config=self.log_config
        )
        
        data_ingest_results = builder.optimize_data_ingest()
        
        # Create custom retrieval configuration
        retrieval_config = RetrievalOptionsConfig(
            retrievers=[
                BaseRetrieverConfig(
                    type=RetrieverType.VECTOR_SIMILARITY,
                    retriever_k=[5],
                    weight=1.0
                )
            ],
            rerankers=[
                RerankerConfig(type=RerankerType.BGE_BASE)
            ],
            top_k=[3],
            optimization=OptimizationConfig(
                n_trials=1,
                n_jobs=1,
                optimization_direction="maximize"
            ),
            evaluation_config=EvaluationConfig(
                type=EvaluatorType.RAGAS,
                test_dataset=str(self.test_dataset_path),
                llm=self.default_llm,
                embeddings=self.default_embeddings
            )
        )
        
        # Run retrieval optimization with custom config
        retrieval_results = builder.optimize_retrieval(retrieval_config)
        
        # Verify results
        self.assertIsNotNone(retrieval_results)
        self.assertIsNotNone(retrieval_results.best_pipeline.retriever_chain)
        self.assertIsNotNone(retrieval_results.best_config)
        self.assertIsNotNone(retrieval_results.best_score)
    
    def test_custom_generation_config(self):
        """Test generation with custom configuration"""
        # Skip if using HuggingFace models for faster testing
        if self.default_llm.type == LLMType.HUGGINGFACE:
            self.skipTest("Skipping test with HuggingFace models for speed")
            
        # First create and optimize data ingest and retrieval
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.test_dataset_path),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,
            log_config=self.log_config
        )
        
        builder.optimize_data_ingest()
        builder.optimize_retrieval()
        
        # Create custom generation configuration
        generation_config = GenerationOptionsConfig(
            llms=[self.default_llm],
            eval_data_set_path=str(self.test_dataset_path),
            optimization=OptimizationConfig(
                n_trials=1,
                n_jobs=1,
                optimization_direction="maximize"
            ),
            evaluation_config=EvaluationConfig(
                type=EvaluatorType.SIMILARITY,
                test_dataset=str(self.test_dataset_path),
                llm=self.default_llm,
                embeddings=self.default_embeddings
            )
        )
        
        # Run generation optimization with custom config
        generation_results = builder.optimize_generation(generation_config)
        
        # Verify results
        self.assertIsNotNone(generation_results)
        self.assertIsNotNone(generation_results.best_config)
        self.assertIsNotNone(generation_results.best_score)
    
    def test_yaml_loading(self):
        """Test loading configuration from YAML file"""
        # Skip if using HuggingFace models for faster testing
        if self.default_llm.type == LLMType.HUGGINGFACE:
            self.skipTest("Skipping test with HuggingFace models for speed")
            
        # Create a YAML configuration file
        yaml_path = Path(self.temp_dir.name) / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write("""
data_ingest:
  input_source: {}
  document_loaders:
    - type: unstructured
  chunking_strategies:
    - type: RecursiveCharacterTextSplitter
  chunk_size:
    min: 500
    max: 1500
    stepsize: 500
  chunk_overlap:
    - 50
  embedding_models:
    - type: azure_openai
      model_kwargs:
        model: text-embedding-3-large
  vector_databases:
    - type: chroma
      vectordb_kwargs:
        collection_metadata:
          hnsw:space: cosine
  optimization:
    n_trials: 1
  evaluation_config:
    type: similarity
    test_dataset: {}

retrieval:
  retrievers:
    - type: vector_similarity
      retriever_k:
        - 5
      weight: 1.0
  rerankers:
    - type: BAAI/bge-reranker-base
  top_k:
    - 3
  optimization:
    n_trials: 1
  evaluation_config:
    type: similarity

generation:
  llms:
    - type: azure_openai
      model_kwargs:
        model: gpt-4o
        temperature: 0.1
  optimization:
    n_trials: 1
  evaluation_config:
    type: similarity
            """.format(
                str(self.test_file_path),
                str(self.test_dataset_path)
            ))
        
        # Load config from YAML
        builder = RAGBuilder.from_yaml(str(yaml_path))
        
        # Check that config was loaded correctly
        self.assertIsNotNone(builder)
        self.assertIsNotNone(builder.data_ingest_config)
        self.assertEqual(builder.data_ingest_config.input_source, str(self.test_file_path))
        
        # Run partial optimization (data ingest only for speed)
        results = builder.optimize_data_ingest()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.best_config)
    
    def test_configs_saving(self):
        """Test saving configurations to file"""
        # Skip if using HuggingFace models for faster testing
        if self.default_llm.type == LLMType.HUGGINGFACE:
            self.skipTest("Skipping test with HuggingFace models for speed")
            
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.test_dataset_path),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,
            log_config=self.log_config
        )
        
        # Run only data ingest optimization for speed
        builder.optimize_data_ingest()
        
        # Save configs to file
        config_path = Path(self.temp_dir.name) / "configs.json"
        builder.save_configs(str(config_path))
        
        # Check that file exists
        self.assertTrue(config_path.exists())


if __name__ == "__main__":
    unittest.main() 