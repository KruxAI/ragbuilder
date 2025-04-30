"""
Test script for RAGBuilder v2 to verify basic functionality
"""

import os
import tempfile
from pathlib import Path
import unittest
from dotenv import load_dotenv
from ragbuilder import RAGBuilder
from ragbuilder.config.base import LogConfig, LLMConfig, EmbeddingConfig, OptimizationConfig
from ragbuilder.config.components import (
    LLMType, EmbeddingType, ChunkingStrategy, ParserType
)
from ragbuilder.config.data_ingest import DataIngestOptionsConfig, LoaderConfig, ChunkingStrategyConfig

# Load environment variables from .env file
load_dotenv()

class TestRAGBuilderBasic(unittest.TestCase):
    """Basic tests for RAGBuilder functionality"""
    
    def setUp(self):
        """Set up test environment with a sample text file"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Check if LangChain documentation directory exists
        self.docs_dir_path = Path("SampleInputFiles/md")
        
        if self.docs_dir_path.exists() and any(self.docs_dir_path.iterdir()):
            # Use LangChain documentation directory
            self.test_file_path = str(self.docs_dir_path)
            # Look for test dataset specific to LangChain docs
            self.langchain_dataset_path = Path("SampleInputFiles/rag_test_data_gpt4o_shortlisted_small.csv")
            if self.langchain_dataset_path.exists():
                self.langchain_test_dataset = str(self.langchain_dataset_path)
            else:
                # Create a fallback test dataset
                self.langchain_test_dataset = Path(self.temp_dir.name) / "langchain_test_dataset.csv"
                with open(self.langchain_test_dataset, "w") as f:
                    f.write("""user_input,reference
"What is an agent in LangChain?","Agents use an LLM to determine which actions to take and in what order. An action can either be using a tool and observing its output, or returning a response to the user."
"How do I create a basic agent?","You can create a basic agent by importing the necessary tools and agent types, then using agent_executor.create() with your tools and an LLM."
"What are tools in the context of agents?","Tools are functions that agents can use to interact with the world. They require a name, description and the actual function to run."
"How can I create custom tools for my agent?","You can create custom tools by defining a function and using the @tool decorator with a description of what the tool does."
""")
        else:
            # Create a simple text file for testing as fallback
            self.test_file_path = Path(self.temp_dir.name) / "sample.txt"
            with open(self.test_file_path, "w") as f:
                f.write("""
                # RAGBuilder Test Document
                
                This is a test document for RAGBuilder. It contains some information about neural networks.
                
                ## Neural Networks
                
                Neural networks are computing systems vaguely inspired by the biological neural networks 
                that constitute animal brains. The neural network itself consists of artificial neurons,
                connected through synapses, organized in layers. Neural networks can be trained to recognize
                patterns in data.
                
                ## Types of Neural Networks
                
                1. **Convolutional Neural Networks (CNN)**: Primarily used for image processing, computer vision tasks.
                2. **Recurrent Neural Networks (RNN)**: Used for sequential data like time series or natural language.
                3. **Transformers**: State-of-the-art architecture for NLP tasks, using self-attention mechanisms.
                
                ## Applications
                
                Neural networks are used in various fields including:
                - Image and speech recognition
                - Natural language processing
                - Recommendation systems
                - Medical diagnosis
                """)
            # Create a fallback test dataset
            self.langchain_test_dataset = Path(self.temp_dir.name) / "neural_network_test_dataset.csv"
            with open(self.langchain_test_dataset, "w") as f:
                f.write("""user_input,reference
"What are neural networks?","Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains. They consist of artificial neurons, connected through synapses, organized in layers."
"What types of neural networks are there?","Common types include Convolutional Neural Networks (CNN) for image processing, Recurrent Neural Networks (RNN) for sequential data, and Transformers for NLP tasks."
"What are applications of neural networks?","Neural networks are used in various fields including image and speech recognition, natural language processing, recommendation systems, and medical diagnosis."
""")
        
        # Check for 10K PDF report
        self.pdf_file_path = Path("SampleInputFiles/pdf/uber_10k.pdf")
        if not self.pdf_file_path.exists():
            # Note: We don't create a fallback PDF since that would be complex
            self.pdf_available = False
        else:
            self.pdf_available = True
            # Look for test dataset specific to 10K report
            self.pdf_dataset_path = Path("SampleInputFiles/pdf/uber10k_shortlist.csv")
            if self.pdf_dataset_path.exists():
                self.pdf_test_dataset = str(self.pdf_dataset_path)
            else:
                # Create a fallback test dataset for PDF
                self.pdf_test_dataset = Path(self.temp_dir.name) / "10k_test_dataset.csv"
                with open(self.pdf_test_dataset, "w") as f:
                    f.write("""user_input,reference
"What is the company's business model?","The 10K report describes the company's business model including its primary revenue streams and operational structure."
"What are the main risk factors?","The risk factors include market competition, regulatory challenges, and economic uncertainties as detailed in the report."
"What was the revenue in the past fiscal year?","The company reported specific revenue figures for the past fiscal year as documented in the financial statements section."
""")
        
        # Set up default configurations for testing
        self.log_config = LogConfig(log_level="INFO", log_file=None)
        
        # Use AzureOpenAI embeddings if API key is available, otherwise fall back to HuggingFace
        if os.getenv("AZURE_OPENAI_API_KEY"):
            self.default_embeddings = EmbeddingConfig(
                type=EmbeddingType.AZURE_OPENAI,
                model_kwargs={
                    "model": "text-embedding-3-large"
                }
            )
            
            self.default_llm = LLMConfig(
                type=LLMType.AZURE_OPENAI,
                model_kwargs={
                    "model": "gpt-4o",
                    "temperature": 0.1
                }
            )
            self.using_hf_models = False
        else:
            # Default embedding model that works without API keys
            self.default_embeddings = EmbeddingConfig(
                type=EmbeddingType.HUGGINGFACE,
                model_kwargs={"model_name": "all-MiniLM-L6-v2"}
            )
            
            # Default local LLM that works without API keys
            self.default_llm = LLMConfig(
                type=LLMType.HUGGINGFACE,
                model_kwargs={
                    "repo_id": "google/flan-t5-small",
                    "temperature": 0.1
                }
            )
            self.using_hf_models = True
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_quickstart_init(self):
        """Test initialization with defaults"""
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,  # Use minimal trials for quick testing
            log_config=self.log_config
        )
        
        # Check that builder was initialized correctly
        self.assertIsNotNone(builder)
        self.assertIsNotNone(builder.data_ingest_config)
        self.assertEqual(builder.data_ingest_config.input_source, str(self.test_file_path))
    
    def test_data_ingest_optimization(self):
        """Test data ingestion optimization"""
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.langchain_test_dataset),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,  # Use minimal trials for quick testing
            log_config=self.log_config
        )
        
        # Run data ingest optimization
        results = builder.optimize_data_ingest()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.best_index)
        self.assertIsNotNone(results.best_config)
        self.assertIsNotNone(results.best_score)
    
    def test_markdown_custom_data_ingest(self):
        """Test data ingestion with custom Markdown configuration"""
        # Skip test if not using actual Markdown files
        if not self.docs_dir_path.exists() or not any(self.docs_dir_path.iterdir()):
            self.skipTest("Skipping Markdown test since no Markdown files available")
            
        # Create custom data ingest configuration with Markdown Header splitter
        data_ingest_config = DataIngestOptionsConfig(
            input_source=str(self.docs_dir_path),
            document_loaders=[
                LoaderConfig(type=ParserType.TEXT)
            ],
            chunking_strategies=[
                ChunkingStrategyConfig(
                    type=ChunkingStrategy.MARKDOWN, 
                    chunker_kwargs={
                        "headers_to_split_on": [
                            ("#", "Header 1"),
                            ("##", "Header 2"),
                            ("###", "Header 3")
                        ],
                        "strip_headers": False
                    }
                )
            ],
            embedding_models=[self.default_embeddings],
            optimization=OptimizationConfig(
                n_trials=1,
                n_jobs=1,
                optimization_direction="maximize"
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
        
        # Confirm that we're using the Markdown Header splitter
        self.assertEqual(
            results.best_config.chunking_strategy.type,
            ChunkingStrategy.MARKDOWN
        )
    
    def test_pdf_extraction(self):
        """Test PDF extraction with 10K report"""
        # Skip test if PDF file not available
        if not self.pdf_available:
            self.skipTest("Skipping PDF test since no 10K PDF file available")
        
        # Create custom data ingest configuration for PDF
        data_ingest_config = DataIngestOptionsConfig(
            input_source=str(self.pdf_file_path),
            document_loaders=[
                LoaderConfig(type=ParserType.PYPDF),
                LoaderConfig(type=ParserType.UNSTRUCTURED)  # Test multiple PDF extractors
            ],
            test_dataset=str(self.pdf_test_dataset),
            optimization=OptimizationConfig(
                n_trials=1,
                n_jobs=1,
                optimization_direction="maximize"
            )
        )
        
        # Initialize RAGBuilder with custom configuration
        builder = RAGBuilder(
            data_ingest_config=data_ingest_config,
            default_llm=self.default_llm,
            default_embeddings=self.default_embeddings,
            log_config=self.log_config
        )
        
        # Run data ingest optimization to test PDF extractors
        results = builder.optimize_data_ingest()
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.best_index)
        self.assertIsNotNone(results.best_config)
        self.assertIsNotNone(results.best_score)
        
        # Check that documents were extracted
        self.assertGreater(results.best_index._collection.count(), 0)
        self.assertTrue(len(results.best_index._collection.peek()['metadatas']) > 0)
    
    def test_retrieval_optimization(self):
        """Test retrieval optimization"""
        if self.using_hf_models:
            self.skipTest("Skipping retrieval optimization test with HuggingFace models for speed")
            
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.langchain_test_dataset),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,  # Use minimal trials for quick testing
            log_config=self.log_config
        )
        
        # First run data ingest to get a vectorstore
        data_ingest_results = builder.optimize_data_ingest()
        
        # Now run retrieval optimization
        retrieval_results = builder.optimize_retrieval()
        
        # Verify results
        self.assertIsNotNone(retrieval_results)
        self.assertIsNotNone(retrieval_results.best_pipeline.retriever_chain)
        self.assertIsNotNone(retrieval_results.best_config)
        self.assertIsNotNone(retrieval_results.best_score)
    
    def test_generation_optimization(self):
        """Test generation optimization"""
        if self.using_hf_models:
            self.skipTest("Skipping generation optimization test with HuggingFace models for speed")
            
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.langchain_test_dataset),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,  # Use minimal trials for quick testing
            log_config=self.log_config
        )
        
        # Run data ingest and retrieval first
        builder.optimize_data_ingest()
        builder.optimize_retrieval()
        
        # Now run generation optimization
        generation_results = builder.optimize_generation()
        
        # Verify results
        self.assertIsNotNone(generation_results)
        self.assertIsNotNone(generation_results.best_config)
        self.assertIsNotNone(generation_results.best_score)
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization"""
        if self.using_hf_models:
            self.skipTest("Skipping end-to-end optimization test with HuggingFace models for speed")
            
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            test_dataset=str(self.langchain_test_dataset),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            n_trials=1,  # Use minimal trials for quick testing
            log_config=self.log_config
        )
        
        # Run full optimization
        results = builder.optimize()
        
        # Verify final results
        self.assertIsNotNone(results)
        self.assertIsNotNone(results.data_ingest)
        self.assertIsNotNone(results.retrieval)
        self.assertIsNotNone(results.generation)
        
        # Test query
        response = results.invoke("What are the key components of an agent?")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 0)
    
    # def test_save_and_load(self):
    #     """Test saving and loading an optimized RAG pipeline"""
    #     if self.using_hf_models:
    #         self.skipTest("Skipping save and load test with HuggingFace models for speed")
            
    #     builder = RAGBuilder.from_source_with_defaults(
    #         input_source=str(self.test_file_path),
    #         test_dataset=str(self.langchain_test_dataset),
    #         default_embeddings=self.default_embeddings,
    #         default_llm=self.default_llm,
    #         n_trials=1,  # Use minimal trials for quick testing
    #         log_config=self.log_config
    #     )
        
    #     # Run optimization
    #     builder.optimize()
        
    #     # Save to temporary directory
    #     save_path = Path(self.temp_dir.name) / "rag_pipeline"
    #     builder.save(str(save_path))
        
    #     # Check that files were created
    #     self.assertTrue(save_path.exists())
        
    #     # Load from saved files
    #     loaded_builder = RAGBuilder.load(str(save_path))
        
    #     # Test query with loaded builder
    #     response = loaded_builder.optimization_results.invoke("What are the key components of an agent?")
    #     self.assertIsNotNone(response)
    #     self.assertTrue(len(response) > 0)


if __name__ == "__main__":
    unittest.main() 