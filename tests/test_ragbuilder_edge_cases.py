"""
Test script for RAGBuilder v2 to verify edge cases and error handling
"""

import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from ragbuilder import RAGBuilder
from ragbuilder.config.base import LogConfig, LLMConfig, EmbeddingConfig
from ragbuilder.config.components import LLMType, EmbeddingType, ParserType
from ragbuilder.core.exceptions import DependencyError

class TestRAGBuilderEdgeCases(unittest.TestCase):
    """Tests for RAGBuilder edge cases and error handling"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for test data
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple text file for testing
        self.test_file_path = Path(self.temp_dir.name) / "sample.txt"
        with open(self.test_file_path, "w") as f:
            f.write("""This is a minimal test document.""")
        
        # Check if the LangChain documentation directory exists
        self.md_dir_path = Path("../SampleInputFiles/md")
        self.md_available = self.md_dir_path.exists() and any(self.md_dir_path.iterdir())
        
        # Check if PDF file exists
        self.pdf_file_path = Path("../SampleInputFiles/pdf/uber_10k.pdf")
        self.pdf_available = self.pdf_file_path.exists()
        
        # Get test dataset paths
        self.md_test_dataset = Path("../SampleInputFiles/rag_test_data_gpt4o_shortlisted_small.csv")
        self.pdf_test_dataset = Path("../SampleInputFiles/pdf/uber10k_shortlist.csv")
        
        # Set up test configs
        self.log_config = LogConfig(log_level="INFO", log_file=None)
        
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
    
    def tearDown(self):
        """Clean up temporary directory"""
        self.temp_dir.cleanup()
    
    def test_nonexistent_input_file(self):
        """Test handling of nonexistent input file"""
        non_existent_file = Path(self.temp_dir.name) / "nonexistent.txt"
        
        # Try to initialize with non-existent file
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(non_existent_file),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Optimization should fail, but initialization should not
        self.assertIsNotNone(builder)
        with self.assertRaises(Exception):
            builder.optimize_data_ingest()
    
    def test_empty_input_file(self):
        """Test handling of empty input file"""
        empty_file = Path(self.temp_dir.name) / "empty.txt"
        with open(empty_file, "w") as f:
            f.write("")
        
        # Try to initialize with empty file
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(empty_file),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Should handle empty file gracefully or raise appropriate exception
        try:
            builder.optimize_data_ingest()
            # If it completes, check that we have appropriate results
            self.assertIsNotNone(builder.optimized_store)
        except Exception as e:
            # If it fails, make sure it's a reasonable error message
            self.assertIn("invalid", str(e).lower())
    
    def test_retrieval_before_data_ingest(self):
        """Test error handling when running retrieval before data ingestion"""
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.md_dir_path),
            test_dataset=str(self.md_test_dataset),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Should raise DependencyError or similar
        with self.assertRaises(DependencyError):
            builder.optimize_retrieval()
    
    def test_generation_before_retrieval(self):
        """Test error handling when running generation before retrieval"""
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.md_dir_path),
            test_dataset=str(self.md_test_dataset),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Run data ingest but skip retrieval
        builder.optimize_data_ingest()
        
        # Should raise DependencyError or similar
        with self.assertRaises(DependencyError):
            builder.optimize_generation()
    
    def test_load_nonexistent_pipeline(self):
        """Test error handling when loading a non-existent pipeline"""
        non_existent_path = Path(self.temp_dir.name) / "nonexistent_pipeline"
        
        # Should raise appropriate exception
        with self.assertRaises(Exception):
            RAGBuilder.load(str(non_existent_path))
    
    # def test_save_incomplete_pipeline(self):
    #     """Test saving an incomplete pipeline (missing some optimization steps)"""
    #     builder = RAGBuilder.from_source_with_defaults(
    #         input_source=str(self.test_file_path),
    #         default_embeddings=self.default_embeddings,
    #         default_llm=self.default_llm,
    #         log_config=self.log_config
    #     )
        
    #     # Only run data ingest
    #     builder.optimize_data_ingest()
        
    #     # Should be able to save partial pipeline
    #     save_path = Path(self.temp_dir.name) / "partial_pipeline"
    #     builder.save(str(save_path))
        
    #     # Check that it saved
    #     self.assertTrue(save_path.exists())
        
    #     # Load the partial pipeline
    #     loaded_builder = RAGBuilder.load(str(save_path))
        
    #     # Check that data ingest results were loaded but not others
    #     self.assertIsNotNone(loaded_builder.optimized_store)
    #     self.assertIsNone(loaded_builder.optimized_retriever)
    #     self.assertIsNone(loaded_builder.optimized_generation)
    
    def test_nonexistent_md_directory(self):
        """Test handling of nonexistent Markdown directory"""
        non_existent_dir = Path(self.temp_dir.name) / "nonexistent_md_dir"
        
        # Try to initialize with non-existent directory
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(non_existent_dir),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Optimization should fail, but initialization should not
        self.assertIsNotNone(builder)
        with self.assertRaises(Exception):
            builder.optimize_data_ingest()
    
    def test_empty_md_directory(self):
        """Test handling of empty Markdown directory"""
        empty_dir = Path(self.temp_dir.name) / "empty_md_dir"
        empty_dir.mkdir(exist_ok=True)
        
        # Try to initialize with empty directory
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(empty_dir),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Should handle empty directory gracefully or raise appropriate exception
        with self.assertRaises(Exception) as context:
            builder.optimize_data_ingest()
        
        # Check error message is reasonable
        self.assertTrue(any(keyword in str(context.exception).lower() 
                        for keyword in ["empty", "no files", "not found"]))
    
    # def test_unsupported_pdf_file(self):
    #     """Test handling of a corrupted or unsupported PDF file"""
    #     # Create a file with PDF extension but invalid content
    #     invalid_pdf = Path(self.temp_dir.name) / "invalid.pdf"
    #     with open(invalid_pdf, "w") as f:
    #         f.write("This is not a valid PDF file")
        
    #     # Try to initialize with invalid PDF
    #     builder = RAGBuilder.from_source_with_defaults(
    #         input_source=str(invalid_pdf),
    #         default_embeddings=self.default_embeddings,
    #         default_llm=self.default_llm,
    #         log_config=self.log_config
    #     )
        
    #     # Optimization might fail or produce empty results
    #     try:
    #         results = builder.optimize_data_ingest()
    #         # If it completes, the results might be empty but shouldn't crash
    #         self.assertIsNotNone(results)
    #     except Exception as e:
    #         # If it fails, make sure it's a reasonable error message
    #         self.assertTrue(any(keyword in str(e).lower() 
    #                       for keyword in ["pdf", "parse", "invalid", "format"]))
    
    def test_invalid_yaml_config(self):
        """Test handling of invalid YAML configuration"""
        invalid_yaml_path = Path(self.temp_dir.name) / "invalid_config.yaml"
        with open(invalid_yaml_path, "w") as f:
            f.write("""
            This is not a valid YAML configuration file.
            It's missing proper structure and required fields.
            """)
        
        # Should raise appropriate exception
        with self.assertRaises(Exception):
            RAGBuilder.from_yaml(str(invalid_yaml_path))
    
    def test_malformed_yaml_config(self):
        """Test handling of malformed YAML configuration"""
        malformed_yaml_path = Path(self.temp_dir.name) / "malformed_config.yaml"
        with open(malformed_yaml_path, "w") as f:
            f.write("""
            data_ingest:
              input_source: {}
              # Missing required fields
              document_loaders:
                - invalid_type: something
            """.format(str(self.test_file_path)))
        
        # Should raise appropriate exception
        with self.assertRaises(Exception):
            RAGBuilder.from_yaml(str(malformed_yaml_path))
    
    def test_invoke_without_optimization(self):
        """Test invoking RAG without optimization"""
        builder = RAGBuilder.from_source_with_defaults(
            input_source=str(self.test_file_path),
            default_embeddings=self.default_embeddings,
            default_llm=self.default_llm,
            log_config=self.log_config
        )
        
        # Check that optimization_results is properly initialized
        with self.assertRaises(Exception):
            # Should raise exception because no optimizations have been run
            builder.optimization_results.invoke("What is this document about?")
    
#     def test_invalid_testset_format(self):
#         """Test handling of invalid test dataset format"""
#         # Create an invalid test dataset (missing required columns)
#         invalid_dataset = Path(self.temp_dir.name) / "invalid_testset.csv"
#         with open(invalid_dataset, "w") as f:
#             f.write("""question,answer
# "What is this?","This is a test"
# """)  # Missing user_input and reference columns
        
#         # Try to initialize with invalid test dataset
#         builder = RAGBuilder.from_source_with_defaults(
#             input_source=str(self.test_file_path),
#             test_dataset=str(invalid_dataset),
#             default_embeddings=self.default_embeddings,
#             default_llm=self.default_llm,
#             log_config=self.log_config
#         )
        
#         # Should handle invalid dataset format gracefully
#         with self.assertRaises(Exception) as context:
#             builder.optimize_data_ingest()
        
#         # Check error message mentions dataset format
#         self.assertTrue(any(keyword in str(context.exception).lower() 
#                       for keyword in ["dataset", "format", "column"]))
    
    def test_missing_environment_variables(self):
        """Test handling of missing environment variables"""
        # Temporarily unset any LLM API keys
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "",
            "MISTRAL_API_KEY": "",
            "AZURE_OPENAI_API_KEY": ""
        }, clear=True):
            # Try to use an LLM that requires API keys
            openai_llm = LLMConfig(
                type=LLMType.OPENAI,
                model_kwargs={
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.1
                }
            )
            
            builder = RAGBuilder.from_source_with_defaults(
                input_source=str(self.md_dir_path),
                test_dataset=str(self.md_test_dataset),
                default_embeddings=self.default_embeddings,
                default_llm=openai_llm,
                log_config=self.log_config
            )
            
            # Should fail with appropriate error about missing API keys
            with self.assertRaises(Exception) as context:
                builder.optimize_data_ingest()
                
            # Check error message mentions Missing environment variables
            self.assertIn("missing", str(context.exception).lower())


if __name__ == "__main__":
    unittest.main() 