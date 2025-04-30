"""
Minimal test to verify RAGBuilder can be imported properly
"""

import unittest

class TestRAGBuilderImport(unittest.TestCase):
    """Test that RAGBuilder can be imported correctly"""
    
    def test_import(self):
        """Test importing RAGBuilder"""
        try:
            from ragbuilder import RAGBuilder
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import RAGBuilder: {e}")
    
    def test_config_import(self):
        """Test importing configuration modules"""
        try:
            from ragbuilder.config.base import LogConfig, LLMConfig, EmbeddingConfig
            from ragbuilder.config.components import LLMType, EmbeddingType
            self.assertTrue(True)  # If we get here, import succeeded
        except ImportError as e:
            self.fail(f"Failed to import configuration modules: {e}")

if __name__ == "__main__":
    unittest.main() 