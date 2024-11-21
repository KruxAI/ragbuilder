from typing import Optional, Any
from ragbuilder.config.base import RAGConfig, BaseConfig
from ragbuilder.config.data_ingest import DataIngestOptionsConfig 
from ragbuilder.config.retriever import RetrievalOptionsConfig
from ragbuilder.data_ingest.optimization import run_optimization_from_dict
from .exceptions import DependencyError
import logging

class RAGBuilder:
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig(base=BaseConfig())
        self.logger = logging.getLogger("ragbuilder")
        self._optimized_store = None
        self._optimized_retriever = None

    @classmethod
    def from_yaml(cls, file_path: str) -> 'RAGBuilder':
        """Create RAGBuilder from YAML config"""
        config = RAGConfig.from_yaml(file_path)
        return cls(config)

    def optimize_data_ingest(self, config: Optional[DataIngestOptionsConfig] = None) -> Any:
        """Run data ingestion optimization"""
        if config:
            self.config.data_ingest = config
        elif not self.config.data_ingest:
            self.config.data_ingest = DataIngestOptionsConfig.with_defaults(self.config.base)

        self.logger.info("Starting data ingestion optimization")
        best_config, best_score, best_index = run_optimization_from_dict(
            self.config.data_ingest.model_dump()
        )
        
        self._optimized_store = best_index
        return best_config, best_score, best_index
    
    def optimize_retriever(self, config: RetrievalOptionsConfig):
        if self._optimized_store is None:
            raise DependencyError("Data ingestion must be optimized first")
        
        # TODO: Implement retriever optimization
        pass

    def get_config(self) -> RAGConfig:
        """Get current configuration"""
        return self.config

    def save_config(self, file_path: str) -> None:
        """Save current configuration to YAML"""
        self.config.to_yaml(file_path)
