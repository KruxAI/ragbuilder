from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import Optional, Dict, Any, List
from datetime import timedelta

class ModuleResults(BaseModel):
    # Core results
    best_config: Any
    best_score: float
    best_pipeline: Any
    
    # Study statistics
    n_trials: int
    completed_trials: int
    optimization_time: float
    
    # Performance metrics
    avg_latency: Optional[float] = None
    error_rate: Optional[float] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "n_trials": self.n_trials,
            "completed_trials": self.completed_trials,
            "optimization_time": self.optimization_time,
            "metrics": {
                "best_score": self.best_score,
                "avg_latency": self.avg_latency,
                "error_rate": self.error_rate
            }
        }

class DataIngestResults(ModuleResults):
    best_index: Any  # VectorStore

    def get_vectorstore(self):
        return self.best_index

    def get_config_summary(self) -> Dict[str, Any]:
        # Get embedding type and model info
        if self.best_config.embedding_model.type:
            embedding_type = self.best_config.embedding_model.type
        elif hasattr(self.best_config.embedding_model._initialized_embedding, "__class__"):
            embedding_type = self.best_config.embedding_model._initialized_embedding.__class__.__name__
        else:
            embedding_type = "unknown"

        # Try to get model name from different possible sources
        model_name = None
        if self.best_config.embedding_model.model_kwargs:
            model_name = (self.best_config.embedding_model.model_kwargs.get('model') 
                        or self.best_config.embedding_model.model_kwargs.get('model_name'))
        elif hasattr(self.best_config.embedding_model._initialized_embedding, "model_name"):
            model_name = self.best_config.embedding_model._initialized_embedding.model_name
        elif hasattr(self.best_config.embedding_model._initialized_embedding, "model"):
            model_name = self.best_config.embedding_model._initialized_embedding.model
        
        embedding_model = f"{embedding_type}:{model_name}" if model_name else str(embedding_type)
        
        
        return {
            "document_loader": self.best_config.document_loader.type.value,
            "chunking_strategy": self.best_config.chunking_strategy.type.value,
            "chunk_size": self.best_config.chunk_size,
            "chunk_overlap": self.best_config.chunk_overlap,
            "embedding_model": embedding_model,
            "vector_database": self.best_config.vector_database.type.value
        }

class RetrievalResults(ModuleResults):
    def invoke(self, query: str, **kwargs) -> List[Any]:
        return self.best_pipeline.retriever_chain.invoke(query, **kwargs)

    def get_config_summary(self) -> Dict[str, Any]:
        return {
            "retrievers": [r.type.value for r in self.best_config.retrievers],
            "top_k": self.best_config.top_k,
            "rerankers": [r.type.value for r in self.best_config.rerankers] if self.best_config.rerankers else []
        }

class GenerationResults(ModuleResults):
    best_prompt: str
    
    def invoke(self, question: str, **kwargs) -> str:
        return self.best_pipeline.invoke(question, **kwargs)

    def get_config_summary(self) -> Dict[str, Any]:
        # Get LLM type and model info
        if self.best_config.llm.type:
            model_type = self.best_config.llm.type
        elif hasattr(self.best_config.llm._initialized_llm, "__class__"):
            model_type = self.best_config.llm._initialized_llm.__class__.__name__
        else:
            model_type = "unknown"

        # Try to get model name and temperature from different possible sources
        model_name = None
        temperature = None
        
        if self.best_config.llm.model_kwargs:
            model_name = (self.best_config.llm.model_kwargs.get('model') 
                         or self.best_config.llm.model_kwargs.get('model_name'))
            temperature = self.best_config.llm.model_kwargs.get("temperature")
        elif hasattr(self.best_config.llm._initialized_llm, "model_name"):
            model_name = self.best_config.llm._initialized_llm.model_name
            if hasattr(self.best_config.llm._initialized_llm, "temperature"):
                temperature = self.best_config.llm._initialized_llm.temperature
        elif hasattr(self.best_config.llm._initialized_llm, "model"):
            model_name = self.best_config.llm._initialized_llm.model
            if hasattr(self.best_config.llm._initialized_llm, "temperature"):
                temperature = self.best_config.llm._initialized_llm.temperature

        model = f"{model_type}:{model_name}" if model_name else str(model_type)

        return {
            "model": model,
            "temperature": temperature,
            "prompt_template": self.best_prompt
        }

class OptimizationResults(BaseModel):
    data_ingest: Optional[DataIngestResults] = None
    retrieval: Optional[RetrievalResults] = None
    generation: Optional[GenerationResults] = None

    class Config:
        arbitrary_types_allowed = True

    def summary(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive summary of optimization results"""
        summary = {}
        for module in ['data_ingest', 'retrieval', 'generation']:
            if result := getattr(self, module):
                summary[module] = {
                    "score": result.best_score,
                    "optimization_time": result.optimization_time,
                    "config": result.get_config_summary(),
                    "metrics": {
                        "avg_latency": result.avg_latency,
                        "error_rate": result.error_rate
                    }
                }
        return summary

    def invoke(self, question: str) -> Dict[str, Any]:
        """Run complete RAG pipeline with optimized components"""
        if not all([self.retrieval, self.generation]):
            raise ValueError("Both retrieval and generation optimization required for querying")

        retrieved_docs = self.retrieval.invoke(question)
        result = self.generation.invoke(question)

        return {
            "question": question,
            "answer": result['answer'],
            "context": result['context'],
            "retrieved_documents": retrieved_docs
        } 
    
    def __repr__(self):
        import json
        return json.dumps(self.summary(), indent=4)