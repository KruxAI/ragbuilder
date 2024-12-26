![RagBuilder logo](./assets/ragbuilder_dark.png#gh-dark-mode-only)
![RagBuilder logo](./assets/ragbuilder_light.png#gh-light-mode-only)

# 

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![GitHub release](https://img.shields.io/github/release/KruxAI/ragbuilder.svg)](https://github.com/KruxAI/ragbuilder/releases/)
[![GitHub license](https://badgen.net/github/license/KruxAI/ragbuilder)](https://github.com/KruxAI/ragbuilder/blob/master/LICENSE)
[![GitHub commits](https://badgen.net/github/commits/KruxAI/ragbuilder)](https://github.com/KruxAI/ragbuilder/commit/)


![11926](https://github.com/user-attachments/assets/af9e241a-b648-4b2f-ab2a-3c268c7f1ca8)

RagBuilder is a toolkit that helps you create optimal Production-ready Retrieval-Augmented-Generation (RAG) setup for your data automatically. By performing hyperparameter tuning on various RAG parameters (Eg: chunking strategy: semantic, character etc., chunk size: 1000, 2000 etc.), RagBuilder evaluates these configurations against a test dataset to identify the best-performing setup for your data. Additionally, RagBuilder includes several state-of-the-art, pre-defined RAG templates that have shown strong performance across diverse datasets. So just bring your data, and RagBuilder will generate a production-grade RAG setup in just minutes.


## Features

- **Hyperparameter Tuning**: Efficiently optimize your RAG configurations using Bayesian optimization
- **Pre-defined RAG Templates**: Use state-of-the-art templates that have demonstrated strong performance Eg: Graph retriever, Contextual chunker etc.)
- **Evaluation Dataset Options**: Generate synthetic test dataset or provide your own
- **Component Access**: Direct access to vectorstore, retriever, and generator components
- **API Deployment**: Easily deploy as an API service
- **Project Persistence**: Save and load optimized RAG pipelines


## Installation

```bash
# Create a new venv
uv venv ragbuilder

# Activate the new venv
source ragbuilder/bin/activate

# Install
uv pip install ragbuilder
```

See other installation options here ([link](https://docs.ragbuilder.io/quickstart/#installation))

## Quick Start

```python
from ragbuilder import RAGBuilder

# Initialize and optimize
builder = RAGBuilder.from_source_with_defaults(input_source='https://lilianweng.github.io/posts/2023-06-23-agent/')
results = builder.optimize()

# Run a query through the complete pipeline
response = results.invoke("What is HNSW?")

# View optimization summary
print(results.summary())
```

## Configuration Guide

### Basic Configuration
For most use cases, the default configuration provides good results:

```python
builder = RAGBuilder.from_source_with_defaults(
    input_source='path/to/your/data',
    test_dataset='path/to/test/data'  # Optional
)
```

## Advanced Configuration

For fine-grained control over your RAG pipeline, you can customize every aspect:

````python
from ragbuilder.config import (
    DataIngestOptionsConfig,
    RetrievalOptionsConfig,
    GenerationOptionsConfig
)

# Configure data ingestion
data_ingest_config = DataIngestOptionsConfig(
    input_source="data.pdf",
    document_loaders=[
        {"type": "pymupdf"},
        {"type": "unstructured"}
    ],
    chunking_strategies=[{
        "type": "RecursiveCharacterTextSplitter",
        "chunker_kwargs": {"separators": ["\n\n", "\n", " ", ""]}
    }],
    chunk_size={"min": 500, "max": 2000, "stepsize": 500},
    embedding_models=[{
        "type": "openai",
        "model_kwargs": {"model": "text-embedding-3-large"}
    }]
)

# Configure retrieval
retrieval_config = RetrievalOptionsConfig(
    retrievers=[
        {
            "type": "vector_similarity",
            "retriever_k": [20],
            "weight": 0.5
        },
        {
            "type": "bm25",
            "retriever_k": [20],
            "weight": 0.5
        }
    ],
    rerankers=[{
        "type": "BAAI/bge-reranker-base"
    }],
    top_k=[3, 5]
)

# Initialize with custom configs
builder = RAGBuilder(
    data_ingest_config=data_ingest_config,
    retrieval_config=retrieval_config
)

# Access individual components
vectorstore = results.data_ingest.get_vectorstore()
docs = results.retrieval.invoke("What is RAG?")
answer = results.generation.invoke("What is RAG?")
````

## Component Reference

### Document Loaders
- `pymupdf`: Optimized for PDFs
- `unstructured`: General-purpose loader
- `pypdf`: Alternative PDF loader
- `bs4`: Web page loader
- Custom loaders via `custom_class`

### Chunking Strategies
- `RecursiveCharacterTextSplitter`: Recursive character text splitter
- `CharacterTextSplitter`: Character text splitter
- `MarkdownHeaderTextSplitter`: Markdown-header based splitter
- `HTMLHeaderTextSplitter`: HTML-header based splitter
- `SemanticChunker`: Semantic chunker
- `TokenTextSplitter`: Token-based splitter
- Custom splitters via `custom_class`


### Retrievers
- `vector_similarity`: Vector similarity search
- `vector_mmr`: Vector MMR search
- `bm25`: Keyword-based search using BM25
- `multi_query`: Multi-query retrievers
- `parent_doc_full`: Parent document full-doc retrieval
- `parent_doc_large`: Parent document large-chunks retrieval
- `graph`: Graph-based retrieval (requires Neo4j)
- Custom retrievers via `custom_class`

### Rerankers
- `BAAI/bge-reranker-base`: BGE base reranker
- `mixedbread-ai/mxbai-rerank-base-v1`: mxbai reranker base v1
- `mixedbread-ai/mxbai-rerank-large-v1`: mxbai reranker large v1
- `cohere`: Cohere's reranking model
- `jina`: Jina reranker
- `flashrank`: Flaskrank reranker
- `rankllm`: RankLLM reranker
- `colbert`: Colbert reranker
- Custom rerankers via `custom_class`


## Environment Variables

Create a `.env` file in your project directory:

````env
# Required
OPENAI_API_KEY=your_key_here

# Optional - For additional features
MISTRAL_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here

# For Graph-based RAG
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
````

## Advanced Topics

### Custom Evaluation Metrics
```python
from ragbuilder import EvaluationConfig

config = EvaluationConfig(
    type="custom",
    custom_class="your_module.CustomEvaluator",
    evaluator_kwargs={
        "metrics": ["precision", "recall", "f1_score"]
    }
)
```

### Optimization Configuration
Fine-tune the optimization parameters:
```python
from ragbuilder import OptimizationConfig

config = OptimizationConfig(
    n_trials=20,
    n_jobs=1,
    study_name="my_optimization",
    optimization_direction="maximize"
)
```

## API Deployment

RAGBuilder can be deployed as an API service:

````python
# Initialize and optimize
builder = RAGBuilder.from_source_with_defaults('data.pdf')
results = builder.optimize()

# Deploy as API
builder.serve(host="0.0.0.0", port=8000)
````

Access via:
- `POST /query` - Run queries through the RAG pipeline

## Project Management

Save and load optimized RAG pipelines:

````python
# Save project
builder.save('rag_project/')

# Load existing project
builder = RAGBuilder.load('rag_project/')

# Access components
vectorstore = builder.data_ingest.get_vectorstore()
retriever = builder.retrieval.get_retriever()
generator = builder.generation.get_generator()
````

## Best Practices

1. **Start Simple**
   - Begin with `from_source_with_defaults()`
   - Add complexity only when needed

2. **Test Data Quality**
   - Provide representative test queries
   - Use domain-specific evaluation metrics

3. **Resource Management**
   - Monitor memory usage with large datasets
   - Use chunking for large documents

4. **Production Deployment**
   - Save optimized projects for reuse
   - Monitor API performance metrics
   - Implement rate limiting for API endpoints

## Usage Analytics

We collect anonymous usage metrics to improve RAGBuilder:
- Number of optimization runs
- Success/failure rates
- No personal or business data is collected

To opt-out set `ENABLE_ANALYTICS=False` in `.env`:

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
