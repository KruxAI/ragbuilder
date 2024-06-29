from ragbuilder.router import router as rtr
configs= {
    'framework': 'langchain',
    'description': 'Configuration for a LangChain-based retrieval system',
    'retrieval_model': 'gpt-3.5-turbo',
    'chunking_kwargs': {
            'chunk_strategy': 'RecursiveCharacterTextSplitter',
            'chunk_size': 1000,
            'chunk_overlap': 200
        },
    'loader_kwargs': {'input_path':'https://ashwinaravind.github.io/'},
    'vectorDB_kwargs': {'vectorDB': 'chromaDB'},
    'embedding_kwargs': { 'embedding_model': 'text-embedding-3-large'},
    'retriever_kwargs': {'retrievers': 
                        [{'retriever_type': 'vectorSimilarity', 'search_type': 'similarity', 'search_kwargs': {'k': 5}}], 
                        'contextual_compression_retriever': False, 
                        'compressors': []}}

print(rtr(configs))

configs= {
    'framework': 'langchain_byor',
    'loader_kwargs': {'input_path':'/Users/ashwinaravind/Desktop/kruxgitrepo/ragbuilder/src/ragbuilder/langchain_module/rag/test55.py'}
    }
print(rtr(configs))