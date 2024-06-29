import logging
import itertools
import time
from ragbuilder.langchain_module.common import setup_logging

setup_logging()
logger = logging.getLogger("ragbuilder")

def nuancedCombos(vectorDB, exclude_elements=None):
    logger.info(f"RAG Builder: Generating combinations...")

    if exclude_elements is None:
        exclude_elements = []

    # Define the arrays
    arr_chunking_strategy = ['RecursiveCharacterTextSplitter','CharacterTextSplitter','SemanticChunker','MarkdownHeaderTextSplitter','HTMLHeaderTextSplitter']
    arr_chunk_size = [1000, 2000, 3000]
    arr_embedding_model = ['text-embedding-3-small','text-embedding-3-large','text-embedding-ada-002']
    arr_retriever = ['vectorSimilarity', 'vectorMMR','bm25Retriever','multiQuery','parentDocFullDoc','parentDocLargeChunk']
    arr_llm = ['gpt-3.5-turbo','gpt-4o','gpt-4-turbo']
    arr_contextual_compression = [True, False]
    arr_compressors = ["EmbeddingsRedundantFilter", "EmbeddingsClusteringFilter", "LLMChainFilter", "LongContextReorder", "CrossEncoderReranker"]
    arr_search_kwargs = [{"k": 5}, {"k": 10}, {"k": 20}]
    chunk_overlap = 200
    no_chunk_req_loaders = ['SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter']
    
    # Filter out excluded elements from arrays
    arr_chunking_strategy = [elem for elem in arr_chunking_strategy if elem not in exclude_elements]
    arr_embedding_model = [elem for elem in arr_embedding_model if elem not in exclude_elements]
    arr_retriever = [elem for elem in arr_retriever if elem not in exclude_elements]
    arr_llm = [elem for elem in arr_llm if elem not in exclude_elements]
    arr_compressors = [elem for elem in arr_compressors if elem not in exclude_elements and 'contextualCompression' not in exclude_elements]

    # Handle search kwargs exclusion
    arr_search_kwargs = []
    if 'search_k_5' not in exclude_elements:
        arr_search_kwargs.append({"k": 5})
    if 'search_k_10' not in exclude_elements:
        arr_search_kwargs.append({"k": 10})
    if 'search_k_20' not in exclude_elements:
        arr_search_kwargs.append({"k": 20})

    # Handle Chunking strategy exclusion
    arr_chunk_size = []
    if 'chunk1000' not in exclude_elements:
        arr_chunk_size.append(1000)
    if 'chunk2000' not in exclude_elements:
        arr_chunk_size.append(2000)
    if 'chunk3000' not in exclude_elements:
        arr_chunk_size.append(3000)
    
    # Handle contextual compression exclusion
    if 'contextualCompression' in exclude_elements:
        arr_contextual_compression = [False]
    else:
        arr_contextual_compression = [True, False]

    # Generate combinations
    combinations = []
    for combination in itertools.product(
        arr_chunking_strategy, 
        arr_embedding_model, 
        arr_llm, 
        arr_contextual_compression
    ):
        if combination[0] in no_chunk_req_loaders:
            combinations.append(combination + ([],))
        else:
            for chunk_size in arr_chunk_size:
                combinations.append(combination + ([chunk_size],))
    
    all_combinations = []
    for combination in combinations:
        for retrievers in itertools.combinations(arr_retriever, 1):
            if arr_compressors:
                for compressors in itertools.combinations(arr_compressors, 1):
                    for search_kwargs in arr_search_kwargs:
                        all_combinations.append(combination + (list(retrievers), list(compressors), search_kwargs))
                    for compressors in itertools.combinations(arr_compressors, 2):
                        for search_kwargs in arr_search_kwargs:
                            all_combinations.append(combination + (list(retrievers), list(compressors), search_kwargs))
            else:
                for search_kwargs in arr_search_kwargs:
                    all_combinations.append(combination + (list(retrievers), [], search_kwargs))
        for retrievers in itertools.combinations(arr_retriever, 2):
            if arr_compressors:
                for compressors in itertools.combinations(arr_compressors, 1):
                    for search_kwargs in arr_search_kwargs:
                        all_combinations.append(combination + (list(retrievers), list(compressors), search_kwargs))
                    for compressors in itertools.combinations(arr_compressors, 2):
                        for search_kwargs in arr_search_kwargs:
                            all_combinations.append(combination + (list(retrievers), list(compressors), search_kwargs))
            else:
                for search_kwargs in arr_search_kwargs:
                    all_combinations.append(combination + (list(retrievers), [], search_kwargs))
    
    combination_configs = {}
    start_index = int(time.time())

    for comb in all_combinations:
        chunk_strategy = comb[0]
        embedding_model = comb[1]
        retrieval_model = comb[2]
        contextual_compression = comb[3]
        chunk_size = comb[4]
        retrievers = comb[5]
        compressors = comb[6]
        search_kwargs = comb[7]
        
        retriever_kwargs = {'retrievers': []}
        for retriever in retrievers:
            if retriever == 'vectorSimilarity':
                retriever_entry = {'retriever_type': 'vectorSimilarity', 'search_type': 'similarity', 'search_kwargs': search_kwargs}
            elif retriever == 'vectorMMR':
                retriever_entry = {'retriever_type': 'vectorMMR', 'search_type': 'mmr', 'search_kwargs': search_kwargs}
            else:
                retriever_entry = {'retriever_type': retriever, 'search_type': 'similarity', 'search_kwargs': search_kwargs}
            retriever_kwargs['retrievers'].append(retriever_entry)
        
        chunking_kwargs = {}
        if chunk_strategy not in no_chunk_req_loaders:
            chunking_kwargs = {
                'chunk_strategy': chunk_strategy,
                'chunk_size': chunk_size[0] if chunk_size else None,
                'chunk_overlap': chunk_overlap
            }
        else:
            chunking_kwargs = {
                'chunk_strategy': chunk_strategy
            }
        
        retriever_kwargs["contextual_compression_retriever"] = contextual_compression

        if contextual_compression:
            retriever_kwargs["document_compressor_pipeline"] = compressors
            if "EmbeddingsClusteringFilter" in compressors:
                retriever_kwargs["EmbeddingsClusteringFilter_kwargs"] = {
                    "embeddings": embedding_model, 
                    "num_clusters": 4, 
                    "num_closest": 1, 
                    "sorted": True
                }

        combination_configs[start_index] = {
            'framework': 'langchain',
            'chunking_kwargs': {
                'chunk_strategy': chunking_kwargs['chunk_strategy'],
                'chunk_size': chunking_kwargs.get('chunk_size', None),
                'chunk_overlap': chunk_overlap
            },
            'vectorDB_kwargs' : {'vectorDB': vectorDB},
            'embedding_kwargs': {'embedding_model': embedding_model},
            'retrieval_model': retrieval_model,
            'retriever_kwargs': retriever_kwargs,
            'compressors': compressors if contextual_compression else []
        }
        start_index += 1

    logger.info(f"RAG Builder: Number of RAG combinations : {len(combination_configs)}")
    return combination_configs

# Example usage
# e=["EmbeddingsRedundantFilter", "EmbeddingsClusteringFilter", "LLMChainFilter", "CrossEncoderReranker",'compareTemplates', 'generateSyntheticData', 'gpt-4o', 'gpt-4-turbo', 'search_k_10', 'search_k_20', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'text-embedding-3-small', 'text-embedding-ada-002', 'chunk2000', 'chunk3000', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'SemanticChunker', 'CharacterTextSplitter']
# print(nuancedCombos('chromaDB', e))

combs=[['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk3000', 'chunk1000', 'text-embedding-3-large', 'text-embedding-ada-002', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_20', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk3000', 'text-embedding-3-small', 'text-embedding-3-large', 'vectorSimilarity', 'vectorMMR', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_20', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'SemanticChunker', 'HTMLHeaderTextSplitter', 'chunk3000', 'chunk1000', 'text-embedding-3-large', 'text-embedding-ada-002', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_20', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk3000', 'text-embedding-3-small', 'text-embedding-3-large', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'search_k_10', 'search_k_20', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo']]
# for c in combs:
#     print(len(nuancedCombos('chromaDB', c)))

advanced_combos=[['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder','gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo']]
# for c in advanced_combos:
#     print(len(nuancedCombos('chromaDB', c)))
    # print(nuancedCombos('chromaDB', c))
 
 