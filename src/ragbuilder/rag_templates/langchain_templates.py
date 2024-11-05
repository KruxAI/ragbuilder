import logging
import itertools
import time
from skopt.space import Categorical, Integer
from functools import reduce
from ragbuilder.langchain_module.common import setup_logging, progress_state

setup_logging()
logger = logging.getLogger("ragbuilder")

MAX_MULTI_RETRIEVER_COMBOS=4
MAX_MULTI_COMPRESSOR_COMBOS=4
# RAG Config parameter option values
arr_chunking_strategy = ['RecursiveCharacterTextSplitter','CharacterTextSplitter','SemanticChunker','MarkdownHeaderTextSplitter','HTMLHeaderTextSplitter']
arr_chunk_size = [1000, 2000, 3000]
arr_embedding_model = ['OpenAI:text-embedding-3-small','OpenAI:text-embedding-3-large','OpenAI:text-embedding-ada-002']
retriever_combinations = arr_retriever = ['vectorSimilarity', 'vectorMMR','bm25Retriever','multiQuery','parentDocFullDoc','parentDocLargeChunk','colbertRetriever']
arr_baseline_retrievers = ['vectorSimilarity', 'bm25Retriever']
arr_llm = ['OpenAI:gpt-4o-mini','OpenAI:gpt-4o','OpenAI:gpt-3.5-turbo','OpenAI:gpt-4-turbo']
arr_contextual_compression = [True, False]
compressor_combinations = arr_compressors = [
    "None",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "mixedbread-ai/mxbai-rerank-large-v1",
    "BAAI/bge-reranker-base",
    "flashrank",
    "cohere",
    "jina",
    "colbert",
    "rankllm",
    'EmbeddingsRedundantFilter', 
    'EmbeddingsClusteringFilter', 
    'LLMChainFilter', 
    'LongContextReorder'
]
arr_search_kwargs = ['5', '10', '20']
vectorDB="chromaDB" # Default fallback value

# Other static parameters
chunk_overlap = 200
chunk_step_size=300
no_chunk_req_loaders = ['SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter']
chunk_req_loaders = ['RecursiveCharacterTextSplitter','CharacterTextSplitter']


def init(db='ChromaDB', min=500, max=2000, other_embedding=[], other_llm=[]):
    global arr_chunking_strategy, arr_chunk_size, arr_embedding_model, retriever_combinations, arr_retriever, arr_llm, arr_contextual_compression, \
        arr_baseline_retrievers, compressor_combinations, arr_compressors, arr_search_kwargs, vectorDB
    arr_chunking_strategy = ['RecursiveCharacterTextSplitter','CharacterTextSplitter','SemanticChunker','MarkdownHeaderTextSplitter','HTMLHeaderTextSplitter']
    arr_chunk_size = _get_arr_chunk_size(min, max, step_size=chunk_step_size)
    arr_embedding_model = ['OpenAI:text-embedding-3-small','OpenAI:text-embedding-3-large','OpenAI:text-embedding-ada-002']
    arr_baseline_retrievers = ['vectorSimilarity', 'bm25Retriever']
    arr_retriever = ['vectorSimilarity', 'vectorMMR','bm25Retriever','multiQuery','parentDocFullDoc','parentDocLargeChunk','colbertRetriever']
    retriever_combinations = [retriever for retriever in arr_retriever if retriever not in arr_baseline_retrievers]
    arr_llm = ['OpenAI:gpt-4o-mini','OpenAI:gpt-4o','OpenAI:gpt-3.5-turbo','OpenAI:gpt-4-turbo']
    arr_contextual_compression = [True, False]
    compressor_combinations = arr_compressors = [
        "None",
        "mixedbread-ai/mxbai-rerank-base-v1",
        "mixedbread-ai/mxbai-rerank-large-v1",
        "BAAI/bge-reranker-base",
        "flashrank",
        "cohere",
        "jina",
        "colbert",
        "rankllm",
        'EmbeddingsRedundantFilter', 
        'EmbeddingsClusteringFilter', 
        'LLMChainFilter', 
        'LongContextReorder'
    ]
    arr_search_kwargs = ['5', '10', '20']
    vectorDB = db
    
    if other_embedding:
        arr_embedding_model.extend(other_embedding)

    if other_llm:
        arr_llm.extend(other_llm)
    
    progress_state.reset()

def filter_exclusions(exclude_elements, override_baseline_retrievers=False):
    global arr_chunking_strategy, arr_chunk_size, arr_embedding_model, arr_retriever, arr_baseline_retrievers, retriever_combinations, \
        arr_llm, arr_contextual_compression, arr_compressors, arr_search_kwargs
    
    if exclude_elements is None:
        exclude_elements = []

    # Filter out excluded elements from arrays
    arr_chunking_strategy = [elem for elem in arr_chunking_strategy if elem not in exclude_elements]
    arr_embedding_model = [elem for elem in arr_embedding_model if elem not in exclude_elements]
    arr_retriever = [elem for elem in arr_retriever if elem not in exclude_elements]
    if override_baseline_retrievers:
        arr_baseline_retrievers = [elem for elem in arr_baseline_retrievers if elem not in exclude_elements]
    retriever_combinations = [elem for elem in retriever_combinations if elem not in exclude_elements]
    arr_llm = [elem for elem in arr_llm if elem not in exclude_elements]
    arr_compressors = [elem for elem in arr_compressors if elem not in exclude_elements and 'contextualCompression' not in exclude_elements]

    # Handle search kwargs exclusion
    arr_search_kwargs = []
    if 'search_k_5' not in exclude_elements:
        arr_search_kwargs.append('5')
    if 'search_k_10' not in exclude_elements:
        arr_search_kwargs.append('10')
    if 'search_k_20' not in exclude_elements:
        arr_search_kwargs.append('20')

    # Handle chunk size exclusion
    if set(arr_chunking_strategy).issubset(no_chunk_req_loaders):
        arr_chunk_size=arr_chunk_size[:1] # Trim to 1 value - will be irrelevant since we only have loaders that don't require chunk_size

def count_combos():
    return reduce(lambda x, y: 
                x * len(y), 
                [
                    arr_chunking_strategy, 
                    arr_chunk_size,
                    arr_search_kwargs,
                    arr_embedding_model, 
                    arr_baseline_retrievers,
                    retriever_combinations,
                    compressor_combinations,
                    arr_llm
                ], 1
    )

def _get_arr_chunk_size(min, max, step_size):
    if min==max:
        return [min]
    if max-min<step_size:
        return [min, max]
    chunk_sizes=[]
    for i in range(min, max+1, step_size):
        if max-i<step_size and max-i>25:
            chunk_sizes.extend([i, max])
            break
        chunk_sizes.append(i)
    return chunk_sizes

def _generate_combinations(options):
    combos = options[:]
    for i in range(1, MAX_MULTI_RETRIEVER_COMBOS):
        for item in itertools.combinations(options, i+1):
            multi="|".join(item)
            combos.append(multi)
    return tuple(combos)


def _format_retriever_config(retriever, search_kwargs):
    if retriever == 'vectorMMR':
        retriever_entry = {'retriever_type': 'vectorMMR', 'search_type': 'mmr', 'search_kwargs': search_kwargs}
    else:
        retriever_entry = {'retriever_type': retriever, 'search_type': 'similarity', 'search_kwargs': search_kwargs}
    
    return retriever_entry

def generate_config_for_trial_optuna(trial):
    if len(arr_chunking_strategy) == 1:
        chunking_strategy = arr_chunking_strategy[0]
    else:
        chunking_strategy = trial.suggest_categorical('chunking_strategy', arr_chunking_strategy)
    
    if len(arr_embedding_model) == 1:
        embedding_model = arr_embedding_model[0]
    else:
        embedding_model = trial.suggest_categorical('embedding_model', arr_embedding_model)

    if len(arr_llm) == 1:
        llm = arr_llm[0]
    else:
        llm = trial.suggest_categorical('llm', arr_llm)

    if len(arr_search_kwargs) == 1:
        search_kwargs = arr_search_kwargs[0]
    else:
        search_kwargs = trial.suggest_categorical('search_kwargs', arr_search_kwargs)

    chunking_kwargs = {}
    if chunking_strategy not in no_chunk_req_loaders:
        if len(arr_chunk_size) == 1:
            chunk_size = arr_chunk_size[0]
        else:
            chunk_size = trial.suggest_categorical('chunk_size', arr_chunk_size)
        chunking_kwargs = {
            'chunk_strategy': chunking_strategy,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap
        }
    else:
        chunking_kwargs = {
            'chunk_strategy': chunking_strategy
        }

    # Retriever selection - start with the baseline retrievers
    selected_retrievers = []
    for retriever in arr_baseline_retrievers:
        selected_retrievers.append(retriever)

    # If no baseline retrievers are selected and there is only one retriever combination, select it
    if len(selected_retrievers) == 0 and len(retriever_combinations) == 1:
        selected_retrievers.append(retriever_combinations[0])
    else:
        # Now, select 'n' - the num of retrievers from (MAX_MULTI_RETRIEVER_COMBOS - len(arr_baseline_retrievers)) or len(retriever_combinations) whichever is lesser
        n_retrievers = trial.suggest_int('n_retrievers', 0, min(MAX_MULTI_RETRIEVER_COMBOS - len(arr_baseline_retrievers), len(retriever_combinations)))
        logger.info(f'n_retrievers: {n_retrievers}')

        # Now choose those n retrievers
        for i in range(n_retrievers):
            retriever = trial.suggest_categorical(f'retriever_{i}', retriever_combinations)
            if retriever not in selected_retrievers:
                selected_retrievers.append(retriever)

    retriever_kwargs = {'retrievers': [_format_retriever_config(retriever, search_kwargs) for retriever in selected_retrievers]}
    logger.debug(f'retriever_kwargs: {retriever_kwargs}')

    if any(r["retriever_type"] in ['parentDocFullDoc', 'parentDocLargeChunk'] for r in retriever_kwargs['retrievers']) \
        and chunking_strategy not in chunk_req_loaders:
        chunking_kwargs = {
            'chunk_strategy': 'RecursiveCharacterTextSplitter',
            'chunk_size': arr_chunk_size[0],
            'chunk_overlap': chunk_overlap
        }

    if len(arr_compressors) == 1:
        compressor = 'None'
    elif len(arr_compressors) == 2: 
        compressor = arr_compressors[1] # 0th index is 'None' - pick the only compressor
    else:
        compressor = trial.suggest_categorical('reranker_compressor', arr_compressors)
    
    if compressor == 'None':
        retriever_kwargs["contextual_compression_retriever"] = False
    else:
        retriever_kwargs["contextual_compression_retriever"] = True
        retriever_kwargs["document_compressor_pipeline"] = [compressor]
        if compressor == "EmbeddingsClusteringFilter":
            retriever_kwargs["EmbeddingsClusteringFilter_kwargs"] = {
                "embeddings": embedding_model, 
                "num_clusters": 4, 
                "num_closest": 1, 
                "sorted": True
            }

    config = {
        'framework': 'langchain',
        'chunking_kwargs': chunking_kwargs,
        'vectorDB_kwargs' : {'vectorDB': vectorDB},
        'embedding_kwargs': {'embedding_model': embedding_model},
        'retriever_kwargs': retriever_kwargs,
        'retrieval_model': llm,
        'compressors': [compressor] if compressor != 'None' else []
    }
    return config

def nuancedCombos(exclude_elements=None):
    global arr_compressors
    logger.info(f"Filtering exclusions...")

    filter_exclusions(exclude_elements)

    logger.info(f"RAG Builder: Generating combinations...")    

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
    
    arr_compressors = [compressor for compressor in arr_compressors if compressor not in ['None']]
    all_combinations = []
    for combination in combinations:
        for retrievers in itertools.combinations(arr_retriever, 1):
            chunking_strategy = combination[0]
            if retrievers[0] in ['parentDocFullDoc', 'parentDocLargeChunk'] and chunking_strategy not in chunk_req_loaders:
                continue
            
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
            chunking_strategy = combination[0]
            if any(r in ['parentDocFullDoc', 'parentDocLargeChunk'] for r in retrievers) and chunking_strategy not in chunk_req_loaders:
                continue

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

    # logger.info(f"RAG Builder: Number of RAG combinations : {len(combination_configs)}")
    return combination_configs


# Example usage
# e=["EmbeddingsRedundantFilter", "EmbeddingsClusteringFilter", "LLMChainFilter", "CrossEncoderReranker",'compareTemplates', 'generateSyntheticData', 'gpt-4o', 'gpt-4-turbo', 'search_k_10', 'search_k_20', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'text-embedding-3-small', 'text-embedding-ada-002', 'chunk2000', 'chunk3000', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'SemanticChunker', 'CharacterTextSplitter']
# print(nuancedCombos('chromaDB', e))

# combs=[
# ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk3000', 'chunk1000', 'text-embedding-3-large', 'text-embedding-ada-002', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_20', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk3000', 'text-embedding-3-small', 'text-embedding-3-large', 'vectorSimilarity', 'vectorMMR', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_20', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'SemanticChunker', 'HTMLHeaderTextSplitter', 'chunk3000', 'chunk1000', 'text-embedding-3-large', 'text-embedding-ada-002', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_20', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo'],
# ['CharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'RecursiveCharacterTextSplitter', 'chunk2000', 'chunk3000', 'text-embedding-3-small', 'text-embedding-3-large', 'vectorSimilarity', 'vectorMMR', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'search_k_10', 'search_k_20', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker', 'contextualCompression', 'gpt-4o', 'gpt-4-turbo']]
# for c in combs:
#     print(len(nuancedCombos('chromaDB', c)))

# advanced_combos=[['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder','gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'EmbeddingsClusteringFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsRedundantFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo'],
# ['RecursiveCharacterTextSplitter', 'SemanticChunker', 'MarkdownHeaderTextSplitter', 'HTMLHeaderTextSplitter', 'chunk2000', 'chunk1000', 'text-embedding-3-small', 'text-embedding-ada-002', 'vectorSimilarity', 'bm25Retriever', 'multiQuery', 'parentDocFullDoc', 'parentDocLargeChunk', 'search_k_10', 'search_k_5', 'EmbeddingsClusteringFilter', 'LLMChainFilter', 'LongContextReorder', 'CrossEncoderReranker','gpt-4o', 'gpt-4-turbo']]
# for c in advanced_combos:
#     print(len(nuancedCombos('chromaDB', c)))
    # print(nuancedCombos('chromaDB', c))
 
 