
from llama_index.core.postprocessor import LongContextReorder
from llama_index.postprocessor.longllmlingua import LongLLMLinguaPostprocessor
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer
from llamaindex_module.embedding_model.embedding_model import *
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.jinaai_rerank import JinaRerank
import os
from llama_index.core.postprocessor import LLMRerank
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llamaindex_module.llms.llm import * 
from langchain_module.common import setup_logging
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def getPostProcessors(**kwargs):
    postprocessors = []
    logger.info("getPostProcessors loaded")
    post_processors_args=kwargs['post_processors_args']
    if 'LongContextReorder' in post_processors_args['post_processors']:
        logger.info("LongContextReorder loaded")
        postprocessors.append(LongContextReorder())
    if 'SimilarityPostprocessor' in post_processors_args['post_processors']:
        logger.info("SimilarityPostprocessor loaded")
        postprocessors.append(SimilarityPostprocessor(similarity_cutoff=post_processors_args['SimilarityPostprocessor']['similarity_cutoff']))
    if 'SentenceEmbeddingOptimizer' in post_processors_args['post_processors']:
        logger.info("SentenceEmbeddingOptimizer loaded")
        postprocessors.append(SentenceEmbeddingOptimizer(
            embed_model=getEmbedding(**kwargs),
            percentile_cutoff=post_processors_args['SentenceEmbeddingOptimizer']['percentile_cutoff'],
            threshold_cutoff=post_processors_args['SentenceEmbeddingOptimizer']['threshold_cutoff']))
    if 'CohereRerank' in post_processors_args['post_processors']:
        logger.info("CohereRerank loaded")
        postprocessors.append(CohereRerank(
        top_n=post_processors_args['CohereRerank']['top_n'], model="rerank-english-v2.0"))
    if 'JinaRerank' in post_processors_args['post_processors']:
        logger.info("JinaRerank loaded")
        postprocessors.append(JinaRerank(
        top_n=post_processors_args['JinaRerank']['top_n'], model="jina-reranker-v1-base-en", api_key=os.environ['JINA_API_KEY']))
    if 'LLMRerank' in post_processors_args['post_processors']:
        logger.info("LLMRerank loaded")
        postprocessors.append(LLMRerank(
            choice_batch_size=post_processors_args['LLMRerank']['choice_batch_size'],
            top_n=post_processors_args['LLMRerank']['top_n'],
        ))
    if 'RankGPTRerank' in post_processors_args['post_processors']:
        logger.info("RankGPTRerank loaded")
        postprocessors.append(RankGPTRerank(top_n=post_processors_args['RankGPTRerank']['top_n'], llm=getLLM(**kwargs)))
    return postprocessors
