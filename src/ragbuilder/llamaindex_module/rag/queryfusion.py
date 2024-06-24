from llama_index.core.retrievers import QueryFusionRetriever
from langchain_module.common import setup_logging
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def QueryFusion(merge_retrievers,kwargs):
    logger.info("QueryFusionRetriever loaded")
    retriever = QueryFusionRetriever(
                    merge_retrievers,
                    similarity_top_k=kwargs['retriever_kwargs']['QueryFusionRetreiver']['similarity_top_k'],
                    num_queries= kwargs['retriever_kwargs']['QueryFusionRetreiver'].get('num_queries',1),  # set this to 1 to disable query generation,
                    verbose=False
                    )
    logger.info("QueryFusionRetriever completed")
    return retriever