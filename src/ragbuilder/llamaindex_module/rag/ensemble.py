from llama_index.core import SummaryIndex
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.schema import IndexNode
from langchain_module.common import setup_logging
import logging
setup_logging()
logger = logging.getLogger("ragbuilder")
def EnsembleRetriever(merge_retrievers,kwargs):
    logger.info("EnsembleRetriever loaded")
    nodes=[]
    index_node =[]
    retriever_nodes = []
    merge_retrievers_dict = {}
    for i in range(0,len(merge_retrievers)):
        merge_retrievers_dict["index_id_"+str(i)]=merge_retrievers[i]
        nodes.append(merge_retrievers[i].retrieve(kwargs['prompt_text']))
        index_node.append(IndexNode(
            text=(
                "Retrieves relevant context using retrievers "+str(i)
            ),
            index_id="index_id_"+str(i),
        ))
        retriever_nodes.append(index_node[i])
    summary_index = SummaryIndex(retriever_nodes)
    recursive_retriever = RecursiveRetriever(
        root_id="root",
        retriever_dict={ "root": summary_index.as_retriever(), **merge_retrievers_dict},)
    logger.info("EnsembleRetriever completed")
    return recursive_retriever