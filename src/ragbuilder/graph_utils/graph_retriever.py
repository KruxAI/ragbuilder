from typing import List, Any
from langchain.docstore.document import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from ragbuilder.graph_utils import check_graph_dependencies


class Neo4jGraphRetriever(BaseRetriever):
    """Custom retriever that uses Neo4j graph database for retrieval.
    
    Args:
        graph: Neo4j graph instance
        top_k: Number of documents to retrieve
        max_hops: Maximum number of hops in graph traversal
        graph_weight: Weight for graph-based scores
        embeddings: Embedding model to embed queries
        index_name: Name of the vector index
    """
    def __init__(self, *args, **kwargs):
        check_graph_dependencies()

    # Define fields directly as class variables
    graph: Any
    top_k: int = 3
    max_hops: int = 2
    max_related_docs_per_doc: int = 3
    graph_weight: float = 0.3
    embeddings: Any
    index_name: str = "document_embeddings"

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    query_template: str = f"""
    // First find similar documents using vector index
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node as doc, score as vector_score
    
    // Get entities mentioned in these documents
    MATCH (doc)-[:MENTIONS]->(entity)
    
    // Find other documents through graph traversal (controlled by max_hops)
    MATCH path = (entity)-[r*1..{max_hops}]-(related)
    MATCH (other_doc:Document)-[:MENTIONS]->(related)
    WHERE other_doc <> doc  // Exclude original documents
    
    // Calculate graph-based score
    WITH doc, vector_score, other_doc, entity, related, r, path,
         // Score based on path length (shorter paths score higher)
         1.0 / length(path) as distance_score,
         // Score based on number of shared entities
         size([(other_doc)-[:MENTIONS]->(e) WHERE e IN nodes(path) | e]) as shared_entities
    
    // Return both vector-similar and graph-related documents with scores
    RETURN 
        doc.text as primary_text,
        doc.source as primary_source,
        vector_score,
        collect(DISTINCT {{
            doc_text: other_doc.text,
            doc_source: other_doc.source,
            graph_score: (distance_score * 0.6 + shared_entities * 0.4),
            connection: {{
                from_entity: entity.name,
                from_type: labels(entity)[0],
                path_length: length(path),
                to_entity: related.name,
                to_type: labels(related)[0],
                shared_count: shared_entities
            }}
        }}) as related_docs
    ORDER BY vector_score DESC
    LIMIT $top_k
    """

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.graph.query(
                self.query_template,
                {
                    "index_name": self.index_name,
                    "embedding": query_embedding,
                    "top_k": self.top_k
                }
            )
            
            documents = []
            for result in results:
                # Add the primary document (from vector search)
                primary_content = f"[Vector Search Result - Score: {result['vector_score']:.3f}]\n"
                primary_content += f"{result['primary_text']}\n\nRelated Documents:\n"

                seen_docs = set()
                unique_related_docs = []
                connection_info = {} 
                connection_count = 0
                
                for rel_doc in sorted(
                    result.get('related_docs', []),
                    key=lambda x: x['graph_score'],
                    reverse=True
                ):
                    doc_text = rel_doc['doc_text']
                    if doc_text not in seen_docs:
                        seen_docs.add(doc_text)
                        unique_related_docs.append(rel_doc)
                        connection_info[doc_text] = [rel_doc['connection']]
                        connection_count += 1
                        if connection_count >= self.max_related_docs_per_doc:
                            break
                    else:
                        # Add this connection info to the existing document
                        connection_info[doc_text].append(rel_doc['connection'])
                
                # Add top related documents found through graph connections
                for rel_doc in unique_related_docs:
                    doc_text = rel_doc['doc_text']
                    # Combined score = weighted average of vector and graph scores
                    combined_score = (
                        (1 - self.graph_weight) * result['vector_score'] +
                        self.graph_weight * rel_doc['graph_score']
                    )
                    
                    primary_content += f"\n[Graph-Connected Document - Score: {combined_score:.3f}]\n"
                    primary_content += "Connection Paths:\n"
                    for connection in connection_info[doc_text]:
                        primary_content += f"- {connection['from_type']} '{connection['from_entity']}' → "
                        primary_content += f"{connection['to_type']} '{connection['to_entity']}' "
                        primary_content += f"({connection['path_length']} hops, "
                        primary_content += f"{connection['shared_count']} shared entities)\n"
                    
                    primary_content += f"Document text: {doc_text}\n"
                
                documents.append(
                    Document(
                        page_content=primary_content,
                        metadata={
                            "source": result.get("primary_source"),
                            "vector_score": result.get("vector_score"),
                            "type": "primary"
                        }
                    )
                )
            
            return documents
            
        except Exception as e:
            print(f"Graph retrieval failed: {str(e)}")
            return []
