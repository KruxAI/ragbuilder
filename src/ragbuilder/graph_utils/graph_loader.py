import os
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from langchain.pydantic_v1 import Field, BaseModel
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticToolsParser
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from ragbuilder.graph_utils import check_graph_dependencies

load_dotenv()


class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseModel):
    """A node in the knowledge graph."""
    id: str = Field(..., description="Unique identifier for the node")
    type: str = Field(..., description="Type of the node")
    properties: List[Property] = Field(default_factory=list, description="List of node properties")

class Relationship(BaseModel):
    """A relationship between two nodes in the knowledge graph."""
    source: Optional[Node] = Field(None, description="Source node")
    target: Optional[Node] = Field(None, description="Target node")
    type: Optional[str] = Field(None, description="Type of the relationship")
    properties: List[Property] = Field(default_factory=list, description="List of relationship properties")

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(..., description="List of relationships in the knowledge graph")

def format_property_key(s: str) -> str:
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)

def props_to_dict(props) -> dict:
    """Convert properties to a dictionary."""
    properties = {}
    if not props:
        return {}
    for p in props:
        return {format_property_key(p.key): p.value}

def map_to_base_node(node: Node) -> BaseNode:
    """Map the KnowledgeGraph Node to the base Node."""
    properties = props_to_dict(node.properties) if node.properties else {}
    # Add name property for better Cypher statement generation
    properties["name"] = node.id.title()
    return BaseNode(
        id=node.id.title(), type=node.type.capitalize(), properties=properties
    )

def map_to_base_relationship(rel: Relationship) -> BaseRelationship:
    """Map the KnowledgeGraph Relationship to the base Relationship."""
    source = map_to_base_node(rel.source)
    target = map_to_base_node(rel.target)
    properties = props_to_dict(rel.properties) if rel.properties else {}
    return BaseRelationship(
        source=source, target=target, type=rel.type, properties=properties
    )
def get_extraction_chain(llm,allowed_nodes: Optional[List[str]] = None, allowed_rels: Optional[List[str]] = None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""# Knowledge Graph Instructions
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 5. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
        """),
        ("human", "Use the given format to extract information from the following input: {input}"),
        ("human", "Tip: Make sure to answer in the correct format"),
    ])
    extraction_chain = prompt | llm.with_structured_output(KnowledgeGraph)
    return extraction_chain

def log_retry(retry_state):
    """return the result of the last call attempt"""
    print(f"Failed with {retry_state.outcome}. Retrying attempt {retry_state.attempt_number}...")

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_fixed(1),
    before_sleep=log_retry,
    retry=retry_if_exception_type(ValidationError)  # Retry only on ValidationError
)
def extract_graph_data(llm, document: Document, nodes: Optional[List[str]] = None, rels: Optional[List[str]] = None):
    """Extract graph data with retry logic."""
    extract_chain = get_extraction_chain(llm, nodes, rels)
    return extract_chain.invoke(document.page_content)

def extract_and_store_graph(embeddings, llm, graph, document: Document, nodes: Optional[List[str]] = None, rels: Optional[List[str]] = None) -> None:
    try:

        # Extract entities and relationships using the LLM
        data = extract_graph_data(llm, document, nodes, rels)
        valid_rels = [rel for rel in data.rels if rel.source and rel.target and rel.type]
        
        # Construct a graph document with valid relationships
        graph_document = GraphDocument(
            nodes=[map_to_base_node(node) for node in data.nodes],
            relationships=[map_to_base_relationship(rel) for rel in valid_rels],
            source=document
        )
        
        graph.add_graph_documents([graph_document])

        embedding = embeddings.embed_query(document.page_content)
        
        # Create Document node and store its ID
        result = graph.query("""
        CREATE (d:Document {
            text: $text,
            source: $source,
            embedding: $embedding
        })
        WITH d
        UNWIND $entity_names as entity_name
        MATCH (e)
        WHERE e.name = entity_name
        CREATE (d)-[:MENTIONS]->(e)
        """, {
            "text": document.page_content,
            "source": document.metadata.get("source", ""),
            "embedding": embedding,
            "entity_names": [node.id.title() for node in data.nodes]
        })

        # result = graph.query("""
        # CREATE (d:Document {
        #     text: $text,
        #     source: $source,
        #     embedding: $embedding
        # })
        # RETURN id(d) as doc_id
        # """, {
        #     "text": document.page_content,
        #     "source": document.metadata.get("source", ""),
        #     "embedding": embedding
        # })
        # doc_id = result[0]["doc_id"]

        # # Connect Document node to all entities mentioned in it
        # for node in data.nodes:
        #     graph.query("""
        #     MATCH (d:Document), (e)
        #     WHERE id(d) = $doc_id AND e.name = $entity_name
        #     CREATE (d)-[:MENTIONS]->(e)
        #     """, {
        #         "doc_id": doc_id,
        #         "entity_name": node.id.title()
        #     })
    except ValidationError as e:
        print(f"Failed to extract graph data after retries: {e}")

def load_graph(documents, embeddings, llm):
    check_graph_dependencies()
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD)

    # Get dimensions from first embedding if dimensions attribute not available
    if hasattr(embeddings, 'dimensions') and embeddings.dimensions:
        dims = embeddings.dimensions
    else:
        test_embedding = embeddings.embed_query("test")
        dims = len(test_embedding)

    graph.query("""
    CREATE VECTOR INDEX document_embeddings IF NOT EXISTS
    FOR (d:Document)
    ON d.embedding
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: $dims,
            `vector.similarity_function`: 'cosine'
        }
    }
    """, {
        "dims": dims
    })
    
    from tqdm import tqdm
    for i, d in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(embeddings, llm, graph, d)
    
    return graph

class KnowledgeGraphOutputParser(PydanticToolsParser):
    def parse_result(self, result, partial=False):
        parsed = super().parse_result(result, partial)
        if isinstance(parsed, KnowledgeGraph):
            # Normalize all nodes
            for node in parsed.nodes:
                if not isinstance(node.properties, list):
                    node.properties = [node.properties] if node.properties else []
            
            # Filter and normalize all relationships
            valid_rels = []
            for rel in parsed.rels:
                # Check if source, target, and type are present
                if rel.source and rel.target and rel.type:
                    if not isinstance(rel.properties, list):
                        rel.properties = [rel.properties] if rel.properties else []
                    # Normalize source and target nodes
                    if hasattr(rel.source, 'properties') and not isinstance(rel.source.properties, list):
                        rel.source.properties = [rel.source.properties] if rel.source.properties else []
                    if hasattr(rel.target, 'properties') and not isinstance(rel.target.properties, list):
                        rel.target.properties = [rel.target.properties] if rel.target.properties else []
                    valid_rels.append(rel)
            
            # Update the relationships with only valid ones
            parsed.rels = valid_rels
        return parsed

 