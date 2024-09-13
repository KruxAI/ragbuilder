from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_community.document_loaders import *
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from operator import itemgetter
from langchain import hub
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
load_dotenv()
import os


class Property(BaseModel):
    """A single property consisting of key and value"""
    key: str = Field(..., description="key")
    value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(None, description="List of relationship properties")

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
        return properties
    for p in props:
        properties[format_property_key(p.key)] = p.value
    return properties

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
        ("system", f"""# Knowledge Graph Instructions for GPT-4
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

def extract_and_store_graph(llm,graph,document: Document, nodes:Optional[List[str]] = None, rels:Optional[List[str]]=None) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(llm,nodes, rels)
    data = extract_chain.invoke(document.page_content)
    # Construct a graph document
    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in data.rels],
        source=document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])


def load_graph(documents,llm):
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD)
    from tqdm import tqdm
    for i, d in tqdm(enumerate(documents), total=len(documents)):
        extract_and_store_graph(llm,graph,d)

 