# import os
# from langchain_experimental.llms.ollama_functions import OllamaFunctions
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# import networkx as nx
# from langchain.chains import GraphQAChain
# from langchain_core.documents import Document
# from langchain_community.graphs.networkx_graph import NetworkxEntityGraph
# from langchain_community.llms import Ollama
# from langchain_community.document_loaders import *
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.graphs import Neo4jGraph
# from langchain_core.runnables import  RunnablePassthrough
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_core.output_parsers import StrOutputParser
# import os
# from langchain_community.graphs import Neo4jGraph
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from e import ChatOpenAI
# from langchain_community.chat_models import ChatOllama
# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from neo4j import GraphDatabase
# # from yfiles_jupyter_graphs import GraphWidget
# from langchain_community.vectorstores import Neo4jVector
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import TextLoader
# from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars

# graph = Neo4jGraph()

# llm = Ollama(model='llama3.1:latest',base_url='http://localhost:11434')


# loader = WebBaseLoader('https://ashwinaravind.github.io/')
# documents = loader.load()
# llm_transformer = LLMGraphTransformer(llm=llm)
# graph_documents = llm_transformer.convert_to_graph_documents(documents)
# print(1)
# graph.add_graph_documents(
#     graph_documents,
#     baseEntityLabel=True,
#     include_source=True
# )

# vector_index = Neo4jVector.from_existing_graph(
#     OpenAIEmbeddings(),
#     search_type="hybrid",
#     node_label="Document",
#     text_node_properties=["text"],
#     embedding_node_property="embedding"
# )
# vector_retriever = vector_index.as_retriever()



# from langchain_experimental.graph_transformers import LLMGraphTransformer
# from langchain_openai import ChatOpenAI

# llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

# llm_transformer = LLMGraphTransformer(
#   llm=llm, 
#   node_properties=["description"],
#   relationship_properties=["description"]
# )

# def process_text(text: str) -> List[GraphDocument]:
#     doc = Document(page_content=text)
#     return llm_transformer.convert_to_graph_documents([doc])


# # class Entities(BaseModel):
# #     """Identifying information about entities."""

# #     names: list[str] = Field(
# #         ...,
# #         description="All the person, organization, or business entities that "
# #         "appear in the text",
# #     )

# # prompt = ChatPromptTemplate.from_messages(
# #     [
# #         (
# #             "system",
# #             "You are extracting organization and person entities from the text.",
# #         ),
# #         (
# #             "human",
# #             "Use the given format to extract information from the following "
# #             "input: {question}",
# #         ),
# #     ]
# # )

# # entity_chain = prompt | llm.with_structured_output(Entities)

# # entity_chain.invoke({"question": "what is RAG?"}).names
# # print(entity_chain.invoke({"question": "what is RAG?"}).names)

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders import *
from langchain_community.graphs import Neo4jGraph
from langchain.schema import Document
import tqdm
import pandas as pd
graph = Neo4jGraph(refresh_schema=False)
loader = WebBaseLoader('https://en.wikipedia.org/wiki/Kerala?action=raw')
# loader = WebBaseLoader('https://ashwinaravind.github.io/')

documents = loader.load()
# splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
# documents=splitter.split_documents(documents)
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits=markdown_splitter.split_text(documents[0].page_content)

from langchain_community.graphs.graph_document import (
    Node as BaseNode,
    Relationship as BaseRelationship,
    GraphDocument,
)
from langchain.schema import Document
from typing import List, Dict, Any, Optional
from langchain.pydantic_v1 import Field, BaseModel

class Property(BaseModel):
  """A single property consisting of key and value"""
  key: str = Field(..., description="key")
  value: str = Field(..., description="value")

class Node(BaseNode):
    properties: Optional[List[Property]] = Field(
        None, description="List of node properties")

class Relationship(BaseRelationship):
    properties: Optional[List[Property]] = Field(
        None, description="List of relationship properties"
    )

class KnowledgeGraph(BaseModel):
    """Generate a knowledge graph with entities and relationships."""
    nodes: List[Node] = Field(
        ..., description="List of nodes in the knowledge graph")
    rels: List[Relationship] = Field(
        ..., description="List of relationships in the knowledge graph"
    )

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

import os
from langchain.chains.openai_functions import (
    # create_openai_fn_chain,
    create_structured_output_chain,
)
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# os.environ["OPENAI_API_KEY"] = "sk-"
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None,
    allowed_rels: Optional[List[str]] = None
    ):
    prompt = ChatPromptTemplate.from_messages(
        [(
          "system",
          f"""# Knowledge Graph Instructions for GPT-4
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
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)

def extract_and_store_graph(
    document: Document,
    nodes:Optional[List[str]] = None,
    rels:Optional[List[str]]=None) -> None:
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(nodes, rels)
    data = extract_chain.invoke(document.page_content)['function']
    # Construct a graph document
    graph_document = GraphDocument(
      nodes = [map_to_base_node(node) for node in data.nodes],
      relationships = [map_to_base_relationship(rel) for rel in data.rels],
      source = document
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])

from tqdm import tqdm

for i, d in tqdm(enumerate(documents), total=len(documents)):
    extract_and_store_graph(d)

# Query the knowledge graph in a RAG application
from langchain.chains import GraphCypherQAChain

graph.refresh_schema()

cypher_chain = GraphCypherQAChain.from_llm(
    graph=graph,
    cypher_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
    qa_llm=ChatOpenAI(temperature=0, model="gpt-4o"),
    validate_cypher=True, # Validate relationship directions
    return_intermediate_steps=True,
    output_key = 'answer',
    verbose=True
)
# cypher_chain.invoke({"query": "what is rag?"}))
print(cypher_chain.invoke("where is kerala?"))
 