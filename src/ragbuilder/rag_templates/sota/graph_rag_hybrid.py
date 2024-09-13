code="""from langchain_core.runnables import  RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
from ragbuilder.graph_utils.graph_loader import load_graph 
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
load_dotenv()
import os
def rag_pipeline():
    try:
        NEO4J_URI = os.getenv("NEO4J_URI")
        NEO4J_USER = os.getenv("NEO4J_USER")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        NEO4J_LOAD = os.getenv('NEO4J_LOAD', 'True').lower() == 'true'
        print(OLLAMA_BASE_URL)
        {llm_class}
            
        {loader_class}
            
        {embedding_class}
        
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        documents=splitter.split_documents(docs)

        c=Chroma.from_documents(documents=documents, embedding=embedding, collection_name='testindex-ragbuilder',)
        vector_retriever=c.as_retriever(search_type='similarity', search_kwargs={'k': 100})
        
        if NEO4J_LOAD:
            print("Loading Graph")
            load_graph(documents,llm)

        class Entities(BaseModel):
            '''Identifying information about entities.'''
            names: list[str] = Field(
                ...,
                description="All nodes "
                "appear in the text",
            )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following "
                    "input: {question}",
                ),
            ]
        )
        entity_chain = prompt | llm.with_structured_output(Entities)

        def generate_full_text_query(input: str) -> str:
            words = [el for el in remove_lucene_chars(input).split() if el]
            if not words:
                return ""
            full_text_query = " AND ".join([f"{word}~2" for word in words])
            return full_text_query.strip()
        

        def graph_retriever(question: str) -> str:
            '''
            Collects the neighborhood of entities mentioned
            in the question and searches across all full-text indexes
            '''
            # Step 1: Retrieve all labels from the database
            labels_result = graph.query("CALL db.labels() YIELD label RETURN label")
            all_labels = [record['label'] for record in labels_result]
            
            # Step 2: Create a full-text index for each label if not already existing
            existing_indexes = [index["name"] for index in graph.query("SHOW INDEXES")]

            for label in all_labels:
                index_name = f"index_{label}"
                label = label.replace(' ', '_')
                # Check if index already exists for this label
                if index_name not in existing_indexes:
                    print(f"Index not found for label {label}, creating index...")
                    
                    # Dynamically create the full-text index for this label on the desired properties (e.g., id, name, description)
                    query = f"CREATE FULLTEXT INDEX {index_name} FOR (n:{label}) ON EACH [n.id, n.name, n.description]"
                    graph.query(query)
                    print(f"Full-text index created for label {label}")

            # Step 3: Run the full-text search across all indexes
            result = ""
            entities = entity_chain.invoke({"question": question})
            
            # Check if any entities were found
            if not entities.names:
                print("No entities found")
                return result
            
            # Search across all full-text indexes dynamically
            for entity in entities.names:
                for label in all_labels:
                    index_name = f"index_{label}"
                    try:
                        response = graph.query(
                            f'''CALL db.index.fulltext.queryNodes('{index_name}', $query, {{limit:100}})
                            YIELD node, score
                            CALL {{
                            WITH node
                            MATCH (node)-[r]->(neighbor)
                            RETURN node.id + ' - ' + COALESCE(node.description, '') + ' - ' + type(r) + ' -> ' + neighbor.id + ' - ' + COALESCE(neighbor.description, '') AS output
                            UNION ALL
                            WITH node
                            MATCH (node)<-[r]-(neighbor)
                            RETURN node.id + ' - ' + COALESCE(node.description, '') + ' - ' + type(r) + ' -> ' + neighbor.id + ' - ' + COALESCE(neighbor.description, '') AS output
                            }}
                            RETURN output LIMIT 50
                            ''',
                            {"query": generate_full_text_query(entity)},
                        ) 
                        # Collect results
                        result += "\\n".join([el['output'] for el in response])
                    except Exception as e:
                        print(f"Error querying index {index_name}: {e}")
            
            return result

        def full_retriever(question: str):
            graph_data = graph_retriever(question)
            vector_data = [el.page_content for el in vector_retriever.invoke(question)]
            final_data = f'''Graph data:
        {graph_data}
        vector data:
        {"#Document ". join(vector_data)}
            '''
            return final_data



        prompt = hub.pull("rlm/rag-prompt")
        rag_chain = (
                    RunnableParallel(context=full_retriever, question=RunnablePassthrough())
                        .assign(context=itemgetter("context"))
                        .assign(answer=prompt | llm | StrOutputParser())
                        .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {e}")"""
