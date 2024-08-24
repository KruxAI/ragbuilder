code="""from langchain_community.llms import Ollama
from langchain_community.document_loaders import *
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from operator import itemgetter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain.retrievers import MergerRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field 
from langchain.schema import Document
import json
def rag_pipeline():
    question=RunnablePassthrough()
    try:  
        def format_docs(docs):
            return "\\n".join(doc.page_content for doc in docs) 
        
        {llm_class}
        
        {loader_class}
        
        {embedding_class}
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)
        splits=splitter.split_documents(docs)
        c=Chroma.from_documents(documents=splits, embedding=embedding, collection_name='testindex-ragbuilder',)
        retriever=c.as_retriever(search_type='similarity', search_kwargs={'k': 5})
        docs=retriever.invoke(question)
        def filter(docs):
                try:
                    response_schemas = [
                        ResponseSchema(name="Score", description="Score for the context query relevancy"),
                    ]
                    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

                    class Score(BaseModel):
                        Score: str = Field(description="Yes or No")
                        # doc: str = Field(description="document")x
                    parser = JsonOutputParser(pydantic_object=Score)
                    c_prompt =  PromptTemplate(
                    template='''You are a grader assessing relevance of a retrieved document to a user question. \\n
                    Here is the retrieved document: \\n\\n {context} \\n\\n
                    Here is the user question: {question} \\n
                    If the document contains keywords related to the user question, grade it as relevant. \\n
                    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \\n
                    Give a binary "Score" "yes" or "no" Score to indicate whether the document is relevant to the question. \\n
                    Provide only the binary "Score" as a text varible with a single key "Score" and no premable or explaination.
                    Answer the user query.\\n{format_instructions}\\n''',
                    input_variables=["question", "context"],
                    partial_variables={"format_instructions": parser.get_format_instructions()})
                    rag_chain = (
                    RunnableParallel({"context": lambda x: docs, "question": RunnablePassthrough()})
                        .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                        .assign(answer=c_prompt | llm | StrOutputParser())
                        .pick(["answer", "context"]))
                    llm_output=rag_chain.invoke(question)
                    data = json.loads(llm_output["answer"])
                    # Extract the value associated with the "Score" key
                    score = data.get("Score")
                    if score == "yes":
                        return llm_output['context']
                except Exception as e:
                    return None
        filter_content = []
        for doc in docs:
            f=filter([doc])
            if f != None:
                document = Document(page_content=f)
                filter_content.append(document)
        prompt = hub.pull("rlm/rag-prompt")
        # print(filter_content)
        rag_chain = (
                RunnableParallel({"context": lambda x: filter_content, "question": RunnablePassthrough()})
                    .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                    .assign(answer=prompt | llm | StrOutputParser())
                    .pick(["answer", "context"]))
        return rag_chain
    except Exception as e:
        print(f"An error occurred: {e}")
"""
