from typing import List, Any
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import TextSplitter

class ContextualChunker(TextSplitter):
    def __init__(
        self, 
        chunk_size: int, 
        chunk_overlap: int, 
        llm=None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm or ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
        self._base_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # Create a temporary document to use existing logic
        temp_doc = Document(page_content=text)
        processed_docs = self.split_documents([temp_doc])
        # Return just the text content
        return [doc.page_content for doc in processed_docs]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        splits = splitter.split_documents(documents)
        
        document_array = []
        for i, split in enumerate(splits):
            chunk_length = len(split.page_content)
            if chunk_length < 100:
                document_array.append(split)
                continue
            context_length = int(chunk_length * 0.1)
            
            # Get 10 adjacent chunks around the current chunk (including the current chunk)
            chunk_group = splits[max(0, i - 5):min(len(splits), i + 6)]
            chunk_content = self.format_docs([split])

            # Combine adjacent chunks for context
            chunk_group_content = self.format_docs(chunk_group)

            # Construct the prompt with chunk group as the document context
            prompt = f'''
            You are an expert at situating chunks within a document. When given a chunk and a group of 10 adjacent chunks, 
            your task is to provide a short succinct context to situate the chunk within the overall document (the group of 10 chunks) 
            for the purposes of improving search retrieval of the chunk. Do not ask any questions. Answer only with the short and succinct context.
            <document> 
            {chunk_group_content}
            </document> 
            Here is the chunk we want to situate within the chunk group 
            <chunk> 
            {chunk_content}
            </chunk>
            (question)
            '''
            
            cr_prompt = ChatPromptTemplate.from_template(prompt)
            cr_prompt_chain = cr_prompt | self.llm | StrOutputParser()
            
            context = cr_prompt_chain.invoke({
                'question': 'Please give a short succinct context to situate this chunk within the overall document '
            })

            document = Document(
                page_content=f"{context} {chunk_content}",
                metadata={**split.metadata, "chunk_index": i}
            )
            document_array.append(document)

        return document_array

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

class CustomChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            text = doc.page_content
            start = 0
            while start < len(text):
                end = start + self.chunk_size
                chunk = text[start:end]
                chunked_docs.append(Document(page_content=chunk, metadata=doc.metadata))
                start += self.chunk_size - self.chunk_overlap
        return chunked_docs
