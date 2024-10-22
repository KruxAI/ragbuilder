from typing import List
from langchain.schema import Document

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