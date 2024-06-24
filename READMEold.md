# RAG Builder

Rag builder Toolkit that helps with deciding on the right RAG framework for your. RAG  Builder helps developers iterate through 
different RAG Components and helps them pick the right RAG setting for their Data

## Features

- `v0.0.1` Support for LangChain RAG Components
            Loaders
               - WebBaseLoader
               - UnstructuredFileLoader
               - DirectoryLoader
            Chunking Strategies
               - RecursiveCharacterTextSplitter
               - CharacterTextSplitter
               - SemanticChunker
               - MarkdownHeaderTextSplitter
               - HTMLHeaderTextSplitter
            Vector DB
               - ChromaDB
               - FAISS
            Retreival Strategies
               - Vector Similarity
               - Vector MMR
               - Multi- Query
               - Parent Document - Full Document
               - Parent Document - Large Chunk
               - Merge Retreiver
               - Contextual Compression
                  Compressors
                     - LongContextReorder
                     - EmbeddingsRedundantFilter
                     - EmbeddingsClusteringFilter
                     - LLMChainFilter 
            Embedding Models
               - text-embedding-3-small
               - text-embedding-3-large
               - text-embedding-ada-002
               - mistral-embed

            LLM Models 
               - gpt-3.5-turbo
               - gpt-4o 
               - gpt-4-turbo

## Installation

1. Download and install the latest version of [python](https://www.python.org/downloads/). Open a terminal and check that it is installed.

   Windows
   ```
   py --version
   ```

   Linux/MAC OS
   ```
   python3 --version
   ```

2. Make sure you have upgraded version of pip.

   Windows
   ```
   py -m pip install --upgrade pip
   ```

   Linux/MAC OS
   ```
   python3 -m pip install --upgrade pip
   ```

3. Install ragbuilder using pip.

   Windows
   ```
   pip install rag_builder
   ```

   Linux/MAC OS
   ```
   python3 -m pip install rag_builder
   ```

4. Check that the package was installed

   ```
   pip show rag_builder
   ```
5. Make a Folder to Store your Synthetic data and DB

   ```
   cd mkdir rag_builder
   ```
6. Start Rag Builder

   ```
   rag_builder
   ```

## Usage


   ```
   cd rag_builder
   python src/rag_builder.py

   ```

## API Keys for LLM and Embedding Models
In the src/.env Set the API Keys for your LLM and Embedding Models
