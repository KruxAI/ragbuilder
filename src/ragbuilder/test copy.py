import time
# import ragbuilder.executor as exec 
import json

class RagBuilder:
    def __init__(self, val):
        self.config = val
        self.run_id = int(time.time())
        self.framework = val['framework']
        self.description = val['description']
        self.retrieval_model = val['retrieval_model']
        self.source_ids = val['source_ids']
        self.loader_kwargs = val['loader_kwargs']
        self.chunking_kwargs = val['chunking_kwargs']
        self.vectorDB_kwargs = val['vectorDB_kwargs']
        self.embedding_kwargs = val['embedding_kwargs']
        self.retriever_kwargs = val['retriever_kwargs']
        # print(f"retrieval model: {self.retreival_model}")
        self.rag = self.mergerag(
            framework=self.framework,
            description=self.description,
            retrieval_model=self.retrieval_model,
            source_ids=self.source_ids,
            loader_kwargs=self.loader_kwargs,
            chunking_kwargs=self.chunking_kwargs,
            vectorDB_kwargs=self.vectorDB_kwargs,
            embedding_kwargs=self.embedding_kwargs,
            retriever_kwargs=self.retriever_kwargs
        )

    def mergerag(self, framework, description, retrieval_model, source_ids, loader_kwargs, chunking_kwargs, vectorDB_kwargs, embedding_kwargs, retriever_kwargs):
        # Dummy implementation for illustration purposes
        return f"RAG Object with framework={framework}, description={description}"

    
    # def __repr__(self):
    #     return (
    #             f"    run_id={self.run_id!r},\n"
    #             f"    framework={self.framework!r},\n"
    #             f"    description={self.description!r},\n"
    #             f"    retreival_model={self.retreival_model!r},\n"
    #             f"    source={self.loader_kwargs[1]['input_path']!r},\n"
    #             f"    chunking_strategy={self.chunking_kwargs[1]!r},\n"
    #             f"    vectorDB_kwargs={self.vectorDB_kwargs[1]!r},\n"
    #             f"    embedding_kwargs={self.embedding_kwargs[1][0]['embedding_model']!r},\n"
    #             f"    retriever_kwargs={self.retriever_kwargs[1][self.retreival_model]['retrievers']!r}\n"
    #             f")")
    def __repr__(self):
        return json.dumps(self.config)

# Provided configuration dictionary
config = {
    'framework': "langchain",
    'description': 'ragbuilder: Merge RAG Invoked',
    'source_ids': [1],
    'retrieval_model': 'openai',
    'loader_kwargs': {
        1: {'source': 'test', 'input_path': 'path/to/input'},
    },
    'chunking_kwargs': {
        1: {'chunk_strategy': 'MarkdownHeaderTextSplitter', 'chunk_size': 1000, 'chunk_overlap': 200},
    },
    'embedding_kwargs': {
        1: [{'embedding_model': 'openai'}],
    },
    'vectorDB_kwargs': {1: {'vectorDB': 'CHROMA'}},
    'retriever_kwargs': {
        1: {
            'openai': {
                'retrievers': [
                    {'retriever_type': 'vector', 'search_type': 'similarity', 'search_kwargs': {"k": 5}},
                    {'retriever_type': 'bm25retreiver'},
                    {'retriever_type': 'vector', 'search_type': 'mmr', 'search_kwargs': {"k": 5}}
                ]
            }
        },
        "document_compressor_pipeline": ["EmbeddingsRedundantFilter", "EmbeddingsClusteringFilter", "LLMChainFilter", "LongContextReorder"],
        "EmbeddingsClusteringFilter_kwargs": {"embeddings": "openai", "num_clusters": 4, "num_closest": 1, "sorted": True},
        "contextual_compression_retriever": False
    }
}

# Create an instance of RagBuilder
rag_builder = RagBuilder(config)

# Use repr() to get the string representation
print(repr(rag_builder))


# f_name=generate_data.generate_data(
#             src_data='/Users/aravind/KruxAI/ragbuilder/langchain_for_ragas/'
#             # test_size=5,
#             # generator_llm=ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens=800),
#             # critic_llm = ChatOpenAI(model="gpt-4o", temperature=0.2),
#             # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
#         )