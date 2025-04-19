"""
Generation pipeline module for RAGBuilder.
Provides standardized interface for generation components.
"""
import logging
from typing import Any, Dict, List, Optional
from operator import itemgetter

from ragbuilder.config import GenerationConfig
from ragbuilder.core.exceptions import PipelineError


class GenerationPipeline:
    """
    Standard pipeline for generation components.
    Handles the transformation of retrieval results into LLM-generated answers.
    """
    
    def __init__(
        self, 
        config: GenerationConfig, 
        retriever: Any,
        verbose: bool = False
    ):
        """
        Initialize the generation pipeline.
        
        Args:
            config: Configuration for the generation pipeline
            retriever: Retriever component to use
            verbose: Whether to log detailed information
        """
        self.logger = logging.getLogger("ragbuilder.generation.pipeline")
        self.config = config
        self.retriever = retriever
        self.verbose = verbose
        
        # Create the pipeline
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self) -> Any:
        """
        Create the generation pipeline using LangChain runnables.
        
        Returns:
            LangChain runnable for generation
        """
        try:
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
            from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
            from langchain_core.output_parsers import StrOutputParser

            def format_docs(docs):
                """Format documents for prompt context."""
                return "\n".join(doc.page_content for doc in docs)

            # Get LLM from config
            llm = self.config.llm.llm
            prompt_template = self.config.prompt_template

            # Create chat prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", prompt_template),
                ("user", "{question}"),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
            ])

            # Create RAG chain
            rag_chain = (
                RunnableParallel(context=self.retriever, question=RunnablePassthrough())
                .assign(context=itemgetter("context") | RunnableLambda(format_docs))
                .assign(answer=prompt | llm | StrOutputParser())
                .pick(["answer", "context"])
            )
            return rag_chain
            
        except Exception as e:
            self.logger.error(f"Pipeline creation failed: {str(e)}")
            raise PipelineError(f"Failed to create generation pipeline: {str(e)}")
            
    # def invoke(self, query: str, chat_history: Optional[List] = None) -> Dict[str, Any]:
    #     """
    #     Generate an answer for the given query.
        
    #     Args:
    #         query: User question to answer
    #         chat_history: Optional chat history for contextual generation
            
    #     Returns:
    #         Dictionary with generated answer and context
    #     """
    #     try:
    #         if self.verbose:
    #             self.logger.debug(f"Generating answer for query: {query}")
                
    #         # Prepare input
    #         input_data = {"question": query}
    #         if chat_history:
    #             input_data["chat_history"] = chat_history
                
    #         # Run the pipeline
    #         result = self.pipeline.invoke(input_data)
            
    #         return result
            
    #     except Exception as e:
    #         self.logger.error(f"Generation failed: {str(e)}")
    #         return {
    #             "answer": f"Error generating response: {str(e)}",
    #             "context": [],
    #             "error": str(e)
    #         }
            
    def batch_generate(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries.
        
        Args:
            queries: List of user questions
            
        Returns:
            List of dictionaries with generated answers and contexts
        """
        return [self.pipeline.invoke(query) for query in queries]
            
    def get_config(self) -> GenerationConfig:
        """
        Get the configuration of this pipeline.
        
        Returns:
            GenerationConfig used by this pipeline
        """
        return self.config 