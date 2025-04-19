"""
Retrieval Module

This module provides DSPy modules for retrieval-augmented generation (RAG).
"""

import logging
from typing import List, Dict, Any, Optional, Union
import dspy

from .base import RAGModule
from .enhanced_base import EnhancedRAGModule

# Set up logging
logger = logging.getLogger(__name__)


class RAGSignature(dspy.Signature):
    """Signature for retrieval-augmented generation."""
    
    query = dspy.InputField(desc="User query or question")
    retrieved_documents = dspy.InputField(desc="Retrieved passages or documents relevant to the query")
    answer = dspy.OutputField(desc="Answer to the query based on the retrieved documents")
    citations = dspy.OutputField(desc="Citations to the relevant parts of the retrieved documents")


class EnhancedRAGSignature(dspy.Signature):
    """Signature for enhanced retrieval-augmented generation."""
    
    query = dspy.InputField(desc="User query or question")
    retrieved_documents = dspy.InputField(desc="Retrieved passages or documents relevant to the query")
    answer = dspy.OutputField(desc="Answer to the query based on the retrieved documents")
    citations = dspy.OutputField(desc="Citations to the relevant parts of the retrieved documents")
    missing_information = dspy.OutputField(desc="Information that was missing from the retrieved documents")
    confidence = dspy.OutputField(desc="Confidence in the answer (high, medium, low)")


class RAGImplementation(RAGModule):
    """Module for retrieval-augmented generation."""
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        num_passages: int = 3,
        include_citations: bool = True,
        rerank: bool = False
    ):
        """
        Initialize the RAG module.
        
        Args:
            retriever: Document retriever
            num_passages: Number of passages to retrieve
            include_citations: Whether to include citations in the output
            rerank: Whether to rerank retrieved passages
        """
        super().__init__(
            retriever=retriever,
            num_passages=num_passages,
            rerank=rerank
        )
        self.include_citations = include_citations
        
        # Create predictor
        self.predictor = dspy.Predict(RAGSignature)
    
    async def predict(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Perform retrieval-augmented generation for a query.
        
        Args:
            query: User query
            
        Returns:
            Dict[str, Any]: RAG result
        """
        # Log the request
        logger.info(f"Performing RAG for query: {query[:100]}...")
        
        try:
            # Retrieve documents
            retrieved_docs = await self.retrieve(query)
            
            # Format documents for the predictor
            formatted_docs = "\n\n".join([
                f"Document {i+1}:\n{doc.content if hasattr(doc, 'content') else str(doc)}"
                for i, doc in enumerate(retrieved_docs)
            ])
            
            # Call predictor
            response = await self.call_predictor(
                self.predictor,
                query=query,
                retrieved_documents=formatted_docs
            )
            
            # Process response
            result = {
                "query": query,
                "answer": response.answer,
                "retrieved_documents": [
                    {
                        "content": doc.content if hasattr(doc, "content") else str(doc),
                        "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                    }
                    for doc in retrieved_docs
                ]
            }
            
            # Add citations if requested
            if self.include_citations:
                result["citations"] = response.citations
            
            return result
            
        except Exception as e:
            logger.error(f"Error in RAG: {str(e)}")
            raise


class EnhancedRAG(EnhancedRAGModule):
    """Enhanced module for retrieval-augmented generation with additional capabilities."""
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        num_passages: int = 3,
        include_citations: bool = True,
        filter_threshold: Optional[float] = 0.7,
        include_confidence: bool = True,
        rerank: bool = False
    ):
        """
        Initialize the enhanced RAG module.
        
        Args:
            retriever: Document retriever
            num_passages: Number of passages to retrieve
            include_citations: Whether to include citations in the output
            filter_threshold: Relevance threshold for filtering passages (0-1)
            include_confidence: Whether to include confidence score
            rerank: Whether to rerank retrieved passages
        """
        super().__init__(
            retriever=retriever,
            num_passages=num_passages,
            filter_threshold=filter_threshold,
            rerank=rerank
        )
        self.include_citations = include_citations
        self.include_confidence = include_confidence
        
        # Create predictor
        self.predictor = dspy.Predict(EnhancedRAGSignature)
    
    async def _predict_impl(
        self,
        query: str
    ) -> Dict[str, Any]:
        """
        Implementation of enhanced retrieval-augmented generation for a query.
        
        Args:
            query: User query
            
        Returns:
            Dict[str, Any]: Enhanced RAG result
        """
        # Retrieve documents
        retrieved_docs = await self.retrieve(query)
        
        # Format documents for the predictor
        formatted_docs = "\n\n".join([
            f"Document {i+1}:\n{doc.content if hasattr(doc, 'content') else str(doc)}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        # Call predictor
        response = await self.call_predictor(
            self.predictor,
            query=query,
            retrieved_documents=formatted_docs
        )
        
        # Process response
        result = {
            "query": query,
            "answer": response.answer,
            "missing_information": response.missing_information,
            "retrieved_documents": [
                {
                    "content": doc.content if hasattr(doc, "content") else str(doc),
                    "metadata": doc.metadata if hasattr(doc, "metadata") else {}
                }
                for doc in retrieved_docs
            ]
        }
        
        # Add citations if requested
        if self.include_citations:
            result["citations"] = response.citations
        
        # Add confidence if requested
        if self.include_confidence:
            result["confidence"] = response.confidence
        
        return result


# Common retrieval types with preconfigured settings
class QuestionAnsweringRAG(RAGImplementation):
    """RAG module specifically optimized for question answering."""
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        **kwargs
    ):
        """Initialize the question answering RAG module."""
        super().__init__(
            retriever=retriever,
            num_passages=5,  # More passages for better coverage
            include_citations=True,  # Always include citations
            **kwargs
        )


class SummarizationRAG(RAGImplementation):
    """RAG module specifically optimized for summarization tasks."""
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        **kwargs
    ):
        """Initialize the summarization RAG module."""
        super().__init__(
            retriever=retriever,
            num_passages=10,  # More passages for comprehensive summary
            include_citations=False,  # Citations less important for summarization
            **kwargs
        )


# Export
__all__ = [
    "RAGSignature",
    "EnhancedRAGSignature",
    "RAGImplementation",
    "EnhancedRAG",
    "QuestionAnsweringRAG",
    "SummarizationRAG",
]