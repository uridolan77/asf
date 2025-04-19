"""
Question Answering Module

This module provides DSPy modules for general question answering tasks.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import dspy

from .base import DSPyModuleBase
from .enhanced_base import EnhancedDSPyModuleBase
from .retrieval import RAGImplementation

# Set up logging
logger = logging.getLogger(__name__)


class QASignature(dspy.Signature):
    """Signature for question answering."""
    
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Answer to the question")
    confidence = dspy.OutputField(desc="Confidence in the answer (high, medium, low)")


class AdvancedQASignature(dspy.Signature):
    """Signature for advanced question answering."""
    
    question = dspy.InputField(desc="Question to answer")
    context = dspy.InputField(desc="Optional context to help answer the question")
    answer = dspy.OutputField(desc="Answer to the question")
    confidence = dspy.OutputField(desc="Confidence in the answer (high, medium, low)")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning to arrive at the answer")
    missing_info = dspy.OutputField(desc="Any missing information that would help answer the question better")


class QAModule(DSPyModuleBase):
    """Module for general question answering."""
    
    def __init__(
        self,
        include_confidence: bool = True,
    ):
        """
        Initialize the QA module.
        
        Args:
            include_confidence: Whether to include confidence score
        """
        super().__init__()
        self.include_confidence = include_confidence
        
        # Create predictor
        self.predictor = dspy.Predict(QASignature)
    
    async def predict(
        self,
        question: str
    ) -> Dict[str, Any]:
        """
        Answer a question.
        
        Args:
            question: Question to answer
            
        Returns:
            Dict[str, Any]: QA result
        """
        # Log the request
        logger.info(f"Answering question: {question[:100]}...")
        
        try:
            # Call predictor
            response = await self.call_predictor(
                self.predictor,
                question=question
            )
            
            # Process response
            result = {
                "question": question,
                "answer": response.answer,
            }
            
            # Add confidence if requested
            if self.include_confidence:
                result["confidence"] = response.confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Error in QA: {str(e)}")
            raise


class AdvancedQAModule(EnhancedDSPyModuleBase):
    """Module for advanced question answering with reasoning."""
    
    def __init__(
        self,
        include_confidence: bool = True,
        include_reasoning: bool = True,
        include_missing_info: bool = True,
        rag_module: Optional[RAGImplementation] = None,
    ):
        """
        Initialize the advanced QA module.
        
        Args:
            include_confidence: Whether to include confidence score
            include_reasoning: Whether to include reasoning steps
            include_missing_info: Whether to include missing information
            rag_module: Optional RAG module to provide context
        """
        super().__init__()
        self.include_confidence = include_confidence
        self.include_reasoning = include_reasoning
        self.include_missing_info = include_missing_info
        self.rag_module = rag_module
        
        # Create predictor
        self.predictor = dspy.Predict(AdvancedQASignature)
    
    async def _predict_impl(
        self,
        question: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Answer a question with advanced capabilities.
        
        Args:
            question: Question to answer
            context: Optional context to help answer the question
            
        Returns:
            Dict[str, Any]: Advanced QA result
        """
        # Get context from RAG if available and not provided
        if context is None and self.rag_module is not None:
            try:
                rag_result = await self.rag_module.predict(question)
                context_docs = rag_result.get("retrieved_documents", [])
                if context_docs:
                    context = "\n\n".join([
                        doc.get("content", "") if isinstance(doc, dict) else str(doc)
                        for doc in context_docs
                    ])
            except Exception as e:
                logger.warning(f"Error retrieving context: {str(e)}")
        
        # Default empty context if still None
        context = context or ""
        
        # Call predictor
        response = await self.call_predictor(
            self.predictor,
            question=question,
            context=context
        )
        
        # Process response
        result = {
            "question": question,
            "answer": response.answer,
        }
        
        # Add optional fields
        if self.include_confidence:
            result["confidence"] = response.confidence
        
        if self.include_reasoning:
            result["reasoning"] = response.reasoning
        
        if self.include_missing_info:
            result["missing_info"] = response.missing_info
        
        # Add context if it was used
        if context:
            result["context_used"] = True
        
        return result


class MultiQuestionModule(DSPyModuleBase):
    """Module for handling multiple related questions."""
    
    def __init__(
        self,
        base_qa_module: Optional[QAModule] = None,
        track_previous_answers: bool = True,
    ):
        """
        Initialize the multi-question module.
        
        Args:
            base_qa_module: Base QA module to use for individual questions
            track_previous_answers: Whether to track previous answers
        """
        super().__init__()
        self.base_qa_module = base_qa_module or QAModule()
        self.track_previous_answers = track_previous_answers
        
        # Store conversation history
        self.history: List[Dict[str, Any]] = []
    
    async def predict(
        self,
        question: str,
        reset_history: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a question in a multi-question conversation.
        
        Args:
            question: Question to answer
            reset_history: Whether to reset conversation history
            
        Returns:
            Dict[str, Any]: QA result
        """
        # Reset history if requested
        if reset_history:
            self.history = []
        
        # Log the request
        logger.info(f"Answering question in multi-question conversation: {question[:100]}...")
        
        try:
            # Call base QA module
            result = await self.base_qa_module.predict(question)
            
            # Add to history if tracking
            if self.track_previous_answers:
                self.history.append({
                    "question": question,
                    "answer": result["answer"]
                })
                
                # Add history to result
                result["history"] = self.history
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-question: {str(e)}")
            raise


# Export
__all__ = [
    "QASignature",
    "AdvancedQASignature",
    "QAModule",
    "AdvancedQAModule",
    "MultiQuestionModule",
]