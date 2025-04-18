"""
DSPy Module Base

This module provides base classes for DSPy modules.
"""

import abc
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic, Protocol

import dspy

# Set up logging
logger = logging.getLogger(__name__)


class DSPyModuleProtocol(Protocol):
    """Protocol for DSPy modules."""
    
    async def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction using the module.
        
        Returns:
            Dict[str, Any]: Prediction result
        """
        ...


class DSPyModuleBase(abc.ABC):
    """Base class for DSPy modules."""
    
    def __init__(self):
        """Initialize the module."""
        self._lm = None
    
    @property
    def lm(self):
        """Get language model."""
        return self._lm
    
    @lm.setter
    def lm(self, lm):
        """Set language model."""
        self._lm = lm
    
    async def call_predictor(self, predictor: Any, **kwargs) -> Any:
        """
        Call DSPy predictor.
        
        Args:
            predictor: DSPy predictor
            **kwargs: Keyword arguments for the predictor
            
        Returns:
            Any: Predictor response
        """
        if asyncio.iscoroutinefunction(predictor.__call__):
            return await predictor(**kwargs)
        else:
            return predictor(**kwargs)
    
    @abc.abstractmethod
    async def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction using the module.
        
        Returns:
            Dict[str, Any]: Prediction result
        """
        raise NotImplementedError("Subclasses must implement predict()")


class RAGModule(DSPyModuleBase):
    """Base class for RAG modules."""
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        num_passages: int = 3,
        rerank: bool = False
    ):
        """
        Initialize the RAG module.
        
        Args:
            retriever: Document retriever
            num_passages: Number of passages to retrieve
            rerank: Whether to rerank retrieved passages
        """
        super().__init__()
        self.retriever = retriever
        self.num_passages = num_passages
        self.rerank = rerank
    
    def set_retriever(self, retriever: Any) -> None:
        """
        Set document retriever.
        
        Args:
            retriever: Document retriever
        """
        self.retriever = retriever
    
    async def retrieve(self, query: str) -> List[Any]:
        """
        Retrieve documents relevant to a query.
        
        Args:
            query: Query string
            
        Returns:
            List[Any]: Retrieved documents
        """
        if self.retriever is None:
            raise ValueError("Retriever not set")
        
        try:
            # Try to handle both synchronous and asynchronous retrievers
            if asyncio.iscoroutinefunction(self.retriever):
                docs = await self.retriever(query, k=self.num_passages)
            else:
                docs = self.retriever(query, k=self.num_passages)
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise


class ChainOfThoughtModule(DSPyModuleBase):
    """Base class for chain-of-thought modules."""
    
    def __init__(
        self,
        steps: Optional[List[str]] = None,
        max_steps: int = 5,
        verbose: bool = False
    ):
        """
        Initialize the chain-of-thought module.
        
        Args:
            steps: Pre-defined reasoning steps
            max_steps: Maximum number of reasoning steps
            verbose: Whether to output verbose reasoning
        """
        super().__init__()
        self.steps = steps or []
        self.max_steps = max_steps
        self.verbose = verbose
    
    def add_step(self, step: str) -> None:
        """
        Add a reasoning step.
        
        Args:
            step: Reasoning step
        """
        self.steps.append(step)
    
    def clear_steps(self) -> None:
        """Clear reasoning steps."""
        self.steps = []


# Export
__all__ = [
    "DSPyModuleProtocol",
    "DSPyModuleBase",
    "RAGModule",
    "ChainOfThoughtModule",
]
