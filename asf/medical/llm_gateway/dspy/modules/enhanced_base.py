"""
Enhanced DSPy Module Base

This module provides enhanced base classes for DSPy modules with additional
functionality like tracing, validation, and telemetry.
"""

import abc
import logging
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic

import dspy

from .base import DSPyModuleBase, DSPyModuleProtocol
from ..utils.circuit_breaker import CircuitBreaker, AsyncCircuitBreaker

# Set up logging
logger = logging.getLogger(__name__)


class EnhancedDSPyModuleProtocol(DSPyModuleProtocol):
    """Protocol for enhanced DSPy modules."""
    
    async def validate(self, result: Dict[str, Any]) -> bool:
        """
        Validate the prediction result.
        
        Args:
            result: Prediction result
            
        Returns:
            bool: Whether the result is valid
        """
        ...
    
    async def trace(self, request: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Trace the prediction request and result.
        
        Args:
            request: Prediction request
            result: Prediction result
        """
        ...


class EnhancedDSPyModuleBase(DSPyModuleBase, abc.ABC):
    """Enhanced base class for DSPy modules with additional functionality."""
    
    def __init__(self):
        """Initialize the module."""
        super().__init__()
        self._trace_enabled = False
        self._validation_enabled = False
        self._telemetry_enabled = False
        self._circuit_breaker = None
    
    @property
    def trace_enabled(self) -> bool:
        """Get whether tracing is enabled."""
        return self._trace_enabled
    
    @trace_enabled.setter
    def trace_enabled(self, value: bool):
        """Set whether tracing is enabled."""
        self._trace_enabled = value
    
    @property
    def validation_enabled(self) -> bool:
        """Get whether validation is enabled."""
        return self._validation_enabled
    
    @validation_enabled.setter
    def validation_enabled(self, value: bool):
        """Set whether validation is enabled."""
        self._validation_enabled = value
    
    @property
    def telemetry_enabled(self) -> bool:
        """Get whether telemetry is enabled."""
        return self._telemetry_enabled
    
    @telemetry_enabled.setter
    def telemetry_enabled(self, value: bool):
        """Set whether telemetry is enabled."""
        self._telemetry_enabled = value
    
    @property
    def circuit_breaker(self) -> Optional[Union[CircuitBreaker, AsyncCircuitBreaker]]:
        """Get circuit breaker."""
        return self._circuit_breaker
    
    @circuit_breaker.setter
    def circuit_breaker(self, value: Optional[Union[CircuitBreaker, AsyncCircuitBreaker]]):
        """Set circuit breaker."""
        self._circuit_breaker = value
    
    async def trace(self, request: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Trace the prediction request and result.
        
        Args:
            request: Prediction request
            result: Prediction result
        """
        if not self._trace_enabled:
            return
        
        trace_id = str(uuid.uuid4())
        logger.debug(f"Trace {trace_id}: Request={request}")
        logger.debug(f"Trace {trace_id}: Result={result}")
    
    async def validate(self, result: Dict[str, Any]) -> bool:
        """
        Validate the prediction result.
        
        Args:
            result: Prediction result
            
        Returns:
            bool: Whether the result is valid
        """
        if not self._validation_enabled:
            return True
        
        # Default implementation just checks if result is not empty
        return bool(result)
    
    async def collect_telemetry(self, start_time: float, end_time: float, result: Dict[str, Any]) -> None:
        """
        Collect telemetry for the prediction.
        
        Args:
            start_time: Start time in seconds since epoch
            end_time: End time in seconds since epoch
            result: Prediction result
        """
        if not self._telemetry_enabled:
            return
        
        duration_ms = (end_time - start_time) * 1000
        logger.debug(f"Telemetry: Duration={duration_ms:.2f}ms")
    
    async def call_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call a function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Any: Function result
        """
        if self._circuit_breaker is None:
            return await func(*args, **kwargs)
        
        if isinstance(self._circuit_breaker, AsyncCircuitBreaker):
            return await self._circuit_breaker.call(func, *args, **kwargs)
        else:
            return self._circuit_breaker.call(func, *args, **kwargs)
    
    async def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Make a prediction using the module with enhanced functionality.
        
        Returns:
            Dict[str, Any]: Prediction result
        """
        # Record start time for telemetry
        start_time = time.time()
        
        # Build request dict
        request = {}
        for i, arg in enumerate(args):
            request[f"arg{i}"] = arg
        request.update(kwargs)
        
        try:
            # Call implementation
            result = await self.call_with_circuit_breaker(self._predict_impl, *args, **kwargs)
            
            # Validate result if enabled
            if self._validation_enabled and not await self.validate(result):
                logger.warning("Validation failed for prediction result")
            
            # Record end time and collect telemetry
            end_time = time.time()
            await self.collect_telemetry(start_time, end_time, result)
            
            # Trace request and result if enabled
            await self.trace(request, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {str(e)}")
            raise
    
    @abc.abstractmethod
    async def _predict_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Implementation of the prediction logic.
        
        Returns:
            Dict[str, Any]: Prediction result
        """
        raise NotImplementedError("Subclasses must implement _predict_impl()")


class EnhancedRAGModule(EnhancedDSPyModuleBase):
    """Enhanced base class for RAG modules."""
    
    def __init__(
        self,
        retriever: Optional[Any] = None,
        num_passages: int = 3,
        rerank: bool = False,
        filter_threshold: Optional[float] = None
    ):
        """
        Initialize the enhanced RAG module.
        
        Args:
            retriever: Document retriever
            num_passages: Number of passages to retrieve
            rerank: Whether to rerank retrieved passages
            filter_threshold: Relevance threshold for filtering passages (0-1)
        """
        super().__init__()
        self.retriever = retriever
        self.num_passages = num_passages
        self.rerank = rerank
        self.filter_threshold = filter_threshold
    
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
            
            # Filter passages if threshold is set
            if self.filter_threshold is not None:
                docs = [
                    doc for doc in docs
                    if getattr(doc, "relevance_score", 1.0) >= self.filter_threshold
                ]
            
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise
    
    async def validate(self, result: Dict[str, Any]) -> bool:
        """
        Validate the RAG prediction result.
        
        Args:
            result: Prediction result
            
        Returns:
            bool: Whether the result is valid
        """
        if not super().validate(result):
            return False
        
        # Check if retrieved_documents is in the result
        if "retrieved_documents" not in result:
            logger.warning("Retrieved documents missing from RAG result")
            return False
        
        return True


class EnhancedChainOfThoughtModule(EnhancedDSPyModuleBase):
    """Enhanced base class for chain-of-thought modules."""
    
    def __init__(
        self,
        steps: Optional[List[str]] = None,
        max_steps: int = 5,
        verbose: bool = False,
        validate_steps: bool = False
    ):
        """
        Initialize the enhanced chain-of-thought module.
        
        Args:
            steps: Pre-defined reasoning steps
            max_steps: Maximum number of reasoning steps
            verbose: Whether to output verbose reasoning
            validate_steps: Whether to validate reasoning steps
        """
        super().__init__()
        self.steps = steps or []
        self.max_steps = max_steps
        self.verbose = verbose
        self.validate_steps = validate_steps
    
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
    
    async def validate(self, result: Dict[str, Any]) -> bool:
        """
        Validate the chain-of-thought prediction result.
        
        Args:
            result: Prediction result
            
        Returns:
            bool: Whether the result is valid
        """
        if not super().validate(result):
            return False
        
        # Check if reasoning_steps is in the result and not empty
        if "reasoning_steps" not in result or not result["reasoning_steps"]:
            logger.warning("Reasoning steps missing from chain-of-thought result")
            return False
        
        # Validate steps if enabled
        if self.validate_steps and isinstance(result["reasoning_steps"], list):
            # Make sure the number of steps is within limits
            if len(result["reasoning_steps"]) > self.max_steps:
                logger.warning(f"Too many reasoning steps: {len(result['reasoning_steps'])}")
                return False
        
        return True


# Export
__all__ = [
    "EnhancedDSPyModuleProtocol",
    "EnhancedDSPyModuleBase",
    "EnhancedRAGModule",
    "EnhancedChainOfThoughtModule",
]
