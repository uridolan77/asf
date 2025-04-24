"""
Enhanced Service Abstraction Layer for Domain-Agnostic LLM Gateway.

This module defines an enhanced service abstraction layer that extends the
basic LLMServiceInterface with additional capabilities for caching, resilience,
observability, events, and progress tracking.
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union, TypeVar, Generic, Callable, Awaitable, Type
import asyncio
from datetime import datetime

from asf.conexus.llm_gateway.interfaces.llm_service import LLMServiceInterface
from asf.conexus.llm_gateway.core.models import (
    LLMRequest, LLMResponse, LLMConfig, Context,
    ContentItem, MCPRole, StreamChunk, ToolDefinition, ToolFunction
)

T = TypeVar('T')

class ServiceAbstractionLayer(LLMServiceInterface, ABC):
    """
    Enhanced Service Abstraction Layer for Domain-Agnostic LLM Gateway.
    
    This abstract class extends the basic LLMServiceInterface with additional
    capabilities for caching, resilience, observability, events, and progress tracking.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the service.
        
        This method should be called before using the service to set up
        any necessary resources, connections, or state.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shut down the service.
        
        This method should be called when the service is no longer needed
        to clean up resources and connections.
        """
        pass
    
    # --- Tool/Function Calling ---
    
    @abstractmethod
    async def chat_with_tools(self,
                             messages: List[Dict[str, str]],
                             tools: List[Dict[str, Any]],
                             model: str,
                             params: Dict[str, Any],
                             request_id: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Have a chat-based interaction with the model with tool calling capabilities.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: List of tool definitions
            model: The model to use for the chat
            params: Additional parameters for the chat
            request_id: Optional request ID for tracking
            context: Optional context information
            
        Returns:
            Dictionary containing the response with tool calls
        """
        pass
    
    # --- Caching Hooks ---
    
    @abstractmethod
    async def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        pass
    
    @abstractmethod
    async def store_in_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Optional time-to-live in seconds
        """
        pass
    
    @abstractmethod
    async def invalidate_cache(self, key: str) -> None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        pass
    
    @abstractmethod
    async def clear_cache(self) -> None:
        """
        Clear the entire cache.
        """
        pass
    
    # --- Resilience Hooks ---
    
    @abstractmethod
    async def with_retry(self, 
                        operation: Callable[[], Awaitable[T]], 
                        max_retries: int = 3,
                        retry_delay: float = 1.0,
                        backoff_factor: float = 2.0,
                        retryable_errors: Optional[List[Type[Exception]]] = None) -> T:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Async operation to execute
            max_retries: Maximum number of retries
            retry_delay: Initial delay between retries in seconds
            backoff_factor: Factor to increase delay between retries
            retryable_errors: List of exception types that should trigger a retry
            
        Returns:
            Result of the operation
        """
        pass
    
    @abstractmethod
    async def with_circuit_breaker(self,
                                  operation: Callable[[], Awaitable[T]],
                                  circuit_name: str,
                                  fallback: Optional[Callable[[], Awaitable[T]]] = None) -> T:
        """
        Execute an operation with circuit breaker protection.
        
        Args:
            operation: Async operation to execute
            circuit_name: Name of the circuit breaker
            fallback: Optional fallback operation if circuit is open
            
        Returns:
            Result of the operation or fallback
        """
        pass
    
    @abstractmethod
    async def with_timeout(self,
                          operation: Callable[[], Awaitable[T]],
                          timeout_seconds: float) -> T:
        """
        Execute an operation with a timeout.
        
        Args:
            operation: Async operation to execute
            timeout_seconds: Timeout in seconds
            
        Returns:
            Result of the operation
        """
        pass
    
    # --- Observability Hooks ---
    
    @abstractmethod
    def record_metric(self, 
                     name: str, 
                     value: Union[int, float, str], 
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        pass
    
    @abstractmethod
    def start_span(self, 
                  name: str, 
                  parent_span: Optional[Any] = None,
                  attributes: Optional[Dict[str, str]] = None) -> Any:
        """
        Start a new tracing span.
        
        Args:
            name: Span name
            parent_span: Optional parent span
            attributes: Optional span attributes
            
        Returns:
            Span object
        """
        pass
    
    @abstractmethod
    def end_span(self, span: Any) -> None:
        """
        End a tracing span.
        
        Args:
            span: Span to end
        """
        pass
    
    # --- Event Hooks ---
    
    @abstractmethod
    async def publish_event(self, 
                           event_type: str, 
                           payload: Dict[str, Any]) -> None:
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        pass
    
    @abstractmethod
    async def subscribe_to_events(self,
                                 event_type: str,
                                 handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Subscribe to events.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Event handler function
        """
        pass
    
    # --- Progress Tracking ---
    
    @abstractmethod
    def create_progress_tracker(self,
                               operation_id: str,
                               total_steps: int = 100,
                               operation_type: str = "llm_request") -> Any:
        """
        Create a progress tracker for a long-running operation.
        
        Args:
            operation_id: Unique identifier for the operation
            total_steps: Total number of steps in the operation
            operation_type: Type of operation
            
        Returns:
            Progress tracker object
        """
        pass
    
    @abstractmethod
    def update_progress(self,
                       tracker: Any,
                       step: int,
                       message: str,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the progress of an operation.
        
        Args:
            tracker: Progress tracker object
            step: Current step number
            message: Progress message
            details: Optional details about this update
        """
        pass
    
    # --- Configuration and Status ---
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current service configuration.
        
        Returns:
            Dictionary containing the service configuration
        """
        pass
    
    @abstractmethod
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the service configuration.
        
        Args:
            config: New configuration values
        """
        pass
    
    @abstractmethod
    async def get_health(self) -> Dict[str, Any]:
        """
        Get the health status of the service.
        
        Returns:
            Dictionary containing health information
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the service.
        
        Returns:
            Dictionary containing service statistics
        """
        pass