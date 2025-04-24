"""
Enhanced LLM Service implementation.

This module provides a concrete implementation of the ServiceAbstractionLayer
that uses the LLMGatewayClient for core LLM operations and implements additional
capabilities for caching, resilience, observability, events, and progress tracking.
"""

import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union, TypeVar, Callable, Awaitable, Type

from asf.medical.llm_gateway.services.service_abstraction_layer import ServiceAbstractionLayer
from asf.medical.llm_gateway.core.client import LLMGatewayClient
from asf.medical.llm_gateway.core.factory import ProviderFactory
from asf.medical.llm_gateway.cache.cache_manager import get_cache_manager
from asf.medical.llm_gateway.observability.metrics import get_metrics_service
from asf.medical.llm_gateway.events.event_bus import get_event_bus

# Import components
from asf.medical.llm_gateway.services.components.core_operations import CoreOperationsComponent
from asf.medical.llm_gateway.services.components.caching import CachingComponent
from asf.medical.llm_gateway.services.components.resilience import ResilienceComponent
from asf.medical.llm_gateway.services.components.observability import ObservabilityComponent
from asf.medical.llm_gateway.services.components.events import EventsComponent
from asf.medical.llm_gateway.services.components.progress_tracking import ProgressTrackingComponent

logger = logging.getLogger(__name__)
T = TypeVar('T')

class EnhancedLLMService(ServiceAbstractionLayer):
    """
    Enhanced LLM Service implementation.
    
    This class provides a concrete implementation of the ServiceAbstractionLayer
    that uses the LLMGatewayClient for core LLM operations and implements additional
    capabilities for caching, resilience, observability, events, and progress tracking.
    """
    
    def __init__(
        self,
        gateway_client: Optional[LLMGatewayClient] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_caching: bool = True,
        enable_resilience: bool = True,
        enable_observability: bool = True,
        enable_events: bool = True,
        enable_progress_tracking: bool = True,
        db_session = None
    ):
        """
        Initialize the enhanced LLM service.
        
        Args:
            gateway_client: Optional LLMGatewayClient instance to use
            config: Optional configuration dictionary
            enable_caching: Whether to enable caching
            enable_resilience: Whether to enable resilience patterns
            enable_observability: Whether to enable observability
            enable_events: Whether to enable event publishing
            enable_progress_tracking: Whether to enable progress tracking
            db_session: Optional database session
        """
        self.config = config or {}
        self.service_id = self.config.get('service_id', 'enhanced_llm_service')
        self.db_session = db_session
        
        # Feature flags
        self.enable_caching = enable_caching
        self.enable_resilience = enable_resilience
        self.enable_observability = enable_observability
        self.enable_events = enable_events
        self.enable_progress_tracking = enable_progress_tracking
        
        # Initialize gateway client if not provided
        if gateway_client is None:
            provider_factory = ProviderFactory()
            from asf.medical.llm_gateway.core.models import GatewayConfig
            gateway_config = GatewayConfig(**self.config.get('gateway_config', {}))
            self.gateway_client = LLMGatewayClient(gateway_config, provider_factory, db_session)
        else:
            self.gateway_client = gateway_client
        
        # Initialize components
        self._initialize_components()
        
        logger.info(f"Initialized EnhancedLLMService with features: "
                   f"caching={enable_caching}, resilience={enable_resilience}, "
                   f"observability={enable_observability}, events={enable_events}, "
                   f"progress_tracking={enable_progress_tracking}")
    
    def _initialize_components(self):
        """Initialize components based on feature flags."""
        # Initialize cache manager if caching is enabled
        cache_manager = None
        if self.enable_caching:
            cache_config = self.config.get('cache', {})
            cache_manager = get_cache_manager(
                enable_caching=True,
                similarity_threshold=cache_config.get('similarity_threshold', 0.92),
                max_entries=cache_config.get('max_entries', 10000),
                ttl_seconds=cache_config.get('ttl_seconds', 3600),
                persistence_type=cache_config.get('persistence_type', 'disk'),
                persistence_config=cache_config.get('persistence_config')
            )
        
        # Initialize metrics service if observability is enabled
        metrics_service = None
        if self.enable_observability:
            metrics_service = get_metrics_service()
        
        # Initialize event bus if events are enabled
        event_bus = None
        if self.enable_events:
            event_bus = get_event_bus()
        
        # Initialize components
        self.core = CoreOperationsComponent(self.gateway_client)
        self.caching = CachingComponent(cache_manager, self.enable_caching)
        self.resilience = ResilienceComponent(self.enable_resilience)
        self.observability = ObservabilityComponent(metrics_service, self.enable_observability)
        self.events = EventsComponent(event_bus, self.enable_events)
        self.progress = ProgressTrackingComponent(self.enable_progress_tracking)
    
    async def initialize(self) -> None:
        """
        Initialize the service.
        
        This method should be called before using the service to set up
        any necessary resources, connections, or state.
        """
        # Initialize gateway client
        await self.gateway_client.initialize_resources()
        
        # Initialize caching component
        await self.caching.initialize()
        
        logger.info(f"Initialized {self.service_id} service")
    
    async def shutdown(self) -> None:
        """
        Shut down the service.
        
        This method should be called when the service is no longer needed
        to clean up resources and connections.
        """
        # Shut down gateway client
        await self.gateway_client.close()
        
        # Shut down caching component
        await self.caching.shutdown()
        
        logger.info(f"Shut down {self.service_id} service")
    
    # --- Core LLM Operations ---
    
    async def generate_text(self, prompt: str, model: str, params: Dict[str, Any]) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Generated text as string
        """
        # Start span if observability is enabled
        span = self.observability.start_span("generate_text", attributes={
            "model": model
        })
        
        try:
            # Check cache if enabled
            cache_key = f"text:{model}:{hash(prompt)}:{hash(str(params))}"
            cached_response = await self.caching.get_from_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for generate_text")
                self.observability.record_metric("llm.cache.hit", 1, {"operation": "generate_text", "model": model})
                return cached_response
            else:
                self.observability.record_metric("llm.cache.miss", 1, {"operation": "generate_text", "model": model})
            
            # Create progress tracker if enabled
            progress_tracker = self.progress.create_progress_tracker(
                operation_id=f"generate_text_{model}_{hash(prompt)}",
                total_steps=3,
                operation_type="generate_text"
            )
            self.progress.update_progress(progress_tracker, 1, "Preparing request")
            
            # Generate text with resilience if enabled
            if self.enable_resilience:
                result = await self.resilience.with_retry(
                    lambda: self.core.generate_text(prompt, model, params),
                    max_retries=3,
                    retry_delay=1.0,
                    backoff_factor=2.0
                )
            else:
                result = await self.core.generate_text(prompt, model, params)
            
            # Update progress if enabled
            if progress_tracker:
                self.progress.update_progress(progress_tracker, 3, "Processing response")
            
            # Store in cache if enabled
            await self.caching.store_in_cache(cache_key, result)
            
            # Publish event if enabled
            await self.events.publish_event("llm.text_generated", {
                "model": model,
                "prompt_length": len(prompt),
                "response_length": len(result)
            })
            
            return result
        finally:
            # End span if started
            if span:
                self.observability.end_span(span)
    
    async def generate_stream(self, prompt: str, model: str, params: Dict[str, Any]) -> AsyncIterator[str]:
        """
        Stream text generation from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation
            params: Additional parameters for the generation
            
        Returns:
            Asynchronous iterator of generated text chunks
        """
        # Start span if observability is enabled
        span = self.observability.start_span("generate_stream", attributes={
            "model": model
        })
        
        try:
            # Create progress tracker if enabled
            progress_tracker = self.progress.create_progress_tracker(
                operation_id=f"generate_stream_{model}_{hash(prompt)}",
                total_steps=100,  # We'll update this as chunks come in
                operation_type="generate_stream"
            )
            self.progress.update_progress(progress_tracker, 1, "Preparing request")
            
            # Publish event if enabled
            await self.events.publish_event("llm.stream_started", {
                "model": model,
                "prompt_length": len(prompt)
            })
            
            # Generate stream
            chunk_count = 0
            
            async for chunk in self.core.generate_stream(prompt, model, params):
                chunk_count += 1
                
                # Update progress if enabled
                if progress_tracker:
                    progress_step = min(2 + chunk_count, 99)  # Keep within bounds
                    self.progress.update_progress(
                        progress_tracker, 
                        progress_step, 
                        f"Received chunk {chunk_count}"
                    )
                
                # Record metrics if enabled
                self.observability.record_metric("llm.stream.chunks", 1, {"model": model})
                self.observability.record_metric("llm.stream.chunk_size", len(chunk), {"model": model})
                
                # Yield the chunk
                yield chunk
            
            # Update progress if enabled
            if progress_tracker:
                self.progress.update_progress(progress_tracker, 100, "Stream completed")
            
            # Publish event if enabled
            await self.events.publish_event("llm.stream_completed", {
                "model": model,
                "chunk_count": chunk_count
            })
        finally:
            # End span if started
            if span:
                self.observability.end_span(span)
    
    async def chat(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Have a chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Dictionary containing the response
        """
        # Start span if observability is enabled
        span = self.observability.start_span("chat", attributes={
            "model": model
        })
        
        try:
            # Check cache if enabled
            cache_key = f"chat:{model}:{hash(str(messages))}:{hash(str(params))}"
            cached_response = await self.caching.get_from_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for chat")
                self.observability.record_metric("llm.cache.hit", 1, {"operation": "chat", "model": model})
                return cached_response
            else:
                self.observability.record_metric("llm.cache.miss", 1, {"operation": "chat", "model": model})
            
            # Create progress tracker if enabled
            progress_tracker = self.progress.create_progress_tracker(
                operation_id=f"chat_{model}_{hash(str(messages))}",
                total_steps=3,
                operation_type="chat"
            )
            self.progress.update_progress(progress_tracker, 1, "Preparing request")
            
            # Generate chat response with resilience if enabled
            if self.enable_resilience:
                result = await self.resilience.with_retry(
                    lambda: self.core.chat(messages, model, params),
                    max_retries=3,
                    retry_delay=1.0,
                    backoff_factor=2.0
                )
            else:
                result = await self.core.chat(messages, model, params)
            
            # Update progress if enabled
            if progress_tracker:
                self.progress.update_progress(progress_tracker, 3, "Processing response")
            
            # Store in cache if enabled
            await self.caching.store_in_cache(cache_key, result)
            
            # Publish event if enabled
            await self.events.publish_event("llm.chat_completed", {
                "model": model,
                "message_count": len(messages),
                "response_length": len(result.get("choices", [{}])[0].get("message", {}).get("content", ""))
            })
            
            return result
        finally:
            # End span if started
            if span:
                self.observability.end_span(span)
    
    async def chat_stream(self, messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Have a streaming chat-based interaction with the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: The model to use for the chat
            params: Additional parameters for the chat
            
        Returns:
            Asynchronous iterator of response chunks
        """
        # Start span if observability is enabled
        span = self.observability.start_span("chat_stream", attributes={
            "model": model
        })
        
        try:
            # Create progress tracker if enabled
            progress_tracker = self.progress.create_progress_tracker(
                operation_id=f"chat_stream_{model}_{hash(str(messages))}",
                total_steps=100,  # We'll update this as chunks come in
                operation_type="chat_stream"
            )
            self.progress.update_progress(progress_tracker, 1, "Preparing request")
            
            # Publish event if enabled
            await self.events.publish_event("llm.chat_stream_started", {
                "model": model,
                "message_count": len(messages)
            })
            
            # Generate stream
            chunk_count = 0
            
            async for chunk in self.core.chat_stream(messages, model, params):
                chunk_count += 1
                
                # Update progress if enabled
                if progress_tracker:
                    progress_step = min(2 + chunk_count, 99)  # Keep within bounds
                    self.progress.update_progress(
                        progress_tracker, 
                        progress_step, 
                        f"Received chunk {chunk_count}"
                    )
                
                # Record metrics if enabled
                self.observability.record_metric("llm.stream.chunks", 1, {"model": model})
                
                # Yield the chunk
                yield chunk
            
            # Update progress if enabled
            if progress_tracker:
                self.progress.update_progress(progress_tracker, 100, "Stream completed")
            
            # Publish event if enabled
            await self.events.publish_event("llm.chat_stream_completed", {
                "model": model,
                "chunk_count": chunk_count
            })
        finally:
            # End span if started
            if span:
                self.observability.end_span(span)
    
    async def get_embeddings(self, text: Union[str, List[str]], model: str, params: Optional[Dict[str, Any]] = None) -> List[List[float]]:
        """
        Get embeddings for text.
        
        Args:
            text: The text or list of texts to get embeddings for
            model: The model to use for embeddings
            params: Additional parameters for the embedding generation
            
        Returns:
            List of embedding vectors
        """
        # Start span if observability is enabled
        span = self.observability.start_span("get_embeddings", attributes={
            "model": model,
            "text_count": 1 if isinstance(text, str) else len(text)
        })
        
        try:
            # Check cache if enabled
            cache_key = f"embeddings:{model}:{hash(str(text))}:{hash(str(params))}"
            cached_response = await self.caching.get_from_cache(cache_key)
            if cached_response:
                logger.debug(f"Cache hit for embeddings")
                self.observability.record_metric("llm.cache.hit", 1, {"operation": "embeddings", "model": model})
                return cached_response
            else:
                self.observability.record_metric("llm.cache.miss", 1, {"operation": "embeddings", "model": model})
            
            # Get embeddings with resilience if enabled
            if self.enable_resilience:
                result = await self.resilience.with_retry(
                    lambda: self.core.get_embeddings(text, model, params),
                    max_retries=3,
                    retry_delay=1.0,
                    backoff_factor=2.0
                )
            else:
                result = await self.core.get_embeddings(text, model, params)
            
            # Store in cache if enabled
            await self.caching.store_in_cache(cache_key, result)
            
            # Publish event if enabled
            await self.events.publish_event("llm.embeddings_generated", {
                "model": model,
                "text_count": 1 if isinstance(text, str) else len(text)
            })
            
            return result
        finally:
            # End span if started
            if span:
                self.observability.end_span(span)
    
    # --- Tool/Function Calling ---
    
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
        # Start span if observability is enabled
        span = self.observability.start_span("chat_with_tools", attributes={
            "model": model,
            "tool_count": len(tools)
        })
        
        try:
            # Forward to core component
            return await self.core.chat_with_tools(messages, tools, model, params)
        finally:
            # End span if started
            if span:
                self.observability.end_span(span)
    
    # --- Caching Hooks ---
    
    async def get_from_cache(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        return await self.caching.get_from_cache(key)
    
    async def store_in_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Optional time-to-live in seconds
        """
        await self.caching.store_in_cache(key, value, ttl)
    
    async def invalidate_cache(self, key: str) -> None:
        """
        Invalidate a cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        await self.caching.invalidate_cache(key)
    
    async def clear_cache(self) -> None:
        """
        Clear the entire cache.
        """
        await self.caching.clear_cache()
    
    # --- Resilience Hooks ---
    
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
        return await self.resilience.with_retry(
            operation, max_retries, retry_delay, backoff_factor, retryable_errors
        )
    
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
        return await self.resilience.with_circuit_breaker(
            operation, circuit_name, fallback
        )
    
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
        return await self.resilience.with_timeout(
            operation, timeout_seconds
        )
    
    # --- Observability Hooks ---
    
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
        self.observability.record_metric(name, value, tags)
    
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
        return self.observability.start_span(name, parent_span, attributes)
    
    def end_span(self, span: Any) -> None:
        """
        End a tracing span.
        
        Args:
            span: Span to end
        """
        self.observability.end_span(span)
    
    # --- Event Hooks ---
    
    async def publish_event(self, 
                           event_type: str, 
                           payload: Dict[str, Any]) -> None:
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            payload: Event payload
        """
        await self.events.publish_event(event_type, payload)
    
    async def subscribe_to_events(self,
                                 event_type: str,
                                 handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Subscribe to events.
        
        Args:
            event_type: Type of events to subscribe to
            handler: Event handler function
        """
        await self.events.subscribe_to_events(event_type, handler)
    
    # --- Progress Tracking ---
    
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
        return self.progress.create_progress_tracker(
            operation_id, total_steps, operation_type
        )
    
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
        self.progress.update_progress(tracker, step, message, details)
    
    # --- Configuration and Status ---
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current service configuration.
        
        Returns:
            Dictionary containing the service configuration
        """
        return {
            "service_id": self.service_id,
            "enable_caching": self.enable_caching,
            "enable_resilience": self.enable_resilience,
            "enable_observability": self.enable_observability,
            "enable_events": self.enable_events,
            "enable_progress_tracking": self.enable_progress_tracking,
            "config": self.config
        }
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update the service configuration.
        
        Args:
            config: New configuration values
        """
        # Update feature flags
        if "enable_caching" in config:
            self.enable_caching = config["enable_caching"]
        if "enable_resilience" in config:
            self.enable_resilience = config["enable_resilience"]
        if "enable_observability" in config:
            self.enable_observability = config["enable_observability"]
        if "enable_events" in config:
            self.enable_events = config["enable_events"]
        if "enable_progress_tracking" in config:
            self.enable_progress_tracking = config["enable_progress_tracking"]
        
        # Update service ID
        if "service_id" in config:
            self.service_id = config["service_id"]
        
        # Update config dictionary
        if "config" in config:
            self.config.update(config["config"])
        
        # Re-initialize components
        self._initialize_components()
    
    async def get_health(self) -> Dict[str, Any]:
        """
        Get the health status of the service.
        
        Returns:
            Dictionary containing health information
        """
        return {
            "service_id": self.service_id,
            "status": "operational",
            "components": {
                "caching": {
                    "enabled": self.enable_caching,
                    "status": "operational"
                },
                "resilience": {
                    "enabled": self.enable_resilience,
                    "status": "operational",
                    "circuit_breakers": self.resilience.get_all_circuit_breaker_statuses() if self.enable_resilience else {}
                },
                "observability": {
                    "enabled": self.enable_observability,
                    "status": "operational"
                },
                "events": {
                    "enabled": self.enable_events,
                    "status": "operational"
                },
                "progress_tracking": {
                    "enabled": self.enable_progress_tracking,
                    "status": "operational",
                    "active_operations": len(self.progress.get_all_progress(active_only=True)) if self.enable_progress_tracking else 0
                }
            }
        }
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the service.
        
        Returns:
            Dictionary containing service statistics
        """
        return {
            "service_id": self.service_id,
            "metrics": self.observability.get_metrics() if self.enable_observability else {},
            "cache_stats": await self.caching.get_cache_stats() if self.enable_caching else {},
            "active_operations": self.progress.get_all_progress(active_only=True) if self.enable_progress_tracking else {},
            "recent_events": self.events.get_recent_events(limit=10) if self.enable_events else []
        }
