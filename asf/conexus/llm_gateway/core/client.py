"""
LLM client interface for the Conexus LLM Gateway.

This module provides client interfaces for applications to interact
with the LLM Gateway, including both a core client and a higher-level
client with progress tracking capabilities.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from asf.conexus.llm_gateway.core.models import (
    LLMRequest, 
    LLMResponse, 
    StreamChunk,
    GatewayConfig, 
    InterventionContext, 
    ErrorDetails, 
    ErrorLevel,
    FinishReason,
    PerformanceMetrics,
    ContentItem, 
    ToolUseRequest, 
    ToolDefinition,
    UsageStats,
    LLMConfig,
    BatchLLMRequest,
    BatchLLMResponse
)
from asf.conexus.llm_gateway.core.factory import ProviderFactory, ProviderFactoryError
from asf.conexus.llm_gateway.core.manager import InterventionManager
from asf.conexus.llm_gateway.core.config_loader import ConfigLoader
from asf.conexus.llm_gateway.core.resource_init import setup_resource_layer, shutdown_resource_layer, get_resource_stats
from asf.conexus.llm_gateway.core.errors import ResourceError, CircuitBreakerError

# Import progress tracking functionality if available
try:
    from asf.conexus.llm_gateway.progress.models import ProgressTracker
    from asf.conexus.llm_gateway.progress.service import get_progress_service
    from asf.conexus.llm_gateway.progress import (
        track_llm_progress, get_progress_tracker, OperationType,
        get_progress_registry
    )
    PROGRESS_TRACKING_AVAILABLE = True
except ImportError:
    PROGRESS_TRACKING_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMGatewayClient:
    """
    Core client for the LLM Gateway.
    
    This class provides low-level access to the LLM Gateway functionality,
    including request processing, streaming, and resource management.
    """
    
    def __init__(self, config: GatewayConfig = None, provider_factory: Optional[ProviderFactory] = None, db=None):
        """
        Initialize the LLM Gateway client.
        
        Args:
            config: Gateway configuration
            provider_factory: Provider factory for creating LLM providers
            db: Database session for configuration loading
        """
        self.db = db
        self.config_loader = ConfigLoader(db)
        self._resource_layer_initialized = False

        # Load configuration from database if not provided
        if config is None and db is not None:
            config_dict = self.config_loader.load_config()
            self.config = GatewayConfig(**config_dict)
        else:
            self.config = config

        if self.config is None:
            raise ValueError("Gateway configuration is required. Provide either config or db.")

        # Use a central factory instance
        self.provider_factory = provider_factory or ProviderFactory()
        # Pass the factory to the manager along with the database session
        self.intervention_manager = InterventionManager(self.provider_factory, self.config, self.db)
        # Configure semaphore from GatewayConfig
        batch_limit = self.config.additional_config.get("max_concurrent_batch_requests", 10)
        self._batch_semaphore = asyncio.Semaphore(batch_limit)
        logger.info(f"Initialized LLMGatewayClient with batch concurrency limit: {batch_limit}")

    async def initialize_resources(self):
        """Initialize the resource management layer."""
        if not self._resource_layer_initialized:
            await setup_resource_layer(register_signals=True)
            self._resource_layer_initialized = True
            logger.info("Resource management layer initialized")

    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Process a single LLM request.
        
        Args:
            request: The LLM request to process
            
        Returns:
            The LLM response
        """
        logger.info(f"Processing generate request: {request.initial_context.request_id}")
        try:
            if not self._resource_layer_initialized:
                await self.initialize_resources()
            response = await self.intervention_manager.process_request(request)
            logger.info(f"Request successful: {request.initial_context.request_id}")
            return response
        except ProviderFactoryError as e:
            logger.error(f"Provider creation failed for request {request.initial_context.request_id}: {e}", exc_info=True)
            return self._create_error_response(request, e, "PROVIDER_INIT_FAILED")
        except ResourceError as e:
            logger.error(f"Resource error in request {request.initial_context.request_id}: {e}", exc_info=True)
            return self._create_error_response(request, e, "RESOURCE_ERROR", retryable=True)
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker tripped for request {request.initial_context.request_id}: {e}", exc_info=True)
            return self._create_error_response(request, e, "CIRCUIT_BREAKER_TRIPPED", retryable=True)
        except Exception as e:
            logger.error(f"Request processing failed: {request.initial_context.request_id}", exc_info=True)
            return self._create_error_response(request, e)

    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Process a streaming LLM request.
        
        Args:
            request: The LLM request to process
            
        Yields:
            Stream chunks of the LLM response
        """
        request_id = request.initial_context.request_id
        logger.info(f"Processing stream request: {request_id}")
        try:
            if not self._resource_layer_initialized:
                await self.initialize_resources()
            async for chunk in self.intervention_manager.process_stream(request):
                yield chunk
            logger.info(f"Stream finished successfully: {request_id}")
        except ProviderFactoryError as e:
            logger.error(f"Provider creation failed for stream {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, e, "PROVIDER_INIT_FAILED")
        except ResourceError as e:
            logger.error(f"Resource error in stream {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, e, "RESOURCE_ERROR")
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker tripped for stream {request_id}: {e}", exc_info=True)
            yield self._create_error_chunk(request_id, e, "CIRCUIT_BREAKER_TRIPPED")
        except Exception as e:
            logger.error(f"Stream processing failed: {request_id}", exc_info=True)
            yield self._create_error_chunk(request_id, e)

    async def process_batch(self, batch_request: BatchLLMRequest) -> BatchLLMResponse:
        """
        Process a batch of LLM requests.
        
        Args:
            batch_request: The batch of LLM requests to process
            
        Returns:
            Batch response with individual responses for each request
        """
        batch_start_time = datetime.now(timezone.utc)
        logger.info(f"Processing batch request: {batch_request.batch_id} ({len(batch_request.requests)} requests)")
        if not self._resource_layer_initialized:
            await self.initialize_resources()
        async with self._batch_semaphore:
            tasks = [
                self._process_single_request_in_batch(req)
                for req in batch_request.requests
            ]
            results_with_requests = await asyncio.gather(*tasks, return_exceptions=True)

            valid_responses: List[LLMResponse] = []
            for req, result in results_with_requests:
                if isinstance(result, Exception):
                    logger.warning(f"Sub-request {req.initial_context.request_id} in batch {batch_request.batch_id} failed.", exc_info=result)
                    valid_responses.append(self._create_error_response(req, result))
                else:
                    valid_responses.append(result)

            total_duration_ms = (datetime.now(timezone.utc) - batch_start_time).total_seconds() * 1000
            logger.info(f"Batch processing complete: {batch_request.batch_id}, Duration: {total_duration_ms:.2f}ms")
            return BatchLLMResponse(
                batch_id=batch_request.batch_id,
                responses=valid_responses,
                total_duration_ms=total_duration_ms
            )

    async def _process_single_request_in_batch(self, request: LLMRequest) -> Tuple[LLMRequest, LLMResponse]:
        """
        Process a single request within a batch.
        
        Args:
            request: The request to process
            
        Returns:
            Tuple of the original request and its response
        """
        try:
            response = await self.intervention_manager.process_request(request)
            return request, response
        except Exception as e:
            raise e

    async def close(self):
        """Clean up resources used by the client."""
        logger.info("Shutting down gateway client and cleaning up providers...")
        await self.provider_factory.cleanup_all()
        if self._resource_layer_initialized:
            try:
                await shutdown_resource_layer()
                self._resource_layer_initialized = False
                logger.info("Resource management layer shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down resource management layer: {e}", exc_info=True)
        logger.info("Gateway client shutdown complete.")

    def _create_error_response(self, request: Optional[LLMRequest], error: Exception, default_code: str = "CLIENT_PROCESSING_ERROR", retryable: bool = False) -> LLMResponse:
        """
        Create an error response for a failed request.
        
        Args:
            request: The original request
            error: The exception that occurred
            default_code: Default error code to use
            retryable: Whether the error is retryable
            
        Returns:
            An error response
        """
        request_id = request.initial_context.request_id if request else f"error_{uuid.uuid4()}"
        context = request.initial_context if request else InterventionContext(request_id=request_id)

        error_code = default_code
        message = str(error)
        level = ErrorLevel.ERROR
        provider_details = {"exception_type": type(error).__name__, "details": str(error)}

        if isinstance(error, ResourceError):
            error_code = "RESOURCE_ERROR"
            retryable = True
            if hasattr(error, 'provider_id'):
                provider_details['provider_id'] = error.provider_id
            if hasattr(error, 'resource_type'):
                provider_details['resource_type'] = error.resource_type
            if hasattr(error, 'operation'):
                provider_details['operation'] = error.operation
        elif isinstance(error, CircuitBreakerError):
            error_code = "CIRCUIT_BREAKER_TRIPPED"
            retryable = True
            level = ErrorLevel.WARNING
            if hasattr(error, 'provider_id'):
                provider_details['provider_id'] = error.provider_id
            if hasattr(error, 'failure_count'):
                provider_details['failure_count'] = error.failure_count
            if hasattr(error, 'reset_timeout_seconds'):
                provider_details['reset_timeout_seconds'] = error.reset_timeout_seconds

        error_details = ErrorDetails(
            code=error_code,
            message=message,
            level=level,
            retryable=retryable,
            provider_error_details=provider_details
        )

        duration = (datetime.now(timezone.utc) - context.timestamp_start).total_seconds() * 1000 if context else 0

        return LLMResponse(
            version="1.0",
            request_id=request_id,
            generated_content=None,
            error_details=error_details,
            final_context=context,
            finish_reason=FinishReason.ERROR,
            performance_metrics=PerformanceMetrics(
                total_duration_ms=duration,
            )
        )

    def _create_error_chunk(self, request_id: str, error: Exception, error_code: str = "STREAM_PROCESSING_ERROR") -> StreamChunk:
        """
        Create an error chunk for a streaming response.
        
        Args:
            request_id: ID of the request
            error: The exception that occurred
            error_code: Error code to use
            
        Returns:
            A stream chunk containing error details
        """
        retryable = isinstance(error, (ResourceError, CircuitBreakerError))
        error_details = ErrorDetails(
            code=error_code,
            message=str(error),
            level=ErrorLevel.ERROR,
            retryable=retryable,
            provider_error_details={"exception_type": type(error).__name__}
        )
        return StreamChunk(
            chunk_id=999,
            request_id=request_id,
            finish_reason=FinishReason.ERROR,
            provider_specific_data={"error": error_details.model_dump()}
        )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of providers and resources.
        
        Returns:
            Dictionary with health status for each provider
        """
        status = {}
        if self._resource_layer_initialized:
            try:
                resource_stats = get_resource_stats()
                status["resource_pools"] = resource_stats
            except Exception as e:
                logger.error(f"Failed to get resource pool stats: {e}", exc_info=True)
                status["resource_pools"] = {"error": str(e)}

        provider_configs = getattr(self.config, 'providers', {})
        if not provider_configs:
            logger.warning("No provider configurations found in GatewayConfig for health check.")
            provider_ids_to_check = self.config.allowed_providers
            if provider_ids_to_check and not provider_configs:
                logger.warning("Checking allowed_providers without full configs.")
                for provider_id in provider_ids_to_check:
                    status[provider_id] = {"status": "unknown", "error": "Missing provider configuration"}
                return status
            elif not provider_ids_to_check:
                return {"status": "no_providers_configured"}

        for provider_id, provider_conf in provider_configs.items():
            try:
                provider = await self.provider_factory.get_provider(provider_id, provider_conf, self.config)
                status[provider_id] = await provider.health_check()
            except Exception as e:
                status[provider_id] = {"status": "unhealthy", "provider_id": provider_id, "error": f"Failed to get/check provider: {str(e)}"}
        return status

    async def warmup_providers(self):
        """Pre-initialize configured providers."""
        await self.initialize_resources()
        preload_ids = getattr(self.config, 'preload_providers', [])
        provider_configs = getattr(self.config, 'providers', {})

        logger.info(f"Warming up providers: {preload_ids}")
        warmed_up = []
        for provider_id in preload_ids:
            provider_conf = provider_configs.get(provider_id)
            if provider_conf:
                try:
                    await self.provider_factory.get_provider(provider_id, provider_conf, self.config)
                    warmed_up.append(provider_id)
                except Exception as e:
                    logger.error(f"Failed to warm up provider {provider_id}: {e}")
            else:
                logger.warning(f"Cannot warm up provider {provider_id}: Configuration not found.")
        logger.info(f"Successfully warmed up providers: {warmed_up}")

    async def get_resource_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all resource pools.
        
        Returns:
            Dictionary with statistics for each resource pool
        """
        if not self._resource_layer_initialized:
            return {"status": "resource_layer_not_initialized"}
        try:
            return get_resource_stats()
        except Exception as e:
            logger.error(f"Failed to get resource stats: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


if PROGRESS_TRACKING_AVAILABLE:
    class ProgressTrackingLLMClient(LLMGatewayClient):
        """
        LLM Gateway client with integrated progress tracking.
        
        This class extends the base LLM Gateway client with progress tracking
        capabilities, automatically tracking the progress of all LLM operations.
        """
        
        def __init__(self, config: GatewayConfig = None, provider_factory: Optional[ProviderFactory] = None, db=None):
            """
            Initialize the progress tracking LLM client.
            
            Args:
                config: Gateway configuration
                provider_factory: Provider factory
                db: Database session
            """
            super().__init__(config, provider_factory, db)
            logger.info("Initializing progress tracking LLM client")
            
            # Initialize progress registry
            self.progress_registry = get_progress_registry()
        
        @track_llm_progress(operation_type=OperationType.LLM_REQUEST, total_steps=5)
        async def generate(self, request: LLMRequest) -> LLMResponse:
            """
            Process a single LLM request with progress tracking.
            
            Args:
                request: LLM request
                
            Returns:
                LLM response
            """
            # Get the current tracker
            tracker = get_progress_tracker()
            
            # Step 1: Initialize request
            tracker.update(1, "Initializing request")
            
            # Add request ID to logger context
            logger.info(f"Processing generate request with progress tracking: {request.initial_context.request_id}")
            
            try:
                # Step 2: Pre-processing
                tracker.update(2, "Pre-processing request")
                
                # Step 3: Provider selection
                tracker.update(3, "Selecting provider")
                
                # Step 4: Generating response
                tracker.update(4, "Generating response")
                
                # Delegate to intervention manager
                response = await self.intervention_manager.process_request(request)
                
                # Step 5: Post-processing
                tracker.update(5, "Post-processing response")
                
                logger.info(f"Request successful: {request.initial_context.request_id}")
                return response
            except ProviderFactoryError as e:
                # Handle factory errors specifically
                logger.error(f"Provider creation failed for request {request.initial_context.request_id}: {e}", exc_info=True)
                tracker.fail(f"Provider creation failed: {str(e)}")
                return self._create_error_response(request, e, "PROVIDER_INIT_FAILED")
            except Exception as e:
                # Handle other errors
                logger.error(f"Request processing failed: {request.initial_context.request_id}", exc_info=True)
                tracker.fail(f"Request processing failed: {str(e)}")
                return self._create_error_response(request, e)
        
        @track_llm_progress(operation_type=OperationType.LLM_STREAMING, total_steps=5)
        async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
            """
            Process a streaming request with progress tracking.
            
            Args:
                request: LLM request
                
            Yields:
                Stream chunks
            """
            # Get the current tracker
            tracker = get_progress_tracker()
            
            # Step 1: Initialize request
            tracker.update(1, "Initializing streaming request")
            
            request_id = request.initial_context.request_id
            logger.info(f"Processing stream request with progress tracking: {request_id}")
            
            try:
                # Step 2: Pre-processing
                tracker.update(2, "Pre-processing streaming request")
                
                # Step 3: Provider selection
                tracker.update(3, "Selecting provider for streaming")
                
                # Step 4: Starting stream
                tracker.update(4, "Starting response stream")
                
                # Process the stream
                chunk_count = 0
                async for chunk in self.intervention_manager.process_stream(request):
                    chunk_count += 1
                    
                    # Update progress periodically (every 10 chunks)
                    if chunk_count % 10 == 0:
                        tracker.update(4, f"Streaming response (chunks: {chunk_count})")
                    
                    yield chunk
                
                # Step 5: Completing stream
                tracker.update(5, f"Stream completed (total chunks: {chunk_count})")
                
                logger.info(f"Stream finished successfully: {request_id}")
            except ProviderFactoryError as e:
                logger.error(f"Provider creation failed for stream {request_id}: {e}", exc_info=True)
                tracker.fail(f"Provider creation failed: {str(e)}")
                yield self._create_error_chunk(request_id, e, "PROVIDER_INIT_FAILED")
            except Exception as e:
                logger.error(f"Stream processing failed: {request_id}", exc_info=True)
                tracker.fail(f"Stream processing failed: {str(e)}")
                yield self._create_error_chunk(request_id, e)


class LLMClient:
    """
    Higher-level client interface for interacting with the LLM Gateway.
    
    This class provides an easy-to-use interface for applications to send
    prompts to LLMs via the Gateway, with options for progress tracking.
    """
    
    def __init__(
        self,
        provider_id: str,
        model_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ):
        """
        Initialize an LLM client.
        
        Args:
            provider_id: ID of the provider to use
            model_id: ID of the model to use
            user_id: Optional user ID for attribution
            session_id: Optional session ID for grouping requests
            conversation_id: Optional conversation ID for context
        """
        self.provider_id = provider_id
        self.model_id = model_id
        self.user_id = user_id
        self.session_id = session_id or f"session_{uuid.uuid4().hex}"
        self.conversation_id = conversation_id or f"conv_{uuid.uuid4().hex}"
        
        # Internal state
        self._provider = None  # Will be lazily initialized when needed
        self._gateway_client = None  # Will be lazily initialized when needed
        self._metrics_enabled = True
        self._metrics_prefix = f"llm_client_{provider_id}"
        
        logger.debug(
            f"Initialized LLM client for provider {provider_id}, model {model_id}"
            f" (user: {user_id}, session: {self.session_id}, conversation: {self.conversation_id})"
        )
    
    async def generate(
        self,
        prompt: Union[str, List[ContentItem]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        tools: Optional[List[ToolDefinition]] = None,
        system_prompt: Optional[str] = None,
        stop_sequences: Optional[List[str]] = None,
        track_progress: bool = False,
        progress_task_name: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Union[LLMResponse, AsyncGenerator[StreamChunk, None], Tuple[AsyncGenerator[StreamChunk, None], ProgressTracker]]:
        """
        Generate a response using the configured LLM.
        
        Args:
            prompt: Text prompt or list of content items
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            tools: Optional list of tools the model can use
            system_prompt: Optional system instructions
            stop_sequences: Optional list of sequences to stop generation
            track_progress: Whether to track progress of this generation
            progress_task_name: Optional name for the progress task
            additional_params: Optional additional parameters to pass to the provider
            
        Returns:
            If not streaming: LLMResponse with the generated content
            If streaming without tracking: AsyncGenerator yielding StreamChunks
            If streaming with tracking: Tuple of (AsyncGenerator, ProgressTracker)
        """
        # Ensure provider is initialized
        if not self._gateway_client:
            await self._initialize_client()
        
        # Create the request ID
        request_id = f"req_{uuid.uuid4().hex}"
        
        # Create configuration
        llm_config = LLMConfig(
            model_identifier=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            system_prompt=system_prompt
        )
        
        # Add any additional parameters
        if additional_params:
            for key, value in additional_params.items():
                setattr(llm_config, key, value)
        
        # Create context
        context = InterventionContext(
            request_id=request_id,
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            timestamp_start=datetime.now(timezone.utc)
        )
        
        # Create the request object
        request = LLMRequest(
            version="1.0",
            prompt_content=prompt,
            config=llm_config,
            initial_context=context,
            stream=stream,
            tools=tools
        )
        
        progress_tracker = None
        
        # Create a progress tracker if requested and available
        if track_progress and PROGRESS_TRACKING_AVAILABLE:
            progress_service = get_progress_service()
            progress_tracker = await progress_service.create_tracker(
                task_id=request_id,
                task_type="generation",
                name=progress_task_name or f"Generation with {self.model_id}",
                description=f"Text generation using provider {self.provider_id}, model {self.model_id}",
                user_id=self.user_id,
                metadata={
                    "model_id": self.model_id,
                    "provider_id": self.provider_id,
                    "request_id": request_id,
                    "conversation_id": self.conversation_id,
                    "session_id": self.session_id,
                    "prompt_length": len(prompt) if isinstance(prompt, str) else sum(len(item.text_content or "") for item in prompt),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": stream
                }
            )
            
            await progress_service.update_tracker(
                tracker_id=progress_tracker.id,
                message="Generation started",
                status="starting",
                detail=f"Sending request to provider {self.provider_id}"
            )
        
        # Handle streaming vs. non-streaming differently
        if stream:
            # For streaming, return an async generator
            if track_progress and PROGRESS_TRACKING_AVAILABLE:
                # With tracking, we need to wrap the stream to update progress
                stream_gen = self._track_streaming_progress(
                    self._gateway_client.generate_stream(request),
                    progress_tracker.id if progress_tracker else None,
                    max_tokens
                )
                # Return both the generator and the tracker
                return stream_gen, progress_tracker
            else:
                # Without tracking, just pass through the stream
                return self._gateway_client.generate_stream(request)
        else:
            # For non-streaming, execute the request and return the result
            try:
                if track_progress and PROGRESS_TRACKING_AVAILABLE:
                    await get_progress_service().update_tracker(
                        tracker_id=progress_tracker.id,
                        message="Generating response",
                        status="in_progress"
                    )
                
                response = await self._gateway_client.generate(request)
                
                if track_progress and PROGRESS_TRACKING_AVAILABLE:
                    if response.error_details:
                        # Handle error
                        await get_progress_service().fail_tracker(
                            tracker_id=progress_tracker.id,
                            message=f"Generation failed: {response.error_details.message}",
                            detail=f"Error code: {response.error_details.code}",
                            error={
                                "error_code": response.error_details.code,
                                "message": response.error_details.message,
                                "level": response.error_details.level.value
                            }
                        )
                    else:
                        # Handle success
                        await get_progress_service().complete_tracker(
                            tracker_id=progress_tracker.id,
                            message="Generation completed",
                            detail=f"Generated {response.usage.completion_tokens if response.usage else 'unknown'} tokens",
                            metadata={
                                "token_count": response.usage.completion_tokens if response.usage else None,
                                "completion_percentage": 100.0
                            }
                        )
                
                return response
                
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                
                if track_progress and PROGRESS_TRACKING_AVAILABLE:
                    await get_progress_service().fail_tracker(
                        tracker_id=progress_tracker.id,
                        message=f"Generation failed with exception",
                        detail=str(e),
                        error={
                            "exception": str(e),
                            "type": type(e).__name__
                        }
                    )
                
                raise
    
    async def _initialize_client(self) -> None:
        """Initialize the gateway client."""
        from asf.conexus.llm_gateway.core.factory import ProviderFactory
        from asf.conexus.llm_gateway.core.models import GatewayConfig
        
        # Create a minimal config for the client
        config = GatewayConfig(
            gateway_id="llm_client_gateway",
            default_provider=self.provider_id,
            default_model_identifier=self.model_id
        )
        
        # Create provider factory
        factory = ProviderFactory()
        
        # Create the gateway client with progress tracking if available
        if PROGRESS_TRACKING_AVAILABLE:
            self._gateway_client = ProgressTrackingLLMClient(config=config, provider_factory=factory)
        else:
            self._gateway_client = LLMGatewayClient(config=config, provider_factory=factory)
            
        logger.debug(f"Initialized LLM Gateway client for {self.provider_id}")
    
    async def _track_streaming_progress(
        self,
        stream: AsyncGenerator[StreamChunk, None],
        tracker_id: Optional[str],
        max_tokens: int
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Wrap a streaming generator to track progress.
        
        Args:
            stream: The original stream
            tracker_id: ID of the progress tracker
            max_tokens: Maximum number of tokens to generate
            
        Yields:
            Stream chunks with progress tracking updates
        """
        if not tracker_id or not PROGRESS_TRACKING_AVAILABLE:
            # If no tracker ID or progress tracking not available, just pass through the stream
            async for chunk in stream:
                yield chunk
            return
        
        progress_service = get_progress_service()
        token_count = 0
        chunk_count = 0
        start_time = datetime.utcnow()
        
        try:
            # Update tracker to in_progress
            await progress_service.update_tracker(
                tracker_id=tracker_id,
                message="Generating response",
                status="in_progress",
                metadata={
                    "current_step": 0,
                    "total_steps": max_tokens
                }
            )
            
            # Process the stream
            async for chunk in stream:
                chunk_count += 1
                
                # Update token count if available
                if chunk.usage_update:
                    token_count = chunk.usage_update.completion_tokens
                elif chunk.delta_text:
                    # Rough estimate if usage not provided
                    # This is very approximate, could be improved
                    token_count += len(chunk.delta_text.split()) / 1.5
                
                # Periodically update progress (not every token)
                if chunk_count % 5 == 0 or chunk.finish_reason:
                    # Calculate progress metrics
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    tokens_per_second = token_count / elapsed if elapsed > 0 else 0
                    estimated_total_time = max_tokens / tokens_per_second if tokens_per_second > 0 else None
                    
                    # Update progress tracker
                    completion_pct = min(100.0, (token_count / max_tokens) * 100) if max_tokens > 0 else None
                    await progress_service.update_tracker(
                        tracker_id=tracker_id,
                        message=f"Generated {int(token_count)} tokens",
                        metadata={
                            "token_count": int(token_count),
                            "current_step": int(token_count),
                            "completion_percentage": completion_pct
                        }
                    )
                
                # Handle completion
                if chunk.finish_reason:
                    if chunk.error:
                        # Handle error
                        await progress_service.fail_tracker(
                            tracker_id=tracker_id,
                            message=f"Generation failed: {chunk.error.message}",
                            detail=f"Error code: {chunk.error.code}",
                            error={
                                "error_code": chunk.error.code,
                                "message": chunk.error.message,
                                "level": chunk.error.level.value if chunk.error.level else "error"
                            }
                        )
                    else:
                        # Handle success
                        await progress_service.complete_tracker(
                            tracker_id=tracker_id,
                            message="Generation completed",
                            detail=f"Generated {int(token_count)} tokens",
                            metadata={
                                "token_count": int(token_count),
                                "completion_percentage": 100.0
                            }
                        )
                
                # Pass through the chunk
                yield chunk
                
        except Exception as e:
            logger.error(f"Error during streaming: {e}")
            
            # Update tracker with error
            await progress_service.fail_tracker(
                tracker_id=tracker_id,
                message=f"Streaming failed with exception",
                detail=str(e),
                error={
                    "exception": str(e),
                    "type": type(e).__name__
                }
            )
            
            raise

# Global instances
_llm_gateway_client = None

def get_gateway_client(
    config: Optional[GatewayConfig] = None, 
    provider_factory: Optional[ProviderFactory] = None,
    db=None,
    enable_progress_tracking: bool = False
) -> Union[LLMGatewayClient, ProgressTrackingLLMClient]:
    """
    Get a global LLM Gateway client instance.
    
    Args:
        config: Gateway configuration
        provider_factory: Provider factory
        db: Database session
        enable_progress_tracking: Whether to enable progress tracking
        
    Returns:
        An LLM Gateway client instance
    """
    global _llm_gateway_client
    
    if _llm_gateway_client is None:
        if enable_progress_tracking and PROGRESS_TRACKING_AVAILABLE:
            _llm_gateway_client = ProgressTrackingLLMClient(
                config=config,
                provider_factory=provider_factory,
                db=db
            )
        else:
            _llm_gateway_client = LLMGatewayClient(
                config=config,
                provider_factory=provider_factory,
                db=db
            )
    
    return _llm_gateway_client