"""
LLM Gateway for Conexus.

This module provides a gateway for LLM requests that handles provider selection,
caching, fallbacks, and observability.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast

from asf.conexus.llm_gateway.core.models import (
    LLMRequest, 
    LLMResponse, 
    StreamChunk,
    ErrorDetails,
    ErrorLevel,
    FinishReason,
    UsageStats
)
from asf.conexus.llm_gateway.cache.cache_manager import get_cache_manager
from asf.conexus.llm_gateway.services.provider_registry import get_provider_registry
from asf.conexus.llm_gateway.observability.metrics import record_metric
from asf.conexus.llm_gateway.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class LLMGateway:
    """
    Main gateway for LLM requests.
    
    This class provides a central point for handling LLM requests,
    including provider selection, caching, and observability.
    """
    
    def __init__(
        self,
        enable_cache: bool = True,
        max_retries: int = 2,
        fallback_provider_ids: Optional[List[str]] = None,
        default_provider_id: Optional[str] = None,
    ):
        """
        Initialize LLM Gateway.
        
        Args:
            enable_cache: Whether to use the cache
            max_retries: Maximum number of retries for failed requests
            fallback_provider_ids: Provider IDs to use as fallbacks if the primary provider fails
            default_provider_id: Default provider ID to use if none is specified in the request
        """
        self.enable_cache = enable_cache
        self.max_retries = max_retries
        self.fallback_provider_ids = fallback_provider_ids or []
        self.default_provider_id = default_provider_id
        
        # These will be lazily initialized when needed
        self._registry = None
        self._cache_manager = None
        
        logger.info(
            f"Initialized LLM Gateway with cache_enabled={enable_cache}, "
            f"max_retries={max_retries}"
        )
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate an LLM response for a request.
        
        This method handles provider selection, caching, retries, and fallbacks.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        start_time = time.time()
        cache_lookup_time = 0.0
        hit_cache = False
        
        # Initialize registry and cache manager if not already initialized
        if not self._registry:
            self._registry = get_provider_registry()
        
        # Try to get from cache first if enabled
        if self.enable_cache:
            try:
                cache_start = time.time()
                
                # Lazy initialization
                if not self._cache_manager:
                    self._cache_manager = get_cache_manager()
                
                # Check cache for a match
                cached_response = await self._cache_manager.get_response(request)
                cache_lookup_time = time.time() - cache_start
                
                # If found in cache, return immediately
                if cached_response:
                    logger.info(
                        f"Cache hit for request {request.request_id} "
                        f"(model: {request.model_identifier}, lookup: {cache_lookup_time*1000:.2f}ms)"
                    )
                    
                    # Update metrics
                    record_metric("llm_gateway_cache_hit", 1)
                    record_metric("llm_gateway_cache_lookup_time", cache_lookup_time)
                    
                    # Set hit_cache flag for tracking
                    hit_cache = True
                    
                    # Update response with cache information
                    if cached_response.metadata is None:
                        cached_response.metadata = {}
                    cached_response.metadata["cache_hit"] = True
                    cached_response.metadata["cache_lookup_time_ms"] = round(cache_lookup_time * 1000, 2)
                    
                    # Track overall latency
                    total_time = time.time() - start_time
                    record_metric("llm_gateway_total_latency", total_time)
                    
                    return cached_response
                else:
                    # Cache miss
                    record_metric("llm_gateway_cache_miss", 1)
                    record_metric("llm_gateway_cache_lookup_time", cache_lookup_time)
                    logger.debug(
                        f"Cache miss for request {request.request_id} "
                        f"(model: {request.model_identifier}, lookup: {cache_lookup_time*1000:.2f}ms)"
                    )
            except Exception as e:
                logger.warning(f"Error checking cache: {e}")
                # Continue with normal generation if cache lookup fails
                record_metric("llm_gateway_cache_error", 1)
        
        # Get the provider ID from the request or use the default
        provider_id = request.provider_id or self.default_provider_id
        if not provider_id:
            return self._create_error_response(
                request=request,
                message="No provider specified and no default provider configured",
                error_code="NO_PROVIDER",
                level=ErrorLevel.ERROR
            )
        
        # Get an ordered list of providers to try
        providers_to_try = [provider_id] + [
            p for p in self.fallback_provider_ids if p != provider_id
        ]
        
        # Try each provider in order
        last_error = None
        for attempt, current_provider_id in enumerate(providers_to_try):
            try:
                # Get provider
                provider = await self._registry.get_provider(current_provider_id)
                if not provider:
                    logger.warning(f"Provider {current_provider_id} not found")
                    continue
                
                # Make the request
                logger.info(
                    f"Sending request {request.request_id} to provider {current_provider_id} "
                    f"(model: {request.model_identifier}, attempt: {attempt+1}/{len(providers_to_try)})"
                )
                
                generation_start = time.time()
                response = await provider.generate(request)
                generation_time = time.time() - generation_start
                
                # Record metrics
                record_metric("llm_gateway_provider_latency", generation_time, {"provider": current_provider_id})
                if attempt > 0:
                    record_metric("llm_gateway_fallback_used", 1)
                
                if response.error_details:
                    # Provider returned an error response
                    logger.warning(
                        f"Provider {current_provider_id} returned error: {response.error_details.message} "
                        f"(code: {response.error_details.error_code})"
                    )
                    record_metric("llm_gateway_provider_error", 1, {"provider": current_provider_id})
                    last_error = response.error_details
                    
                    # If error is retryable and we have more providers, continue to next provider
                    if response.error_details.retryable and attempt < len(providers_to_try) - 1:
                        continue
                else:
                    # Successful response
                    logger.info(
                        f"Got successful response from provider {current_provider_id} "
                        f"for request {request.request_id} in {generation_time*1000:.2f}ms"
                    )
                    
                    # Store in cache if enabled and response is valid
                    if self.enable_cache and response.content and not hit_cache:
                        try:
                            if not self._cache_manager:
                                self._cache_manager = get_cache_manager()
                                
                            await self._cache_manager.store_response(request, response)
                        except Exception as e:
                            logger.warning(f"Error storing response in cache: {e}")
                    
                    # Calculate total time
                    total_time = time.time() - start_time
                    record_metric("llm_gateway_total_latency", total_time)
                    
                    # Update response with timing information
                    if response.metadata is None:
                        response.metadata = {}
                    response.metadata["cache_hit"] = False
                    response.metadata["provider_latency_ms"] = round(generation_time * 1000, 2)
                    response.metadata["total_latency_ms"] = round(total_time * 1000, 2)
                    response.metadata["cache_lookup_time_ms"] = round(cache_lookup_time * 1000, 2) if cache_lookup_time > 0 else 0
                    
                    return response
            except Exception as e:
                logger.error(f"Error with provider {current_provider_id}: {e}")
                record_metric("llm_gateway_provider_exception", 1, {"provider": current_provider_id})
                last_error = ErrorDetails(
                    message=f"Provider error: {str(e)}",
                    error_code="PROVIDER_EXCEPTION",
                    level=ErrorLevel.ERROR,
                    source=current_provider_id
                )
        
        # If we get here, all providers failed
        record_metric("llm_gateway_all_providers_failed", 1)
        
        # Create an error response
        error_response = self._create_error_response(
            request=request,
            message=last_error.message if last_error else "All providers failed",
            error_code=last_error.error_code if last_error else "ALL_PROVIDERS_FAILED",
            level=ErrorLevel.ERROR,
            source=last_error.source if last_error else None
        )
        
        # Update with timing information
        if error_response.metadata is None:
            error_response.metadata = {}
        error_response.metadata["cache_hit"] = False
        error_response.metadata["total_latency_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return error_response
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for a request.
        
        This method handles provider selection and retries for streaming requests.
        Note that caching is not supported for streaming requests.
        
        Args:
            request: The LLM request
            
        Yields:
            Stream chunks from the provider
        """
        # Initialize registry if not already initialized
        if not self._registry:
            self._registry = get_provider_registry()
            
        # Get the provider ID from the request or use the default
        provider_id = request.provider_id or self.default_provider_id
        if not provider_id:
            # Return a single error chunk
            yield StreamChunk(
                index=0,
                request_id=request.request_id,
                error=ErrorDetails(
                    message="No provider specified and no default provider configured",
                    error_code="NO_PROVIDER",
                    level=ErrorLevel.ERROR
                )
            )
            return
            
        # Get an ordered list of providers to try
        providers_to_try = [provider_id] + [
            p for p in self.fallback_provider_ids if p != provider_id
        ]
        
        # Try each provider in order
        last_error = None
        for attempt, current_provider_id in enumerate(providers_to_try):
            try:
                # Get provider
                provider = await self._registry.get_provider(current_provider_id)
                if not provider:
                    logger.warning(f"Provider {current_provider_id} not found")
                    continue
                    
                # Make the streaming request
                logger.info(
                    f"Sending streaming request {request.request_id} to provider {current_provider_id} "
                    f"(model: {request.model_identifier}, attempt: {attempt+1}/{len(providers_to_try)})"
                )
                
                # Record attempt
                if attempt > 0:
                    record_metric("llm_gateway_streaming_fallback_used", 1)
                    
                # Track success
                success = True
                chunk_count = 0
                generation_start = time.time()
                
                # Stream the response
                async for chunk in provider.generate_stream(request):
                    chunk_count += 1
                    
                    # If there's an error in the chunk, we need to try the next provider
                    if chunk.error:
                        success = False
                        last_error = chunk.error
                        logger.warning(
                            f"Provider {current_provider_id} returned error in stream: "
                            f"{chunk.error.message} (code: {chunk.error.error_code})"
                        )
                        record_metric("llm_gateway_streaming_provider_error", 1, {"provider": current_provider_id})
                        
                        # Only break if the error is not retryable or we've exhausted all providers
                        if not chunk.error.retryable or attempt >= len(providers_to_try) - 1:
                            # Pass the error chunk to the client
                            yield chunk
                        break
                        
                    # Pass the chunk to the client
                    yield chunk
                    
                generation_time = time.time() - generation_start
                record_metric("llm_gateway_streaming_provider_latency", generation_time, {"provider": current_provider_id})
                record_metric("llm_gateway_streaming_chunk_count", chunk_count, {"provider": current_provider_id})
                
                # If successful, we're done
                if success:
                    logger.info(
                        f"Completed streaming from provider {current_provider_id} "
                        f"for request {request.request_id} in {generation_time*1000:.2f}ms "
                        f"({chunk_count} chunks)"
                    )
                    return
                    
                # If not successful but we have more providers, continue to the next one
                if attempt < len(providers_to_try) - 1:
                    logger.info(f"Trying next provider for streaming request {request.request_id}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error with streaming provider {current_provider_id}: {e}")
                record_metric("llm_gateway_streaming_provider_exception", 1, {"provider": current_provider_id})
                last_error = ErrorDetails(
                    message=f"Provider streaming error: {str(e)}",
                    error_code="PROVIDER_STREAMING_EXCEPTION",
                    level=ErrorLevel.ERROR,
                    source=current_provider_id
                )
                
                # If we have more providers, continue to the next one
                if attempt < len(providers_to_try) - 1:
                    continue
                    
                # Otherwise, return an error chunk
                yield StreamChunk(
                    index=0,
                    request_id=request.request_id,
                    error=last_error
                )
        
        # If we get here, all providers failed
        record_metric("llm_gateway_streaming_all_providers_failed", 1)
        
        # Return a final error chunk if we didn't already
        if last_error:
            yield StreamChunk(
                index=0,
                request_id=request.request_id,
                error=last_error
            )
        else:
            # This should not happen, but just in case
            yield StreamChunk(
                index=0,
                request_id=request.request_id,
                error=ErrorDetails(
                    message="All streaming providers failed",
                    error_code="ALL_STREAMING_PROVIDERS_FAILED",
                    level=ErrorLevel.ERROR
                )
            )

    def _create_error_response(
        self,
        request: LLMRequest,
        message: str,
        error_code: str,
        level: ErrorLevel = ErrorLevel.ERROR,
        source: Optional[str] = None
    ) -> LLMResponse:
        """
        Create an error response.
        
        Args:
            request: The original request
            message: Error message
            error_code: Error code
            level: Error level
            source: Error source
            
        Returns:
            An LLMResponse with error details
        """
        return LLMResponse(
            request_id=request.request_id,
            provider_id=request.provider_id,
            model_id=request.model_identifier,
            content=None,
            created_at=datetime.now(timezone.utc),
            error_details=ErrorDetails(
                message=message,
                error_code=error_code,
                level=level,
                source=source or "llm_gateway"
            )
        )


# Global instance
_llm_gateway = None


def get_gateway() -> LLMGateway:
    """
    Get the global LLM Gateway instance.
    
    Returns:
        The global LLM Gateway instance
    """
    global _llm_gateway
    
    if _llm_gateway is None:
        _llm_gateway = LLMGateway()
        
    return _llm_gateway