"""
Caching intervention for LLM Gateway.

This intervention implements request/response caching to improve performance
and reduce costs by avoiding duplicate requests to LLM providers.
"""

import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any

from asf.medical.llm_gateway.core.cache import get_cache
from asf.medical.llm_gateway.core.models import (
    LLMRequest,
    LLMResponse,
    InterventionContext,
    CacheConfig,
    CacheMetadata
)
from asf.medical.llm_gateway.interventions.base import BaseIntervention

logger = logging.getLogger(__name__)


class CachingIntervention(BaseIntervention):
    """
    Intervention that provides caching for LLM requests/responses.
    
    This intervention:
    1. Checks if a request can be served from cache before passing to provider
    2. Stores responses in the cache after provider execution
    3. Updates cache statistics and adds cache metadata to responses
    """
    
    name = "cache_intervention"
    hook_type = "pre_post"  # Run both before and after provider execution
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the caching intervention.
        
        Args:
            config: Dictionary containing cache configuration
        """
        super().__init__(config)
        
        # Convert the config dict to a CacheConfig pydantic model
        cache_config_dict = self.config.get("cache_config", {})
        self.cache_config = CacheConfig(
            enabled=cache_config_dict.get("enabled", True),
            cache_ttl_seconds=cache_config_dict.get("cache_ttl_seconds", 3600),
            use_redis=cache_config_dict.get("use_redis", False),
            redis_url=cache_config_dict.get("redis_url"),
            exclude_models=cache_config_dict.get("exclude_models", []),
            exclude_providers=cache_config_dict.get("exclude_providers", []),
            cache_embeddings=cache_config_dict.get("cache_embeddings", True)
        )
        
        # Get the global cache instance with our config
        self.cache = get_cache(self.cache_config)
        logger.info(f"Initialized cache intervention (enabled: {self.cache_config.enabled})")
    
    async def initialize_async(self):
        """
        Asynchronous initialization - nothing to do here as the cache is initialized
        in the constructor.
        """
        pass
    
    def _should_skip_cache(self, request: LLMRequest) -> bool:
        """
        Determine if caching should be skipped for this request.
        
        Args:
            request: The LLM request to check
            
        Returns:
            True if caching should be skipped, False otherwise
        """
        # Return early if caching is disabled globally
        if not self.cache_config.enabled:
            return True
        
        # Check if the model is in the excluded models list
        model_id = request.config.model_identifier
        if model_id in self.cache_config.exclude_models:
            logger.debug(f"Skipping cache for excluded model: {model_id}")
            return True
        
        # Check if streaming is requested (we don't cache streaming requests)
        if request.stream:
            logger.debug(f"Skipping cache for streaming request: {request.initial_context.request_id}")
            return True
        
        # Check if the request has specific markers to bypass cache
        if request.extensions.experimental_features.get("skip_cache", False):
            logger.debug(f"Skipping cache due to skip_cache flag: {request.initial_context.request_id}")
            return True
        
        # If temperature is 0, we always cache (deterministic)
        # If temperature > 0, we might not want to cache depending on config
        if request.config.temperature and request.config.temperature > 0:
            # Skip cache for non-deterministic requests if configured to do so
            if self.config.get("skip_cache_for_non_deterministic", False):
                logger.debug(f"Skipping cache for non-deterministic request (temp={request.config.temperature})")
                return True
        
        # Determine the provider if available
        provider_id = request.config.model_kwargs.get("provider_id")
        if provider_id in self.cache_config.exclude_providers:
            logger.debug(f"Skipping cache for excluded provider: {provider_id}")
            return True
        
        return False
    
    async def process_request(self, request: LLMRequest, context: InterventionContext) -> LLMRequest:
        """
        Check if the request can be served from cache.
        
        If a cached response is found, block further processing and store the
        cached response in the context to be returned later.
        
        Args:
            request: The incoming LLM request
            context: The mutable intervention context
            
        Returns:
            The unchanged request (caching doesn't modify the request itself)
        """
        # Store processing start time for metrics
        start_time = time.time()
        request_id = request.initial_context.request_id
        
        # Check if we should skip cache for this request
        if self._should_skip_cache(request):
            logger.debug(f"Cache check skipped for request {request_id}")
            return request
        
        # Try to get the response from cache
        cached_response = await self.cache.get(request)
        
        if cached_response is not None:
            # Cache hit! Store the cached response in the context
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Cache hit for request {request_id} (took {elapsed_ms:.2f}ms)")
            
            # Store the cached response directly in the context
            context.intervention_data.set("cached_response", cached_response)
            
            # Signal to the intervention manager to skip provider execution
            context.intervention_data.set("skip_provider_execution", True)
            context.intervention_data.set("skip_provider_reason", "cache_hit")
            
            # Add cache metadata to track this was a cache hit
            stored_at = getattr(cached_response, "_stored_at", datetime.utcnow())
            ttl = self.cache_config.cache_ttl_seconds
            remaining_ttl = max(0, ttl - int((datetime.utcnow() - stored_at).total_seconds()))
            
            # Attach cache metadata to the context for later use
            context.intervention_data.set("cache_metadata", CacheMetadata(
                cache_hit=True,
                cache_key=context.intervention_data.get("cache_key"),
                ttl_seconds_remaining=remaining_ttl,
                stored_at=stored_at
            ))
        else:
            # Cache miss, will proceed with normal provider execution
            elapsed_ms = (time.time() - start_time) * 1000
            logger.debug(f"Cache miss for request {request_id} (took {elapsed_ms:.2f}ms)")
        
        return request
    
    async def process_response(self, response: LLMResponse, context: InterventionContext) -> LLMResponse:
        """
        Process the provider response, which might be:
        1. A cached response we retrieved earlier, or
        2. A fresh response from the provider that we need to cache
        
        Args:
            response: The LLM response (either from cache or from provider)
            context: The mutable intervention context
            
        Returns:
            The LLM response, with cache metadata added if it was a cache hit
        """
        # Check if we have a cached response from the pre-intervention phase
        cached_response = context.intervention_data.get("cached_response")
        
        if cached_response is not None:
            # This was a cache hit, already processed in pre-intervention
            logger.debug(f"Returning cached response for request {response.request_id}")
            
            # Get the cache metadata we prepared earlier
            cache_metadata = context.intervention_data.get("cache_metadata")
            
            # Create a new response with the cache metadata attached
            # We need to create a new response because LLMResponse is frozen (immutable)
            updated_response = LLMResponse(
                **{**cached_response.model_dump(), "cache_metadata": cache_metadata}
            )
            
            return updated_response
        
        # This was a fresh response from the provider, check if we should cache it
        request_id = response.request_id
        
        # Reconstruct the original request from the context
        # The original request is not directly available in the response,
        # but we can recreate it from the context and response info
        # In a production system, you might want to pass the original request in the context
        # during the pre-intervention phase
        
        # Skip caching if there was an error
        if response.error_details is not None:
            logger.debug(f"Not caching response with error for request {request_id}")
            return response
        
        # Get the original request from the context if available
        original_request = context.intervention_data.get("original_request")
        
        if original_request is None:
            logger.warning(f"Cannot cache response for request {request_id}: Original request not found in context")
            return response
        
        # Check if we should skip cache for this request
        if self._should_skip_cache(original_request):
            return response
        
        # Store the response in the cache
        try:
            # Record the storage time for TTL calculation later
            setattr(response, "_stored_at", datetime.utcnow())
            await self.cache.set(original_request, response)
            logger.debug(f"Cached response for request {request_id}")
            
            # Add cache metadata to indicate this was just cached
            cache_metadata = CacheMetadata(
                cache_hit=False,
                cache_key=None,  # We don't need to expose the key here
                ttl_seconds_remaining=self.cache_config.cache_ttl_seconds,
                stored_at=datetime.utcnow()
            )
            
            # Create a new response with the cache metadata attached
            updated_response = LLMResponse(
                **{**response.model_dump(), "cache_metadata": cache_metadata}
            )
            
            return updated_response
            
        except Exception as e:
            logger.error(f"Error caching response for request {request_id}: {e}")
        
        return response
    
    async def cleanup(self) -> None:
        """
        Clean up resources - nothing specific needed here as the cache
        cleanup is handled by the cache service itself.
        """
        pass