"""
Cached provider wrapper for the LLM Gateway.

This module provides a wrapper for LLM providers that adds caching capabilities.
"""

import hashlib
import json
import logging
from typing import AsyncGenerator, Dict, Any, Optional, Type

from asf.conexus.llm_gateway.providers.base import BaseProvider
from asf.conexus.llm_gateway.core.models import LLMRequest, LLMResponse, StreamChunk, ProviderConfig

# Import caching utilities - adjust import paths based on your project structure
try:
    from asf.medical.llm_gateway.dspy.utils.caching import (
        CacheInterface, DiskCache, RedisCache, NullCache, create_cache
    )
except ImportError:
    # Create simplified versions if the imports aren't available
    class CacheInterface:
        def get(self, key: str) -> Optional[Any]: ...
        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...
        def delete(self, key: str) -> None: ...
        def clear(self) -> None: ...
        def close(self) -> None: ...
    
    class DiskCache(CacheInterface):
        def __init__(self, cache_dir: str = "./cache", **kwargs): ...
        def get(self, key: str) -> Optional[Any]: return None
        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: pass
        def delete(self, key: str) -> None: pass
        def clear(self) -> None: pass
        def close(self) -> None: pass
    
    class RedisCache(CacheInterface):
        def __init__(self, host: str = "localhost", port: int = 6379, **kwargs): ...
        def get(self, key: str) -> Optional[Any]: return None
        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: pass
        def delete(self, key: str) -> None: pass
        def clear(self) -> None: pass
        def close(self) -> None: pass
    
    class NullCache(CacheInterface):
        def get(self, key: str) -> Optional[Any]: return None
        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: pass
        def delete(self, key: str) -> None: pass
        def clear(self) -> None: pass
        def close(self) -> None: pass
    
    def create_cache(backend: str = "disk", **kwargs) -> CacheInterface:
        if backend == "disk": return DiskCache(**kwargs)
        elif backend == "redis": return RedisCache(**kwargs)
        else: return NullCache()

logger = logging.getLogger(__name__)


class CachedProviderWrapper(BaseProvider):
    """
    Provider wrapper that adds caching capabilities to any LLM provider.
    
    This wrapper intercepts requests to the underlying provider and
    caches responses to improve performance and reduce costs.
    """
    
    def __init__(
        self,
        provider: BaseProvider,
        provider_id: str,
        cache: Optional[CacheInterface] = None,
        cache_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the cached provider wrapper.
        
        Args:
            provider: The underlying provider to wrap
            provider_id: Provider identifier
            cache: Optional pre-configured cache instance
            cache_config: Cache configuration if no cache instance is provided
        """
        # Initialize with the wrapped provider's config
        super().__init__(provider.provider_config)
        
        self.provider = provider
        self.provider_id = provider_id
        
        # Set up cache
        self.cache = cache or self._setup_cache(cache_config or {})
        
        # Default TTL (12 hours)
        self.default_ttl = cache_config.get("default_ttl", 12 * 60 * 60) if cache_config else 12 * 60 * 60
        
        # Track cache hits/misses for metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"Initialized cached provider wrapper for '{provider_id}'")
    
    def _setup_cache(self, config: Dict[str, Any]) -> CacheInterface:
        """
        Set up the cache based on configuration.
        
        Args:
            config: Cache configuration
            
        Returns:
            CacheInterface: Cache instance
        """
        backend = config.get("backend", "disk")
        logger.info(f"Setting up {backend} cache for provider '{self.provider_id}'")
        
        try:
            return create_cache(backend=backend, **config.get("options", {}))
        except Exception as e:
            logger.warning(f"Failed to create {backend} cache: {str(e)}. Using NullCache instead.")
            return NullCache()
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        """
        Generate a cache key for an LLM request.
        
        Args:
            request: LLM request
            
        Returns:
            str: Cache key
        """
        # Create a dict with all the elements we want to include in the cache key
        key_data = {
            "model": request.config.model_identifier,
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_tokens,
            "prompt": request.prompt_content,
            "system_message": request.system_message,
            "stop_sequences": request.config.stop_sequences
        }
        
        # Create a deterministic JSON representation for hashing
        key_json = json.dumps(key_data, sort_keys=True)
        
        # Hash the key data
        hash_key = hashlib.md5(key_json.encode()).hexdigest()
        
        # Add prefix for easy identification
        return f"llm:{self.provider_id}:{hash_key}"
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response, using the cache if available.
        
        Args:
            request: LLM request
            
        Returns:
            LLMResponse: Response from the LLM
        """
        # Check if caching is disabled for this request
        if request.config.additional_params.get("disable_cache", False):
            logger.debug("Caching disabled for this request")
            return await self.provider.generate(request)
        
        # Generate cache key
        cache_key = self._get_cache_key(request)
        
        # Try to get from cache
        cached_response = self.cache.get(cache_key)
        
        if cached_response:
            self.cache_hits += 1
            logger.debug(f"Cache hit for key '{cache_key}'")
            
            # Update request_id to ensure uniqueness
            cached_response.request_id = request.request_id
            return cached_response
        
        # Cache miss, forward to provider
        self.cache_misses += 1
        logger.debug(f"Cache miss for key '{cache_key}'")
        
        # Get response from provider
        response = await self.provider.generate(request)
        
        # Cache the response
        ttl = request.config.additional_params.get("cache_ttl", self.default_ttl)
        self.cache.set(cache_key, response, ttl=ttl)
        
        return response
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response.
        
        Note: Streaming responses are not cached because they are
        intended for real-time interaction.
        
        Args:
            request: LLM request
            
        Yields:
            StreamChunk: Stream chunks
        """
        # Just forward to the provider without caching
        async for chunk in self.provider.generate_stream(request):
            yield chunk
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            self.cache.close()
        except Exception as e:
            logger.warning(f"Error closing cache: {e}")
        
        await self.provider.cleanup()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check."""
        # Get the underlying provider's health check
        provider_health = await self.provider.health_check()
        
        # Add cache metrics
        cache_health = {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0
        }
        
        return {
            "provider": provider_health,
            "cache": cache_health
        }
    
    def get_model_info(self, model_identifier: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        return self.provider.get_model_info(model_identifier)
    
    @property
    def supported_models(self) -> Dict[str, Any]:
        """Get all models supported by this provider."""
        return self.provider.supported_models


def create_cached_provider(
    provider: BaseProvider,
    provider_id: str,
    cache_config: Optional[Dict[str, Any]] = None
) -> CachedProviderWrapper:
    """
    Create a cached provider wrapper.
    
    Args:
        provider: Provider to wrap
        provider_id: Provider identifier
        cache_config: Cache configuration
        
    Returns:
        CachedProviderWrapper: Cached provider wrapper
    """
    return CachedProviderWrapper(
        provider=provider,
        provider_id=provider_id,
        cache_config=cache_config
    )