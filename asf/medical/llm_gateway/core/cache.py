"""
Cache service for LLM Gateway requests.

This module provides a caching mechanism for LLM requests to improve performance
and reduce costs by avoiding duplicate requests to LLM providers.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, Tuple

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from asf.medical.llm_gateway.core.models import (
    LLMRequest, LLMResponse, CacheConfig
)

logger = logging.getLogger(__name__)

class CacheKey:
    """
    Helper class for generating and parsing cache keys.
    """
    
    @staticmethod
    def generate_key(request: LLMRequest) -> str:
        """
        Generate a cache key for a request based on its content and configuration.
        
        Args:
            request: The LLM request to generate a key for
            
        Returns:
            A string key that uniquely identifies this request for caching
        """
        # Extract the components that make this request unique
        model_id = request.model_config.model_id
        provider_id = request.model_config.provider_id
        temperature = request.model_config.temperature
        max_tokens = request.model_config.max_tokens
        
        # For messages/prompt content, we need to normalize and hash
        content_hash = CacheKey._hash_content(request)
        
        # Include system fingerprint if present
        system_fingerprint = request.initial_context.provider_specific_data.get("system_fingerprint", "")
        
        # Combine all components into a key
        key_components = [
            f"model:{model_id}",
            f"provider:{provider_id}",
            f"temp:{temperature}",
            f"max_tokens:{max_tokens}",
            f"content:{content_hash}",
        ]
        
        if system_fingerprint:
            key_components.append(f"fingerprint:{system_fingerprint}")
        
        # Check if tools are present and add their signatures to the key
        if request.model_config.tools:
            tools_hash = CacheKey._hash_tools(request.model_config.tools)
            key_components.append(f"tools:{tools_hash}")
            
        # Include any additional parameters that affect the output
        for key, value in request.model_config.model_kwargs.items():
            # Skip parameters that don't affect output deterministically
            if key in ['stream', 'user', 'request_timeout', 'headers']:
                continue
                
            # Hash complex values, use simple values directly
            if isinstance(value, (dict, list, tuple, set)):
                param_value = hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()[:8]
            else:
                param_value = str(value)
                
            key_components.append(f"{key}:{param_value}")
        
        # Generate the final key
        full_key = "llm_cache:" + ":".join(key_components)
        short_key = "llm_cache:" + hashlib.sha256(full_key.encode('utf-8')).hexdigest()
        
        logger.debug(f"Generated cache key: {short_key} from request {request.initial_context.request_id}")
        return short_key
    
    @staticmethod
    def _hash_content(request: LLMRequest) -> str:
        """
        Hash the content of the request (messages or prompt).
        
        Args:
            request: The LLM request
            
        Returns:
            A hash string representing the content
        """
        # Extract and normalize content
        content_str = ""
        
        if request.messages:
            # For chat models, serialize messages
            for msg in request.messages:
                role = msg.role
                # Handle different content types
                if isinstance(msg.content, str):
                    content_text = msg.content
                elif isinstance(msg.content, list):
                    # Handle multiple content parts (text, images, etc.)
                    content_parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and "text" in part:
                            content_parts.append(part["text"])
                        elif isinstance(part, dict) and "image_url" in part:
                            # Hash image URLs or base64 data
                            img_data = part["image_url"]
                            if isinstance(img_data, dict) and "url" in img_data:
                                content_parts.append(f"image:{hashlib.sha256(img_data['url'].encode('utf-8')).hexdigest()[:16]}")
                            else:
                                content_parts.append(f"image:{hashlib.sha256(str(img_data).encode('utf-8')).hexdigest()[:16]}")
                    content_text = " ".join(content_parts)
                else:
                    content_text = str(msg.content)
                    
                content_str += f"{role}:{content_text}\n"
        elif request.prompt:
            # For completion models, use the prompt directly
            content_str = request.prompt
            
        # Generate a hash of the content
        content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        return content_hash
        
    @staticmethod
    def _hash_tools(tools: List[Dict[str, Any]]) -> str:
        """
        Hash tool definitions to include in the cache key.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            A hash string representing the tools
        """
        # Sort and serialize tools to ensure consistent hashing
        tools_json = json.dumps(tools, sort_keys=True)
        return hashlib.sha256(tools_json.encode('utf-8')).hexdigest()[:16]


class LLMCacheBase:
    """
    Base class for LLM request caching implementations.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize the cache with configuration.
        
        Args:
            config: Cache configuration including TTL values
        """
        self.config = config
        self.ttl = config.cache_ttl_seconds
        self.enabled = config.enabled
        self._hit_count = 0
        self._miss_count = 0
        
    async def get(self, request: LLMRequest) -> Optional[LLMResponse]:
        """
        Get a cached response for a request if available.
        
        Args:
            request: The LLM request to get a cached response for
            
        Returns:
            Cached LLMResponse if available, None otherwise
        """
        if not self.enabled:
            return None
            
        cache_key = CacheKey.generate_key(request)
        result = await self._get(cache_key)
        
        if result is not None:
            self._hit_count += 1
            logger.info(f"Cache hit for request {request.initial_context.request_id} (hits: {self._hit_count}, misses: {self._miss_count})")
            return result
            
        self._miss_count += 1
        logger.info(f"Cache miss for request {request.initial_context.request_id} (hits: {self._hit_count}, misses: {self._miss_count})")
        return None
        
    async def set(self, request: LLMRequest, response: LLMResponse) -> None:
        """
        Store a response in the cache.
        
        Args:
            request: The original LLM request
            response: The response to cache
        """
        if not self.enabled:
            return
            
        if not response.generated_content or response.error_details:
            # Don't cache error responses or empty content
            return
            
        cache_key = CacheKey.generate_key(request)
        await self._set(cache_key, response)
        
    async def _get(self, key: str) -> Optional[LLMResponse]:
        """
        Get a value from the cache by key. Must be implemented by subclasses.
        
        Args:
            key: The cache key to retrieve
            
        Returns:
            Cached LLMResponse if available, None otherwise
        """
        raise NotImplementedError("Subclasses must implement _get()")
        
    async def _set(self, key: str, value: LLMResponse) -> None:
        """
        Store a value in the cache. Must be implemented by subclasses.
        
        Args:
            key: The cache key
            value: The LLMResponse to cache
        """
        raise NotImplementedError("Subclasses must implement _set()")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "total": total_requests,
            "hit_rate": round(hit_rate * 100, 2),
            "enabled": self.enabled,
            "ttl_seconds": self.ttl
        }
        
    async def clear(self) -> int:
        """
        Clear all cached items.
        
        Returns:
            Number of items cleared
        """
        raise NotImplementedError("Subclasses must implement clear()")


class InMemoryLLMCache(LLMCacheBase):
    """
    In-memory implementation of LLM request caching.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize the in-memory cache with configuration.
        
        Args:
            config: Cache configuration
        """
        super().__init__(config)
        self._cache: Dict[str, Tuple[LLMResponse, float]] = {}
        self._lock = asyncio.Lock()
        logger.info("Initialized in-memory LLM cache")
        
    async def _get(self, key: str) -> Optional[LLMResponse]:
        """
        Get a value from the in-memory cache by key.
        
        Args:
            key: The cache key to retrieve
            
        Returns:
            Cached LLMResponse if available and not expired, None otherwise
        """
        async with self._lock:
            if key not in self._cache:
                return None
                
            value, timestamp = self._cache[key]
            now = time.time()
            
            if now - timestamp > self.ttl:
                # Expired item, remove it
                del self._cache[key]
                return None
                
            return value
            
    async def _set(self, key: str, value: LLMResponse) -> None:
        """
        Store a value in the in-memory cache.
        
        Args:
            key: The cache key
            value: The LLMResponse to cache
        """
        async with self._lock:
            self._cache[key] = (value, time.time())
            
    async def clear(self) -> int:
        """
        Clear all cached items.
        
        Returns:
            Number of items cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def get_size(self) -> int:
        """
        Get the current number of items in the cache.
        
        Returns:
            Number of cached items
        """
        return len(self._cache)


class RedisLLMCache(LLMCacheBase):
    """
    Redis-based implementation of LLM request caching.
    """
    
    def __init__(self, config: CacheConfig):
        """
        Initialize the Redis cache with configuration.
        
        Args:
            config: Cache configuration including Redis connection settings
        """
        super().__init__(config)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis cache requires redis.asyncio package")
            
        self.redis_url = config.redis_url or "redis://localhost:6379/0"
        self._redis = aioredis.from_url(self.redis_url)
        logger.info(f"Initialized Redis LLM cache with URL: {self.redis_url}")
        
    async def _get(self, key: str) -> Optional[LLMResponse]:
        """
        Get a value from the Redis cache by key.
        
        Args:
            key: The cache key to retrieve
            
        Returns:
            Cached LLMResponse if available, None otherwise
        """
        try:
            value = await self._redis.get(key)
            if not value:
                return None
                
            # Deserialize the cached response
            response_dict = json.loads(value)
            return LLMResponse.model_validate(response_dict)
        except Exception as e:
            logger.error(f"Error retrieving from Redis cache: {e}")
            return None
            
    async def _set(self, key: str, value: LLMResponse) -> None:
        """
        Store a value in the Redis cache.
        
        Args:
            key: The cache key
            value: The LLMResponse to cache
        """
        try:
            # Serialize the response to JSON
            serialized = json.dumps(value.model_dump())
            await self._redis.setex(key, self.ttl, serialized)
        except Exception as e:
            logger.error(f"Error storing in Redis cache: {e}")
            
    async def clear(self) -> int:
        """
        Clear all cached items with the llm_cache prefix.
        
        Returns:
            Number of items cleared
        """
        try:
            # Find keys with the prefix
            cursor = b'0'
            count = 0
            
            while cursor:
                cursor, keys = await self._redis.scan(cursor=cursor, match="llm_cache:*")
                if keys:
                    await self._redis.delete(*keys)
                    count += len(keys)
                    
                if cursor == b'0':
                    break
                    
            return count
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
            return 0
            
    async def get_size(self) -> int:
        """
        Get the current number of items in the cache.
        
        Returns:
            Number of cached items with the llm_cache prefix
        """
        try:
            # Count keys with the prefix
            cursor = b'0'
            count = 0
            
            while cursor:
                cursor, keys = await self._redis.scan(cursor=cursor, match="llm_cache:*")
                count += len(keys)
                
                if cursor == b'0':
                    break
                    
            return count
        except Exception as e:
            logger.error(f"Error getting Redis cache size: {e}")
            return 0


def create_cache(config: CacheConfig) -> LLMCacheBase:
    """
    Factory function to create the appropriate cache implementation.
    
    Args:
        config: Cache configuration
        
    Returns:
        LLMCacheBase implementation based on the configuration
    """
    if not config.enabled:
        logger.info("Cache is disabled in configuration")
        return InMemoryLLMCache(config)
        
    if config.use_redis:
        try:
            return RedisLLMCache(config)
        except ImportError:
            logger.warning("Redis not available, falling back to in-memory cache")
            return InMemoryLLMCache(config)
    
    return InMemoryLLMCache(config)


# Global cache instance
_CACHE_INSTANCE: Optional[LLMCacheBase] = None


def get_cache(config: Optional[CacheConfig] = None) -> LLMCacheBase:
    """
    Get the global cache instance, initializing if needed.
    
    Args:
        config: Optional cache configuration, used only on first call
        
    Returns:
        The global LLMCacheBase instance
    """
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        if config is None:
            # Default configuration if none provided
            config = CacheConfig(
                enabled=True,
                cache_ttl_seconds=3600,  # 1 hour default
                use_redis=False,
                redis_url=None
            )
        _CACHE_INSTANCE = create_cache(config)
    return _CACHE_INSTANCE


def init_cache(config: CacheConfig) -> LLMCacheBase:
    """
    Initialize the global cache with specific configuration.
    
    Args:
        config: Cache configuration
        
    Returns:
        The initialized LLMCacheBase instance
    """
    global _CACHE_INSTANCE
    _CACHE_INSTANCE = create_cache(config)
    return _CACHE_INSTANCE