"""
Distributed cache module for the Medical Research Synthesizer.

This module provides a Redis-based distributed caching system for the application.
It supports key-value storage with TTL (time-to-live) and automatic serialization/deserialization.
"""

import json
import time
import pickle
import asyncio
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable, TypeVar, cast, Union
from datetime import datetime, timedelta

import redis.asyncio as redis
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from asf.medical.core.config import settings
from asf.medical.core.exceptions import CacheError, CacheConnectionError, CacheOperationError
from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')

class CacheInterface(ABC):
    """
    Abstract base class for cache implementations.
    
    This interface defines the methods that all cache implementations must provide.
    It supports basic key-value operations with TTL and namespace support.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set a value in the cache with optional TTL.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (default: 1 hour)
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        pass
    
    @abstractmethod
    async def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear all values in the given namespace or the entire cache.
        
        Args:
            namespace: Optional namespace to clear (default: clear all)
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        pass


class RedisCache(CacheInterface):
    """
    Redis-based distributed cache implementation.
    
    This implementation uses Redis for distributed caching, making it suitable
    for production environments with multiple workers or distributed systems.
    """
    
    def __init__(self, redis_url: str = None, namespace: str = ""):
        """
        Initialize the Redis cache.
        
        Args:
            redis_url: Redis connection URL (default: from settings)
            namespace: Optional namespace prefix for all keys
        """
        self.redis_url = redis_url or settings.REDIS_URL
        if not self.redis_url:
            raise CacheConnectionError("Redis URL not configured")
        
        self.namespace = namespace
        self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
        logger.info(f"Initialized Redis cache with URL: {self.redis_url}")
    
    def _get_full_key(self, key: str) -> str:
        """Get the full key with namespace."""
        return f"{self.namespace}:{key}" if self.namespace else key
    
    async def _serialize(self, value: Any) -> bytes:
        """Serialize a value for storage in Redis."""
        try:
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Error serializing value: {str(e)}", exc_info=e)
            raise CacheOperationError(
                "serialize",
                f"Error serializing value: {str(e)}",
                details={"error": str(e)}
            )
    
    async def _deserialize(self, value: bytes) -> Any:
        """Deserialize a value from Redis."""
        if value is None:
            return None
        
        try:
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Error deserializing value: {str(e)}", exc_info=e)
            raise CacheOperationError(
                "deserialize",
                f"Error deserializing value: {str(e)}",
                details={"error": str(e)}
            )
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        full_key = self._get_full_key(key)
        
        try:
            value = await self.redis_client.get(full_key)
            return await self._deserialize(value)
        except redis.RedisError as e:
            logger.error(f"Redis error getting key {key}: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"key": key, "error": str(e)})
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set a value in the cache with optional TTL.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (default: 1 hour)
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        full_key = self._get_full_key(key)
        serialized_value = await self._serialize(value)
        
        try:
            await self.redis_client.set(full_key, serialized_value, ex=ttl)
        except redis.RedisError as e:
            logger.error(f"Redis error setting key {key}: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"key": key, "error": str(e)})
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: The cache key
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        full_key = self._get_full_key(key)
        
        try:
            await self.redis_client.delete(full_key)
        except redis.RedisError as e:
            logger.error(f"Redis error deleting key {key}: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"key": key, "error": str(e)})
    
    async def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear all values in the given namespace or the entire cache.
        
        Args:
            namespace: Optional namespace to clear (default: clear all)
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        ns = namespace or self.namespace
        
        try:
            if not ns:
                # Warning: This will clear the entire Redis database
                await self.redis_client.flushdb()
                return
            
            # Clear only keys in the namespace
            pattern = f"{ns}:*"
            cursor = 0
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis_client.delete(*keys)
                if cursor == 0:
                    break
        except redis.RedisError as e:
            logger.error(f"Redis error clearing namespace {ns}: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"namespace": ns, "error": str(e)})
    
    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the cache.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key exists, False otherwise
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        full_key = self._get_full_key(key)
        
        try:
            return bool(await self.redis_client.exists(full_key))
        except redis.RedisError as e:
            logger.error(f"Redis error checking key {key}: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"key": key, "error": str(e)})
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from the cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary mapping keys to values (missing keys are omitted)
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        if not keys:
            return {}
        
        full_keys = [self._get_full_key(key) for key in keys]
        
        try:
            values = await self.redis_client.mget(full_keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = await self._deserialize(value)
            
            return result
        except redis.RedisError as e:
            logger.error(f"Redis error getting multiple keys: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"keys": keys, "error": str(e)})
    
    async def set_many(self, mapping: Dict[str, Any], ttl: int = 3600) -> None:
        """
        Set multiple values in the cache.
        
        Args:
            mapping: Dictionary mapping keys to values
            ttl: Time-to-live in seconds (default: 1 hour)
            
        Raises:
            CacheError: If there's an error accessing the cache
        """
        if not mapping:
            return
        
        # Redis doesn't support TTL with mset, so we use a pipeline
        try:
            async with self.redis_client.pipeline() as pipe:
                for key, value in mapping.items():
                    full_key = self._get_full_key(key)
                    serialized_value = await self._serialize(value)
                    pipe.set(full_key, serialized_value, ex=ttl)
                
                await pipe.execute()
        except redis.RedisError as e:
            logger.error(f"Redis error setting multiple keys: {str(e)}", exc_info=e)
            raise CacheConnectionError(f"Redis error: {str(e)}", details={"keys": list(mapping.keys()), "error": str(e)})


# Create a cache decorator
def redis_cached(
    ttl: int = 3600,
    key_prefix: str = "",
    namespace: str = "",
    key_builder: Optional[Callable[..., str]] = None,
):
    """
    Decorator for caching function results in Redis.
    
    Args:
        ttl: Time-to-live in seconds (default: 1 hour)
        key_prefix: Prefix for cache keys
        namespace: Namespace for cache keys
        key_builder: Function to build cache keys (default: based on args and kwargs)
        
    Returns:
        Decorated function
    """
    # Initialize the Redis cache
    redis_cache = RedisCache(namespace=namespace)
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Build the cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Default key builder
                arg_str = ":".join(str(arg) for arg in args if not isinstance(arg, Request))
                kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if k != "db" and not isinstance(v, AsyncSession))
                func_name = func.__module__ + "." + func.__name__
                cache_key = f"{key_prefix}:{func_name}:{arg_str}:{kwarg_str}"
            
            # Try to get from cache
            cached_result = await redis_cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {cache_key}")
                return cached_result
            
            # Call the function
            logger.debug(f"Cache miss: {cache_key}")
            result = await func(*args, **kwargs)
            
            # Cache the result
            await redis_cache.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


# FastAPI dependency for getting a Redis cache
def get_redis_cache(namespace: str = "") -> RedisCache:
    """
    Get a Redis cache instance.
    
    Args:
        namespace: Optional namespace for the cache
        
    Returns:
        Redis cache instance
    """
    return RedisCache(namespace=namespace)
