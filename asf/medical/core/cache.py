"""
Cache module for the Medical Research Synthesizer.

This module provides a caching layer for the API.
"""

import logging
import json
import hashlib
import time
import re
from typing import Dict, Any, Optional
from functools import wraps
import threading

# Set up logging
logger = logging.getLogger(__name__)

class LRUCache:
    """
    LRU (Least Recently Used) cache implementation.

    This class provides an in-memory LRU cache with a maximum size.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of items in the cache (default: 1000)
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Update access time
                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Check if we need to evict an item
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Find the least recently used item
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]

                # Remove it
                del self.cache[oldest_key]
                del self.access_times[oldest_key]

            # Add the new item
            self.cache[key] = value
            self.access_times[key] = time.time()

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            True if the key was deleted, False otherwise
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                return True
            return False

    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def get_size(self) -> int:
        """
        Get the current size of the cache.

        Returns:
            Number of items in the cache
        """
        with self.lock:
            return len(self.cache)

class CacheManager:
    """
    Cache manager for the Medical Research Synthesizer.

    This class provides a caching layer for the API, with support for
    in-memory LRU caching and optional Redis caching. It supports different
    TTLs for different types of data, cache namespaces, and cache invalidation
    patterns.
    """

    _instance = None

    # Default TTLs for different types of data (in seconds)
    DEFAULT_TTLS = {
        "search": 3600,  # 1 hour
        "analysis": 7200,  # 2 hours
        "knowledge_base": 86400,  # 24 hours
        "user": 1800,  # 30 minutes
        "default": 3600  # 1 hour
    }

    def __new__(cls, **_):
        """
        Create a singleton instance of the cache manager.

        Returns:
            CacheManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        max_size: int = 1000,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        namespace: str = "asf:medical:"
    ):
        """
        Initialize the cache manager.

        Args:
            max_size: Maximum number of items in the in-memory cache (default: 1000)
            redis_url: Redis URL for distributed caching (default: None)
            default_ttl: Default TTL in seconds (default: 3600)
            namespace: Cache namespace prefix (default: "asf:medical:")
        """
        if self._initialized:
            return

        self.local_cache = LRUCache(max_size)
        self.redis = None
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0
        }

        # Initialize Redis if URL is provided
        if redis_url:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(redis_url)
                logger.info(f"Redis cache initialized: {redis_url}")
            except ImportError:
                logger.warning("redis-py package not installed. Redis caching disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {str(e)}")

        self._initialized = True
        logger.info("Cache manager initialized")

    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            namespace: Optional namespace override

        Returns:
            Cached value or None if not found
        """
        # Apply namespace
        namespaced_key = f"{namespace or self.namespace}{key}"

        # Try local cache first
        value = self.local_cache.get(namespaced_key)
        if value is not None:
            logger.debug(f"Cache hit (local): {namespaced_key}")
            self.stats["hits"] += 1
            self.stats["local_hits"] += 1
            return value

        # Try Redis if available
        if self.redis:
            try:
                value = await self.redis.get(namespaced_key)
                if value:
                    # Deserialize and update local cache
                    value = json.loads(value)
                    self.local_cache.set(namespaced_key, value)
                    logger.debug(f"Cache hit (Redis): {namespaced_key}")
                    self.stats["hits"] += 1
                    self.stats["redis_hits"] += 1
                    return value
            except Exception as e:
                logger.error(f"Error getting value from Redis: {str(e)}")

        logger.debug(f"Cache miss: {namespaced_key}")
        self.stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: Optional[str] = None, data_type: Optional[str] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (default: based on data_type or self.default_ttl)
            namespace: Optional namespace override
            data_type: Type of data for TTL selection (search, analysis, knowledge_base, user)
        """
        # Apply namespace
        namespaced_key = f"{namespace or self.namespace}{key}"

        # Determine TTL
        if ttl is None:
            if data_type and data_type in self.DEFAULT_TTLS:
                ttl = self.DEFAULT_TTLS[data_type]
            else:
                ttl = self.default_ttl

        # Update local cache
        self.local_cache.set(namespaced_key, value)

        # Update Redis if available
        if self.redis:
            try:
                # Serialize value
                serialized = json.dumps(value)

                # Set in Redis with TTL
                await self.redis.set(
                    namespaced_key,
                    serialized,
                    ex=ttl
                )
                logger.debug(f"Set in cache: {namespaced_key} (TTL={ttl}s)")
            except Exception as e:
                logger.error(f"Error setting value in Redis: {str(e)}")

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key
            namespace: Optional namespace override

        Returns:
            True if the key was deleted, False otherwise
        """
        # Apply namespace
        namespaced_key = f"{namespace or self.namespace}{key}"

        # Delete from local cache
        local_deleted = self.local_cache.delete(namespaced_key)

        # Delete from Redis if available
        redis_deleted = False
        if self.redis:
            try:
                redis_deleted = await self.redis.delete(namespaced_key) > 0
                if redis_deleted:
                    logger.debug(f"Deleted from Redis: {namespaced_key}")
            except Exception as e:
                logger.error(f"Error deleting value from Redis: {str(e)}")

        return local_deleted or redis_deleted

    async def clear(self, namespace: Optional[str] = None) -> None:
        """
        Clear the cache.

        Args:
            namespace: Optional namespace to clear (default: all)
        """
        if namespace:
            # Clear only the specified namespace
            await self.delete_pattern(f"{namespace}*")
            logger.info(f"Cleared cache for namespace: {namespace}")
        else:
            # Clear local cache
            self.local_cache.clear()

            # Clear Redis if available
            if self.redis:
                try:
                    await self.redis.flushdb()
                    logger.info("Cleared Redis cache")
                except Exception as e:
                    logger.error(f"Error clearing Redis cache: {str(e)}")

        # Reset stats
        self.stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0
        }

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "search:*")

        Returns:
            Number of keys deleted
        """
        count = 0

        # Delete from Redis if available
        if self.redis:
            try:
                # Get all keys matching the pattern
                keys = await self.redis.keys(pattern)

                # Delete the keys
                if keys:
                    count = await self.redis.delete(*keys)
                    logger.debug(f"Deleted {count} keys matching pattern: {pattern}")
            except Exception as e:
                logger.error(f"Error deleting keys by pattern: {str(e)}")

        # For local cache, we need to iterate through all keys
        with self.local_cache.lock:
            # Find keys matching the pattern
            keys_to_delete = []
            for key in list(self.local_cache.cache.keys()):
                if self._match_pattern(key, pattern):
                    keys_to_delete.append(key)

            # Delete the keys
            for key in keys_to_delete:
                self.local_cache.delete(key)
                count += 1

        return count

    async def count_pattern(self, pattern: str) -> int:
        """
        Count the number of keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "search:*")

        Returns:
            Number of keys matching the pattern
        """
        count = 0

        # Count in Redis if available
        if self.redis:
            try:
                # Get all keys matching the pattern
                keys = await self.redis.keys(pattern)
                count += len(keys)
            except Exception as e:
                logger.error(f"Error counting keys by pattern in Redis: {str(e)}")

        # Count in local cache
        with self.local_cache.lock:
            # Find keys matching the pattern
            for key in list(self.local_cache.cache.keys()):
                if self._match_pattern(key, pattern):
                    count += 1

        return count

    def _match_pattern(self, key: str, pattern: str) -> bool:
        """
        Check if a key matches a pattern.

        Args:
            key: Cache key
            pattern: Pattern to match (supports * wildcard)

        Returns:
            True if the key matches the pattern, False otherwise
        """
        # Convert Redis pattern to regex
        regex_pattern = pattern.replace("*", ".*").replace("?", ".") + "$"
        return bool(re.match(regex_pattern, key))

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = self.stats.copy()

        # Add additional stats
        stats["local_size"] = self.local_cache.get_size()
        stats["local_max_size"] = self.local_cache.max_size

        # Add Redis stats if available
        if self.redis:
            try:
                info = await self.redis.info()
                stats["redis_used_memory"] = info.get("used_memory_human", "N/A")
                stats["redis_keys"] = info.get("db0", {}).get("keys", 0)
            except Exception as e:
                logger.error(f"Error getting Redis stats: {str(e)}")

        return stats

    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from a prefix and arguments.

        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Cache key
        """
        # Create a string representation of the arguments
        args_str = json.dumps(args, sort_keys=True)
        kwargs_str = json.dumps(kwargs, sort_keys=True)

        # Create a hash of the arguments
        hash_input = f"{args_str}:{kwargs_str}"
        args_hash = hashlib.md5(hash_input.encode()).hexdigest()

        # Create the key
        return f"{prefix}:{args_hash}"

# Create a singleton instance
cache_manager = CacheManager()

def cached(
    prefix: str,
    ttl: Optional[int] = None,
    cache_exceptions: bool = False,
    namespace: Optional[str] = None,
    data_type: Optional[str] = None
):
    """
    Decorator for caching function results.

    Args:
        prefix: Key prefix
        ttl: TTL in seconds (default: based on data_type or cache_manager.default_ttl)
        cache_exceptions: Whether to cache exceptions (default: False)
        namespace: Optional namespace override
        data_type: Type of data for TTL selection (search, analysis, knowledge_base, user)

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate a cache key
            key = cache_manager.generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await cache_manager.get(key, namespace)
            if cached_value is not None:
                # Check if it's an exception
                if isinstance(cached_value, dict) and cached_value.get("__exception__"):
                    # Reconstruct the exception
                    exc_type = cached_value["type"]
                    exc_msg = cached_value["message"]

                    # Raise the exception
                    if exc_type == "ValueError":
                        raise ValueError(exc_msg)
                    elif exc_type == "KeyError":
                        raise KeyError(exc_msg)
                    elif exc_type == "ValidationError":
                        from asf.medical.core.exceptions import ValidationError
                        raise ValidationError(exc_msg)
                    elif exc_type == "ResourceNotFoundError":
                        from asf.medical.core.exceptions import ResourceNotFoundError
                        raise ResourceNotFoundError("Resource", exc_msg)
                    else:
                        raise Exception(exc_msg)

                return cached_value

            try:
                # Call the function
                result = await func(*args, **kwargs)

                # Cache the result
                await cache_manager.set(key, result, ttl, namespace, data_type)

                return result
            except Exception as e:
                if cache_exceptions:
                    # Cache the exception
                    exc_data = {
                        "__exception__": True,
                        "type": type(e).__name__,
                        "message": str(e)
                    }

                    await cache_manager.set(key, exc_data, ttl, namespace, data_type)

                # Re-raise the exception
                raise

        return wrapper

    return decorator
