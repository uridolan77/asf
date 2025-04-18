"""
Caching Utilities

This module provides caching utilities for DSPy API calls.
"""

import os
import abc
import json
import pickle
import hashlib
import logging
import functools
from typing import Dict, Any, Optional, Union, Protocol, runtime_checkable, TypeVar, Generic

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)


@runtime_checkable
class CacheInterface(Protocol):
    """Protocol for cache implementations."""
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if found, None otherwise
        """
        ...
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (optional)
        """
        ...
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        ...
    
    def clear(self) -> None:
        """Clear the cache."""
        ...
    
    def close(self) -> None:
        """Close the cache connection."""
        ...


class DiskCache:
    """Cache implementation using disk storage."""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        max_size_mb: int = 1024,  # 1GB
        serializer: str = "pickle"
    ):
        """
        Initialize the disk cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size_mb: Maximum cache size in MB
            serializer: Serialization format ("pickle" or "json")
        """
        self.cache_dir = os.path.abspath(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        if serializer not in ["pickle", "json"]:
            raise ValueError(f"Unsupported serializer: {serializer}")
        self.serializer = serializer
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.debug(f"Initialized disk cache in {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            str: File path
        """
        # Create a hash of the key to avoid invalid filenames
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.{self.serializer}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if found, None otherwise
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, "rb") as f:
                if self.serializer == "pickle":
                    return pickle.load(f)
                else:  # json
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_path}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (not supported for disk cache)
        """
        if ttl is not None:
            logger.warning("TTL is not supported for disk cache")
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Check if we need to cleanup
        self._check_size()
        
        cache_path = self._get_cache_path(key)
        
        try:
            if self.serializer == "pickle":
                with open(cache_path, "wb") as f:
                    pickle.dump(value, f)
            else:  # json
                with open(cache_path, "w") as f:
                    json.dump(value, f)
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_path}: {str(e)}")
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        cache_path = self._get_cache_path(key)
        
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_path}: {str(e)}")
    
    def clear(self) -> None:
        """Clear the cache."""
        if not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith(f".{self.serializer}"):
                cache_path = os.path.join(self.cache_dir, filename)
                try:
                    os.remove(cache_path)
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_path}: {str(e)}")
    
    def _check_size(self) -> None:
        """Check if the cache is too large and clean up if necessary."""
        if not os.path.exists(self.cache_dir):
            return
        
        # Get total size
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in os.listdir(self.cache_dir)
            if os.path.isfile(os.path.join(self.cache_dir, f))
        )
        
        if total_size > self.max_size_bytes:
            logger.warning(f"Cache size ({total_size / 1024 / 1024:.2f}MB) exceeds limit, cleaning up")
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up the cache by removing the oldest files."""
        if not os.path.exists(self.cache_dir):
            return
        
        # Get files with modification time
        cache_files = [
            (f, os.path.getmtime(os.path.join(self.cache_dir, f)))
            for f in os.listdir(self.cache_dir)
            if f.endswith(f".{self.serializer}") and os.path.isfile(os.path.join(self.cache_dir, f))
        ]
        
        # Sort by modification time (oldest first)
        cache_files.sort(key=lambda x: x[1])
        
        # Remove oldest files until cache is under limit
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f, _ in cache_files
        )
        
        for filename, _ in cache_files:
            if total_size <= self.max_size_bytes * 0.8:  # Remove files until below 80% of limit
                break
            
            cache_path = os.path.join(self.cache_dir, filename)
            file_size = os.path.getsize(cache_path)
            
            try:
                os.remove(cache_path)
                total_size -= file_size
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_path}: {str(e)}")
    
    def close(self) -> None:
        """Close the cache connection (no-op for disk cache)."""
        pass


class RedisCache:
    """Cache implementation using Redis."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        prefix: str = "dspy_cache:",
        serializer: str = "pickle"
    ):
        """
        Initialize the Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database
            prefix: Key prefix
            serializer: Serialization format ("pickle" or "json")
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not available. Please install the redis package.")
        
        self.prefix = prefix
        
        if serializer not in ["pickle", "json"]:
            raise ValueError(f"Unsupported serializer: {serializer}")
        self.serializer = serializer
        
        # Connect to Redis
        self.redis = redis.Redis(
            host=host,
            port=port,
            password=password,
            db=db
        )
        
        logger.debug(f"Initialized Redis cache at {host}:{port}")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Cached value if found, None otherwise
        """
        prefixed_key = f"{self.prefix}{key}"
        
        try:
            value = self.redis.get(prefixed_key)
            if value is None:
                return None
            
            if self.serializer == "pickle":
                return pickle.loads(value)
            else:  # json
                return json.loads(value)
        except Exception as e:
            logger.warning(f"Failed to get value from Redis: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        prefixed_key = f"{self.prefix}{key}"
        
        try:
            if self.serializer == "pickle":
                serialized_value = pickle.dumps(value)
            else:  # json
                serialized_value = json.dumps(value).encode("utf-8")
            
            if ttl is not None:
                self.redis.setex(prefixed_key, ttl, serialized_value)
            else:
                self.redis.set(prefixed_key, serialized_value)
        except Exception as e:
            logger.warning(f"Failed to set value in Redis: {str(e)}")
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        prefixed_key = f"{self.prefix}{key}"
        
        try:
            self.redis.delete(prefixed_key)
        except Exception as e:
            logger.warning(f"Failed to delete value from Redis: {str(e)}")
    
    def clear(self) -> None:
        """Clear the cache (all keys with the prefix)."""
        try:
            keys = self.redis.keys(f"{self.prefix}*")
            if keys:
                self.redis.delete(*keys)
        except Exception as e:
            logger.warning(f"Failed to clear Redis cache: {str(e)}")
    
    def close(self) -> None:
        """Close the Redis connection."""
        try:
            self.redis.close()
        except Exception as e:
            logger.warning(f"Failed to close Redis connection: {str(e)}")


class NullCache:
    """Cache implementation that doesn't actually cache anything."""
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache (always returns None).
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Any]: Always None
        """
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache (no-op).
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        pass
    
    def delete(self, key: str) -> None:
        """
        Delete a value from the cache (no-op).
        
        Args:
            key: Cache key
        """
        pass
    
    def clear(self) -> None:
        """Clear the cache (no-op)."""
        pass
    
    def close(self) -> None:
        """Close the cache connection (no-op)."""
        pass


def create_cache(
    backend: str = "disk",
    **kwargs
) -> Union[DiskCache, RedisCache, NullCache]:
    """
    Create a cache instance.
    
    Args:
        backend: Cache backend ("disk", "redis", or "none")
        **kwargs: Backend-specific arguments
        
    Returns:
        Union[DiskCache, RedisCache, NullCache]: Cache instance
    """
    if backend == "disk":
        return DiskCache(**kwargs)
    elif backend == "redis":
        return RedisCache(**kwargs)
    elif backend == "none":
        return NullCache()
    else:
        raise ValueError(f"Unsupported cache backend: {backend}")


# Export
__all__ = [
    "CacheInterface",
    "DiskCache",
    "RedisCache",
    "NullCache",
    "create_cache",
]
