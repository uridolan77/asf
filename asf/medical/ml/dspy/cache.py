Enhanced Cache Implementations

This module provides enhanced cache implementations with better error handling,
connection pooling, and retry logic for production use.

import logging
import json
import os
import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod

# Set up logging
logger = logging.getLogger(__name__)

# Check for Redis availability
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheInterface(ABC):
    Abstract interface for cache implementations.
    
    This interface defines the methods that all cache implementations must provide.
    
    @abstractmethod
    async def get(self, key: str) -> Any:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: The cached value, or None if not found
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration time in seconds
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """
        Clear the entire cache.
        """
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            int: Number of keys cleared
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        Close the cache connection.
        
        Args:
        
        
        Returns:
            Description of return value
        pass


class EnhancedDiskCache(CacheInterface):
    Enhanced disk-based cache implementation.
    
    This implementation provides a disk-based cache with improved error handling
    and support for pattern-based clearing.
    
    def __init__(self, directory: str = ".cache"):
        """
        Initialize the disk cache.
        
        Args:
            directory: Directory to store cache files
        """
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self._lock = asyncio.Lock()
        logger.info(f"Initialized disk cache in directory: {directory}")
    
    async def get(self, key: str) -> Any:
        """
        Get a value from the disk cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: The cached value, or None if not found
        """
        file_path = self._get_file_path(key)
        try:
            if not os.path.exists(file_path):
                return None
            
            # Check if expired
            if await self._is_expired(file_path):
                await self.delete(key)
                return None
            
            async with self._lock:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    return data.get('value')
        except Exception as e:
            logger.warning(f"Error reading from disk cache: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """
        Set a value in the disk cache.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration time in seconds
        """
        file_path = self._get_file_path(key)
        try:
            # Create data structure with value and expiration
            data = {
                'value': value,
                'expire': time.time() + expire if expire else None
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write to file
            async with self._lock:
                with open(file_path, 'w') as f:
                    json.dump(data, f)
        except Exception as e:
            logger.warning(f"Error writing to disk cache: {str(e)}")
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the disk cache.
        
        Args:
            key: Cache key
        """
        file_path = self._get_file_path(key)
        try:
            if os.path.exists(file_path):
                async with self._lock:
                    os.remove(file_path)
        except Exception as e:
            logger.warning(f"Error deleting from disk cache: {str(e)}")
    
    async def clear(self) -> None:
        """
        Clear the entire disk cache.
        """
        try:
            async with self._lock:
                for root, dirs, files in os.walk(self.directory):
                    for file in files:
                        if file.endswith('.cache'):
                            os.remove(os.path.join(root, file))
            logger.info("Disk cache cleared")
        except Exception as e:
            logger.warning(f"Error clearing disk cache: {str(e)}")
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching a pattern.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            int: Number of keys cleared
        """
        count = 0
        try:
            async with self._lock:
                for root, dirs, files in os.walk(self.directory):
                    for file in files:
                        if file.endswith('.cache') and pattern in file:
                            os.remove(os.path.join(root, file))
                            count += 1
            logger.info(f"Cleared {count} items from disk cache matching pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Error clearing disk cache with pattern: {str(e)}")
        return count
    
    def close(self) -> None:
        Close the disk cache.
        
        No action needed for disk cache.
        
        Args:
        
        
        Returns:
            Description of return value
        pass
    
    def _get_file_path(self, key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            key: Cache key
            
        Returns:
            str: File path
        """
        # Create a safe filename from the key
        safe_key = "".join(c if c.isalnum() else "_" for c in key)
        return os.path.join(self.directory, f"{safe_key}.cache")
    
    async def _is_expired(self, file_path: str) -> bool:
        """
        Check if a cache entry is expired.
        
        Args:
            file_path: Path to the cache file
            
        Returns:
            bool: True if expired, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                expire = data.get('expire')
                if expire is not None and time.time() > expire:
                    return True
                return False
        except Exception:
            # If we can't read the file, consider it expired
            return True


class EnhancedRedisCache(CacheInterface):
    Enhanced Redis-based cache implementation.
    
    This implementation provides a Redis-based cache with connection pooling,
    retry logic, and improved error handling.
    
    def __init__(
        self,
        url: str,
        password: Optional[str] = None,
        connection_pool_size: int = 10,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        max_retries: int = 3
    ):
        """
        Initialize the Redis cache.
        
        Args:
            url: Redis connection URL
            password: Optional Redis password
            connection_pool_size: Size of the connection pool
            socket_timeout: Timeout for socket operations
            socket_connect_timeout: Timeout for connection
            max_retries: Maximum number of retries for operations
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package is not installed. Install with 'pip install redis'")
        
        # Create connection pool for better performance under concurrent load
        self.pool = redis.ConnectionPool.from_url(
            url, 
            password=password,
            max_connections=connection_pool_size,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout
        )
        self.redis = redis.Redis(connection_pool=self.pool)
        self._lock = asyncio.Lock()
        self.max_retries = max_retries
        
        logger.info(f"Initialized Redis cache with connection pool size: {connection_pool_size}")
    
    async def get(self, key: str) -> Any:
        """
        Get a value from Redis asynchronously.
        
        Args:
            key: Cache key
            
        Returns:
            Any: The cached value, or None if not found
        """
        loop = asyncio.get_event_loop()
        try:
            # Execute in threadpool to avoid blocking
            value = await loop.run_in_executor(None, lambda: self.redis.get(key))
            if value:
                return json.loads(value)
            return None
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Redis get error for key {key}: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """
        Set a value in Redis asynchronously with retry mechanism.
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration time in seconds
        """
        loop = asyncio.get_event_loop()
        serialized = json.dumps(value)
        
        # Tenacity-style retry for transient Redis failures
        for attempt in range(self.max_retries):
            try:
                if expire:
                    await loop.run_in_executor(
                        None, lambda: self.redis.setex(key, expire, serialized)
                    )
                else:
                    await loop.run_in_executor(
                        None, lambda: self.redis.set(key, serialized)
                    )
                return
            except redis.RedisError as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Redis set failed after {self.max_retries} attempts: {str(e)}")
                    raise
                wait_time = 0.1 * (2 ** attempt)  # Exponential backoff
                await asyncio.sleep(wait_time)
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from Redis asynchronously.
        
        Args:
            key: Cache key
        """
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, lambda: self.redis.delete(key))
        except redis.RedisError as e:
            logger.warning(f"Redis delete error for key {key}: {str(e)}")
    
    async def clear(self) -> None:
        """
        Clear the Redis database asynchronously with concurrency control.
        """
        async with self._lock:  # Prevent multiple concurrent clear operations
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, lambda: self.redis.flushdb())
                logger.info("Redis cache cleared successfully")
            except redis.RedisError as e:
                logger.error(f"Redis clear error: {str(e)}")
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching a pattern with a cursor-based approach.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            int: Number of keys cleared
        """
        async with self._lock:
            loop = asyncio.get_event_loop()
            try:
                count = 0
                cursor = 0
                while True:
                    # Use scan instead of keys for better performance on large DBs
                    cursor, keys = await loop.run_in_executor(
                        None, lambda: self.redis.scan(cursor, f"*{pattern}*", 100)
                    )
                    if keys:
                        await loop.run_in_executor(None, lambda: self.redis.delete(*keys))
                        count += len(keys)
                    if cursor == 0:
                        break
                logger.info(f"Cleared {count} keys matching pattern: {pattern}")
                return count
            except redis.RedisError as e:
                logger.error(f"Redis clear_pattern error: {str(e)}")
                return 0
    
    def close(self) -> None:
        Close the Redis connection pool.
        
        Args:
        
        
        Returns:
            Description of return value
        if hasattr(self, 'pool'):
            self.pool.disconnect()
            logger.debug("Redis connection pool closed")


class NullCache(CacheInterface):
    No-op cache implementation when caching is disabled.
    
    async def get(self, key: str) -> Any:
        """
        Get a value from the null cache (always returns None).
        
        Args:
            key: Cache key
            
        Returns:
            None: Always returns None
        """
        return None
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """
        Set a value in the null cache (no-op).
        
        Args:
            key: Cache key
            value: Value to cache
            expire: Optional expiration time in seconds
        """
        pass
    
    async def delete(self, key: str) -> None:
        """
        Delete a value from the null cache (no-op).
        
        Args:
            key: Cache key
        """
        pass
    
    async def clear(self) -> None:
        """
        Clear the null cache (no-op).
        """
        pass
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear keys matching a pattern (no-op).
        
        Args:
            pattern: Pattern to match
            
        Returns:
            int: Always returns 0
        """
        return 0
    
    def close(self) -> None:
        Close the null cache (no-op).
        
        Args:
        
        
        Returns:
            Description of return value
        pass


# Factory function to create the appropriate cache
def create_cache(
    cache_type: str,
    cache_directory: str = ".cache",
    redis_url: str = "redis://localhost:6379/0",
    redis_password: Optional[str] = None,
    connection_pool_size: int = 10
) -> CacheInterface:
    """
    Create a cache instance based on the specified type.
    
    Args:
        cache_type: Type of cache ('disk', 'redis', 'null')
        cache_directory: Directory for disk cache
        redis_url: Redis connection URL
        redis_password: Redis password
        connection_pool_size: Redis connection pool size
        
    Returns:
        CacheInterface: The cache instance
        
    Raises:
        ValueError: If the cache type is invalid
    """
    if cache_type.lower() == 'disk':
        return EnhancedDiskCache(directory=cache_directory)
    elif cache_type.lower() == 'redis':
        if not REDIS_AVAILABLE:
            logger.warning("Redis package not installed. Falling back to disk cache.")
            return EnhancedDiskCache(directory=cache_directory)
        return EnhancedRedisCache(
            url=redis_url,
            password=redis_password,
            connection_pool_size=connection_pool_size
        )
    elif cache_type.lower() == 'null':
        return NullCache()
    else:
        raise ValueError(f"Invalid cache type: {cache_type}")


# Export all classes and functions
__all__ = [
    'CacheInterface',
    'EnhancedDiskCache',
    'EnhancedRedisCache',
    'NullCache',
    'create_cache'
]
