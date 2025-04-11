"""
Enhanced Cache Manager for the Medical Research Synthesizer.

This module provides an enhanced caching layer with mandatory Redis support
for production environments, ensuring consistent state across multiple instances.
"""

import os
import json
import time
import logging
import hashlib
import asyncio
from typing import Dict, List, Any, Optional, Union, Set
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedCacheManager:
    """
    Enhanced cache manager for the Medical Research Synthesizer.
    
    This class provides a caching layer that ensures consistent state across
    multiple instances by using Redis as the primary cache store. It supports
    different TTLs for different types of data, cache namespaces, and cache
    invalidation patterns.
    
    In production environments, Redis is mandatory to ensure consistent state.
    In development environments, a local cache can be used as a fallback.
    """
    
    _instance = None
    _initialized = False
    
    # Default TTLs for different types of data (in seconds)
    DEFAULT_TTLS = {
        "search": 3600,  # 1 hour
        "analysis": 7200,  # 2 hours
        "knowledge_base": 86400,  # 24 hours
        "user": 1800,  # 30 minutes
        "task": 86400,  # 24 hours
        "progress": 3600,  # 1 hour
        "model": 3600,  # 1 hour
        "prediction": 7200,  # 2 hours
        "explanation": 7200,  # 2 hours
        "default": 3600  # 1 hour
    }
    
    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the cache manager.
        
        Returns:
            EnhancedCacheManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(EnhancedCacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        namespace: str = "asf:medical:",
        local_cache_size: int = 1000,
        environment: str = "development"
    ):
        """
        Initialize the enhanced cache manager.
        
        Args:
            redis_url: Redis URL for distributed caching (default: from env var REDIS_URL)
            default_ttl: Default TTL in seconds (default: 3600)
            namespace: Cache namespace prefix (default: "asf:medical:")
            local_cache_size: Maximum number of items in the local cache (default: 1000)
            environment: Environment (development or production) (default: from env var ENVIRONMENT)
        """
        if self._initialized:
            return
        
        # Get Redis URL from environment variable if not provided
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        
        # Get environment from environment variable if not provided
        self.environment = environment or os.environ.get("ENVIRONMENT", "development")
        
        # Check if Redis is required but not available
        if self.environment.lower() == "production" and not self.redis_url:
            logger.warning("Redis is required in production environment but no Redis URL provided.")
            logger.warning("Set REDIS_URL environment variable or provide redis_url parameter.")
            logger.warning("Falling back to local cache, but this may cause inconsistent state across instances.")
        
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.local_cache_size = local_cache_size
        self.redis = None
        self.local_cache = {}
        self.local_expiry = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
        # Initialize Redis if URL is provided
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,  # Automatically decode responses to strings
                    socket_timeout=5.0,     # Socket timeout
                    socket_connect_timeout=5.0,  # Connection timeout
                    retry_on_timeout=True,  # Retry on timeout
                    health_check_interval=30  # Health check interval
                )
                logger.info(f"Redis cache initialized: {self.redis_url}")
            except ImportError:
                logger.error("redis-py package not installed. Install with: pip install redis")
                if self.environment.lower() == "production":
                    raise ImportError("redis-py package is required in production environment")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {str(e)}")
                if self.environment.lower() == "production":
                    raise RuntimeError(f"Failed to initialize Redis cache: {str(e)}")
        
        self._initialized = True
        logger.info(f"Enhanced cache manager initialized in {self.environment} environment")
    
    async def ping_redis(self) -> bool:
        """
        Check if Redis is available.
        
        Returns:
            bool: True if Redis is available, False otherwise
        """
        if not self.redis:
            return False
        
        try:
            return await self.redis.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False
    
    async def get(self, key: str, namespace: Optional[str] = None, data_type: Optional[str] = None) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            namespace: Optional namespace override
            data_type: Type of data for logging
            
        Returns:
            Cached value or None if not found
        """
        # Apply namespace
        namespaced_key = f"{namespace or self.namespace}{key}"
        
        # Try Redis first if available
        if self.redis:
            try:
                value = await self.redis.get(namespaced_key)
                if value is not None:
                    try:
                        # Try to deserialize JSON
                        value = json.loads(value)
                        
                        # Update local cache
                        self._update_local_cache(namespaced_key, value)
                        
                        logger.debug(f"Cache hit (Redis): {namespaced_key} ({data_type or 'unknown'})")
                        self.stats["hits"] += 1
                        self.stats["redis_hits"] += 1
                        return value
                    except json.JSONDecodeError:
                        # Not JSON, return as is
                        self._update_local_cache(namespaced_key, value)
                        
                        logger.debug(f"Cache hit (Redis, non-JSON): {namespaced_key} ({data_type or 'unknown'})")
                        self.stats["hits"] += 1
                        self.stats["redis_hits"] += 1
                        return value
            except Exception as e:
                logger.error(f"Error getting value from Redis: {str(e)}")
                self.stats["errors"] += 1
                
                # Fall back to local cache if Redis fails
                if namespaced_key in self.local_cache:
                    # Check if expired
                    if self.local_expiry.get(namespaced_key, 0) > time.time():
                        logger.debug(f"Cache hit (local fallback): {namespaced_key} ({data_type or 'unknown'})")
                        self.stats["hits"] += 1
                        self.stats["local_hits"] += 1
                        return self.local_cache[namespaced_key]
        else:
            # No Redis, try local cache
            if namespaced_key in self.local_cache:
                # Check if expired
                if self.local_expiry.get(namespaced_key, 0) > time.time():
                    logger.debug(f"Cache hit (local): {namespaced_key} ({data_type or 'unknown'})")
                    self.stats["hits"] += 1
                    self.stats["local_hits"] += 1
                    return self.local_cache[namespaced_key]
        
        logger.debug(f"Cache miss: {namespaced_key} ({data_type or 'unknown'})")
        self.stats["misses"] += 1
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        namespace: Optional[str] = None, 
        data_type: Optional[str] = None
    ) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (default: based on data_type or self.default_ttl)
            namespace: Optional namespace override
            data_type: Type of data for TTL selection
            
        Returns:
            bool: True if the value was set, False otherwise
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
        self._update_local_cache(namespaced_key, value, ttl)
        
        # Update Redis if available
        if self.redis:
            try:
                # Serialize value
                if isinstance(value, (dict, list, tuple, set, bool, int, float)) or value is None:
                    serialized = json.dumps(value)
                else:
                    serialized = str(value)
                
                # Set in Redis with TTL
                await self.redis.set(
                    namespaced_key,
                    serialized,
                    ex=ttl
                )
                logger.debug(f"Set in cache: {namespaced_key} (TTL={ttl}s, {data_type or 'unknown'})")
                self.stats["sets"] += 1
                return True
            except Exception as e:
                logger.error(f"Error setting value in Redis: {str(e)}")
                self.stats["errors"] += 1
                return False
        else:
            logger.debug(f"Set in local cache only: {namespaced_key} (TTL={ttl}s, {data_type or 'unknown'})")
            self.stats["sets"] += 1
            return True
    
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.
        
        Args:
            key: Cache key
            namespace: Optional namespace override
            
        Returns:
            bool: True if the key was deleted, False otherwise
        """
        # Apply namespace
        namespaced_key = f"{namespace or self.namespace}{key}"
        
        # Delete from local cache
        local_deleted = self._delete_from_local_cache(namespaced_key)
        
        # Delete from Redis if available
        redis_deleted = False
        if self.redis:
            try:
                redis_deleted = await self.redis.delete(namespaced_key) > 0
                if redis_deleted:
                    logger.debug(f"Deleted from Redis: {namespaced_key}")
                    self.stats["deletes"] += 1
            except Exception as e:
                logger.error(f"Error deleting value from Redis: {str(e)}")
                self.stats["errors"] += 1
        
        return local_deleted or redis_deleted
    
    async def delete_pattern(self, pattern: str, namespace: Optional[str] = None) -> int:
        """
        Delete all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "user:*")
            namespace: Optional namespace override
            
        Returns:
            int: Number of keys deleted
        """
        # Apply namespace
        namespaced_pattern = f"{namespace or self.namespace}{pattern}"
        
        # Delete from Redis if available
        count = 0
        if self.redis:
            try:
                # Get all keys matching the pattern
                keys = await self.redis.keys(namespaced_pattern)
                
                if keys:
                    # Delete all keys
                    count = await self.redis.delete(*keys)
                    logger.debug(f"Deleted {count} keys matching pattern: {namespaced_pattern}")
                    self.stats["deletes"] += count
                    
                    # Also delete from local cache
                    for key in keys:
                        self._delete_from_local_cache(key)
            except Exception as e:
                logger.error(f"Error deleting keys matching pattern from Redis: {str(e)}")
                self.stats["errors"] += 1
        
        # If no Redis or Redis failed, try to delete from local cache
        if count == 0:
            # Delete from local cache
            local_count = 0
            for key in list(self.local_cache.keys()):
                if key.startswith(namespaced_pattern.replace("*", "")):
                    if self._delete_from_local_cache(key):
                        local_count += 1
            
            if local_count > 0:
                logger.debug(f"Deleted {local_count} keys matching pattern from local cache: {namespaced_pattern}")
                self.stats["deletes"] += local_count
                count = local_count
        
        return count
    
    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear all keys in the cache.
        
        Args:
            namespace: Optional namespace to clear (default: all keys in the default namespace)
            
        Returns:
            int: Number of keys cleared
        """
        # Determine pattern to clear
        pattern = f"{namespace or self.namespace}*"
        
        # Clear from Redis if available
        count = 0
        if self.redis:
            try:
                # Get all keys matching the pattern
                keys = await self.redis.keys(pattern)
                
                if keys:
                    # Delete all keys
                    count = await self.redis.delete(*keys)
                    logger.info(f"Cleared {count} keys from Redis: {pattern}")
                    self.stats["deletes"] += count
                    
                    # Also clear from local cache
                    for key in list(self.local_cache.keys()):
                        if key.startswith(namespace or self.namespace):
                            self._delete_from_local_cache(key)
            except Exception as e:
                logger.error(f"Error clearing keys from Redis: {str(e)}")
                self.stats["errors"] += 1
        
        # If no Redis or Redis failed, try to clear from local cache
        if count == 0:
            # Clear from local cache
            local_count = 0
            for key in list(self.local_cache.keys()):
                if key.startswith(namespace or self.namespace):
                    if self._delete_from_local_cache(key):
                        local_count += 1
            
            if local_count > 0:
                logger.info(f"Cleared {local_count} keys from local cache: {pattern}")
                self.stats["deletes"] += local_count
                count = local_count
        
        return count
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        stats = self.stats.copy()
        
        # Calculate hit rate
        total = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total if total > 0 else 0
        
        # Add Redis info if available
        if self.redis:
            try:
                info = await self.redis.info()
                stats["redis_info"] = {
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "connected_clients": info.get("connected_clients", "unknown"),
                    "uptime_in_seconds": info.get("uptime_in_seconds", "unknown"),
                    "total_commands_processed": info.get("total_commands_processed", "unknown")
                }
            except Exception as e:
                logger.error(f"Error getting Redis info: {str(e)}")
                stats["redis_info"] = {"error": str(e)}
        
        # Add local cache info
        stats["local_cache_info"] = {
            "size": len(self.local_cache),
            "max_size": self.local_cache_size
        }
        
        return stats
    
    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate a cache key from a prefix and arguments.
        
        Args:
            prefix: Key prefix
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: Cache key
        """
        # Convert args to strings
        arg_strings = []
        for arg in args:
            if hasattr(arg, "__dict__"):
                # For objects, use their __dict__
                arg_strings.append(str(arg.__dict__))
            else:
                arg_strings.append(str(arg))
        
        # Convert kwargs to strings
        kwarg_strings = [f"{k}={v}" for k, v in sorted(kwargs.items())]
        
        # Combine all strings
        key_string = f"{prefix}:{':'.join(arg_strings)}:{':'.join(kwarg_strings)}"
        
        # Hash the key string to get a fixed-length key
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_local_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Update the local cache with a value.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (default: None, meaning no expiry)
        """
        # Check if we need to evict an item
        if len(self.local_cache) >= self.local_cache_size and key not in self.local_cache:
            # Find the oldest item
            oldest_key = min(self.local_expiry.items(), key=lambda x: x[1])[0]
            
            # Remove it
            self._delete_from_local_cache(oldest_key)
        
        # Add the new item
        self.local_cache[key] = value
        
        # Set expiry if TTL is provided
        if ttl is not None:
            self.local_expiry[key] = time.time() + ttl
    
    def _delete_from_local_cache(self, key: str) -> bool:
        """
        Delete a value from the local cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key was deleted, False otherwise
        """
        if key in self.local_cache:
            del self.local_cache[key]
            if key in self.local_expiry:
                del self.local_expiry[key]
            return True
        return False
    
    async def _maintenance(self) -> None:
        """
        Perform maintenance tasks.
        
        This method is called periodically to clean up expired items from the local cache.
        """
        # Get current time
        now = time.time()
        
        # Find expired items
        expired_keys = [
            key for key, expiry in self.local_expiry.items()
            if expiry <= now
        ]
        
        # Remove expired items
        for key in expired_keys:
            self._delete_from_local_cache(key)
        
        if expired_keys:
            logger.debug(f"Removed {len(expired_keys)} expired items from local cache")

# Create a singleton instance
enhanced_cache_manager = EnhancedCacheManager()

def enhanced_cached(
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
        data_type: Type of data for TTL selection
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if explicitly requested
            skip_cache = kwargs.pop("skip_cache", False)
            if skip_cache:
                return await func(*args, **kwargs)
            
            # Generate a cache key
            key = enhanced_cache_manager.generate_key(prefix, *args, **kwargs)
            
            # Try to get from cache
            cached_result = await enhanced_cache_manager.get(key, namespace, data_type)
            if cached_result is not None:
                # Check if it's a cached exception
                if isinstance(cached_result, dict) and "__exception__" in cached_result:
                    if cache_exceptions:
                        # Re-raise the cached exception
                        exception_class = cached_result["__exception__"]["class"]
                        exception_args = cached_result["__exception__"]["args"]
                        
                        # Try to get the exception class
                        try:
                            exception_type = eval(exception_class)
                            raise exception_type(*exception_args)
                        except (NameError, SyntaxError):
                            # Fall back to generic exception
                            raise Exception(f"{exception_class}: {exception_args}")
                    else:
                        # Ignore cached exception and recompute
                        pass
                else:
                    # Return cached result
                    return cached_result
            
            try:
                # Call the function
                result = await func(*args, **kwargs)
                
                # Cache the result
                await enhanced_cache_manager.set(key, result, ttl, namespace, data_type)
                
                return result
            except Exception as e:
                if cache_exceptions:
                    # Cache the exception
                    exception_data = {
                        "__exception__": {
                            "class": e.__class__.__name__,
                            "args": [str(arg) for arg in e.args]
                        }
                    }
                    await enhanced_cache_manager.set(key, exception_data, ttl, namespace, data_type)
                
                # Re-raise the exception
                raise
        
        return wrapper
    
    return decorator
