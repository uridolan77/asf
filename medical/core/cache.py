"""
Cache module for the Medical Research Synthesizer.

This module provides a unified caching system with both local and distributed (Redis)
cache implementations. The caching system improves performance and reduces redundant
computations across the application.

Classes:
    CacheInterface: Abstract base class defining the cache interface.
    LRUCache: Local in-memory LRU cache implementation.
    RedisCache: Distributed Redis-based cache implementation.
    CacheManager: Unified cache manager supporting both local and distributed caching.

Functions:
    cached: Decorator for caching function results.
    get_cache_manager: Get the singleton instance of the CacheManager.
    get_cache_key: Generate a cache key based on input parameters.
"""
import os
import json
import time
import pickle
import logging
import hashlib
import threading
import functools
from abc import ABC, abstractmethod
from functools import wraps
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)
T = TypeVar('T')

class CacheInterface(ABC):
    """
    Abstract base class for cache implementations.
    This interface defines the methods that all cache implementations must provide.
    """

    @abstractmethod
    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The cache key.
            namespace: Optional namespace for the cache key.

        Returns:
            The cached value or None if not found.
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  namespace: Optional[str] = None) -> bool:
        """
        Set a value in the cache.

        Args:
            key: The cache key.
            value: The value to cache.
            ttl: Time-to-live in seconds.
            namespace: Optional namespace for the cache key.

        Returns:
            True if the value was successfully cached, False otherwise.
        """
        pass

    @abstractmethod
    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The cache key.
            namespace: Optional namespace for the cache key.

        Returns:
            True if the key was deleted, False otherwise.
        """
        pass

    @abstractmethod
    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            namespace: Optional namespace to clear.

        Returns:
            Number of keys cleared.
        """
        pass

class LRUCache:
    """
    LRU (Least Recently Used) cache implementation.
    This class provides an in-memory LRU cache with a maximum size.

    Attributes:
        max_size (int): Maximum number of items in the cache.
        cache (Dict[str, Any]): Dictionary to store cached items.
        access_times (Dict[str, float]): Dictionary to track access times of cached items.
        expiry_times (Dict[str, float]): Dictionary to track expiry times of cached items.
        lock (threading.RLock): Lock for thread-safe operations.
    """
    def __init__(self, max_size: int = 1000):
        """
        Initialize the LRU cache.

        Args:
            max_size (int): Maximum number of items in the cache. Defaults to 1000.
        """
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, float] = {}
        self.expiry_times: Dict[str, float] = {}
        self.lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key (str): Cache key.

        Returns:
            Optional[Any]: Cached value or None if not found.
        """
        with self.lock:
            if key in self.cache:
                # Check if key has expired
                if key in self.expiry_times and self.expiry_times[key] <= time.time():
                    self.delete(key)
                    return None

                self.access_times[key] = time.time()
                return self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key (str): Cache key.
            value (Any): Value to cache.
            ttl (Optional[int]): Time-to-live in seconds. None means no expiration.
        """
        with self.lock:
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
                self.delete(oldest_key)

            self.cache[key] = value
            self.access_times[key] = time.time()

            if ttl is not None:
                self.expiry_times[key] = time.time() + ttl

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key (str): Cache key.

        Returns:
            bool: True if the key was deleted, False otherwise.
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                if key in self.expiry_times:
                    del self.expiry_times[key]
                return True
            return False

    def clear(self) -> int:
        """
        Clear the cache.

        Returns:
            int: Number of items cleared.
        """
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            self.expiry_times.clear()
            return count

    def get_size(self) -> int:
        """
        Get the current size of the cache.

        Returns:
            int: Number of items in the cache.
        """
        with self.lock:
            return len(self.cache)

    def cleanup_expired(self) -> int:
        """
        Remove all expired items from the cache.

        Returns:
            int: Number of items removed.
        """
        with self.lock:
            now = time.time()
            expired_keys = [key for key, expiry in self.expiry_times.items() if expiry <= now]

            for key in expired_keys:
                self.delete(key)

            return len(expired_keys)

class RedisCache(CacheInterface):
    """
    Redis-based distributed cache implementation.
    This implementation uses Redis for distributed caching, making it suitable
    for production environments with multiple workers or distributed systems.
    """

    DEFAULT_TTL = 3600  # 1 hour

    def __init__(self, redis_url: Optional[str] = None, namespace: str = ""):
        """
        Initialize the Redis cache.

        Args:
            redis_url: Redis connection URL (default: from environment)
            namespace: Optional namespace prefix for all keys
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not installed. Install with: pip install redis")

        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        if not self.redis_url:
            raise ValueError("Redis URL not configured")

        self.namespace = namespace
        self.redis_client = redis.from_url(
            self.redis_url,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
            health_check_interval=30
        )
        logger.info(f"Initialized Redis cache with URL: {self.redis_url}")

    def _get_namespaced_key(self, key: str, namespace: Optional[str] = None) -> str:
        """
        Get the namespaced key.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            Namespaced key
        """
        ns = namespace or self.namespace
        return f"{ns}:{key}" if ns else key

    async def _serialize(self, value: Any) -> bytes:
        """
        Serialize a value for storage in Redis.

        Args:
            value: The value to serialize

        Returns:
            Serialized bytes
        """
        try:
            if isinstance(value, (dict, list, tuple, set, bool, int, float)) or value is None:
                return json.dumps(value).encode('utf-8')
            return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Error serializing value: {str(e)}")
            raise ValueError(f"Error serializing value: {str(e)}")

    async def _deserialize(self, value: bytes) -> Any:
        """
        Deserialize a value from Redis.

        Args:
            value: The value to deserialize

        Returns:
            Deserialized value
        """
        if value is None:
            return None
        try:
            # First try JSON deserialization
            try:
                return json.loads(value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If that fails, try pickle
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"Error deserializing value: {str(e)}")
            raise ValueError(f"Error deserializing value: {str(e)}")

    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            The cached value or None if not found
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        try:
            value = await self.redis_client.get(namespaced_key)
            return await self._deserialize(value)
        except Exception as e:
            logger.error(f"Redis error getting key {key}: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  namespace: Optional[str] = None) -> bool:
        """
        Set a value in the cache with optional TTL.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds (default: class DEFAULT_TTL)
            namespace: Optional namespace override

        Returns:
            True if successful, False otherwise
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        ttl = ttl if ttl is not None else self.DEFAULT_TTL

        try:
            serialized_value = await self._serialize(value)
            await self.redis_client.set(namespaced_key, serialized_value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Redis error setting key {key}: {str(e)}")
            return False

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            True if key was deleted, False otherwise
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        try:
            count = await self.redis_client.delete(namespaced_key)
            return count > 0
        except Exception as e:
            logger.error(f"Redis error deleting key {key}: {str(e)}")
            return False

    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear all values in the given namespace or the entire cache.

        Args:
            namespace: Optional namespace to clear (default: instance namespace)

        Returns:
            Number of keys cleared
        """
        ns = namespace or self.namespace
        try:
            if not ns:
                # Warning: This will clear the entire Redis database
                await self.redis_client.flushdb()
                return 1  # We don't know the exact count

            # Clear only keys in the namespace
            pattern = f"{ns}:*"
            count = 0
            cursor = 0

            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted = await self.redis_client.delete(*keys)
                    count += deleted
                if cursor == 0:
                    break

            return count
        except Exception as e:
            logger.error(f"Redis error clearing namespace {ns}: {str(e)}")
            return 0

    async def exists(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            True if the key exists, False otherwise
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        try:
            return bool(await self.redis_client.exists(namespaced_key))
        except Exception as e:
            logger.error(f"Redis error checking key {key}: {str(e)}")
            return False

    async def get_many(self, keys: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Get multiple values from the cache.

        Args:
            keys: List of cache keys
            namespace: Optional namespace override

        Returns:
            Dictionary mapping keys to values (missing keys are omitted)
        """
        if not keys:
            return {}

        namespaced_keys = [self._get_namespaced_key(key, namespace) for key in keys]
        try:
            values = await self.redis_client.mget(namespaced_keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = await self._deserialize(value)
            return result
        except Exception as e:
            logger.error(f"Redis error getting multiple keys: {str(e)}")
            return {}

    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None,
                       namespace: Optional[str] = None) -> bool:
        """
        Set multiple values in the cache.

        Args:
            mapping: Dictionary mapping keys to values
            ttl: Time-to-live in seconds (default: class DEFAULT_TTL)
            namespace: Optional namespace override

        Returns:
            True if successful, False otherwise
        """
        if not mapping:
            return True

        ttl = ttl if ttl is not None else self.DEFAULT_TTL

        try:
            async with self.redis_client.pipeline() as pipe:
                for key, value in mapping.items():
                    namespaced_key = self._get_namespaced_key(key, namespace)
                    serialized_value = await self._serialize(value)
                    pipe.set(namespaced_key, serialized_value, ex=ttl)
                await pipe.execute()
            return True
        except Exception as e:
            logger.error(f"Redis error setting multiple keys: {str(e)}")
            return False

    async def ping(self) -> bool:
        """
        Check if the Redis server is reachable.

        Returns:
            True if the Redis server is reachable, False otherwise
        """
        try:
            return await self.redis_client.ping()
        except Exception as e:
            logger.error(f"Redis ping failed: {str(e)}")
            return False

class LocalCache(CacheInterface):
    """
    Local in-memory cache implementation using LRUCache.
    This implementation is suitable for development environments or as a fallback.
    """

    def __init__(self, max_size: int = 1000, namespace: str = ""):
        """
        Initialize the local cache.

        Args:
            max_size: Maximum cache size (default: 1000)
            namespace: Optional namespace prefix for all keys
        """
        self.lru_cache = LRUCache(max_size)
        self.namespace = namespace

    def _get_namespaced_key(self, key: str, namespace: Optional[str] = None) -> str:
        """Get the namespaced key."""
        ns = namespace or self.namespace
        return f"{ns}:{key}" if ns else key

    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            Cached value or None if not found
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        return self.lru_cache.get(namespaced_key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None,
                  namespace: Optional[str] = None) -> bool:
        """
        Set a value in the cache.

        Args:
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
            namespace: Optional namespace override

        Returns:
            True if successful
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        self.lru_cache.set(namespaced_key, value, ttl)
        return True

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            True if the key was deleted
        """
        namespaced_key = self._get_namespaced_key(key, namespace)
        return self.lru_cache.delete(namespaced_key)

    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            namespace: Optional namespace to clear

        Returns:
            Number of keys cleared
        """
        if not namespace and not self.namespace:
            return self.lru_cache.clear()

        # Clear only keys in the namespace
        ns = namespace or self.namespace
        if not ns:
            return 0

        count = 0
        with self.lru_cache.lock:
            keys_to_delete = [
                key for key in list(self.lru_cache.cache.keys())
                if key.startswith(f"{ns}:")
            ]

            for key in keys_to_delete:
                if self.lru_cache.delete(key):
                    count += 1

        return count

    async def cleanup_expired(self) -> int:
        """
        Remove all expired items from the cache.

        Returns:
            Number of items removed
        """
        return self.lru_cache.cleanup_expired()

class CacheManager:
    """
    Unified cache manager supporting both local and distributed caching.
    This class provides a unified interface to cache operations with fallback mechanisms.

    In production environments, Redis is used as the primary cache.
    In development environments, a local cache is used as a fallback.
    """

    _instance = None

    # Default TTL values by data type
    DEFAULT_TTLS = {
        "search": 3600,         # 1 hour
        "analysis": 7200,       # 2 hours
        "knowledge_base": 86400, # 24 hours
        "user": 1800,           # 30 minutes
        "task": 86400,          # 24 hours
        "progress": 3600,       # 1 hour
        "model": 3600,          # 1 hour
        "prediction": 7200,     # 2 hours
        "explanation": 7200,    # 2 hours
        "default": 3600         # 1 hour
    }

    def __new__(cls, *args, **kwargs):
        """Create a singleton instance of the cache manager."""
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,
        namespace: str = "asf:medical:",
        local_cache_size: int = 1000,
        environment: str = None
    ):
        """
        Initialize the CacheManager.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            namespace: Namespace prefix for cache keys
            local_cache_size: Size of the local cache
            environment: Environment name
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.environment = environment or os.environ.get("ENVIRONMENT", "development")
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self.default_ttl = default_ttl
        self.namespace = namespace
        self.local_cache_size = local_cache_size

        # Set up the local cache always
        self.local_cache = LocalCache(max_size=local_cache_size, namespace=namespace)

        # Set up Redis cache if available
        self.redis_cache = None
        if self.redis_url and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisCache(redis_url=self.redis_url, namespace=namespace)
                logger.info(f"Redis cache initialized: {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Redis cache: {str(e)}")
                if self.environment.lower() == "production":
                    raise RuntimeError(f"Failed to initialize Redis cache in production: {str(e)}")
        elif self.environment.lower() == "production":
            logger.warning("Redis is required in production environment but not configured properly.")

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "local_hits": 0,
            "redis_hits": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }

        self._initialized = True
        logger.info(f"Cache manager initialized in {self.environment} environment")

    async def get(
        self,
        key: str,
        namespace: Optional[str] = None,
        data_type: Optional[str] = None
    ) -> Optional[Any]:
        """
        Retrieve a value from the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override
            data_type: Data type for logging purposes

        Returns:
            Cached value or None if not found
        """
        # Try Redis first if available
        if self.redis_cache:
            try:
                value = await self.redis_cache.get(key, namespace)
                if value is not None:
                    logger.debug(f"Cache hit (Redis): {key} ({data_type or 'unknown'})")
                    self.stats["hits"] += 1
                    self.stats["redis_hits"] += 1
                    return value
            except Exception as e:
                logger.error(f"Redis cache error: {str(e)}")
                self.stats["errors"] += 1

        # Fall back to local cache
        try:
            value = await self.local_cache.get(key, namespace)
            if value is not None:
                logger.debug(f"Cache hit (local): {key} ({data_type or 'unknown'})")
                self.stats["hits"] += 1
                self.stats["local_hits"] += 1
                return value
        except Exception as e:
            logger.error(f"Local cache error: {str(e)}")
            self.stats["errors"] += 1

        logger.debug(f"Cache miss: {key} ({data_type or 'unknown'})")
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
            key: The cache key
            value: The value to cache
            ttl: Time-to-live in seconds
            namespace: Optional namespace override
            data_type: Data type for TTL selection

        Returns:
            True if successful in at least one cache
        """
        # Determine TTL
        if ttl is None:
            if data_type in self.DEFAULT_TTLS:
                ttl = self.DEFAULT_TTLS[data_type]
            else:
                ttl = self.default_ttl

        success = False

        # Set in local cache
        try:
            local_success = await self.local_cache.set(key, value, ttl, namespace)
            success = success or local_success
        except Exception as e:
            logger.error(f"Local cache error: {str(e)}")
            self.stats["errors"] += 1

        # Set in Redis if available
        if self.redis_cache:
            try:
                redis_success = await self.redis_cache.set(key, value, ttl, namespace)
                success = success or redis_success
            except Exception as e:
                logger.error(f"Redis cache error: {str(e)}")
                self.stats["errors"] += 1

        if success:
            logger.debug(f"Set in cache: {key} (TTL={ttl}s, {data_type or 'unknown'})")
            self.stats["sets"] += 1

        return success

    async def delete(self, key: str, namespace: Optional[str] = None) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: The cache key
            namespace: Optional namespace override

        Returns:
            True if deleted from at least one cache
        """
        success = False

        # Delete from local cache
        try:
            local_success = await self.local_cache.delete(key, namespace)
            success = success or local_success
        except Exception as e:
            logger.error(f"Local cache error: {str(e)}")
            self.stats["errors"] += 1

        # Delete from Redis if available
        if self.redis_cache:
            try:
                redis_success = await self.redis_cache.delete(key, namespace)
                success = success or redis_success
            except Exception as e:
                logger.error(f"Redis cache error: {str(e)}")
                self.stats["errors"] += 1

        if success:
            logger.debug(f"Deleted from cache: {key}")
            self.stats["deletes"] += 1

        return success

    async def clear(self, namespace: Optional[str] = None) -> int:
        """
        Clear cache entries.

        Args:
            namespace: Optional namespace to clear

        Returns:
            Number of keys cleared
        """
        count = 0

        # Clear local cache
        try:
            local_count = await self.local_cache.clear(namespace)
            count = max(count, local_count)
        except Exception as e:
            logger.error(f"Local cache error: {str(e)}")
            self.stats["errors"] += 1

        # Clear Redis if available
        if self.redis_cache:
            try:
                redis_count = await self.redis_cache.clear(namespace)
                count = max(count, redis_count)
            except Exception as e:
                logger.error(f"Redis cache error: {str(e)}")
                self.stats["errors"] += 1

        if count > 0:
            logger.info(f"Cleared {count} keys from cache{' in namespace ' + namespace if namespace else ''}")
            self.stats["deletes"] += count

        return count

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()

        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        else:
            stats["hit_rate"] = 0

        # Add Redis stats if available
        if self.redis_cache:
            try:
                is_connected = await self.redis_cache.ping()
                stats["redis_connected"] = is_connected
                if is_connected and hasattr(self.redis_cache.redis_client, "info"):
                    info = await self.redis_cache.redis_client.info()
                    stats["redis_info"] = {
                        "used_memory_human": info.get("used_memory_human", "unknown"),
                        "connected_clients": info.get("connected_clients", "unknown"),
                        "uptime_in_seconds": info.get("uptime_in_seconds", "unknown")
                    }
            except Exception as e:
                logger.error(f"Error getting Redis info: {str(e)}")
                stats["redis_connected"] = False

        # Add local cache info
        stats["local_cache_info"] = {
            "size": self.local_cache.lru_cache.get_size(),
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
            Cache key string
        """
        key_parts = [prefix]

        # Add args to key
        for arg in args:
            if arg is None:
                key_parts.append("None")
            elif hasattr(arg, "__dict__"):
                key_parts.append(str(hash(frozenset(arg.__dict__.items()))))
            else:
                key_parts.append(str(arg))

        # Add kwargs to key
        for k, v in sorted(kwargs.items()):
            if v is None:
                key_parts.append(f"{k}=None")
            elif hasattr(v, "__dict__"):
                key_parts.append(f"{k}={hash(frozenset(v.__dict__.items()))}")
            else:
                key_parts.append(f"{k}={v}")

        # Create unique hash from key parts
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def maintenance(self) -> None:
        """
        Perform maintenance tasks on the caches, such as removing expired items.
        This should be called periodically from a background task.
        """
        # Clean up expired items from local cache
        await self.local_cache.cleanup_expired()

# Singleton instance
_cache_manager = None

def get_cache_manager(**kwargs) -> CacheManager:
    """
    Get the singleton CacheManager instance.

    Args:
        **kwargs: Initialization parameters passed to CacheManager

    Returns:
        The CacheManager instance
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(**kwargs)
    return _cache_manager

def get_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key based on input parameters.

    Args:
        *args: Positional arguments to include in the cache key.
        **kwargs: Keyword arguments to include in the cache key.

    Returns:
        str: A unique cache key.
    """
    key_parts = []

    # Add args to key
    for arg in args:
        if arg is None:
            key_parts.append("None")
        elif hasattr(arg, "__dict__"):
            # For objects, use their __dict__ representation
            key_parts.append(str(sorted(arg.__dict__.items())))
        else:
            key_parts.append(str(arg))

    # Add sorted kwargs to key
    for k, v in sorted(kwargs.items()):
        if v is None:
            key_parts.append(f"{k}=None")
        elif hasattr(v, "__dict__"):
            key_parts.append(f"{k}={sorted(v.__dict__.items())}")
        else:
            key_parts.append(f"{k}={v}")

    # Create a unique hash
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def cached(
    prefix: str = "",
    ttl: Optional[int] = None,
    namespace: Optional[str] = None,
    data_type: Optional[str] = None,
    cache_exceptions: bool = False
):
    """
    Decorator for caching function results.

    Args:
        prefix: Key prefix for the cache key
        ttl: Time-to-live in seconds
        namespace: Optional cache namespace
        data_type: Type of data for TTL selection
        cache_exceptions: Whether to cache exceptions

    Returns:
        Decorated function
    """
    cache_mgr = get_cache_manager()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            skip_cache = kwargs.pop("skip_cache", False)
            if skip_cache:
                return await func(*args, **kwargs)

            # Generate cache key
            key = cache_mgr.generate_key(
                f"{prefix or func.__module__}.{func.__name__}",
                *args,
                **kwargs
            )

            # Try to get from cache
            cached_result = await cache_mgr.get(key, namespace, data_type)

            if cached_result is not None:
                # Handle cached exceptions
                if isinstance(cached_result, dict) and "__exception__" in cached_result:
                    if cache_exceptions:
                        exception_class = cached_result["__exception__"]["class"]
                        exception_args = cached_result["__exception__"]["args"]

                        try:
                            exception_type = eval(exception_class)
                            raise exception_type(*exception_args)
                        except (NameError, SyntaxError):
                            raise Exception(f"{exception_class}: {exception_args}")

                return cached_result

            # Execute function and cache result
            try:
                result = await func(*args, **kwargs)
                await cache_mgr.set(key, result, ttl, namespace, data_type)
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
                    await cache_mgr.set(key, exception_data, ttl, namespace, data_type)
                raise

        return wrapper

    return decorator

def clear_cache(keys: Optional[List[str]] = None, namespace: Optional[str] = None) -> int:
    """
    Clear the cache for specific keys or an entire namespace.

    Args:
        keys: List of specific keys to clear
        namespace: Namespace to clear

    Returns:
        Number of keys cleared
    """
    cache_mgr = get_cache_manager()

    async def _clear():
        count = 0

        if keys:
            for key in keys:
                if await cache_mgr.delete(key, namespace):
                    count += 1
            return count
        else:
            return await cache_mgr.clear(namespace)

    # Execute asynchronously if in an event loop, otherwise use a new loop
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new task in the running loop
            return asyncio.create_task(_clear())
        else:
            return loop.run_until_complete(_clear())
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return 0

# Enhanced cache functionality is imported in cache_init.py
# to avoid circular imports

# Mock implementation of redis_cached
def redis_cached(prefix="", ttl=None, namespace=None, data_type=None, cache_exceptions=False):
    """Mock implementation of redis_cached decorator."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator