"""
Cache Persistence for LLM Gateway

This module provides implementations of cache persistence mechanisms,
including disk-based storage and Redis storage to preserve the cache
between application restarts.
"""

import asyncio
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)

class CachePersistence(ABC):
    """Abstract base class for cache persistence implementations."""
    
    @abstractmethod
    async def store(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Store data with the given key.
        
        Args:
            key: Cache key
            data: Data to store
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data for the given key.
        
        Args:
            key: Cache key
            
        Returns:
            Retrieved data, or None if not found
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete data for the given key.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def list_keys(self) -> List[str]:
        """
        List all cache keys.
        
        Returns:
            List of all cache keys
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            Success status
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Cache statistics
        """
        pass

class DiskPersistence(CachePersistence):
    """
    Disk-based cache persistence implementation.
    
    Stores cache entries as files in a directory structure.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_entries: int = 10000,
        serialization_format: str = "json",
        compaction_threshold: int = 1000,
    ):
        """
        Initialize disk persistence.
        
        Args:
            cache_dir: Directory for cache files
            max_entries: Maximum number of entries to store
            serialization_format: Format for serializing data ("json" or "pickle")
            compaction_threshold: Number of operations before cleaning old entries
        """
        self.cache_dir = Path(cache_dir)
        self.max_entries = max_entries
        self.serialization_format = serialization_format
        self.compaction_threshold = compaction_threshold
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.cache_dir / "_metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Track operation count for triggering compaction
        self.operation_count = 0
        
        # Log initialization
        logger.info(f"Initialized disk cache persistence at {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """
        Get path for cached entry.
        
        Args:
            key: Cache key
            
        Returns:
            Path to cache file
        """
        # Create a directory structure to avoid too many files in one directory
        # Use first 2 chars of key for subdirectory
        if len(key) >= 2:
            subdir = key[:2]
        else:
            subdir = key + "_"
            
        # Create subdirectory if needed
        (self.cache_dir / subdir).mkdir(exist_ok=True)
        
        # Determine file extension based on serialization format
        extension = ".json" if self.serialization_format == "json" else ".pickle"
        
        return self.cache_dir / subdir / f"{key}{extension}"
    
    def _get_metadata_path(self, key: str) -> Path:
        """
        Get path for metadata file.
        
        Args:
            key: Cache key
            
        Returns:
            Path to metadata file
        """
        return self.metadata_dir / f"{key}.meta"
    
    async def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """
        Serialize data to bytes.
        
        Args:
            data: Data to serialize
            
        Returns:
            Serialized data as bytes
        """
        if self.serialization_format == "json":
            return json.dumps(data).encode('utf-8')
        else:
            return pickle.dumps(data)
    
    async def _deserialize_data(self, data: bytes) -> Dict[str, Any]:
        """
        Deserialize data from bytes.
        
        Args:
            data: Serialized data
            
        Returns:
            Deserialized data
        """
        if self.serialization_format == "json":
            return json.loads(data.decode('utf-8'))
        else:
            return pickle.loads(data)
    
    async def _update_metadata(self, key: str) -> None:
        """
        Update metadata for a cache entry.
        
        Args:
            key: Cache key
        """
        metadata_path = self._get_metadata_path(key)
        metadata = {
            "key": key,
            "last_access": datetime.now().timestamp(),
            "access_count": 1
        }
        
        # Update existing metadata if it exists
        try:
            if await aiofiles.os.path.exists(metadata_path):
                async with aiofiles.open(metadata_path, 'rb') as f:
                    existing_metadata = json.loads(await f.read())
                    metadata["access_count"] = existing_metadata.get("access_count", 0) + 1
        except Exception as e:
            logger.warning(f"Error reading metadata for {key}: {e}")
        
        # Write updated metadata
        try:
            async with aiofiles.open(metadata_path, 'w') as f:
                await f.write(json.dumps(metadata))
        except Exception as e:
            logger.warning(f"Error updating metadata for {key}: {e}")
    
    async def _perform_compaction(self) -> None:
        """Perform cache compaction by removing least recently used entries."""
        # Only compact if we've exceeded the threshold
        if self.operation_count < self.compaction_threshold:
            return
        
        logger.info("Starting cache compaction")
        start_time = time.time()
        
        try:
            # Reset operation count
            self.operation_count = 0
            
            # Get list of all metadata files
            metadata_files = list(self.metadata_dir.glob("*.meta"))
            if not metadata_files:
                logger.info("No entries to compact")
                return
                
            # Read all metadata
            all_metadata = []
            for meta_path in metadata_files:
                try:
                    async with aiofiles.open(meta_path, 'r') as f:
                        metadata = json.loads(await f.read())
                        all_metadata.append(metadata)
                except Exception as e:
                    logger.warning(f"Error reading metadata from {meta_path}: {e}")
            
            # Sort by last access time (oldest first)
            all_metadata.sort(key=lambda x: x.get("last_access", 0))
            
            # If we have more entries than max_entries, delete the oldest ones
            entries_to_remove = len(all_metadata) - self.max_entries
            if entries_to_remove <= 0:
                logger.info(f"Cache compaction not needed (entries: {len(all_metadata)}, max: {self.max_entries})")
                return
                
            logger.info(f"Compacting cache: removing {entries_to_remove} oldest entries")
            
            # Delete oldest entries
            for i in range(entries_to_remove):
                if i < len(all_metadata):
                    key = all_metadata[i].get("key")
                    if key:
                        await self.delete(key)
                        
            logger.info(f"Cache compaction complete in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error during cache compaction: {e}")
    
    async def store(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Store data with the given key.
        
        Args:
            key: Cache key
            data: Data to store
            
        Returns:
            Success status
        """
        try:
            # Increment operation count
            self.operation_count += 1
            
            # Serialize data
            serialized_data = await self._serialize_data(data)
            
            # Write to file
            cache_path = self._get_cache_path(key)
            async with aiofiles.open(cache_path, 'wb') as f:
                await f.write(serialized_data)
            
            # Update metadata
            await self._update_metadata(key)
            
            # Perform compaction if needed
            if self.operation_count >= self.compaction_threshold:
                asyncio.create_task(self._perform_compaction())
            
            return True
        except Exception as e:
            logger.error(f"Error storing cache entry {key}: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data for the given key.
        
        Args:
            key: Cache key
            
        Returns:
            Retrieved data, or None if not found
        """
        try:
            # Increment operation count
            self.operation_count += 1
            
            # Check if file exists
            cache_path = self._get_cache_path(key)
            if not await aiofiles.os.path.exists(cache_path):
                return None
            
            # Read from file
            async with aiofiles.open(cache_path, 'rb') as f:
                serialized_data = await f.read()
            
            # Deserialize data
            data = await self._deserialize_data(serialized_data)
            
            # Update metadata (in background task)
            asyncio.create_task(self._update_metadata(key))
            
            return data
        except Exception as e:
            logger.error(f"Error retrieving cache entry {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete data for the given key.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        try:
            # Increment operation count
            self.operation_count += 1
            
            # Delete cache file
            cache_path = self._get_cache_path(key)
            if await aiofiles.os.path.exists(cache_path):
                await aiofiles.os.remove(cache_path)
            
            # Delete metadata file
            metadata_path = self._get_metadata_path(key)
            if await aiofiles.os.path.exists(metadata_path):
                await aiofiles.os.remove(metadata_path)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting cache entry {key}: {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """
        List all cache keys.
        
        Returns:
            List of all cache keys
        """
        try:
            # Get list of all metadata files
            keys = []
            metadata_files = list(self.metadata_dir.glob("*.meta"))
            
            # Extract keys from metadata files
            for meta_path in metadata_files:
                key = meta_path.stem  # Remove extension
                keys.append(key)
            
            return keys
        except Exception as e:
            logger.error(f"Error listing cache keys: {e}")
            return []
    
    async def clear(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            Success status
        """
        try:
            # Get list of all keys
            keys = await self.list_keys()
            
            # Delete each key
            for key in keys:
                await self.delete(key)
            
            # Reset operation count
            self.operation_count = 0
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Cache statistics
        """
        try:
            # Get list of all metadata files
            metadata_files = list(self.metadata_dir.glob("*.meta"))
            
            # Calculate total size
            total_size = 0
            for subdir in self.cache_dir.iterdir():
                if subdir.is_dir() and subdir.name != "_metadata":
                    for cache_file in subdir.iterdir():
                        if cache_file.is_file():
                            total_size += cache_file.stat().st_size
            
            # Get access statistics from metadata
            access_counts = []
            last_access_times = []
            
            for meta_path in metadata_files:
                try:
                    async with aiofiles.open(meta_path, 'r') as f:
                        metadata = json.loads(await f.read())
                        access_counts.append(metadata.get("access_count", 0))
                        last_access_times.append(metadata.get("last_access", 0))
                except Exception as e:
                    logger.warning(f"Error reading metadata from {meta_path}: {e}")
            
            # Calculate statistics
            avg_access_count = sum(access_counts) / len(access_counts) if access_counts else 0
            
            # Get current time for age calculation
            now = datetime.now().timestamp()
            ages = [now - t for t in last_access_times]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                "entry_count": len(metadata_files),
                "total_size_bytes": total_size,
                "avg_entry_size_bytes": total_size / len(metadata_files) if metadata_files else 0,
                "avg_access_count": avg_access_count,
                "avg_age_seconds": avg_age,
                "max_entries": self.max_entries,
                "storage_directory": str(self.cache_dir),
                "serialization_format": self.serialization_format,
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "storage_directory": str(self.cache_dir)
            }

class RedisPersistence(CachePersistence):
    """
    Redis-based cache persistence implementation.
    
    Stores cache entries in a Redis database.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "llm_cache:",
        ttl_seconds: Optional[int] = None,
    ):
        """
        Initialize Redis persistence.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for Redis entries
            ttl_seconds: Time-to-live for entries (None for no expiry)
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        
        # Initialize Redis client lazily (only when needed)
        self._redis = None
        
        # Log initialization
        logger.info(f"Initialized Redis cache persistence at {redis_url}")
    
    @property
    async def redis(self):
        """Get or create Redis client."""
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self.redis_url)
                # Verify connection
                await self._redis.ping()
                logger.debug("Redis client connected")
            except ImportError:
                raise ImportError(
                    "Redis package is not installed. "
                    "Install it with: pip install redis"
                )
            except Exception as e:
                logger.error(f"Error connecting to Redis: {e}")
                raise
                
        return self._redis
    
    def _get_full_key(self, key: str) -> str:
        """
        Get full Redis key with prefix.
        
        Args:
            key: Cache key
            
        Returns:
            Full Redis key
        """
        return f"{self.prefix}{key}"
    
    def _get_metadata_key(self, key: str) -> str:
        """
        Get Redis key for metadata.
        
        Args:
            key: Cache key
            
        Returns:
            Redis metadata key
        """
        return f"{self.prefix}meta:{key}"
    
    async def _update_metadata(self, key: str) -> None:
        """
        Update metadata for a cache entry.
        
        Args:
            key: Cache key
        """
        redis_client = await self.redis
        metadata_key = self._get_metadata_key(key)
        
        try:
            # Get existing metadata
            existing_metadata = await redis_client.get(metadata_key)
            
            if existing_metadata:
                metadata = json.loads(existing_metadata)
                metadata["access_count"] = metadata.get("access_count", 0) + 1
                metadata["last_access"] = datetime.now().timestamp()
            else:
                metadata = {
                    "key": key,
                    "last_access": datetime.now().timestamp(),
                    "access_count": 1
                }
            
            # Update metadata
            await redis_client.set(metadata_key, json.dumps(metadata))
            
            # Set TTL if configured
            if self.ttl_seconds is not None:
                await redis_client.expire(metadata_key, self.ttl_seconds)
                
        except Exception as e:
            logger.warning(f"Error updating metadata for {key}: {e}")
    
    async def store(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Store data with the given key.
        
        Args:
            key: Cache key
            data: Data to store
            
        Returns:
            Success status
        """
        try:
            redis_client = await self.redis
            full_key = self._get_full_key(key)
            
            # Serialize data
            serialized_data = json.dumps(data)
            
            # Store in Redis
            await redis_client.set(full_key, serialized_data)
            
            # Set TTL if configured
            if self.ttl_seconds is not None:
                await redis_client.expire(full_key, self.ttl_seconds)
            
            # Update metadata
            await self._update_metadata(key)
            
            return True
        except Exception as e:
            logger.error(f"Error storing cache entry {key}: {e}")
            return False
    
    async def retrieve(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve data for the given key.
        
        Args:
            key: Cache key
            
        Returns:
            Retrieved data, or None if not found
        """
        try:
            redis_client = await self.redis
            full_key = self._get_full_key(key)
            
            # Get from Redis
            serialized_data = await redis_client.get(full_key)
            
            if not serialized_data:
                return None
            
            # Deserialize data
            data = json.loads(serialized_data)
            
            # Update metadata in background
            asyncio.create_task(self._update_metadata(key))
            
            return data
        except Exception as e:
            logger.error(f"Error retrieving cache entry {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete data for the given key.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        try:
            redis_client = await self.redis
            full_key = self._get_full_key(key)
            metadata_key = self._get_metadata_key(key)
            
            # Delete both data and metadata
            await redis_client.delete(full_key, metadata_key)
            
            return True
        except Exception as e:
            logger.error(f"Error deleting cache entry {key}: {e}")
            return False
    
    async def list_keys(self) -> List[str]:
        """
        List all cache keys.
        
        Returns:
            List of all cache keys
        """
        try:
            redis_client = await self.redis
            
            # Use pattern matching to find all keys with prefix
            pattern = f"{self.prefix}*"
            full_keys = await redis_client.keys(pattern)
            
            # Filter out metadata keys and strip prefix
            keys = []
            metadata_prefix = f"{self.prefix}meta:"
            for full_key in full_keys:
                if isinstance(full_key, bytes):
                    full_key = full_key.decode('utf-8')
                if not full_key.startswith(metadata_prefix):
                    key = full_key[len(self.prefix):]
                    keys.append(key)
            
            return keys
        except Exception as e:
            logger.error(f"Error listing cache keys: {e}")
            return []
    
    async def clear(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            Success status
        """
        try:
            redis_client = await self.redis
            
            # Get all keys with prefix
            pattern = f"{self.prefix}*"
            keys_to_delete = await redis_client.keys(pattern)
            
            if keys_to_delete:
                # Delete all keys
                await redis_client.delete(*keys_to_delete)
            
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.
        
        Returns:
            Cache statistics
        """
        try:
            redis_client = await self.redis
            
            # Count data keys
            data_pattern = f"{self.prefix}[^m][^e][^t][^a].*"
            data_keys = await redis_client.keys(data_pattern)
            
            # Count metadata keys
            meta_pattern = f"{self.prefix}meta:*"
            meta_keys = await redis_client.keys(meta_pattern)
            
            # Calculate total size (approximate)
            total_size = 0
            for key in data_keys:
                try:
                    total_size += await redis_client.strlen(key)
                except:
                    pass
            
            # Get access statistics from metadata
            access_counts = []
            last_access_times = []
            
            for meta_key in meta_keys:
                try:
                    metadata_json = await redis_client.get(meta_key)
                    if metadata_json:
                        metadata = json.loads(metadata_json)
                        access_counts.append(metadata.get("access_count", 0))
                        last_access_times.append(metadata.get("last_access", 0))
                except Exception as e:
                    logger.warning(f"Error reading metadata: {e}")
            
            # Calculate statistics
            avg_access_count = sum(access_counts) / len(access_counts) if access_counts else 0
            
            # Get current time for age calculation
            now = datetime.now().timestamp()
            ages = [now - t for t in last_access_times]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                "entry_count": len(data_keys),
                "total_size_bytes": total_size,
                "avg_entry_size_bytes": total_size / len(data_keys) if data_keys else 0,
                "avg_access_count": avg_access_count,
                "avg_age_seconds": avg_age,
                "ttl_seconds": self.ttl_seconds,
                "redis_url": self.redis_url.split("@")[-1],  # Hide credentials
                "prefix": self.prefix
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {
                "error": str(e),
                "redis_url": self.redis_url.split("@")[-1]  # Hide credentials
            }