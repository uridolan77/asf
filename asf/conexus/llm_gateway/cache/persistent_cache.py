"""
Persistent Cache Implementation for LLM Gateway

This module provides persistent storage options for the semantic cache,
allowing cache data to be preserved between application restarts.
"""

import os
import json
import logging
import asyncio
import aiofiles
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import pickle
import aioredis
import contextlib
import shutil

from asf.conexus.llm_gateway.cache.semantic_cache import CacheEntry

logger = logging.getLogger(__name__)

class BaseCacheStore:
    """Base class for persistent cache stores."""
    
    async def save_entry(self, entry_id: str, entry: CacheEntry) -> bool:
        """
        Save a cache entry to persistent storage.
        
        Args:
            entry_id: Unique identifier for the entry
            entry: Cache entry to save
            
        Returns:
            True if save was successful, False otherwise
        """
        raise NotImplementedError("Cache store must implement save_entry")
    
    async def load_entry(self, entry_id: str) -> Optional[CacheEntry]:
        """
        Load a cache entry from persistent storage.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            Loaded cache entry or None if not found
        """
        raise NotImplementedError("Cache store must implement load_entry")
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a cache entry from persistent storage.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            True if deletion was successful, False otherwise
        """
        raise NotImplementedError("Cache store must implement delete_entry")
    
    async def list_entries(self) -> List[str]:
        """
        List all entry IDs in the cache.
        
        Returns:
            List of entry IDs
        """
        raise NotImplementedError("Cache store must implement list_entries")
    
    async def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            True if clear was successful, False otherwise
        """
        raise NotImplementedError("Cache store must implement clear")
    
    async def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save cache metadata.
        
        Args:
            metadata: Dictionary of metadata to save
            
        Returns:
            True if save was successful, False otherwise
        """
        raise NotImplementedError("Cache store must implement save_metadata")
    
    async def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load cache metadata.
        
        Returns:
            Dictionary of metadata or None if not found
        """
        raise NotImplementedError("Cache store must implement load_metadata")
    
    async def close(self) -> None:
        """Close the cache store and release resources."""
        pass


class DiskCacheStore(BaseCacheStore):
    """
    Disk-based cache store that persists cache entries to files.
    
    This store saves cache entries as JSON files on disk, allowing
    the cache to persist between application restarts.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_age_days: int = 30,
        cleanup_interval_hours: float = 24.0,
        compression: bool = True
    ):
        """
        Initialize disk cache store.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_days: Maximum age of cache entries in days
            cleanup_interval_hours: Interval between cache cleanup runs
            compression: Whether to compress cache entries
        """
        # Convert string path to Path object if needed
        self.cache_dir = Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
        self.entries_dir = self.cache_dir / "entries"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.entries_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_age_days = max_age_days
        self.cleanup_interval_hours = cleanup_interval_hours
        self.compression = compression
        self.last_cleanup_time = 0
        
        # Initialize async resources
        self._lock = asyncio.Lock()
        self._closed = False
        
        logger.info(
            f"Initialized disk cache store at {self.cache_dir} "
            f"(max age: {max_age_days} days, cleanup interval: {cleanup_interval_hours} hours)"
        )
    
    def _get_entry_path(self, entry_id: str) -> Path:
        """
        Get file path for a cache entry.
        
        Args:
            entry_id: Cache entry ID
            
        Returns:
            Path to the cache entry file
        """
        # Use first few characters of ID to create subdirectories to avoid
        # too many files in a single directory
        if len(entry_id) > 4:
            subdir = entry_id[:2]
            subdir_path = self.entries_dir / subdir
            subdir_path.mkdir(exist_ok=True)
            return subdir_path / f"{entry_id}.json"
        else:
            return self.entries_dir / f"{entry_id}.json"
    
    async def save_entry(self, entry_id: str, entry: CacheEntry) -> bool:
        """
        Save a cache entry to a file.
        
        Args:
            entry_id: Unique identifier for the entry
            entry: Cache entry to save
            
        Returns:
            True if save was successful, False otherwise
        """
        if self._closed:
            return False
            
        entry_path = self._get_entry_path(entry_id)
        
        try:
            # Convert CacheEntry to dict
            entry_dict = entry.dict()
            
            # Convert datetime objects to ISO strings for JSON serialization
            for key in ['created_at', 'last_accessed']:
                if key in entry_dict and isinstance(entry_dict[key], datetime):
                    entry_dict[key] = entry_dict[key].isoformat()
            
            async with aiofiles.open(entry_path, 'w') as f:
                await f.write(json.dumps(entry_dict))
            return True
        except Exception as e:
            logger.error(f"Error saving cache entry {entry_id}: {str(e)}")
            return False
    
    async def load_entry(self, entry_id: str) -> Optional[CacheEntry]:
        """
        Load a cache entry from a file.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            Loaded cache entry or None if not found
        """
        if self._closed:
            return None
            
        entry_path = self._get_entry_path(entry_id)
        
        if not entry_path.exists():
            return None
            
        try:
            async with aiofiles.open(entry_path, 'r') as f:
                content = await f.read()
                entry_dict = json.loads(content)
                
                # Convert ISO strings back to datetime objects
                for key in ['created_at', 'last_accessed']:
                    if key in entry_dict and isinstance(entry_dict[key], str):
                        try:
                            entry_dict[key] = datetime.fromisoformat(entry_dict[key])
                        except ValueError:
                            # If parsing fails, use current time
                            entry_dict[key] = datetime.utcnow()
                
                # Create CacheEntry from dict
                return CacheEntry(**entry_dict)
        except Exception as e:
            logger.error(f"Error loading cache entry {entry_id}: {str(e)}")
            return None
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a cache entry file.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if self._closed:
            return False
            
        entry_path = self._get_entry_path(entry_id)
        
        if not entry_path.exists():
            return True  # Already deleted
            
        try:
            entry_path.unlink()
            return True
        except Exception as e:
            logger.error(f"Error deleting cache entry {entry_id}: {str(e)}")
            return False
    
    async def list_entries(self) -> List[str]:
        """
        List all entry IDs in the cache.
        
        Returns:
            List of entry IDs
        """
        if self._closed:
            return []
            
        # Check if cleanup is needed
        await self._maybe_cleanup()
        
        entry_ids = []
        
        # Recursively find all JSON files in the entries directory
        for root, dirs, files in os.walk(self.entries_dir):
            for file in files:
                if file.endswith('.json'):
                    # Extract entry ID from filename
                    entry_id = file.rsplit('.', 1)[0]
                    entry_ids.append(entry_id)
        
        return entry_ids
    
    async def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            True if clear was successful, False otherwise
        """
        if self._closed:
            return False
            
        try:
            # Delete and recreate entries directory
            shutil.rmtree(self.entries_dir)
            self.entries_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error clearing disk cache: {str(e)}")
            return False
    
    async def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save cache metadata to a file.
        
        Args:
            metadata: Dictionary of metadata to save
            
        Returns:
            True if save was successful, False otherwise
        """
        if self._closed:
            return False
            
        try:
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(metadata))
            return True
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
            return False
    
    async def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load cache metadata from a file.
        
        Returns:
            Dictionary of metadata or None if not found
        """
        if self._closed:
            return None
            
        if not self.metadata_file.exists():
            return None
            
        try:
            async with aiofiles.open(self.metadata_file, 'r') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
            return None
    
    async def _maybe_cleanup(self) -> None:
        """
        Perform cleanup of old cache entries if needed.
        """
        # Check if cleanup is needed
        current_time = time.time()
        if current_time - self.last_cleanup_time < self.cleanup_interval_hours * 3600:
            return
            
        # Acquire lock to prevent concurrent cleanups
        async with self._lock:
            # Check again in case another task did cleanup while we were waiting
            if current_time - self.last_cleanup_time < self.cleanup_interval_hours * 3600:
                return
                
            logger.info("Starting disk cache cleanup")
            
            # Calculate cutoff time
            cutoff_time = datetime.utcnow() - timedelta(days=self.max_age_days)
            
            # Find all JSON files recursively
            deleted_count = 0
            total_count = 0
            
            for root, dirs, files in os.walk(self.entries_dir):
                for file in files:
                    if not file.endswith('.json'):
                        continue
                        
                    total_count += 1
                    file_path = Path(root) / file
                    
                    try:
                        # Check file modification time
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_time:
                            # Delete old file
                            file_path.unlink()
                            deleted_count += 1
                    except Exception as e:
                        logger.warning(f"Error checking cache file {file_path}: {str(e)}")
            
            # Update last cleanup time
            self.last_cleanup_time = current_time
            
            logger.info(f"Disk cache cleanup complete: deleted {deleted_count} of {total_count} entries")
    
    async def close(self) -> None:
        """Close the cache store."""
        self._closed = True


class RedisCacheStore(BaseCacheStore):
    """
    Redis-based cache store for distributed caching.
    
    This store uses Redis as a backend for cache storage, allowing
    for distributed caching across multiple instances.
    """
    
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "llm_cache:",
        ttl_seconds: int = 86400 * 30,  # 30 days
        pool_size: int = 10
    ):
        """
        Initialize Redis cache store.
        
        Args:
            redis_url: Redis connection URL
            prefix: Key prefix for all cache entries
            ttl_seconds: Time-to-live for cache entries
            pool_size: Size of connection pool
        """
        self.redis_url = redis_url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.pool_size = pool_size
        self.client = None
        
        # Keys for metadata
        self.entry_set_key = f"{self.prefix}entries"
        self.metadata_key = f"{self.prefix}metadata"
        
        logger.info(
            f"Initialized Redis cache store at {self.redis_url} "
            f"(prefix: {self.prefix}, TTL: {ttl_seconds} seconds)"
        )
    
    async def connect(self) -> None:
        """Connect to Redis server."""
        if self.client is not None:
            return
            
        try:
            self.client = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.client = None
            raise
    
    def _get_entry_key(self, entry_id: str) -> str:
        """
        Get Redis key for a cache entry.
        
        Args:
            entry_id: Cache entry ID
            
        Returns:
            Redis key
        """
        return f"{self.prefix}entry:{entry_id}"
    
    @contextlib.asynccontextmanager
    async def _get_client(self):
        """
        Get Redis client, connecting if needed.
        """
        if self.client is None:
            await self.connect()
            
        if self.client is None:
            raise RuntimeError("Failed to connect to Redis")
            
        try:
            yield self.client
        except Exception as e:
            logger.error(f"Redis operation failed: {str(e)}")
            raise
    
    async def save_entry(self, entry_id: str, entry: CacheEntry) -> bool:
        """
        Save a cache entry to Redis.
        
        Args:
            entry_id: Unique identifier for the entry
            entry: Cache entry to save
            
        Returns:
            True if save was successful, False otherwise
        """
        # Convert CacheEntry to dict
        entry_dict = entry.dict()
        
        # Convert datetime objects to ISO strings
        for key in ['created_at', 'last_accessed']:
            if key in entry_dict and isinstance(entry_dict[key], datetime):
                entry_dict[key] = entry_dict[key].isoformat()
        
        # Serialize to JSON
        entry_json = json.dumps(entry_dict)
        
        try:
            async with self._get_client() as client:
                # Save entry with TTL
                key = self._get_entry_key(entry_id)
                await client.set(key, entry_json, ex=self.ttl_seconds)
                
                # Add to entry set
                await client.sadd(self.entry_set_key, entry_id)
                
                return True
        except Exception as e:
            logger.error(f"Error saving cache entry to Redis {entry_id}: {str(e)}")
            return False
    
    async def load_entry(self, entry_id: str) -> Optional[CacheEntry]:
        """
        Load a cache entry from Redis.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            Loaded cache entry or None if not found
        """
        try:
            async with self._get_client() as client:
                key = self._get_entry_key(entry_id)
                entry_json = await client.get(key)
                
                if not entry_json:
                    return None
                
                # Deserialize from JSON
                entry_dict = json.loads(entry_json)
                
                # Convert ISO strings back to datetime objects
                for key in ['created_at', 'last_accessed']:
                    if key in entry_dict and isinstance(entry_dict[key], str):
                        try:
                            entry_dict[key] = datetime.fromisoformat(entry_dict[key])
                        except ValueError:
                            entry_dict[key] = datetime.utcnow()
                
                # Create CacheEntry from dict
                entry = CacheEntry(**entry_dict)
                
                # Update TTL since the entry was accessed
                await client.expire(key, self.ttl_seconds)
                
                return entry
        except Exception as e:
            logger.error(f"Error loading cache entry from Redis {entry_id}: {str(e)}")
            return None
    
    async def delete_entry(self, entry_id: str) -> bool:
        """
        Delete a cache entry from Redis.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            async with self._get_client() as client:
                key = self._get_entry_key(entry_id)
                await client.delete(key)
                await client.srem(self.entry_set_key, entry_id)
                return True
        except Exception as e:
            logger.error(f"Error deleting cache entry from Redis {entry_id}: {str(e)}")
            return False
    
    async def list_entries(self) -> List[str]:
        """
        List all entry IDs in the cache.
        
        Returns:
            List of entry IDs
        """
        try:
            async with self._get_client() as client:
                entries = await client.smembers(self.entry_set_key)
                return list(entries)
        except Exception as e:
            logger.error(f"Error listing cache entries from Redis: {str(e)}")
            return []
    
    async def clear(self) -> bool:
        """
        Clear all entries from the cache.
        
        Returns:
            True if clear was successful, False otherwise
        """
        try:
            async with self._get_client() as client:
                # Get all entry IDs
                entry_ids = await client.smembers(self.entry_set_key)
                
                if not entry_ids:
                    return True
                
                # Delete all entries
                keys = [self._get_entry_key(entry_id) for entry_id in entry_ids]
                
                # Use pipeline for efficiency
                pipe = client.pipeline()
                for key in keys:
                    pipe.delete(key)
                pipe.delete(self.entry_set_key)
                await pipe.execute()
                
                return True
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {str(e)}")
            return False
    
    async def save_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Save cache metadata to Redis.
        
        Args:
            metadata: Dictionary of metadata to save
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            async with self._get_client() as client:
                await client.set(self.metadata_key, json.dumps(metadata))
                return True
        except Exception as e:
            logger.error(f"Error saving cache metadata to Redis: {str(e)}")
            return False
    
    async def load_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load cache metadata from Redis.
        
        Returns:
            Dictionary of metadata or None if not found
        """
        try:
            async with self._get_client() as client:
                metadata_json = await client.get(self.metadata_key)
                if metadata_json:
                    return json.loads(metadata_json)
                return None
        except Exception as e:
            logger.error(f"Error loading cache metadata from Redis: {str(e)}")
            return None
    
    async def close(self) -> None:
        """Close Redis connections."""
        if self.client is not None:
            await self.client.close()
            self.client = None