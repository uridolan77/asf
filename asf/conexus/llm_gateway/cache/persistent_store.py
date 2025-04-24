"""
Persistent storage backends for semantic cache.

This module provides persistent storage implementations for the semantic cache,
allowing cache data to be stored on disk and loaded on application startup.
"""

import abc
import asyncio
import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel

from asf.conexus.llm_gateway.cache.semantic_cache import CacheEntry

logger = logging.getLogger(__name__)

class BaseCacheStore(abc.ABC):
    """Base class for persistent cache storage."""
    
    @abc.abstractmethod
    async def save_entry(self, entry_id: str, entry: CacheEntry) -> None:
        """
        Save a cache entry to persistent storage.
        
        Args:
            entry_id: Unique identifier for the entry
            entry: Cache entry to save
        """
        pass
    
    @abc.abstractmethod
    async def load_entry(self, entry_id: str) -> Optional[CacheEntry]:
        """
        Load a cache entry from persistent storage.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            Loaded cache entry or None if not found
        """
        pass
    
    @abc.abstractmethod
    async def delete_entry(self, entry_id: str) -> None:
        """
        Delete a cache entry from persistent storage.
        
        Args:
            entry_id: Unique identifier for the entry
        """
        pass
    
    @abc.abstractmethod
    async def list_entries(self) -> List[str]:
        """
        List all entry IDs in persistent storage.
        
        Returns:
            List of entry IDs
        """
        pass
    
    @abc.abstractmethod
    async def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save cache metadata to persistent storage.
        
        Args:
            metadata: Dictionary of metadata to save
        """
        pass
    
    @abc.abstractmethod
    async def load_metadata(self) -> Dict[str, Any]:
        """
        Load cache metadata from persistent storage.
        
        Returns:
            Dictionary of metadata
        """
        pass
    
    @abc.abstractmethod
    async def clear(self) -> None:
        """Clear all entries from persistent storage."""
        pass
    
    @abc.abstractmethod
    async def close(self) -> None:
        """Close the persistent store and release resources."""
        pass
    
class DiskCacheStore(BaseCacheStore):
    """
    Disk-based persistent cache storage.
    
    This implementation saves cache entries to disk as individual files
    in a specified directory structure.
    """
    
    def __init__(
        self, 
        cache_dir: str = None,
        use_pickle: bool = False
    ):
        """
        Initialize disk-based cache store.
        
        Args:
            cache_dir: Directory to store cache files (default: ~/.llm_gateway/cache)
            use_pickle: Whether to use pickle for serialization (faster but less portable)
        """
        # Default to ~/.llm_gateway/cache if not specified
        if not cache_dir:
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".llm_gateway", "cache")
            
        # Create cache directory structure
        self.cache_dir = cache_dir
        self.entries_dir = os.path.join(self.cache_dir, "entries")
        self.metadata_path = os.path.join(self.cache_dir, "metadata.json")
        self.use_pickle = use_pickle
        
        # Create directories if they don't exist
        os.makedirs(self.entries_dir, exist_ok=True)
        
        logger.info(f"Initialized disk cache store at {self.cache_dir}")
    
    def _get_entry_path(self, entry_id: str) -> str:
        """
        Get file path for a cache entry.
        
        Args:
            entry_id: Entry ID
            
        Returns:
            File path for the entry
        """
        # Replace potentially problematic characters in entry ID with safe ones
        safe_id = entry_id.replace(":", "_").replace("/", "_").replace("\\", "_")
        
        # Return path with appropriate extension
        extension = ".pkl" if self.use_pickle else ".json"
        return os.path.join(self.entries_dir, f"{safe_id}{extension}")
    
    async def save_entry(self, entry_id: str, entry: CacheEntry) -> None:
        """
        Save a cache entry to disk.
        
        Args:
            entry_id: Unique identifier for the entry
            entry: Cache entry to save
        """
        entry_path = self._get_entry_path(entry_id)
        
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            if self.use_pickle:
                # Use pickle for binary serialization
                await loop.run_in_executor(
                    None, 
                    lambda: pickle.dump(entry, open(entry_path, "wb"))
                )
            else:
                # Use JSON for text serialization (more portable)
                await loop.run_in_executor(
                    None, 
                    lambda: open(entry_path, "w").write(entry.json())
                )
        except Exception as e:
            logger.error(f"Error saving cache entry to {entry_path}: {str(e)}")
            raise
    
    async def load_entry(self, entry_id: str) -> Optional[CacheEntry]:
        """
        Load a cache entry from disk.
        
        Args:
            entry_id: Unique identifier for the entry
            
        Returns:
            Loaded cache entry or None if not found
        """
        entry_path = self._get_entry_path(entry_id)
        
        # Check if file exists
        if not os.path.exists(entry_path):
            return None
        
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            if self.use_pickle:
                # Use pickle for binary deserialization
                return await loop.run_in_executor(
                    None, 
                    lambda: pickle.load(open(entry_path, "rb"))
                )
            else:
                # Use JSON for text deserialization
                content = await loop.run_in_executor(
                    None, 
                    lambda: open(entry_path, "r").read()
                )
                return CacheEntry.parse_raw(content)
        except Exception as e:
            logger.error(f"Error loading cache entry from {entry_path}: {str(e)}")
            return None
    
    async def delete_entry(self, entry_id: str) -> None:
        """
        Delete a cache entry from disk.
        
        Args:
            entry_id: Unique identifier for the entry
        """
        entry_path = self._get_entry_path(entry_id)
        
        # Check if file exists
        if not os.path.exists(entry_path):
            return
        
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            await loop.run_in_executor(None, os.remove, entry_path)
        except Exception as e:
            logger.error(f"Error deleting cache entry {entry_path}: {str(e)}")
            raise
    
    async def list_entries(self) -> List[str]:
        """
        List all entry IDs in the cache directory.
        
        Returns:
            List of entry IDs
        """
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            # Get all files in entries directory
            extension = ".pkl" if self.use_pickle else ".json"
            files = await loop.run_in_executor(
                None,
                lambda: [f for f in os.listdir(self.entries_dir) if f.endswith(extension)]
            )
            
            # Convert filenames back to entry IDs
            entry_ids = [
                f.replace("_", ":").replace(extension, "") for f in files
            ]
            
            return entry_ids
        except Exception as e:
            logger.error(f"Error listing cache entries: {str(e)}")
            return []
    
    async def save_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Save cache metadata to disk.
        
        Args:
            metadata: Dictionary of metadata to save
        """
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            # Add timestamp
            metadata["last_updated"] = datetime.utcnow().isoformat()
            
            # Write metadata to file
            await loop.run_in_executor(
                None,
                lambda: open(self.metadata_path, "w").write(json.dumps(metadata, indent=2))
            )
        except Exception as e:
            logger.error(f"Error saving cache metadata: {str(e)}")
            raise
    
    async def load_metadata(self) -> Dict[str, Any]:
        """
        Load cache metadata from disk.
        
        Returns:
            Dictionary of metadata
        """
        # Check if file exists
        if not os.path.exists(self.metadata_path):
            return {}
        
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            content = await loop.run_in_executor(
                None,
                lambda: open(self.metadata_path, "r").read()
            )
            return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading cache metadata: {str(e)}")
            return {}
    
    async def clear(self) -> None:
        """Clear all entries from disk cache."""
        # Use thread pool for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        
        try:
            # Delete entries directory and recreate it
            await loop.run_in_executor(None, lambda: shutil.rmtree(self.entries_dir, ignore_errors=True))
            await loop.run_in_executor(None, lambda: os.makedirs(self.entries_dir, exist_ok=True))
            
            # Delete metadata file
            if os.path.exists(self.metadata_path):
                await loop.run_in_executor(None, os.remove, self.metadata_path)
                
            logger.info(f"Disk cache cleared at {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close disk cache store (no-op for disk-based store)."""
        # Nothing to do for disk-based store
        pass