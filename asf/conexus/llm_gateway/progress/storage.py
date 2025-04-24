"""
Storage layer for progress trackers.

This module provides interfaces and implementations for persisting
progress tracking information.
"""

import abc
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from asf.conexus.llm_gateway.progress.models import ProgressTracker

logger = logging.getLogger(__name__)


class ProgressStorage(abc.ABC):
    """Abstract base class for progress tracker storage implementations."""
    
    @abc.abstractmethod
    async def save_tracker(self, tracker: ProgressTracker) -> None:
        """
        Save a progress tracker.
        
        Args:
            tracker: The tracker to save
        """
        pass
    
    @abc.abstractmethod
    async def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """
        Get a progress tracker by ID.
        
        Args:
            tracker_id: ID of the tracker
            
        Returns:
            The tracker if found, None otherwise
        """
        pass
    
    @abc.abstractmethod
    async def get_trackers_by_task_id(self, task_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of trackers for the task
        """
        pass
    
    @abc.abstractmethod
    async def get_trackers_by_user_id(self, user_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of trackers for the user
        """
        pass
    
    @abc.abstractmethod
    async def get_expired_trackers(self, as_of: datetime) -> List[ProgressTracker]:
        """
        Get all trackers that have expired as of the specified time.
        
        Args:
            as_of: Reference time for expiration check
            
        Returns:
            List of expired trackers
        """
        pass
    
    @abc.abstractmethod
    async def delete_tracker(self, tracker_id: str) -> bool:
        """
        Delete a progress tracker.
        
        Args:
            tracker_id: ID of the tracker to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass


class InMemoryProgressStorage(ProgressStorage):
    """In-memory implementation of progress tracker storage."""
    
    def __init__(self):
        """Initialize the storage."""
        self.trackers: Dict[str, ProgressTracker] = {}
    
    async def save_tracker(self, tracker: ProgressTracker) -> None:
        """
        Save a progress tracker.
        
        Args:
            tracker: The tracker to save
        """
        self.trackers[tracker.id] = tracker
    
    async def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """
        Get a progress tracker by ID.
        
        Args:
            tracker_id: ID of the tracker
            
        Returns:
            The tracker if found, None otherwise
        """
        return self.trackers.get(tracker_id)
    
    async def get_trackers_by_task_id(self, task_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of trackers for the task
        """
        return [t for t in self.trackers.values() if t.task_id == task_id]
    
    async def get_trackers_by_user_id(self, user_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of trackers for the user
        """
        if not user_id:
            return []
        return [t for t in self.trackers.values() if t.user_id == user_id]
    
    async def get_expired_trackers(self, as_of: datetime) -> List[ProgressTracker]:
        """
        Get all trackers that have expired as of the specified time.
        
        Args:
            as_of: Reference time for expiration check
            
        Returns:
            List of expired trackers
        """
        return [
            t for t in self.trackers.values()
            if t.expires_at and t.expires_at <= as_of
        ]
    
    async def delete_tracker(self, tracker_id: str) -> bool:
        """
        Delete a progress tracker.
        
        Args:
            tracker_id: ID of the tracker to delete
            
        Returns:
            True if deleted, False if not found
        """
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]
            return True
        return False


class FileProgressStorage(ProgressStorage):
    """File-based implementation of progress tracker storage."""
    
    def __init__(self, storage_dir: str):
        """
        Initialize the storage.
        
        Args:
            storage_dir: Directory to store tracker files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
    
    def _get_tracker_path(self, tracker_id: str) -> str:
        """Get file path for a tracker."""
        return os.path.join(self.storage_dir, f"{tracker_id}.json")
    
    async def save_tracker(self, tracker: ProgressTracker) -> None:
        """
        Save a progress tracker.
        
        Args:
            tracker: The tracker to save
        """
        # Serialize the tracker to JSON
        tracker_data = tracker.model_dump()
        
        # Write to file
        with open(self._get_tracker_path(tracker.id), "w") as f:
            json.dump(tracker_data, f)
    
    async def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """
        Get a progress tracker by ID.
        
        Args:
            tracker_id: ID of the tracker
            
        Returns:
            The tracker if found, None otherwise
        """
        path = self._get_tracker_path(tracker_id)
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, "r") as f:
                tracker_data = json.load(f)
            
            return ProgressTracker.model_validate(tracker_data)
        except Exception as e:
            logger.error(f"Error loading tracker {tracker_id}: {e}")
            return None
    
    async def get_all_trackers(self) -> List[ProgressTracker]:
        """Get all trackers."""
        trackers = []
        
        # List all JSON files in the storage directory
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith(".json"):
                continue
                
            tracker_id = filename[:-5]  # Remove .json extension
            tracker = await self.get_tracker(tracker_id)
            
            if tracker:
                trackers.append(tracker)
        
        return trackers
    
    async def get_trackers_by_task_id(self, task_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of trackers for the task
        """
        all_trackers = await self.get_all_trackers()
        return [t for t in all_trackers if t.task_id == task_id]
    
    async def get_trackers_by_user_id(self, user_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of trackers for the user
        """
        if not user_id:
            return []
            
        all_trackers = await self.get_all_trackers()
        return [t for t in all_trackers if t.user_id == user_id]
    
    async def get_expired_trackers(self, as_of: datetime) -> List[ProgressTracker]:
        """
        Get all trackers that have expired as of the specified time.
        
        Args:
            as_of: Reference time for expiration check
            
        Returns:
            List of expired trackers
        """
        all_trackers = await self.get_all_trackers()
        return [
            t for t in all_trackers
            if t.expires_at and t.expires_at <= as_of
        ]
    
    async def delete_tracker(self, tracker_id: str) -> bool:
        """
        Delete a progress tracker.
        
        Args:
            tracker_id: ID of the tracker to delete
            
        Returns:
            True if deleted, False if not found
        """
        path = self._get_tracker_path(tracker_id)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False


# Factory function to get configured storage instance
def get_progress_storage() -> ProgressStorage:
    """
    Get the configured progress storage implementation.
    
    Returns:
        A progress storage implementation
    """
    # Check environment variable for storage type
    storage_type = os.environ.get("LLMGATEWAY_PROGRESS_STORAGE", "memory")
    
    if storage_type == "file":
        storage_dir = os.environ.get(
            "LLMGATEWAY_PROGRESS_DIR", 
            os.path.join(os.path.dirname(__file__), "trackers")
        )
        return FileProgressStorage(storage_dir)
    else:
        return InMemoryProgressStorage()