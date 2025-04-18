"""
Progress registry for the LLM Gateway.

This module provides a registry for managing multiple progress trackers,
allowing for centralized access to progress information across the application.
"""

import logging
import threading
from typing import Dict, List, Optional, Any

from .tracker import ProgressTracker
from .models import ProgressDetails, OperationType

# Set up logging
logger = logging.getLogger(__name__)


class ProgressRegistry:
    """
    Registry for managing multiple progress trackers.
    
    This class provides a centralized registry for progress trackers,
    allowing for easy access to progress information across the application.
    
    Attributes:
        trackers: Dictionary of progress trackers, keyed by operation ID
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProgressRegistry, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, cache_manager=None):
        """
        Initialize the progress registry.
        
        Args:
            cache_manager: Cache manager to use for saving progress
        """
        # Only initialize once
        if getattr(self, "_initialized", False):
            return
            
        self.trackers: Dict[str, ProgressTracker] = {}
        self.cache_manager = cache_manager
        self._lock = threading.RLock()
        self._initialized = True
        
        logger.debug("Progress registry initialized")
    
    def register(self, tracker: ProgressTracker) -> None:
        """
        Register a progress tracker.
        
        Args:
            tracker: Progress tracker to register
        """
        with self._lock:
            self.trackers[tracker.operation_id] = tracker
            logger.debug(f"Registered progress tracker for operation {tracker.operation_id}")
    
    def unregister(self, operation_id: str) -> None:
        """
        Unregister a progress tracker.
        
        Args:
            operation_id: Operation ID of the tracker to unregister
        """
        with self._lock:
            if operation_id in self.trackers:
                del self.trackers[operation_id]
                logger.debug(f"Unregistered progress tracker for operation {operation_id}")
    
    def get_tracker(self, operation_id: str) -> Optional[ProgressTracker]:
        """
        Get a progress tracker by operation ID.
        
        Args:
            operation_id: Operation ID of the tracker to get
            
        Returns:
            Progress tracker or None if not found
        """
        with self._lock:
            return self.trackers.get(operation_id)
    
    def create_tracker(
        self,
        operation_id: str,
        operation_type: OperationType = OperationType.GENERAL,
        total_steps: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        on_update=None
    ) -> ProgressTracker:
        """
        Create and register a new progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation being tracked
            total_steps: Total number of steps in the operation
            metadata: Additional metadata about the operation
            on_update: Callback function to call when progress is updated
            
        Returns:
            Newly created progress tracker
        """
        with self._lock:
            # Check if tracker already exists
            if operation_id in self.trackers:
                logger.warning(f"Progress tracker for operation {operation_id} already exists")
                return self.trackers[operation_id]
            
            # Create new tracker
            tracker = ProgressTracker(
                operation_id=operation_id,
                operation_type=operation_type,
                total_steps=total_steps,
                metadata=metadata,
                on_update=on_update,
                cache_manager=self.cache_manager
            )
            
            # Register tracker
            self.trackers[operation_id] = tracker
            logger.debug(f"Created and registered progress tracker for operation {operation_id}")
            
            return tracker
    
    def get_all_trackers(self) -> List[ProgressTracker]:
        """
        Get all registered progress trackers.
        
        Returns:
            List of all registered progress trackers
        """
        with self._lock:
            return list(self.trackers.values())
    
    def get_active_trackers(self) -> List[ProgressTracker]:
        """
        Get all active progress trackers.
        
        Returns:
            List of all active progress trackers
        """
        with self._lock:
            return [
                tracker for tracker in self.trackers.values()
                if tracker.status not in ("completed", "failed", "cancelled")
            ]
    
    def get_all_progress(self) -> Dict[str, ProgressDetails]:
        """
        Get progress details for all registered trackers.
        
        Returns:
            Dictionary mapping operation IDs to progress details
        """
        with self._lock:
            return {
                operation_id: tracker.get_progress_details()
                for operation_id, tracker in self.trackers.items()
            }
    
    def cleanup(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old completed trackers.
        
        Args:
            max_age_seconds: Maximum age in seconds for completed trackers
            
        Returns:
            Number of trackers removed
        """
        import time
        
        with self._lock:
            now = time.time()
            to_remove = []
            
            for operation_id, tracker in self.trackers.items():
                if tracker.status in ("completed", "failed", "cancelled"):
                    if tracker.end_time and (now - tracker.end_time) > max_age_seconds:
                        to_remove.append(operation_id)
            
            for operation_id in to_remove:
                del self.trackers[operation_id]
            
            logger.debug(f"Cleaned up {len(to_remove)} old progress trackers")
            return len(to_remove)


# Singleton instance
_progress_registry = None


def get_progress_registry(cache_manager=None) -> ProgressRegistry:
    """
    Get the singleton instance of the ProgressRegistry.
    
    Args:
        cache_manager: Cache manager to use for saving progress
        
    Returns:
        ProgressRegistry instance
    """
    global _progress_registry
    if _progress_registry is None:
        with ProgressRegistry._lock:
            if _progress_registry is None:
                _progress_registry = ProgressRegistry(cache_manager=cache_manager)
    return _progress_registry
