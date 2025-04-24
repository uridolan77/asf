"""
Progress tracking for the Conexus LLM Gateway.

This module provides functionality for tracking and reporting on the
progress of LLM requests, which can be useful for long-running requests
and improving user experience.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Union, Callable

from asf.conexus.llm_gateway.observability.metrics import gauge_set

logger = logging.getLogger(__name__)


class ProgressState(str, Enum):
    """States for progress tracking."""
    QUEUED = "queued"
    STARTING = "starting"
    THINKING = "thinking"
    GENERATING = "generating"
    COMPLETING = "completing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProgressTracker:
    """
    Tracks the progress of LLM requests.
    
    This class maintains the state, progress percentage, and other metadata
    for tracking the progress of LLM requests.
    """
    
    def __init__(self, request_id: str, timeout_seconds: float = 300.0):
        """
        Initialize the progress tracker.
        
        Args:
            request_id: Unique identifier for the request
            timeout_seconds: Maximum time to track the request
        """
        self.request_id = request_id
        self.timeout_seconds = timeout_seconds
        
        # Track state and progress
        self._state = ProgressState.QUEUED
        self._progress = 0.0  # 0.0 to 1.0
        self._message = "Request queued"
        
        # Times for different phases
        self._created_at = time.time()
        self._started_at: Optional[float] = None
        self._completed_at: Optional[float] = None
        self._thinking_start: Optional[float] = None
        self._generating_start: Optional[float] = None
        
        # Tokens info
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        
        # Completion info
        self._completion_text = ""
        self._token_count = 0
        
        # Metrics
        self._record_metrics()
    
    @property
    def state(self) -> ProgressState:
        """Get the current state."""
        return self._state
    
    @property
    def progress(self) -> float:
        """Get the progress percentage (0.0 to 1.0)."""
        return self._progress
    
    @property
    def message(self) -> str:
        """Get the current progress message."""
        return self._message
    
    @property
    def elapsed_seconds(self) -> float:
        """Get the elapsed time in seconds."""
        if self._completed_at:
            # Use completion time if completed
            return self._completed_at - self._created_at
        else:
            # Otherwise use current time
            return time.time() - self._created_at
    
    @property
    def thinking_seconds(self) -> Optional[float]:
        """Get the thinking time in seconds."""
        if not self._thinking_start:
            return None
            
        if self._generating_start:
            # If generating has started, thinking is done
            return self._generating_start - self._thinking_start
        else:
            # Otherwise, thinking is still in progress
            return time.time() - self._thinking_start
    
    @property
    def generating_seconds(self) -> Optional[float]:
        """Get the generating time in seconds."""
        if not self._generating_start:
            return None
            
        if self._completed_at:
            # If completed, use completion time
            return self._completed_at - self._generating_start
        else:
            # Otherwise still generating
            return time.time() - self._generating_start
    
    @property
    def is_completed(self) -> bool:
        """Check if the request is completed (success, failure, or cancellation)."""
        return self._state in {
            ProgressState.COMPLETED, 
            ProgressState.FAILED, 
            ProgressState.CANCELLED
        }
    
    @property
    def is_active(self) -> bool:
        """Check if the request is still active."""
        return not self.is_completed
    
    @property
    def tokens_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens
        }
    
    @property
    def times(self) -> Dict[str, Union[float, None]]:
        """Get timing information."""
        return {
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "thinking_start": self._thinking_start,
            "generating_start": self._generating_start,
            "elapsed_seconds": self.elapsed_seconds,
            "thinking_seconds": self.thinking_seconds,
            "generating_seconds": self.generating_seconds
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert progress tracker to a dictionary."""
        return {
            "request_id": self.request_id,
            "state": self._state,
            "progress": self._progress,
            "message": self._message,
            "created_at": self._created_at,
            "started_at": self._started_at,
            "completed_at": self._completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "thinking_seconds": self.thinking_seconds,
            "generating_seconds": self.generating_seconds,
            "tokens": {
                "prompt": self._prompt_tokens,
                "completion": self._completion_tokens,
                "total": self._total_tokens
            },
            "is_completed": self.is_completed,
            "is_active": self.is_active
        }
    
    def update_state(
        self, 
        state: ProgressState, 
        message: Optional[str] = None,
        progress: Optional[float] = None
    ) -> None:
        """
        Update the state and message.
        
        Args:
            state: New state
            message: Optional progress message
            progress: Optional progress percentage (0.0 to 1.0)
        """
        # Update state with appropriate timestamps
        self._state = state
        
        # Set default messages for states if not provided
        if message is None:
            if state == ProgressState.QUEUED:
                message = "Request queued"
            elif state == ProgressState.STARTING:
                message = "Request starting"
            elif state == ProgressState.THINKING:
                message = "Thinking..."
            elif state == ProgressState.GENERATING:
                message = "Generating response..."
            elif state == ProgressState.COMPLETING:
                message = "Finalizing response..."
            elif state == ProgressState.COMPLETED:
                message = "Request completed"
            elif state == ProgressState.CANCELLED:
                message = "Request cancelled"
            elif state == ProgressState.FAILED:
                message = "Request failed"
        
        self._message = message or self._message
        
        # Update progress if provided
        if progress is not None:
            self._progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        elif state == ProgressState.COMPLETED:
            self._progress = 1.0
        
        # Update timestamps
        if state == ProgressState.STARTING and self._started_at is None:
            self._started_at = time.time()
        elif state == ProgressState.THINKING and self._thinking_start is None:
            self._thinking_start = time.time()
        elif state == ProgressState.GENERATING and self._generating_start is None:
            self._generating_start = time.time()
        elif state in {ProgressState.COMPLETED, ProgressState.CANCELLED, ProgressState.FAILED} and self._completed_at is None:
            self._completed_at = time.time()
            
        # Record metrics
        self._record_metrics()
            
        logger.debug(f"Progress for {self.request_id}: {state}, {progress}, {message}")
    
    def update_tokens(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None
    ) -> None:
        """
        Update token counts.
        
        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens
        """
        if prompt_tokens is not None:
            self._prompt_tokens = prompt_tokens
            
        if completion_tokens is not None:
            self._completion_tokens = completion_tokens
            
        if total_tokens is not None:
            self._total_tokens = total_tokens
        elif prompt_tokens is not None or completion_tokens is not None:
            self._total_tokens = self._prompt_tokens + self._completion_tokens
            
        # Record metrics
        self._record_metrics()
    
    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """
        Update progress percentage and optionally message.
        
        Args:
            progress: Progress percentage (0.0 to 1.0)
            message: Optional progress message
        """
        self._progress = max(0.0, min(1.0, progress))  # Clamp between 0 and 1
        
        if message:
            self._message = message
            
        # Record metrics
        self._record_metrics()
        
        logger.debug(f"Progress for {self.request_id}: {progress}, {message}")
    
    def mark_started(self, message: Optional[str] = None) -> None:
        """Mark the request as started."""
        self.update_state(ProgressState.STARTING, message)
    
    def mark_thinking(self, message: Optional[str] = None) -> None:
        """Mark the request as in thinking phase."""
        self.update_state(ProgressState.THINKING, message)
    
    def mark_generating(self, message: Optional[str] = None) -> None:
        """Mark the request as in generating phase."""
        self.update_state(ProgressState.GENERATING, message)
    
    def mark_completing(self, message: Optional[str] = None) -> None:
        """Mark the request as completing."""
        self.update_state(ProgressState.COMPLETING, message)
    
    def mark_completed(self, message: Optional[str] = None) -> None:
        """Mark the request as completed successfully."""
        self.update_state(ProgressState.COMPLETED, message, 1.0)
    
    def mark_cancelled(self, message: Optional[str] = None) -> None:
        """Mark the request as cancelled."""
        self.update_state(ProgressState.CANCELLED, message)
    
    def mark_failed(self, message: Optional[str] = None) -> None:
        """Mark the request as failed."""
        self.update_state(ProgressState.FAILED, message)
    
    def is_timed_out(self) -> bool:
        """Check if the request has timed out."""
        if not self.is_active:  # Completed requests can't time out
            return False
            
        return (time.time() - self._created_at) > self.timeout_seconds
    
    def _record_metrics(self) -> None:
        """Record metrics for this tracker."""
        try:
            # Record state as a gauge
            gauge_set(
                "llm_gateway_request_state", 
                self._state_to_number(),
                {"request_id": self.request_id}
            )
            
            # Record progress
            gauge_set(
                "llm_gateway_request_progress",
                self._progress,
                {"request_id": self.request_id}
            )
            
            # Record tokens
            gauge_set(
                "llm_gateway_request_tokens",
                self._prompt_tokens,
                {"request_id": self.request_id, "type": "prompt"}
            )
            gauge_set(
                "llm_gateway_request_tokens",
                self._completion_tokens,
                {"request_id": self.request_id, "type": "completion"}
            )
            gauge_set(
                "llm_gateway_request_tokens",
                self._total_tokens,
                {"request_id": self.request_id, "type": "total"}
            )
            
        except Exception as e:
            # Don't let metric recording failures affect functionality
            logger.warning(f"Error recording progress metrics: {e}")
            
    def _state_to_number(self) -> int:
        """Convert state to a numeric value for metrics."""
        state_values = {
            ProgressState.QUEUED: 0,
            ProgressState.STARTING: 1,
            ProgressState.THINKING: 2,
            ProgressState.GENERATING: 3,
            ProgressState.COMPLETING: 4,
            ProgressState.COMPLETED: 5,
            ProgressState.CANCELLED: -1,
            ProgressState.FAILED: -2
        }
        return state_values.get(self._state, 0)


class ProgressRegistry:
    """
    Registry for managing progress trackers.
    
    This class maintains a collection of progress trackers for multiple
    requests and provides methods for creating, accessing, and cleaning them up.
    """
    
    def __init__(self, max_trackers: int = 10000, cleanup_interval_seconds: float = 300.0):
        """
        Initialize the progress registry.
        
        Args:
            max_trackers: Maximum number of trackers to store
            cleanup_interval_seconds: Interval for cleaning up old trackers
        """
        self.max_trackers = max_trackers
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        self._trackers: Dict[str, ProgressTracker] = {}
        self._lock = Lock()
        self._running_cleanup = False
    
    def create_tracker(self, request_id: str, timeout_seconds: float = 300.0) -> ProgressTracker:
        """
        Create a new progress tracker.
        
        Args:
            request_id: Unique identifier for the request
            timeout_seconds: Maximum time to track the request
            
        Returns:
            The newly created tracker
        """
        with self._lock:
            # Check if we need to remove old entries
            if len(self._trackers) >= self.max_trackers:
                # Simple approach: remove oldest trackers first
                oldest_time = time.time()
                oldest_id = None
                
                for rid, tracker in self._trackers.items():
                    if not tracker.is_active and tracker._created_at < oldest_time:
                        oldest_time = tracker._created_at
                        oldest_id = rid
                        
                # If we found an inactive tracker, remove it
                if oldest_id:
                    del self._trackers[oldest_id]
            
            # Create new tracker
            tracker = ProgressTracker(request_id, timeout_seconds)
            self._trackers[request_id] = tracker
            
            # Schedule cleanup if not already running
            if not self._running_cleanup:
                asyncio.create_task(self._periodic_cleanup())
            
            return tracker
    
    def get_tracker(self, request_id: str) -> Optional[ProgressTracker]:
        """
        Get a progress tracker by request ID.
        
        Args:
            request_id: The request ID to look up
            
        Returns:
            The tracker if found, None otherwise
        """
        return self._trackers.get(request_id)
    
    def update_tracker(
        self, 
        request_id: str,
        state: Optional[ProgressState] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None
    ) -> Optional[ProgressTracker]:
        """
        Update a progress tracker if it exists.
        
        Args:
            request_id: The request ID to update
            state: New state (optional)
            progress: New progress value (optional)
            message: New message (optional)
            
        Returns:
            The updated tracker if found, None otherwise
        """
        tracker = self.get_tracker(request_id)
        if not tracker:
            return None
            
        if state is not None:
            tracker.update_state(state, message, progress)
        elif progress is not None:
            tracker.update_progress(progress, message)
            
        return tracker
    
    def remove_tracker(self, request_id: str) -> bool:
        """
        Remove a tracker from the registry.
        
        Args:
            request_id: The request ID to remove
            
        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if request_id in self._trackers:
                del self._trackers[request_id]
                return True
            return False
    
    def get_all_trackers(self) -> Dict[str, ProgressTracker]:
        """
        Get all active trackers.
        
        Returns:
            Dictionary of request IDs to trackers
        """
        return {rid: tracker for rid, tracker in self._trackers.items() if tracker.is_active}
    
    def get_all_trackers_as_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all active trackers as dictionaries.
        
        Returns:
            Dictionary of request IDs to tracker dictionaries
        """
        return {rid: tracker.to_dict() for rid, tracker in self._trackers.items() if tracker.is_active}
    
    def cleanup(self) -> int:
        """
        Clean up old, completed trackers.
        
        Returns:
            Number of trackers removed
        """
        with self._lock:
            # Get IDs of trackers to remove (completed and older than 10 minutes)
            cutoff_time = time.time() - 600  # 10 minutes
            to_remove = []
            
            for rid, tracker in self._trackers.items():
                if not tracker.is_active and (tracker._completed_at or 0) < cutoff_time:
                    to_remove.append(rid)
                elif tracker.is_timed_out():  # Also clean up timed-out trackers
                    tracker.mark_failed("Request timed out")
                    if len(to_remove) < 100:  # Limit to removing 100 trackers at once
                        to_remove.append(rid)
            
            # Remove the identified trackers
            for rid in to_remove:
                del self._trackers[rid]
                
            return len(to_remove)
    
    async def _periodic_cleanup(self) -> None:
        """Run periodic cleanup of old trackers."""
        with self._lock:
            if self._running_cleanup:
                return
            self._running_cleanup = True
            
        try:
            while True:
                await asyncio.sleep(self.cleanup_interval_seconds)
                count = self.cleanup()
                if count > 0:
                    logger.debug(f"Cleaned up {count} old progress trackers")
        except Exception as e:
            logger.error(f"Error in progress tracker cleanup: {e}")
        finally:
            with self._lock:
                self._running_cleanup = False


# Singleton instance
_progress_registry = None


def get_progress_registry() -> ProgressRegistry:
    """
    Get the global progress registry instance.
    
    Returns:
        The progress registry singleton
    """
    global _progress_registry
    if _progress_registry is None:
        _progress_registry = ProgressRegistry()
    return _progress_registry


# Convenience functions
def create_tracker(request_id: str, timeout_seconds: float = 300.0) -> ProgressTracker:
    """Create a new progress tracker."""
    return get_progress_registry().create_tracker(request_id, timeout_seconds)


def get_tracker(request_id: str) -> Optional[ProgressTracker]:
    """Get a tracker by request ID."""
    return get_progress_registry().get_tracker(request_id)


def update_tracker(
    request_id: str,
    state: Optional[ProgressState] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None
) -> Optional[ProgressTracker]:
    """Update a tracker if it exists."""
    return get_progress_registry().update_tracker(request_id, state, progress, message)