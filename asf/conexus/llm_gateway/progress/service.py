"""
Progress tracking service for the Conexus LLM Gateway.

This module provides services for creating, updating, and retrieving
progress trackers for long-running operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union

from asf.conexus.llm_gateway.progress.models import (
    ProgressTracker,
    ProgressStatus,
    ProgressUpdate
)
from asf.conexus.llm_gateway.progress.storage import ProgressStorage, get_progress_storage

logger = logging.getLogger(__name__)


class ProgressTrackingService:
    """
    Service for tracking progress of long-running operations.
    
    This service provides functionality to create, update, and retrieve
    progress trackers for tasks such as model fine-tuning, batch 
    processing, and other long-running operations.
    """
    
    def __init__(self, storage: ProgressStorage):
        """
        Initialize the progress tracking service.
        
        Args:
            storage: Storage interface for progress trackers
        """
        self.storage = storage
        self._cleanup_task = None
    
    async def create_tracker(
        self,
        task_id: str,
        task_type: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        retention_hours: Optional[int] = None
    ) -> ProgressTracker:
        """
        Create a new progress tracker.
        
        Args:
            task_id: ID of the task to track
            task_type: Type of task (e.g., "fine-tuning", "batch-processing")
            name: User-friendly name for the task
            description: Description of the task
            user_id: ID of the user who initiated the task
            metadata: Additional metadata for the tracker
            retention_hours: How long to keep this tracker after completion
            
        Returns:
            The created tracker
        """
        # Create a new tracker
        tracker = ProgressTracker(
            task_id=task_id,
            task_type=task_type,
            name=name or f"{task_type.title()} task",
            description=description,
            user_id=user_id
        )
        
        # Set expiration if specified
        if retention_hours is not None:
            tracker.expires_at = datetime.utcnow() + timedelta(hours=retention_hours)
        
        # Set initial metadata
        if metadata:
            for key, value in metadata.items():
                if key in ["total_steps", "current_step", "step_name", 
                          "completion_percentage", "estimated_completion_time",
                          "model_id", "provider_id", "token_count",
                          "request_id", "conversation_id"]:
                    # These are direct attributes of metadata
                    setattr(tracker.metadata, key, value)
                else:
                    # Other keys go into additional
                    tracker.metadata.additional[key] = value
        
        # Add initial update
        tracker.add_update(
            message=f"Created tracker for {task_type} task",
            status=ProgressStatus.QUEUED
        )
        
        # Save to storage
        await self.storage.save_tracker(tracker)
        
        logger.info(f"Created progress tracker {tracker.id} for task {task_id} of type {task_type}")
        return tracker
    
    async def get_tracker(self, tracker_id: str) -> Optional[ProgressTracker]:
        """
        Get a progress tracker by ID.
        
        Args:
            tracker_id: ID of the tracker to retrieve
            
        Returns:
            The tracker if found, None otherwise
        """
        return await self.storage.get_tracker(tracker_id)
    
    async def get_trackers_for_task(self, task_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific task.
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of trackers for the task
        """
        return await self.storage.get_trackers_by_task_id(task_id)
    
    async def get_trackers_for_user(self, user_id: str) -> List[ProgressTracker]:
        """
        Get all progress trackers for a specific user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of trackers for the user
        """
        return await self.storage.get_trackers_by_user_id(user_id)
    
    async def update_tracker(
        self,
        tracker_id: str,
        message: str,
        status: Optional[Union[ProgressStatus, str]] = None,
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressUpdate]:
        """
        Update a progress tracker.
        
        Args:
            tracker_id: ID of the tracker to update
            message: User-friendly progress message
            status: New status (if changing)
            detail: Technical details
            metadata: Additional metadata to update
            
        Returns:
            The created update if successful, None if tracker not found
        """
        # Get the tracker
        tracker = await self.storage.get_tracker(tracker_id)
        if not tracker:
            logger.warning(f"Attempted to update non-existent tracker: {tracker_id}")
            return None
        
        # Add the update
        update = tracker.add_update(
            message=message,
            status=status,
            detail=detail,
            metadata=metadata
        )
        
        # Save the updated tracker
        await self.storage.save_tracker(tracker)
        
        logger.debug(f"Updated tracker {tracker_id}: {message}")
        return update
    
    async def complete_tracker(
        self,
        tracker_id: str,
        message: str = "Task completed",
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
        result_url: Optional[str] = None
    ) -> Optional[ProgressTracker]:
        """
        Mark a tracker as completed.
        
        Args:
            tracker_id: ID of the tracker to complete
            message: Completion message
            detail: Technical details
            metadata: Additional metadata to update
            result: Result data to store
            result_url: URL to access results
            
        Returns:
            The updated tracker if successful, None if not found
        """
        # Get the tracker
        tracker = await self.storage.get_tracker(tracker_id)
        if not tracker:
            logger.warning(f"Attempted to complete non-existent tracker: {tracker_id}")
            return None
        
        # Add completion update
        tracker.add_update(
            message=message,
            status=ProgressStatus.COMPLETED,
            detail=detail,
            metadata=metadata
        )
        
        # Set completion time if not already set
        if not tracker.completed_at:
            tracker.completed_at = datetime.utcnow()
        
        # Store result data if provided
        if result is not None:
            tracker.result = result
        
        # Store result URL if provided
        if result_url is not None:
            tracker.result_url = result_url
        
        # Save the updated tracker
        await self.storage.save_tracker(tracker)
        
        logger.info(f"Completed tracker {tracker_id}: {message}")
        return tracker
    
    async def fail_tracker(
        self,
        tracker_id: str,
        message: str = "Task failed",
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressTracker]:
        """
        Mark a tracker as failed.
        
        Args:
            tracker_id: ID of the tracker to fail
            message: Failure message
            detail: Technical details
            metadata: Additional metadata to update
            error: Error details to store
            
        Returns:
            The updated tracker if successful, None if not found
        """
        # Get the tracker
        tracker = await self.storage.get_tracker(tracker_id)
        if not tracker:
            logger.warning(f"Attempted to fail non-existent tracker: {tracker_id}")
            return None
        
        # Add failure update
        tracker.add_update(
            message=message,
            status=ProgressStatus.FAILED,
            detail=detail,
            metadata=metadata
        )
        
        # Set completion time if not already set
        if not tracker.completed_at:
            tracker.completed_at = datetime.utcnow()
        
        # Store error data if provided
        if error is not None:
            tracker.error = error
        
        # Save the updated tracker
        await self.storage.save_tracker(tracker)
        
        logger.info(f"Failed tracker {tracker_id}: {message}")
        return tracker
    
    async def cancel_tracker(
        self,
        tracker_id: str,
        message: str = "Task cancelled",
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[ProgressTracker]:
        """
        Mark a tracker as cancelled.
        
        Args:
            tracker_id: ID of the tracker to cancel
            message: Cancellation message
            detail: Technical details
            metadata: Additional metadata to update
            
        Returns:
            The updated tracker if successful, None if not found
        """
        # Get the tracker
        tracker = await self.storage.get_tracker(tracker_id)
        if not tracker:
            logger.warning(f"Attempted to cancel non-existent tracker: {tracker_id}")
            return None
        
        # Add cancellation update
        tracker.add_update(
            message=message,
            status=ProgressStatus.CANCELLED,
            detail=detail,
            metadata=metadata
        )
        
        # Set completion time if not already set
        if not tracker.completed_at:
            tracker.completed_at = datetime.utcnow()
        
        # Save the updated tracker
        await self.storage.save_tracker(tracker)
        
        logger.info(f"Cancelled tracker {tracker_id}: {message}")
        return tracker
    
    async def delete_tracker(self, tracker_id: str) -> bool:
        """
        Delete a progress tracker.
        
        Args:
            tracker_id: ID of the tracker to delete
            
        Returns:
            True if deleted, False if not found
        """
        result = await self.storage.delete_tracker(tracker_id)
        if result:
            logger.info(f"Deleted tracker {tracker_id}")
        else:
            logger.warning(f"Attempted to delete non-existent tracker: {tracker_id}")
        
        return result
    
    async def start_cleanup_task(self, interval_seconds: int = 3600) -> None:
        """
        Start a background task to clean up expired trackers.
        
        Args:
            interval_seconds: How often to run the cleanup task
        """
        if self._cleanup_task is not None:
            logger.warning("Cleanup task already running")
            return
        
        async def cleanup_loop() -> None:
            """Background loop for cleaning up expired trackers."""
            while True:
                try:
                    # Get expired trackers
                    now = datetime.utcnow()
                    expired = await self.storage.get_expired_trackers(now)
                    
                    if expired:
                        logger.info(f"Cleaning up {len(expired)} expired trackers")
                        
                        # Delete each expired tracker
                        for tracker in expired:
                            await self.storage.delete_tracker(tracker.id)
                    
                    # Sleep until next cleanup
                    await asyncio.sleep(interval_seconds)
                
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")
                    await asyncio.sleep(interval_seconds)
        
        # Start the cleanup task
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info(f"Started tracker cleanup task with interval {interval_seconds}s")
    
    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            
            self._cleanup_task = None
            logger.info("Stopped tracker cleanup task")


# Singleton pattern for service instance
_progress_service_instance = None


def get_progress_service() -> ProgressTrackingService:
    """
    Get the singleton progress tracking service instance.
    
    Returns:
        The progress tracking service
    """
    global _progress_service_instance
    
    if _progress_service_instance is None:
        # Create a new instance
        storage = get_progress_storage()
        _progress_service_instance = ProgressTrackingService(storage)
    
    return _progress_service_instance