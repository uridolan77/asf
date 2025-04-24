"""
Data models for progress tracking in the Conexus LLM Gateway.

This module defines the core data models used for tracking the progress
of long-running LLM operations.
"""

import copy
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ProgressStatus(str, Enum):
    """Enumeration of possible progress tracker statuses."""
    
    QUEUED = "queued"
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ProgressInfo(BaseModel):
    """Information about the progress tracking itself."""
    
    update_count: int = Field(0, description="Number of updates to this tracker")
    last_update_time: Optional[datetime] = Field(None, description="Time of the last update")
    update_interval_seconds: Optional[float] = Field(None, description="Average interval between updates")
    time_elapsed_seconds: Optional[float] = Field(None, description="Total time elapsed since creation")
    time_remaining_seconds: Optional[float] = Field(None, description="Estimated time remaining")


class ProgressMetadata(BaseModel):
    """Metadata for progress tracking."""
    
    # Step tracking
    total_steps: Optional[int] = Field(None, description="Total number of steps")
    current_step: Optional[int] = Field(None, description="Current step number")
    step_name: Optional[str] = Field(None, description="Name of the current step")
    
    # Completion estimation
    completion_percentage: Optional[float] = Field(None, description="Percentage complete (0-100)")
    estimated_completion_time: Optional[datetime] = Field(None, description="Estimated time of completion")
    
    # LLM-specific
    model_id: Optional[str] = Field(None, description="ID of the LLM model being used")
    provider_id: Optional[str] = Field(None, description="ID of the LLM provider")
    token_count: Optional[int] = Field(None, description="Current token count")
    
    # Request context
    request_id: Optional[str] = Field(None, description="ID of the associated request")
    conversation_id: Optional[str] = Field(None, description="ID of the associated conversation")
    
    # Additional custom metadata
    additional: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProgressUpdate(BaseModel):
    """An update to a progress tracker."""
    
    id: str = Field(default_factory=lambda: f"update_{uuid.uuid4().hex}", description="Unique ID of the update")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the update")
    message: str = Field(..., description="User-friendly progress message")
    status: Optional[ProgressStatus] = Field(None, description="New status (if changing)")
    detail: Optional[str] = Field(None, description="Technical details")
    metadata_updates: Optional[Dict[str, Any]] = Field(None, description="Metadata updates")


class ProgressTracker(BaseModel):
    """A tracker for progress of a long-running operation."""
    
    # Core identification
    id: str = Field(default_factory=lambda: f"progress_{uuid.uuid4().hex}", description="Unique ID of the tracker")
    task_id: str = Field(..., description="ID of the task being tracked")
    task_type: str = Field(..., description="Type of task being tracked")
    user_id: Optional[str] = Field(None, description="ID of the user who initiated the task")
    name: str = Field(..., description="User-friendly name for the task")
    description: Optional[str] = Field(None, description="Description of the task")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Time when tracker was created")
    started_at: Optional[datetime] = Field(None, description="Time when task started")
    completed_at: Optional[datetime] = Field(None, description="Time when task completed")
    expires_at: Optional[datetime] = Field(None, description="Time when tracker expires")
    
    # Status tracking
    status: ProgressStatus = Field(default=ProgressStatus.QUEUED, description="Current status")
    updates: List[ProgressUpdate] = Field(default_factory=list, description="History of updates")
    
    # Result/error handling
    result: Optional[Dict[str, Any]] = Field(None, description="Result data when complete")
    result_url: Optional[str] = Field(None, description="URL to access results when complete")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")
    
    # Metadata
    metadata: ProgressMetadata = Field(default_factory=ProgressMetadata, description="Progress metadata")
    info: ProgressInfo = Field(default_factory=ProgressInfo, description="Progress tracking information")
    
    def add_update(
        self,
        message: str,
        status: Optional[Union[ProgressStatus, str]] = None,
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProgressUpdate:
        """
        Add a progress update to this tracker.
        
        Args:
            message: User-friendly progress message
            status: New status (if status is changing)
            detail: Optional technical details
            metadata: Optional metadata to update
            
        Returns:
            The created update
        """
        now = datetime.utcnow()
        
        # Convert string status to enum if needed
        if status is not None and isinstance(status, str):
            status = ProgressStatus(status)
        
        # Create the update
        update = ProgressUpdate(
            message=message,
            status=status,
            detail=detail,
            metadata_updates=metadata
        )
        
        # Add to updates list
        self.updates.append(update)
        
        # Update status if provided
        if status is not None:
            old_status = self.status
            self.status = status
            
            # Handle status transitions
            if old_status != status:
                if status in [ProgressStatus.STARTING, ProgressStatus.IN_PROGRESS]:
                    # Starting or resuming
                    if not self.started_at:
                        self.started_at = now
                elif status == ProgressStatus.COMPLETED:
                    # Completing
                    self.completed_at = now
        
        # Update metadata if provided
        if metadata:
            for key, value in metadata.items():
                if key in ["total_steps", "current_step", "step_name", 
                          "completion_percentage", "estimated_completion_time",
                          "model_id", "provider_id", "token_count",
                          "request_id", "conversation_id"]:
                    # These are direct attributes of metadata
                    setattr(self.metadata, key, value)
                else:
                    # Other keys go into additional
                    self.metadata.additional[key] = value
        
        # Update progress tracking info
        self.info.update_count += 1
        
        # Calculate update interval
        if self.info.last_update_time:
            interval = (now - self.info.last_update_time).total_seconds()
            if self.info.update_interval_seconds is None:
                self.info.update_interval_seconds = interval
            else:
                # Exponential moving average for smoothing
                self.info.update_interval_seconds = 0.7 * self.info.update_interval_seconds + 0.3 * interval
        
        # Update timing info
        self.info.last_update_time = now
        self.info.time_elapsed_seconds = (now - self.created_at).total_seconds()
        
        # Estimate remaining time if we have percentage and interval data
        if (self.metadata.completion_percentage is not None and 
            self.metadata.completion_percentage > 0 and 
            self.info.update_interval_seconds is not None):
            
            # Simple linear extrapolation
            pct_remaining = 100.0 - self.metadata.completion_percentage
            updates_remaining = pct_remaining / (self.metadata.completion_percentage / self.info.update_count)
            self.info.time_remaining_seconds = updates_remaining * self.info.update_interval_seconds
        
        return update
    
    def get_latest_update(self) -> Optional[ProgressUpdate]:
        """Get the most recent update, if any."""
        if not self.updates:
            return None
        return self.updates[-1]
    
    def get_progress_percentage(self) -> float:
        """
        Get the current progress percentage.
        
        Returns:
            Progress percentage between 0 and 100
        """
        # If explicitly set in metadata, use that
        if self.metadata.completion_percentage is not None:
            return self.metadata.completion_percentage
        
        # If steps are defined, calculate from steps
        if self.metadata.current_step is not None and self.metadata.total_steps is not None:
            if self.metadata.total_steps > 0:
                return min(100.0, (self.metadata.current_step / self.metadata.total_steps) * 100.0)
        
        # Default percentages based on status
        if self.status == ProgressStatus.QUEUED:
            return 0.0
        elif self.status == ProgressStatus.STARTING:
            return 5.0
        elif self.status == ProgressStatus.IN_PROGRESS:
            return 50.0
        elif self.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
            return 100.0
        elif self.status == ProgressStatus.PAUSED:
            return 50.0
        
        return 0.0