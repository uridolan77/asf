"""
Progress tracker implementation for the LLM Gateway.

This module provides the base ProgressTracker class for tracking the progress
of long-running operations, such as complex LLM requests, batch processing,
and model fine-tuning.
"""

import time
import enum
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable

from .models import ProgressDetails, ProgressStep, OperationType

# Set up logging
logger = logging.getLogger(__name__)


class ProgressState(str, enum.Enum):
    """States for a progress tracker."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProgressError(Exception):
    """Base exception for progress tracking errors."""
    pass


class ProgressTracker:
    """
    Base progress tracker class for monitoring long-running operations.
    
    This class provides functionality for tracking the progress of operations,
    including updating status, recording messages, and calculating completion percentage.
    It also supports saving progress to a cache for retrieval by other components.
    
    Attributes:
        operation_id: Unique identifier for the operation
        operation_type: Type of operation being tracked
        total_steps: Total number of steps in the operation
        current_step: Current step number (0-based)
        status: Current status of the operation
        message: Current progress message
        start_time: Start time of the operation
        end_time: End time of the operation (if completed)
        steps: List of progress steps
        metadata: Additional metadata about the operation
    """
    
    def __init__(
        self,
        operation_id: str,
        operation_type: Union[OperationType, str] = OperationType.GENERAL,
        total_steps: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
        on_update: Optional[Callable[[ProgressDetails], None]] = None,
        cache_manager = None
    ):
        """
        Initialize the progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation being tracked
            total_steps: Total number of steps in the operation
            metadata: Additional metadata about the operation
            on_update: Callback function to call when progress is updated
            cache_manager: Cache manager to use for saving progress
        """
        self.operation_id = operation_id
        
        # Convert string to enum if needed
        if isinstance(operation_type, str):
            try:
                self.operation_type = OperationType(operation_type)
            except ValueError:
                logger.warning(f"Unknown operation type: {operation_type}, using CUSTOM")
                self.operation_type = OperationType.CUSTOM
        else:
            self.operation_type = operation_type
            
        self.total_steps = max(1, total_steps)  # Ensure at least 1 step
        self.current_step = 0
        self.status = ProgressState.PENDING
        self.message = ""
        self.start_time = time.time()
        self.end_time = None
        self.steps = []
        self.metadata = metadata or {}
        self.on_update = on_update
        self.cache_manager = cache_manager
        
        # For calculating estimated time remaining
        self._step_times = []
        
        # Add initial step
        self._add_step(0, "Operation initialized")
        
        logger.debug(f"Progress tracker initialized for operation {operation_id}")
    
    def update(self, step: int, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the progress tracker with a new step and message.
        
        Args:
            step: Current step number (0-based)
            message: Progress message
            details: Additional details about this update
        """
        # Validate step
        step = max(0, min(step, self.total_steps))
        
        # Update state
        if self.status == ProgressState.PENDING and step > 0:
            self.status = ProgressState.RUNNING
        
        # Update step and message
        self.current_step = step
        self.message = message
        
        # Add step to history
        self._add_step(step, message, details)
        
        # Check if completed
        if self.current_step >= self.total_steps:
            self.complete(message)
        
        # Log update
        logger.debug(f"Progress update for {self.operation_id}: {step}/{self.total_steps} - {message}")
        
        # Call update callback if provided
        if self.on_update:
            try:
                self.on_update(self.get_progress_details())
            except Exception as e:
                logger.error(f"Error in progress update callback: {str(e)}")
        
        # Save progress to cache if available
        if self.cache_manager:
            asyncio.create_task(self.save_progress())
    
    def complete(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the operation as completed.
        
        Args:
            message: Completion message
            details: Additional details about completion
        """
        self.current_step = self.total_steps
        self.status = ProgressState.COMPLETED
        self.message = message
        self.end_time = time.time()
        
        # Add final step
        self._add_step(self.total_steps, message, details)
        
        logger.info(f"Operation {self.operation_id} completed: {message}")
        
        # Call update callback if provided
        if self.on_update:
            try:
                self.on_update(self.get_progress_details())
            except Exception as e:
                logger.error(f"Error in progress update callback: {str(e)}")
        
        # Save progress to cache if available
        if self.cache_manager:
            asyncio.create_task(self.save_progress())
    
    def fail(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the operation as failed.
        
        Args:
            message: Failure message
            details: Additional details about the failure
        """
        self.status = ProgressState.FAILED
        self.message = message
        self.end_time = time.time()
        
        # Add failure step
        self._add_step(self.current_step, f"FAILED: {message}", details)
        
        logger.error(f"Operation {self.operation_id} failed: {message}")
        
        # Call update callback if provided
        if self.on_update:
            try:
                self.on_update(self.get_progress_details())
            except Exception as e:
                logger.error(f"Error in progress update callback: {str(e)}")
        
        # Save progress to cache if available
        if self.cache_manager:
            asyncio.create_task(self.save_progress())
    
    def cancel(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Mark the operation as cancelled.
        
        Args:
            message: Cancellation message
            details: Additional details about the cancellation
        """
        self.status = ProgressState.CANCELLED
        self.message = message
        self.end_time = time.time()
        
        # Add cancellation step
        self._add_step(self.current_step, f"CANCELLED: {message}", details)
        
        logger.info(f"Operation {self.operation_id} cancelled: {message}")
        
        # Call update callback if provided
        if self.on_update:
            try:
                self.on_update(self.get_progress_details())
            except Exception as e:
                logger.error(f"Error in progress update callback: {str(e)}")
        
        # Save progress to cache if available
        if self.cache_manager:
            asyncio.create_task(self.save_progress())
    
    def get_progress_details(self) -> ProgressDetails:
        """
        Get the progress details as a ProgressDetails object.
        
        Returns:
            ProgressDetails object containing progress details
        """
        now = time.time()
        elapsed = now - self.start_time
        
        # Calculate estimated time remaining
        estimated_remaining = None
        if self.current_step > 0 and self.current_step < self.total_steps:
            if len(self._step_times) >= 2:
                # Calculate average time per step based on recent steps
                recent_steps = min(10, len(self._step_times))
                avg_time_per_step = sum(self._step_times[-recent_steps:]) / recent_steps
                
                # Estimate remaining time
                remaining_steps = self.total_steps - self.current_step
                estimated_remaining = avg_time_per_step * remaining_steps
        
        return ProgressDetails(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            total_steps=self.total_steps,
            current_step=self.current_step,
            status=self.status,
            message=self.message,
            percent_complete=self.get_percent_complete(),
            start_time=datetime.fromtimestamp(self.start_time),
            end_time=datetime.fromtimestamp(self.end_time) if self.end_time else None,
            elapsed_time=elapsed,
            estimated_time_remaining=estimated_remaining,
            steps=[
                ProgressStep(
                    step_number=step["step"],
                    message=step["message"],
                    timestamp=datetime.fromtimestamp(step["timestamp"]),
                    details=step.get("details")
                )
                for step in self.steps
            ],
            metadata=self.metadata
        )
    
    def get_percent_complete(self) -> float:
        """
        Calculate the percentage of completion.
        
        Returns:
            Percentage of completion (0-100)
        """
        if self.total_steps <= 0:
            return 100.0 if self.status == ProgressState.COMPLETED else 0.0
        
        return min(100.0, (self.current_step / self.total_steps) * 100.0)
    
    async def save_progress(self) -> None:
        """
        Save the progress details to the cache.
        
        This allows other components to retrieve the progress information.
        """
        if not self.cache_manager:
            return
        
        progress_key = f"progress:{self.operation_id}"
        try:
            await self.cache_manager.set(
                progress_key,
                self.get_progress_details().dict(),
                ttl=3600,  # 1 hour TTL
                data_type="progress"
            )
            logger.debug(f"Saved progress for {self.operation_id}")
        except Exception as e:
            logger.error(f"Failed to save progress for {self.operation_id}: {str(e)}")
    
    @classmethod
    async def get_progress(cls, operation_id: str, cache_manager) -> Optional[ProgressDetails]:
        """
        Get the progress details for an operation from the cache.
        
        Args:
            operation_id: Operation ID
            cache_manager: Cache manager to use for retrieving progress
            
        Returns:
            ProgressDetails object or None if not found
        """
        if not cache_manager:
            return None
            
        progress_key = f"progress:{operation_id}"
        try:
            progress_data = await cache_manager.get(
                progress_key,
                data_type="progress"
            )
            
            if progress_data:
                return ProgressDetails(**progress_data)
            
            return None
        except Exception as e:
            logger.error(f"Failed to get progress for {operation_id}: {str(e)}")
            return None
    
    def _add_step(self, step: int, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a step to the progress history.
        
        Args:
            step: Step number
            message: Step message
            details: Additional details about this step
        """
        now = time.time()
        
        # Calculate time since last step
        if self.steps:
            last_step_time = self.steps[-1]["timestamp"]
            step_duration = now - last_step_time
            self._step_times.append(step_duration)
        
        # Add step to history
        self.steps.append({
            "step": step,
            "message": message,
            "timestamp": now,
            "details": details
        })
        
        # Limit history size
        if len(self.steps) > 100:
            self.steps = self.steps[-100:]
