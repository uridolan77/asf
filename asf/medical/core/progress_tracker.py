"""
Progress tracking module for the Medical Research Synthesizer.

This module provides a base class for tracking the progress of long-running operations,
such as ML model training, knowledge base creation, and data exports.

Classes:
    ProgressTracker: Base class for tracking operation progress.
"""

import time
import logging
from typing import Dict, Any, Optional
from .enhanced_cache import enhanced_cache_manager

logger = logging.getLogger(__name__)

class ProgressTracker:
    """
    Base progress tracker class for monitoring long-running operations.
    
    This class provides functionality for tracking the progress of operations,
    including updating status, recording messages, and calculating completion percentage.
    It also supports saving progress to a cache for retrieval by other components.
    """
    
    def __init__(self, operation_id: str, total_steps: int = 100):
        """
        Initialize the progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            total_steps: Total number of steps in the operation
        """
        self.operation_id = operation_id
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "pending"
        self.message = ""
        self.start_time = time.time()
    
    def update(self, step: int, message: str):
        """
        Update the progress tracker with a new step and message.
        
        Args:
            step: Current step number
            message: Progress message
        """
        self.current_step = step
        self.message = message
        if self.current_step >= self.total_steps:
            self.status = "completed"
        logger.debug(f"Progress update for {self.operation_id}: {step}/{self.total_steps} - {message}")
    
    def complete(self, message: str):
        """
        Mark the operation as completed.
        
        Args:
            message: Completion message
        """
        self.current_step = self.total_steps
        self.status = "completed"
        self.message = message
        logger.info(f"Operation {self.operation_id} completed: {message}")
    
    def fail(self, message: str):
        """
        Mark the operation as failed.
        
        Args:
            message: Failure message
        """
        self.status = "failed"
        self.message = message
        logger.error(f"Operation {self.operation_id} failed: {message}")
    
    def get_progress_details(self) -> Dict[str, Any]:
        """
        Get the progress details as a dictionary.
        
        Returns:
            Dictionary containing progress details
        """
        return {
            "operation_id": self.operation_id,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "status": self.status,
            "message": self.message,
            "percent_complete": self.get_percent_complete(),
            "elapsed_time": time.time() - self.start_time
        }
    
    def get_percent_complete(self) -> float:
        """
        Calculate the percentage of completion.
        
        Returns:
            Percentage of completion (0-100)
        """
        if self.total_steps <= 0:
            return 100.0 if self.status == "completed" else 0.0
        return min(100.0, (self.current_step / self.total_steps) * 100.0)
    
    async def save_progress(self):
        """
        Save the progress details to the cache.
        
        This allows other components to retrieve the progress information.
        """
        progress_key = f"progress:{self.operation_id}"
        try:
            await enhanced_cache_manager.set(
                progress_key,
                self.get_progress_details(),
                ttl=3600,  # 1 hour TTL
                data_type="progress"
            )
            logger.debug(f"Saved progress for {self.operation_id}")
        except Exception as e:
            logger.error(f"Failed to save progress for {self.operation_id}: {str(e)}")
    
    @classmethod
    async def get_progress(cls, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the progress details for an operation from the cache.
        
        Args:
            operation_id: Operation ID
            
        Returns:
            Progress details or None if not found
        """
        progress_key = f"progress:{operation_id}"
        try:
            return await enhanced_cache_manager.get(
                progress_key,
                data_type="progress"
            )
        except Exception as e:
            logger.error(f"Failed to get progress for {operation_id}: {str(e)}")
            return None
