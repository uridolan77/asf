"""
Progress tracking component for the Enhanced LLM Service.

This module provides progress tracking functionality for the Enhanced LLM Service,
allowing for monitoring and reporting on long-running operations.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from asf.medical.llm_gateway.progress.tracker import ProgressTracker

logger = logging.getLogger(__name__)

class ProgressTrackingComponent:
    """
    Progress tracking component for the Enhanced LLM Service.
    
    This class provides progress tracking functionality for the Enhanced LLM Service,
    allowing for monitoring and reporting on long-running operations.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the progress tracking component.
        
        Args:
            enabled: Whether progress tracking is enabled
        """
        self.enabled = enabled
        
        # Store progress trackers
        self._progress_trackers: Dict[str, Dict[str, Any]] = {}
    
    def create_progress_tracker(self,
                               operation_id: str,
                               total_steps: int = 100,
                               operation_type: str = "llm_request") -> Any:
        """
        Create a progress tracker for a long-running operation.
        
        Args:
            operation_id: Unique identifier for the operation
            total_steps: Total number of steps in the operation
            operation_type: Type of operation
            
        Returns:
            Progress tracker object
        """
        if not self.enabled:
            return None
        
        try:
            # Create a new progress tracker
            tracker = {
                "operation_id": operation_id,
                "operation_type": operation_type,
                "total_steps": total_steps,
                "current_step": 0,
                "status": "created",
                "message": "Operation created",
                "start_time": time.time(),
                "update_time": time.time(),
                "end_time": None,
                "updates": []
            }
            
            # Store the tracker
            self._progress_trackers[operation_id] = tracker
            
            # Log tracker creation
            logger.debug(f"Created progress tracker for operation {operation_id} of type {operation_type}")
            
            return tracker
        except Exception as e:
            logger.error(f"Error creating progress tracker: {str(e)}")
            return None
    
    def update_progress(self,
                       tracker: Any,
                       step: int,
                       message: str,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the progress of an operation.
        
        Args:
            tracker: Progress tracker object
            step: Current step number
            message: Progress message
            details: Optional details about this update
        """
        if not self.enabled or tracker is None:
            return
        
        try:
            # Ensure tracker is a dictionary
            if not isinstance(tracker, dict) or "operation_id" not in tracker:
                logger.warning("Invalid progress tracker provided")
                return
            
            # Get operation ID
            operation_id = tracker["operation_id"]
            
            # Ensure tracker exists
            if operation_id not in self._progress_trackers:
                logger.warning(f"Progress tracker for operation {operation_id} not found")
                return
            
            # Get the stored tracker
            stored_tracker = self._progress_trackers[operation_id]
            
            # Update tracker
            stored_tracker["current_step"] = step
            stored_tracker["message"] = message
            stored_tracker["update_time"] = time.time()
            
            # Calculate progress percentage
            progress_pct = min(100, int(100 * step / stored_tracker["total_steps"]))
            
            # Check if operation is complete
            if progress_pct >= 100:
                stored_tracker["status"] = "completed"
                stored_tracker["end_time"] = time.time()
            else:
                stored_tracker["status"] = "in_progress"
            
            # Add update to history
            update = {
                "step": step,
                "message": message,
                "time": time.time(),
                "progress_pct": progress_pct
            }
            
            if details:
                update["details"] = details
            
            stored_tracker["updates"].append(update)
            
            # Log progress update
            logger.debug(f"Updated progress for operation {operation_id}: {progress_pct}% - {message}")
        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}")
    
    def get_progress(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the progress of an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            
        Returns:
            Dictionary containing progress information or None if not found
        """
        if not self.enabled or operation_id not in self._progress_trackers:
            return None
        
        return self._progress_trackers[operation_id]
    
    def get_all_progress(self, active_only: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get the progress of all operations.
        
        Args:
            active_only: Whether to only return active operations
            
        Returns:
            Dictionary mapping operation IDs to progress information
        """
        if not self.enabled:
            return {}
        
        if not active_only:
            return self._progress_trackers
        
        # Filter for active operations
        return {
            op_id: tracker
            for op_id, tracker in self._progress_trackers.items()
            if tracker.get("status") != "completed"
        }
    
    def cancel_operation(self, operation_id: str) -> bool:
        """
        Cancel an operation.
        
        Args:
            operation_id: Unique identifier for the operation
            
        Returns:
            True if the operation was cancelled, False otherwise
        """
        if not self.enabled or operation_id not in self._progress_trackers:
            return False
        
        try:
            # Get the tracker
            tracker = self._progress_trackers[operation_id]
            
            # Update tracker
            tracker["status"] = "cancelled"
            tracker["end_time"] = time.time()
            tracker["message"] = "Operation cancelled"
            
            # Add update to history
            update = {
                "step": tracker["current_step"],
                "message": "Operation cancelled",
                "time": time.time(),
                "progress_pct": min(100, int(100 * tracker["current_step"] / tracker["total_steps"]))
            }
            
            tracker["updates"].append(update)
            
            # Log cancellation
            logger.debug(f"Cancelled operation {operation_id}")
            
            return True
        except Exception as e:
            logger.error(f"Error cancelling operation: {str(e)}")
            return False
    
    def cleanup_completed_operations(self, max_age_seconds: float = 3600.0) -> int:
        """
        Clean up completed operations.
        
        Args:
            max_age_seconds: Maximum age of completed operations to keep
            
        Returns:
            Number of operations cleaned up
        """
        if not self.enabled:
            return 0
        
        try:
            # Get current time
            current_time = time.time()
            
            # Find operations to clean up
            to_remove = []
            for op_id, tracker in self._progress_trackers.items():
                if tracker.get("status") in ("completed", "cancelled"):
                    end_time = tracker.get("end_time")
                    if end_time and current_time - end_time > max_age_seconds:
                        to_remove.append(op_id)
            
            # Remove operations
            for op_id in to_remove:
                del self._progress_trackers[op_id]
            
            # Log cleanup
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} completed operations")
            
            return len(to_remove)
        except Exception as e:
            logger.error(f"Error cleaning up completed operations: {str(e)}")
            return 0
