"""
Progress tracking system for the Conexus LLM Gateway.

This module provides functionality for tracking the progress of long-running LLM operations.
It allows clients to monitor the status of requests, get partial results, and receive
notifications when operations complete.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable

logger = logging.getLogger(__name__)


class ProgressStatus(str, Enum):
    """Status values for tracking progress of operations."""
    PENDING = "pending"        # Operation is waiting to start
    INITIALIZING = "initializing"  # Operation is starting up
    RUNNING = "running"        # Operation is actively running
    COMPLETED = "completed"    # Operation completed successfully
    FAILED = "failed"          # Operation failed with an error
    CANCELED = "canceled"      # Operation was canceled by user
    TIMEOUT = "timeout"        # Operation timed out
    UNKNOWN = "unknown"        # Status cannot be determined


class ProgressTracker:
    """
    Manages tracking progress for long-running operations.
    
    This class provides methods for creating, updating, and retrieving
    progress information for operations such as complex LLM requests,
    batch processing jobs, or model fine-tuning.
    """
    
    def __init__(self):
        """Initialize the progress tracker."""
        # Main storage for progress records
        self._progress_records: Dict[str, Dict[str, Any]] = {}
        
        # Track active/completed status
        self._active_operations: Set[str] = set()
        self._completed_operations: Set[str] = set()
        
        # Callback subscriptions
        self._callbacks: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        logger.info("Progress tracker initialized")
    
    async def create_operation(self, 
                               operation_type: str, 
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new tracked operation.
        
        Args:
            operation_type: Type of operation (e.g., "llm_request", "batch_processing")
            metadata: Additional metadata about the operation
            
        Returns:
            Operation ID (unique identifier)
        """
        operation_id = str(uuid.uuid4())
        created_time = datetime.utcnow().isoformat()
        
        progress_record = {
            "operation_id": operation_id,
            "operation_type": operation_type,
            "status": ProgressStatus.PENDING.value,
            "progress_percent": 0,
            "created_at": created_time,
            "updated_at": created_time,
            "metadata": metadata or {},
            "steps": [],
            "results": {},
            "error": None
        }
        
        async with self._lock:
            self._progress_records[operation_id] = progress_record
            self._active_operations.add(operation_id)
        
        logger.debug(f"Created operation {operation_id} of type {operation_type}")
        return operation_id
    
    async def update_operation(self, 
                               operation_id: str, 
                               status: Optional[Union[ProgressStatus, str]] = None,
                               progress_percent: Optional[float] = None,
                               message: Optional[str] = None,
                               result: Optional[Any] = None,
                               error: Optional[Dict[str, Any]] = None,
                               metadata_update: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the status of an operation.
        
        Args:
            operation_id: ID of the operation to update
            status: New status value
            progress_percent: Completion percentage (0-100)
            message: Status message
            result: Result data for the operation
            error: Error information if failed
            metadata_update: Additional metadata to update
            
        Returns:
            True if successful, False if operation not found
        """
        async with self._lock:
            if operation_id not in self._progress_records:
                logger.warning(f"Attempted to update unknown operation {operation_id}")
                return False
            
            record = self._progress_records[operation_id]
            record["updated_at"] = datetime.utcnow().isoformat()
            
            # Update status if provided
            if status is not None:
                status_value = status.value if isinstance(status, ProgressStatus) else status
                record["status"] = status_value
                
                # Track completion
                if status_value in (ProgressStatus.COMPLETED.value, 
                                   ProgressStatus.FAILED.value, 
                                   ProgressStatus.CANCELED.value,
                                   ProgressStatus.TIMEOUT.value):
                    if operation_id in self._active_operations:
                        self._active_operations.remove(operation_id)
                        self._completed_operations.add(operation_id)
            
            # Update progress percentage if provided
            if progress_percent is not None:
                # Ensure value is in valid range
                progress_percent = max(0, min(100, progress_percent))
                record["progress_percent"] = progress_percent
            
            # Add a step entry if message provided
            if message:
                step = {
                    "time": datetime.utcnow().isoformat(),
                    "message": message
                }
                record["steps"].append(step)
            
            # Update result if provided
            if result is not None:
                record["results"] = result
            
            # Update error if provided
            if error is not None:
                record["error"] = error
            
            # Update metadata if provided
            if metadata_update:
                record["metadata"].update(metadata_update)
        
        # Trigger any callbacks (outside lock to avoid deadlocks)
        await self._trigger_callbacks(operation_id, self._progress_records[operation_id])
        
        return True
    
    async def add_step(self, 
                       operation_id: str, 
                       message: str,
                       status: Optional[Union[ProgressStatus, str]] = None,
                       progress_percent: Optional[float] = None) -> bool:
        """
        Add a step to the operation's progress.
        
        Args:
            operation_id: ID of the operation
            message: Step description message
            status: Optional status update
            progress_percent: Optional progress percentage update
            
        Returns:
            True if successful, False if operation not found
        """
        return await self.update_operation(
            operation_id=operation_id,
            status=status,
            progress_percent=progress_percent,
            message=message
        )
    
    async def complete_operation(self, 
                                operation_id: str, 
                                result: Any,
                                message: Optional[str] = None) -> bool:
        """
        Mark an operation as successfully completed.
        
        Args:
            operation_id: ID of the operation
            result: Result data for the operation
            message: Optional completion message
            
        Returns:
            True if successful, False if operation not found
        """
        return await self.update_operation(
            operation_id=operation_id,
            status=ProgressStatus.COMPLETED,
            progress_percent=100.0,
            message=message or "Operation completed successfully",
            result=result
        )
    
    async def fail_operation(self, 
                            operation_id: str, 
                            error_message: str,
                            error_code: Optional[str] = None,
                            error_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark an operation as failed.
        
        Args:
            operation_id: ID of the operation
            error_message: Description of the error
            error_code: Error code identifier
            error_details: Additional error details
            
        Returns:
            True if successful, False if operation not found
        """
        error = {
            "message": error_message,
            "code": error_code or "unknown_error",
            "timestamp": datetime.utcnow().isoformat(),
            "details": error_details or {}
        }
        
        return await self.update_operation(
            operation_id=operation_id,
            status=ProgressStatus.FAILED,
            message=f"Operation failed: {error_message}",
            error=error
        )
    
    async def cancel_operation(self, operation_id: str) -> bool:
        """
        Mark an operation as canceled.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            True if successful, False if operation not found
        """
        return await self.update_operation(
            operation_id=operation_id,
            status=ProgressStatus.CANCELED,
            message="Operation canceled by user",
        )
    
    async def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of an operation.
        
        Args:
            operation_id: ID of the operation
            
        Returns:
            Status record or None if not found
        """
        async with self._lock:
            if operation_id not in self._progress_records:
                return None
            
            # Return a copy to avoid external modification
            return json.loads(json.dumps(self._progress_records[operation_id]))
    
    async def get_active_operations(self, 
                                   operation_type: Optional[str] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of active operations.
        
        Args:
            operation_type: Optional filter by operation type
            limit: Maximum number of results to return
            
        Returns:
            List of active operation status records
        """
        result = []
        
        async with self._lock:
            for op_id in self._active_operations:
                if len(result) >= limit:
                    break
                    
                record = self._progress_records[op_id]
                
                # Filter by type if specified
                if operation_type and record["operation_type"] != operation_type:
                    continue
                    
                # Return a copy
                result.append(json.loads(json.dumps(record)))
                
        return result
    
    async def get_recent_completed_operations(self,
                                             operation_type: Optional[str] = None,
                                             limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get a list of recently completed operations.
        
        Args:
            operation_type: Optional filter by operation type
            limit: Maximum number of results to return
            
        Returns:
            List of completed operation status records
        """
        result = []
        
        async with self._lock:
            # Get all completed records
            completed_records = [
                self._progress_records[op_id] for op_id in self._completed_operations
                if op_id in self._progress_records
            ]
            
            # Filter by type if needed
            if operation_type:
                completed_records = [
                    r for r in completed_records if r["operation_type"] == operation_type
                ]
                
            # Sort by updated_at timestamp (most recent first)
            completed_records.sort(key=lambda r: r["updated_at"], reverse=True)
            
            # Take up to the limit
            result = json.loads(json.dumps(completed_records[:limit]))
                
        return result
    
    async def subscribe_to_updates(self, 
                                  operation_id: str, 
                                  callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """
        Subscribe to updates for an operation.
        
        Args:
            operation_id: ID of the operation
            callback: Function to call when operation is updated
            
        Returns:
            True if subscription was successful, False otherwise
        """
        async with self._lock:
            if operation_id not in self._progress_records:
                logger.warning(f"Attempted to subscribe to unknown operation {operation_id}")
                return False
                
            if operation_id not in self._callbacks:
                self._callbacks[operation_id] = []
                
            self._callbacks[operation_id].append(callback)
            
        return True
    
    async def unsubscribe_from_updates(self,
                                      operation_id: str,
                                      callback: Callable[[str, Dict[str, Any]], None]) -> bool:
        """
        Unsubscribe from updates for an operation.
        
        Args:
            operation_id: ID of the operation
            callback: Callback function to remove
            
        Returns:
            True if unsubscription was successful, False otherwise
        """
        async with self._lock:
            if operation_id not in self._callbacks:
                return False
                
            try:
                self._callbacks[operation_id].remove(callback)
                
                # Clean up empty callback lists
                if not self._callbacks[operation_id]:
                    del self._callbacks[operation_id]
                    
                return True
            except ValueError:
                # Callback wasn't in the list
                return False
    
    async def cleanup_old_operations(self, max_age_minutes: int = 60 * 24) -> int:
        """
        Remove old completed operations to free up memory.
        
        Args:
            max_age_minutes: Maximum age in minutes for completed operations
            
        Returns:
            Number of operations removed
        """
        now = datetime.utcnow()
        removed_count = 0
        
        async with self._lock:
            to_remove = []
            
            for op_id in self._completed_operations:
                if op_id not in self._progress_records:
                    to_remove.append(op_id)
                    continue
                    
                record = self._progress_records[op_id]
                updated_at = datetime.fromisoformat(record["updated_at"])
                
                # Calculate age in minutes
                age_minutes = (now - updated_at).total_seconds() / 60
                
                if age_minutes > max_age_minutes:
                    to_remove.append(op_id)
            
            # Remove from sets and dictionaries
            for op_id in to_remove:
                if op_id in self._progress_records:
                    del self._progress_records[op_id]
                    removed_count += 1
                    
                self._completed_operations.remove(op_id)
                
                if op_id in self._callbacks:
                    del self._callbacks[op_id]
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} old operations")
            
        return removed_count
    
    async def _trigger_callbacks(self, operation_id: str, status: Dict[str, Any]) -> None:
        """
        Trigger callbacks for an operation update.
        
        Args:
            operation_id: ID of the operation
            status: Current operation status
        """
        callbacks = []
        
        async with self._lock:
            if operation_id in self._callbacks:
                # Make a copy of the callbacks list to avoid issues if callbacks modify it
                callbacks = list(self._callbacks[operation_id])
        
        # Execute callbacks outside the lock
        for callback in callbacks:
            try:
                callback(operation_id, status)
            except Exception as e:
                logger.error(f"Error in operation callback: {e}")


# Singleton instance for global access
_progress_tracker = None


def get_progress_tracker() -> ProgressTracker:
    """
    Get the global progress tracker instance.
    
    Returns:
        The progress tracker singleton
    """
    global _progress_tracker
    if _progress_tracker is None:
        _progress_tracker = ProgressTracker()
    return _progress_tracker