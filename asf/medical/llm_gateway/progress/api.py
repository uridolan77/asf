"""
API for progress tracking in the LLM Gateway.

This module provides API endpoints for accessing progress information,
allowing clients to monitor the progress of long-running operations.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends

from .registry import get_progress_registry
from .models import ProgressDetails, OperationType

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/progress", tags=["progress"])


# Dependency for getting the progress registry
def get_registry():
    """Get the progress registry."""
    return get_progress_registry()


@router.get("/operations", response_model=List[str])
async def list_operations(
    registry=Depends(get_registry),
    status: Optional[str] = None,
    operation_type: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000)
):
    """
    List all operation IDs.
    
    Args:
        status: Filter by status
        operation_type: Filter by operation type
        limit: Maximum number of operations to return
        
    Returns:
        List of operation IDs
    """
    trackers = registry.get_all_trackers()
    
    # Apply filters
    if status:
        trackers = [t for t in trackers if t.status == status]
    
    if operation_type:
        trackers = [t for t in trackers if t.operation_type == operation_type]
    
    # Sort by start time (newest first)
    trackers.sort(key=lambda t: t.start_time, reverse=True)
    
    # Apply limit
    trackers = trackers[:limit]
    
    return [t.operation_id for t in trackers]


@router.get("/operations/{operation_id}", response_model=ProgressDetails)
async def get_operation_progress(operation_id: str, registry=Depends(get_registry)):
    """
    Get progress details for a specific operation.
    
    Args:
        operation_id: Operation ID
        
    Returns:
        Progress details for the operation
    """
    tracker = registry.get_tracker(operation_id)
    
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
    
    return tracker.get_progress_details()


@router.get("/active", response_model=List[ProgressDetails])
async def list_active_operations(
    registry=Depends(get_registry),
    operation_type: Optional[str] = None
):
    """
    List all active operations.
    
    Args:
        operation_type: Filter by operation type
        
    Returns:
        List of progress details for active operations
    """
    trackers = registry.get_active_trackers()
    
    # Apply filters
    if operation_type:
        trackers = [t for t in trackers if t.operation_type == operation_type]
    
    # Sort by start time (newest first)
    trackers.sort(key=lambda t: t.start_time, reverse=True)
    
    return [t.get_progress_details() for t in trackers]


@router.get("/summary", response_model=Dict[str, Any])
async def get_progress_summary(registry=Depends(get_registry)):
    """
    Get a summary of all operations.
    
    Returns:
        Summary of all operations
    """
    trackers = registry.get_all_trackers()
    
    # Count by status
    status_counts = {}
    for tracker in trackers:
        status = tracker.status
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Count by operation type
    type_counts = {}
    for tracker in trackers:
        op_type = tracker.operation_type
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
    
    # Calculate overall statistics
    total = len(trackers)
    active = len([t for t in trackers if t.status not in ("completed", "failed", "cancelled")])
    completed = len([t for t in trackers if t.status == "completed"])
    failed = len([t for t in trackers if t.status == "failed"])
    
    return {
        "total": total,
        "active": active,
        "completed": completed,
        "failed": failed,
        "by_status": status_counts,
        "by_type": type_counts
    }


@router.delete("/operations/{operation_id}")
async def delete_operation(operation_id: str, registry=Depends(get_registry)):
    """
    Delete a specific operation from the registry.
    
    Args:
        operation_id: Operation ID
        
    Returns:
        Success message
    """
    tracker = registry.get_tracker(operation_id)
    
    if not tracker:
        raise HTTPException(status_code=404, detail=f"Operation {operation_id} not found")
    
    registry.unregister(operation_id)
    
    return {"message": f"Operation {operation_id} deleted"}


@router.post("/cleanup")
async def cleanup_operations(
    max_age_seconds: int = Query(3600, ge=60, le=86400),
    registry=Depends(get_registry)
):
    """
    Clean up old completed operations.
    
    Args:
        max_age_seconds: Maximum age in seconds for completed operations
        
    Returns:
        Number of operations removed
    """
    removed = registry.cleanup(max_age_seconds)
    
    return {"removed": removed}
