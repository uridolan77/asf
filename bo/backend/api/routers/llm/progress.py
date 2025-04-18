"""
Progress tracking router for LLM Gateway.

This module provides API endpoints for accessing progress information
for LLM operations, allowing clients to monitor the progress of
long-running operations.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Query, Depends, status

from ...auth import get_current_user, User
from ...models.llm import (
    ProgressDetailsResponse,
    ProgressSummaryResponse,
    ProgressOperationResponse
)

# Import progress tracking components
from asf.medical.llm_gateway.progress import (
    get_progress_registry,
    ProgressDetails,
    OperationType
)

router = APIRouter(prefix="/progress", tags=["llm-progress"])

logger = logging.getLogger(__name__)


# Check if progress tracking is available
try:
    PROGRESS_TRACKING_AVAILABLE = True
    progress_registry = get_progress_registry()
except ImportError:
    PROGRESS_TRACKING_AVAILABLE = False
    progress_registry = None
    logger.warning("Progress tracking is not available. Some endpoints will be disabled.")


# Dependency for getting the progress registry
def get_registry():
    """Get the progress registry."""
    if not PROGRESS_TRACKING_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Progress tracking is not available. Please check your installation."
        )
    return get_progress_registry()


@router.get("/", response_model=Dict[str, Any])
async def progress_root(current_user: User = Depends(get_current_user)):
    """
    Root endpoint for progress tracking API.
    
    Returns information about available progress tracking endpoints.
    """
    if not PROGRESS_TRACKING_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "Progress tracking is not available. Please check your installation."
        }
    
    return {
        "status": "ok",
        "message": "Progress tracking API is operational",
        "endpoints": [
            {
                "path": "/api/llm/progress/operations",
                "description": "List all operation IDs"
            },
            {
                "path": "/api/llm/progress/operations/{operation_id}",
                "description": "Get progress details for a specific operation"
            },
            {
                "path": "/api/llm/progress/active",
                "description": "List all active operations"
            },
            {
                "path": "/api/llm/progress/summary",
                "description": "Get a summary of all operations"
            }
        ]
    }


@router.get("/operations", response_model=List[ProgressOperationResponse])
async def list_operations(
    current_user: User = Depends(get_current_user),
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
        List of operation IDs with basic information
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
    
    # Convert to response model
    operations = []
    for tracker in trackers:
        progress = tracker.get_progress_details()
        operations.append(
            ProgressOperationResponse(
                operation_id=progress.operation_id,
                operation_type=progress.operation_type,
                status=progress.status,
                percent_complete=progress.percent_complete,
                message=progress.message,
                start_time=progress.start_time.isoformat(),
                end_time=progress.end_time.isoformat() if progress.end_time else None
            )
        )
    
    return operations


@router.get("/operations/{operation_id}", response_model=ProgressDetailsResponse)
async def get_operation_progress(
    operation_id: str,
    current_user: User = Depends(get_current_user),
    registry=Depends(get_registry)
):
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
    
    # Get progress details
    progress = tracker.get_progress_details()
    
    # Convert to response model
    return ProgressDetailsResponse(
        operation_id=progress.operation_id,
        operation_type=progress.operation_type,
        total_steps=progress.total_steps,
        current_step=progress.current_step,
        status=progress.status,
        message=progress.message,
        percent_complete=progress.percent_complete,
        start_time=progress.start_time.isoformat(),
        end_time=progress.end_time.isoformat() if progress.end_time else None,
        elapsed_time=progress.elapsed_time,
        estimated_time_remaining=progress.estimated_time_remaining,
        steps=[
            {
                "step_number": step.step_number,
                "message": step.message,
                "timestamp": step.timestamp.isoformat(),
                "details": step.details
            }
            for step in progress.steps
        ],
        metadata=progress.metadata
    )


@router.get("/active", response_model=List[ProgressOperationResponse])
async def list_active_operations(
    current_user: User = Depends(get_current_user),
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
    
    # Convert to response model
    operations = []
    for tracker in trackers:
        progress = tracker.get_progress_details()
        operations.append(
            ProgressOperationResponse(
                operation_id=progress.operation_id,
                operation_type=progress.operation_type,
                status=progress.status,
                percent_complete=progress.percent_complete,
                message=progress.message,
                start_time=progress.start_time.isoformat(),
                end_time=progress.end_time.isoformat() if progress.end_time else None
            )
        )
    
    return operations


@router.get("/summary", response_model=ProgressSummaryResponse)
async def get_progress_summary(
    current_user: User = Depends(get_current_user),
    registry=Depends(get_registry)
):
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
    
    return ProgressSummaryResponse(
        total=total,
        active=active,
        completed=completed,
        failed=failed,
        by_status=status_counts,
        by_type=type_counts
    )


@router.delete("/operations/{operation_id}")
async def delete_operation(
    operation_id: str,
    current_user: User = Depends(get_current_user),
    registry=Depends(get_registry)
):
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
    current_user: User = Depends(get_current_user),
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
