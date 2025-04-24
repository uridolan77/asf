"""
API endpoints for progress tracking in the Conexus LLM Gateway.

This module provides REST API endpoints for tracking the progress
of long-running LLM operations.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Path, Query, BackgroundTasks, status
from pydantic import BaseModel, Field

from asf.conexus.llm_gateway.progress.models import (
    ProgressTracker,
    ProgressUpdate,
    ProgressStatus,
    ProgressMetadata
)
from asf.conexus.llm_gateway.progress.service import get_progress_service, ProgressTrackingService
from asf.conexus.llm_gateway.api.auth import get_user_id

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/progress",
    tags=["progress"],
    responses={404: {"description": "Not found"}}
)


# Request/Response models for the API
class CreateTrackerRequest(BaseModel):
    """Request model for creating a progress tracker."""
    task_id: str = Field(..., description="Unique ID of the task to track")
    task_type: str = Field(..., description="Type of task being tracked (e.g., generation, fine-tuning)")
    name: Optional[str] = Field(None, description="User-friendly name for the task")
    description: Optional[str] = Field(None, description="Description of the task")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the tracker")
    retention_hours: Optional[int] = Field(None, description="How long to keep this tracker after completion")


class TrackerResponse(BaseModel):
    """Response model for a progress tracker."""
    id: str = Field(..., description="Unique ID of the tracker")
    task_id: str = Field(..., description="ID of the task being tracked")
    task_type: str = Field(..., description="Type of task")
    user_id: Optional[str] = Field(None, description="ID of the user who initiated the task")
    name: Optional[str] = Field(None, description="User-friendly name for the task")
    description: Optional[str] = Field(None, description="Description of the task")
    status: str = Field(..., description="Current status of the task")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")
    created_at: str = Field(..., description="ISO timestamp when tracker was created")
    started_at: Optional[str] = Field(None, description="ISO timestamp when task started")
    completed_at: Optional[str] = Field(None, description="ISO timestamp when task completed")
    result_url: Optional[str] = Field(None, description="URL to access results when complete")
    has_result: bool = Field(False, description="Whether the tracker has result data")
    has_error: bool = Field(False, description="Whether the tracker has error data")
    latest_message: Optional[str] = Field(None, description="Latest progress message")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")
    info: Dict[str, Any] = Field({}, description="Progress tracking information")


class UpdateRequest(BaseModel):
    """Request model for updating a progress tracker."""
    message: str = Field(..., description="User-friendly progress message")
    status: Optional[str] = Field(None, description="New status (if changing)")
    detail: Optional[str] = Field(None, description="Technical details")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update")


class CompletionRequest(BaseModel):
    """Request model for completing a progress tracker."""
    message: str = Field("Task completed", description="User-friendly completion message")
    detail: Optional[str] = Field(None, description="Technical details")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update")
    result: Optional[Dict[str, Any]] = Field(None, description="Result data")
    result_url: Optional[str] = Field(None, description="URL to access results")


class FailureRequest(BaseModel):
    """Request model for marking a tracker as failed."""
    message: str = Field("Task failed", description="User-friendly failure message")
    detail: Optional[str] = Field(None, description="Technical details")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update")
    error: Optional[Dict[str, Any]] = Field(None, description="Error details")


class CancellationRequest(BaseModel):
    """Request model for cancelling a tracker."""
    message: str = Field("Task cancelled", description="User-friendly cancellation message")
    detail: Optional[str] = Field(None, description="Technical details")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata to update")


def _convert_tracker_to_response(tracker: ProgressTracker) -> TrackerResponse:
    """Convert a progress tracker to a response model."""
    # Get latest message if available
    latest_update = tracker.get_latest_update()
    latest_message = latest_update.message if latest_update else None
    
    return TrackerResponse(
        id=tracker.id,
        task_id=tracker.task_id,
        task_type=tracker.task_type,
        user_id=tracker.user_id,
        name=tracker.name,
        description=tracker.description,
        status=tracker.status.value,
        progress_percentage=tracker.get_progress_percentage(),
        created_at=tracker.created_at.isoformat(),
        started_at=tracker.started_at.isoformat() if tracker.started_at else None,
        completed_at=tracker.completed_at.isoformat() if tracker.completed_at else None,
        result_url=tracker.result_url,
        has_result=tracker.result is not None,
        has_error=tracker.error is not None,
        latest_message=latest_message,
        metadata={
            "total_steps": tracker.metadata.total_steps,
            "current_step": tracker.metadata.current_step,
            "step_name": tracker.metadata.step_name,
            "completion_percentage": tracker.metadata.completion_percentage,
            "estimated_completion_time": tracker.metadata.estimated_completion_time.isoformat() 
                                        if tracker.metadata.estimated_completion_time else None,
            "model_id": tracker.metadata.model_id,
            "provider_id": tracker.metadata.provider_id,
            "token_count": tracker.metadata.token_count,
            "request_id": tracker.metadata.request_id,
            "conversation_id": tracker.metadata.conversation_id,
            **tracker.metadata.additional
        },
        info={
            "update_count": tracker.info.update_count,
            "last_update_time": tracker.info.last_update_time.isoformat() if tracker.info.last_update_time else None,
            "update_interval_seconds": tracker.info.update_interval_seconds,
            "time_elapsed_seconds": tracker.info.time_elapsed_seconds,
            "time_remaining_seconds": tracker.info.time_remaining_seconds
        }
    )


@router.post("", response_model=TrackerResponse, status_code=status.HTTP_201_CREATED)
async def create_tracker(
    request: CreateTrackerRequest,
    user_id: Optional[str] = Depends(get_user_id),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> TrackerResponse:
    """
    Create a new progress tracker.
    
    Creates a new tracker to monitor the progress of a long-running task.
    """
    try:
        tracker = await progress_service.create_tracker(
            task_id=request.task_id,
            task_type=request.task_type,
            name=request.name,
            description=request.description,
            user_id=user_id,
            metadata=request.metadata,
            retention_hours=request.retention_hours
        )
        
        return _convert_tracker_to_response(tracker)
    except Exception as e:
        logger.error(f"Error creating progress tracker: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tracker: {str(e)}"
        )


@router.get("/{tracker_id}", response_model=TrackerResponse)
async def get_tracker(
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> TrackerResponse:
    """
    Get a progress tracker by ID.
    
    Retrieves the current state of a progress tracker.
    """
    tracker = await progress_service.get_tracker(tracker_id)
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    return _convert_tracker_to_response(tracker)


@router.get("", response_model=List[TrackerResponse])
async def get_trackers_for_task(
    task_id: Optional[str] = Query(None, description="ID of the task"),
    user_id: Optional[str] = Depends(get_user_id),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> List[TrackerResponse]:
    """
    List progress trackers.
    
    Retrieves trackers filtered by task ID or for the current user.
    """
    if task_id:
        # Get trackers for specific task
        trackers = await progress_service.get_trackers_for_task(task_id)
    elif user_id:
        # Get trackers for current user
        trackers = await progress_service.get_trackers_for_user(user_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either task_id or user authentication is required"
        )
        
    return [_convert_tracker_to_response(tracker) for tracker in trackers]


@router.get("/{tracker_id}/result", response_model=Dict[str, Any])
async def get_tracker_result(
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> Dict[str, Any]:
    """
    Get the result data for a completed tracker.
    
    Returns the result data stored in the tracker.
    """
    tracker = await progress_service.get_tracker(tracker_id)
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    if tracker.status != ProgressStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tracker is not completed (current status: {tracker.status})"
        )
        
    if tracker.result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No result data available for tracker {tracker_id}"
        )
        
    return tracker.result


@router.get("/{tracker_id}/error", response_model=Dict[str, Any])
async def get_tracker_error(
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> Dict[str, Any]:
    """
    Get the error details for a failed tracker.
    
    Returns the error details stored in the tracker.
    """
    tracker = await progress_service.get_tracker(tracker_id)
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    if tracker.status != ProgressStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tracker is not failed (current status: {tracker.status})"
        )
        
    if tracker.error is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No error details available for tracker {tracker_id}"
        )
        
    return tracker.error


@router.put("/{tracker_id}", response_model=TrackerResponse)
async def update_tracker(
    request: UpdateRequest,
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> TrackerResponse:
    """
    Update a progress tracker.
    
    Updates the status and metadata of an existing tracker.
    """
    # Convert status string to enum if provided
    status = None
    if request.status:
        try:
            status = ProgressStatus(request.status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {request.status}"
            )
    
    # Update the tracker
    update = await progress_service.update_tracker(
        tracker_id=tracker_id,
        message=request.message,
        status=status,
        detail=request.detail,
        metadata=request.metadata
    )
    
    if update is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    # Get the updated tracker
    tracker = await progress_service.get_tracker(tracker_id)
    return _convert_tracker_to_response(tracker)


@router.put("/{tracker_id}/complete", response_model=TrackerResponse)
async def complete_tracker(
    request: CompletionRequest,
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> TrackerResponse:
    """
    Mark a tracker as completed.
    
    Sets the tracker status to completed and optionally stores result data.
    """
    tracker = await progress_service.complete_tracker(
        tracker_id=tracker_id,
        message=request.message,
        detail=request.detail,
        metadata=request.metadata,
        result=request.result,
        result_url=request.result_url
    )
    
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    return _convert_tracker_to_response(tracker)


@router.put("/{tracker_id}/fail", response_model=TrackerResponse)
async def fail_tracker(
    request: FailureRequest,
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> TrackerResponse:
    """
    Mark a tracker as failed.
    
    Sets the tracker status to failed and optionally stores error details.
    """
    tracker = await progress_service.fail_tracker(
        tracker_id=tracker_id,
        message=request.message,
        detail=request.detail,
        metadata=request.metadata,
        error=request.error
    )
    
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    return _convert_tracker_to_response(tracker)


@router.put("/{tracker_id}/cancel", response_model=TrackerResponse)
async def cancel_tracker(
    request: CancellationRequest,
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> TrackerResponse:
    """
    Mark a tracker as cancelled.
    
    Sets the tracker status to cancelled.
    """
    tracker = await progress_service.cancel_tracker(
        tracker_id=tracker_id,
        message=request.message,
        detail=request.detail,
        metadata=request.metadata
    )
    
    if tracker is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    return _convert_tracker_to_response(tracker)


@router.delete("/{tracker_id}")
async def delete_tracker(
    tracker_id: str = Path(..., description="ID of the tracker"),
    progress_service: ProgressTrackingService = Depends(get_progress_service)
) -> Dict[str, bool]:
    """
    Delete a progress tracker.
    
    Permanently removes the tracker.
    """
    success = await progress_service.delete_tracker(tracker_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracker with ID {tracker_id} not found"
        )
        
    return {"deleted": True}