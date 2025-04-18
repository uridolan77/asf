"""API endpoints for messaging tasks.

This module provides API endpoints for publishing tasks to the message broker,
including analysis tasks and export tasks.
"""
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from ...core.config import settings
from ...core.logging_config import get_logger
from ...core.messaging.producer import get_message_producer
from ...core.messaging.schemas import TaskType
from ..dependencies import get_current_user
from ...storage.models import User
logger = get_logger(__name__)
router = APIRouter(prefix="/v1/messaging", tags=["messaging"])
class SearchTaskRequest(BaseModel):
    """Request model for search tasks."""
    analysis_type: str = Field(..., description="Type of analysis to perform (contradictions, bias, trends)")
    study_ids: Optional[List[str]] = Field(None, description="List of study IDs for contradiction analysis")
    study_id: Optional[str] = Field(None, description="Study ID for bias analysis")
    topic: Optional[str] = Field(None, description="Topic for trend analysis")
    time_range: Optional[str] = Field(None, description="Time range for analysis")


class ExportTaskRequest(BaseModel):
    """Request model for export tasks."""
    export_type: str = Field(..., description="Type of export (results, analysis)")
    format: str = Field(..., description="Export format (json, csv, excel, pdf)")
    result_id: Optional[str] = Field(None, description="Result ID for results export")
    analysis_id: Optional[str] = Field(None, description="Analysis ID for analysis export")


class TaskResponse(BaseModel):
    """Response model for task operations."""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Task message")


@router.post("/analysis", response_model=TaskResponse)
async def publish_analysis_task(
    request: SearchTaskRequest,
    current_user: User = Depends(get_current_user)
):
    """Publish an analysis task to the message broker.

    This endpoint publishes an analysis task to the message broker for asynchronous
    processing. The task type is determined based on the analysis_type field in the request.

    Args:
        request: Analysis task request containing the analysis type and parameters
        current_user: The authenticated user making the request

    Returns:
        TaskResponse containing the task ID and status

    Raises:
        HTTPException: If RabbitMQ is disabled or if the request is invalid
    """
    if not settings.RABBITMQ_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RabbitMQ messaging is disabled"
        )
    # Determine the task type based on the analysis type
    if request.analysis_type == "contradictions":
        task_type = TaskType.ANALYZE_CONTRADICTIONS
        if not request.study_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study IDs are required for contradiction analysis"
            )
    elif request.analysis_type == "bias":
        task_type = TaskType.ANALYZE_BIAS
        if not request.study_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Study ID is required for bias analysis"
            )
    elif request.analysis_type == "trends":
        task_type = TaskType.ANALYZE_TRENDS
        if not request.topic:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Topic is required for trend analysis"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid analysis type: {request.analysis_type}"
        )
    # Create task data
    task_data = {
        "user_id": current_user.id
    }
    if request.study_ids:
        task_data["study_ids"] = request.study_ids
    if request.study_id:
        task_data["study_id"] = request.study_id
    if request.topic:
        task_data["topic"] = request.topic
    if request.time_range:
        task_data["time_range"] = request.time_range
    # Get the message producer
    producer = get_message_producer()
    try:
        # Publish the task
        task_id = await producer.publish_task(
            task_type=task_type,
            task_data=task_data
        )
        return TaskResponse(
            task_id=task_id,
            task_type=task_type,
            status="pending",
            message=f"Analysis task published: {request.analysis_type}"
        )
    except Exception as e:
        logger.error(f"Error publishing analysis task: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error publishing analysis task: {str(e)}"
        )
@router.post("/export", response_model=TaskResponse)
async def publish_export_task(
    request: ExportTaskRequest,
    current_user: User = Depends(get_current_user)
):
    """Publish an export task to the message broker.

    This endpoint publishes an export task to the message broker for asynchronous
    processing. The task type is determined based on the export_type field in the request.

    Args:
        request: Export task request containing the export type and parameters
        current_user: The authenticated user making the request

    Returns:
        TaskResponse containing the task ID and status

    Raises:
        HTTPException: If RabbitMQ is disabled or if the request is invalid
    """
    if not settings.RABBITMQ_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RabbitMQ messaging is disabled"
        )
    # Determine the task type based on the export type
    if request.export_type == "results":
        task_type = TaskType.EXPORT_RESULTS
        if not request.result_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Result ID is required for results export"
            )
    elif request.export_type == "analysis":
        task_type = TaskType.EXPORT_ANALYSIS
        if not request.analysis_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Analysis ID is required for analysis export"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid export type: {request.export_type}"
        )
    # Create task data
    task_data = {
        "format": request.format,
        "user_id": current_user.id
    }
    if request.result_id:
        task_data["result_id"] = request.result_id
    if request.analysis_id:
        task_data["analysis_id"] = request.analysis_id
    # Get the message producer
    producer = get_message_producer()
    try:
        # Publish the task
        task_id = await producer.publish_task(
            task_type=task_type,
            task_data=task_data
        )
        return TaskResponse(
            task_id=task_id,
            task_type=task_type,
            status="pending",
            message=f"Export task published: {request.export_type}"
        )
    except Exception as e:
        logger.error(f"Error publishing export task: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error publishing export task: {str(e)}"
        )
@router.get("/task/{task_id}", response_model=Dict[str, Any])
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get the status of a task.

    This endpoint retrieves the current status of a task by its ID.

    Args:
        task_id: The unique identifier of the task
        current_user: The authenticated user making the request

    Returns:
        Dictionary containing the task status information
    """
    # Note: current_user is used for authorization but not directly in the function body
    # This is a placeholder implementation
    # In a real implementation, this would query a task status repository
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Task is pending",
        "progress": 0
    }