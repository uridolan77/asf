API endpoints for messaging tasks.
This module provides API endpoints for publishing tasks to the message broker.
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from asf.medical.core.config import settings
from asf.medical.core.logging_config import get_logger
from asf.medical.core.messaging.producer import get_message_producer
from asf.medical.core.messaging.schemas import TaskType
from asf.medical.api.dependencies import get_current_user
from asf.medical.storage.models import User
logger = get_logger(__name__)
router = APIRouter(prefix="/v1/messaging", tags=["messaging"])
class SearchTaskRequest(BaseModel):
    Request model for search tasks.
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
    """
    Publish an export task to the message broker.
    Args:
        request: Export task request
        current_user: Current user
    Returns:
        Task response
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
    """
    Get the status of a task.
    Args:
        task_id: Task ID
        current_user: Current user
    Returns:
        Task status
    """
    # This is a placeholder implementation
    # In a real implementation, this would query a task status repository
    return {
        "task_id": task_id,
        "status": "pending",
        "message": "Task is pending",
        "progress": 0
    }