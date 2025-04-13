"""Task Management API Router

This module provides API endpoints for monitoring and managing tasks.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from asf.medical.core.logging_config import get_logger
from asf.medical.core.task_storage import task_storage

logger = get_logger(__name__)

router = APIRouter(
    prefix="/v1/tasks",
    tags=["tasks"]
)

class TaskInfo(BaseModel):
    """Task information model."""
    id: str
    status: str
    name: str
    created_at: str
    updated_at: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    """Task response model."""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None

@router.delete("/{task_id}", response_model=TaskResponse)
async def delete_task(task_id: str):
    """
    Delete a task.

    Args:
        task_id: Task ID

    Returns:
        Response with status and message
    """
    # Delete task from the persistent task storage
    success = task_storage.delete_task(task_id)

    if not success:
        return TaskResponse(
            status="error",
            message=f"Task not found or could not be deleted: {task_id}"
        )

    return TaskResponse(
        status="success",
        message=f"Task {task_id} deleted successfully"
    )


@router.delete("/", response_model=TaskResponse)
async def delete_all_tasks(
    status: Optional[str] = Query(None, description="Filter by status (completed, failed, processing)"),
    older_than: Optional[int] = Query(None, description="Delete tasks older than this many seconds")
):
    """
    Delete all tasks.

    Args:
        status: Filter by status (completed, failed, processing)
        older_than: Delete tasks older than this many seconds

    Returns:
        Response with status and message
    """
    # Delete tasks from the persistent task storage
    count = task_storage.delete_tasks(status=status, older_than=older_than)

    return TaskResponse(
        status="success",
        message=f"Deleted {count} tasks"
    )