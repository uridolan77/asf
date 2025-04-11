"""
Task Management API Router

This module provides API endpoints for monitoring and managing tasks.
"""

from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from asf.medical.core.logging_config import get_logger
from asf.medical.core.persistent_task_storage import task_storage

logger = get_logger(__name__)

router = APIRouter(
    prefix="/v1/tasks",
    tags=["tasks"]
)

class TaskInfo(BaseModel):
    """Task information."""
    task_id: str
    status: str
    progress: Optional[int] = None
    updated_at: Optional[float] = None
    created_at: Optional[float] = None
    completed_at: Optional[float] = None
    failed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

class TaskListResponse(BaseModel):
    """Task list response."""
    tasks: List[TaskInfo]
    total: int

class TaskResponse(BaseModel):
    """Task response."""
    status: str
    message: str
    data: Dict[str, Any] = {}

# Routes
@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    status: Optional[str] = Query(None, description="Filter by status (completed, failed, processing)"),
    limit: int = Query(100, description="Maximum number of tasks to return"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """
    List tasks.

    Args:
        status: Filter by status (completed, failed, processing)
        limit: Maximum number of tasks to return
        offset: Offset for pagination

    Returns:
        List of tasks
    """
    # Get tasks from the persistent task storage
    tasks = task_storage.get_tasks(status=status, limit=limit, offset=offset)

    # Convert tasks to TaskInfo objects
    task_infos = [
        TaskInfo(
            task_id=task_id,
            status=task_info["status"],
            progress=task_info.get("progress"),
            updated_at=task_info.get("updated_at"),
            created_at=task_info.get("created_at"),
            completed_at=task_info.get("completed_at"),
            failed_at=task_info.get("failed_at"),
            error=task_info.get("error"),
            result=task_info.get("result"),
            metadata=task_info.get("metadata")
        )
        for task_id, task_info in tasks.items()
    ]

    return TaskListResponse(tasks=task_infos, total=len(task_infos))


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """
    Get task information.

    Args:
        task_id: Task ID

    Returns:
        Task information
    """
    # Get task from the persistent task storage
    task_info = task_storage.get_task(task_id)

    if not task_info:
        return TaskResponse(
            status="error",
            message=f"Task not found: {task_id}"
        )

    return TaskResponse(
        status="success",
        message="Task found",
        data=task_info
    )


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