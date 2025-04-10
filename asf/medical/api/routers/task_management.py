"""
Task Management API Router

This module provides API endpoints for monitoring and managing tasks.
"""

import logging
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel

from asf.medical.api.auth import get_current_user
from asf.medical.core.persistent_task_storage import task_storage

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/v1/tasks",
    tags=["tasks"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}},
)

# Models
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
    try:
        # Get all tasks
        all_tasks = task_storage.list_tasks(limit=0)  # No limit
        
        # Filter by status if specified
        if status:
            all_tasks = [task for task in all_tasks if task.get("status") == status]
        
        # Sort by updated_at (newest first)
        all_tasks.sort(key=lambda x: x.get("updated_at", 0), reverse=True)
        
        # Apply pagination
        paginated_tasks = all_tasks[offset:offset + limit]
        
        # Convert to TaskInfo objects
        task_infos = []
        for task in paginated_tasks:
            task_id = task.get("task_id", "")
            task_info = TaskInfo(
                task_id=task_id,
                status=task.get("status", "unknown"),
                progress=task.get("progress"),
                updated_at=task.get("updated_at"),
                created_at=task.get("created_at"),
                completed_at=task.get("completed_at"),
                failed_at=task.get("failed_at"),
                error=task.get("error"),
                result=task.get("result"),
                metadata=task.get("metadata", {})
            )
            task_infos.append(task_info)
        
        return TaskListResponse(
            tasks=task_infos,
            total=len(all_tasks)
        )
    except Exception as e:
        logger.error(f"Error listing tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing tasks: {str(e)}",
        )

@router.get("/{task_id}", response_model=TaskInfo)
async def get_task(task_id: str):
    """
    Get task information.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task information
    """
    try:
        # Get task
        task = task_storage.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}",
            )
        
        # Add task_id to task
        task["task_id"] = task_id
        
        return TaskInfo(**task)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task: {str(e)}",
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
    try:
        # Check if task exists
        task = task_storage.get_task_status(task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}",
            )
        
        # Delete task
        task_storage.delete_task_status(task_id)
        
        return TaskResponse(
            status="success",
            message=f"Task deleted: {task_id}",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting task: {str(e)}",
        )

@router.delete("/", response_model=TaskResponse)
async def delete_all_tasks(
    status: Optional[str] = Query(None, description="Filter by status (completed, failed, processing)"),
    older_than: Optional[int] = Query(None, description="Delete tasks older than this many seconds"),
):
    """
    Delete all tasks.
    
    Args:
        status: Filter by status (completed, failed, processing)
        older_than: Delete tasks older than this many seconds
        
    Returns:
        Response with status and message
    """
    try:
        # Get all tasks
        all_tasks = task_storage.list_tasks(limit=0)  # No limit
        
        # Filter by status if specified
        if status:
            all_tasks = [task for task in all_tasks if task.get("status") == status]
        
        # Filter by age if specified
        if older_than:
            import time
            current_time = time.time()
            all_tasks = [
                task for task in all_tasks 
                if current_time - task.get("updated_at", 0) > older_than
            ]
        
        # Delete tasks
        deleted_count = 0
        for task in all_tasks:
            task_id = task.get("task_id", "")
            if task_id:
                task_storage.delete_task_status(task_id)
                deleted_count += 1
        
        return TaskResponse(
            status="success",
            message=f"Deleted {deleted_count} tasks",
            data={"deleted_count": deleted_count}
        )
    except Exception as e:
        logger.error(f"Error deleting tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting tasks: {str(e)}",
        )
