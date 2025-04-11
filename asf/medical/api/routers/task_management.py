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

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/tasks",
    tags=["tasks"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}},
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
    Get task information.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task information
    Delete a task.
    
    Args:
        task_id: Task ID
        
    Returns:
        Response with status and message
    Delete all tasks.
    
    Args:
        status: Filter by status (completed, failed, processing)
        older_than: Delete tasks older than this many seconds
        
    Returns:
        Response with status and message