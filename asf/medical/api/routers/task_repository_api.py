"""
API endpoints for task repository.

This module provides API endpoints for managing tasks in the database.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query, Path
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from asf.medical.core.logging_config import get_logger
from asf.medical.storage.database import get_db_session
from asf.medical.storage.repositories.task_repository import TaskRepository
from asf.medical.storage.models.task import TaskStatus, TaskPriority
from asf.medical.api.dependencies import get_current_user, get_admin_user
from asf.medical.storage.models import User
from asf.medical.api.websockets.task_updates import task_update_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/db-tasks", tags=["db-tasks"])

class TaskListResponse(BaseModel):
    """Task list response."""
    tasks: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    counts: Dict[str, int]

class TaskResponse(BaseModel):
    """Task response."""
    task: Dict[str, Any]
    events: List[Dict[str, Any]]

class TaskActionResponse(BaseModel):
    """Task action response."""
    task: Dict[str, Any]
    message: str

class DeadLetterListResponse(BaseModel):
    """Dead letter list response."""
    messages: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int

class DeadLetterActionResponse(BaseModel):
    """Dead letter action response."""
    message: str

class CleanupResponse(BaseModel):
    """Cleanup response."""
    tasks_deleted: int
    messages_deleted: int
    message: str

@router.get("/", response_model=TaskListResponse)
async def get_tasks(
    status: Optional[str] = Query(None, description="Filter by task status"),
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    limit: int = Query(100, description="Maximum number of tasks to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get tasks.
    
    Args:
        status: Filter by task status
        user_id: Filter by user ID
        limit: Maximum number of tasks to return
        offset: Offset for pagination
        current_user: Current user
        db: Database session
        
    Returns:
        List of tasks
    """
    # Only admins can view tasks for other users
    if user_id is not None and user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to view tasks for other users"
        )
    
    # Use the current user's ID if no user ID is specified
    if user_id is None:
        user_id = current_user.id
    
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Get tasks by user ID
        if status:
            try:
                task_status = TaskStatus(status)
                tasks = await task_repository.get_tasks_by_status(db, task_status, limit, offset)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid task status: {status}"
                )
        else:
            tasks = await task_repository.get_tasks_by_user_id(db, user_id, limit, offset)
        
        # Get task count by status
        task_counts = await task_repository.get_task_count_by_status(db)
        
        return TaskListResponse(
            tasks=[task.to_dict() for task in tasks],
            total=len(tasks),
            limit=limit,
            offset=offset,
            counts=task_counts
        )
    except Exception as e:
        logger.error(f"Error getting tasks: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting tasks: {str(e)}"
        )

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str = Path(..., description="Task ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get a task by ID.
    
    Args:
        task_id: Task ID
        current_user: Current user
        db: Database session
        
    Returns:
        Task details
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Get the task
        task, events = await task_repository.get_task_with_events(db, task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}"
            )
        
        # Only the task owner or an admin can view the task
        if task.user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this task"
            )
        
        return TaskResponse(
            task=task.to_dict(),
            events=[event.to_dict() for event in events]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task: {str(e)}"
        )

@router.post("/{task_id}/cancel", response_model=TaskActionResponse)
async def cancel_task(
    task_id: str = Path(..., description="Task ID"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Cancel a task.
    
    Args:
        task_id: Task ID
        current_user: Current user
        db: Database session
        
    Returns:
        Cancelled task
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Get the task
        task = await task_repository.get_task_by_id(db, task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}"
            )
        
        # Only the task owner or an admin can cancel the task
        if task.user_id != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to cancel this task"
            )
        
        # Check if the task is already completed or failed
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel task with status: {task.status.value}"
            )
        
        # Check if the task is cancellable
        if not task.cancellable:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task is not cancellable"
            )
        
        # Cancel the task
        task = await task_repository.cancel_task(db, task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel task"
            )
        
        # Notify clients about the cancellation
        await task_update_manager.broadcast_task_cancelled(task)
        
        return TaskActionResponse(
            task=task.to_dict(),
            message="Task cancelled successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling task: {str(e)}"
        )

@router.delete("/{task_id}", response_model=DeadLetterActionResponse)
async def delete_task(
    task_id: str = Path(..., description="Task ID"),
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete a task.
    
    Args:
        task_id: Task ID
        current_user: Current admin user
        db: Database session
        
    Returns:
        Success message
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Get the task
        task = await task_repository.get_task_by_id(db, task_id)
        
        if not task:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found: {task_id}"
            )
        
        # Delete the task
        await db.delete(task)
        await db.commit()
        
        return DeadLetterActionResponse(
            message=f"Task {task_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting task: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting task: {str(e)}"
        )

@router.get("/dead-letters", response_model=DeadLetterListResponse)
async def get_dead_letters(
    reprocessed: Optional[bool] = Query(None, description="Filter by reprocessed status"),
    limit: int = Query(100, description="Maximum number of messages to return"),
    offset: int = Query(0, description="Offset for pagination"),
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get dead letter messages.
    
    Args:
        reprocessed: Filter by reprocessed status
        limit: Maximum number of messages to return
        offset: Offset for pagination
        current_user: Current admin user
        db: Database session
        
    Returns:
        List of dead letter messages
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Get dead letter messages
        messages = await task_repository.get_dead_letter_messages(db, limit, offset, reprocessed)
        
        return DeadLetterListResponse(
            messages=[message.to_dict() for message in messages],
            total=len(messages),
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Error getting dead letter messages: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dead letter messages: {str(e)}"
        )

@router.post("/dead-letters/{message_id}/reprocess", response_model=DeadLetterActionResponse)
async def reprocess_dead_letter(
    message_id: int = Path(..., description="Message ID"),
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Reprocess a dead letter message.
    
    Args:
        message_id: Message ID
        current_user: Current admin user
        db: Database session
        
    Returns:
        Success message
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Get the dead letter message
        message = await task_repository.mark_dead_letter_as_reprocessed(db, message_id)
        
        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dead letter message not found: {message_id}"
            )
        
        # TODO: Implement actual reprocessing logic
        # This would involve publishing the message back to the original exchange
        
        return DeadLetterActionResponse(
            message=f"Dead letter message {message_id} marked as reprocessed"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing dead letter message: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reprocessing dead letter message: {str(e)}"
        )

@router.delete("/dead-letters/{message_id}", response_model=DeadLetterActionResponse)
async def delete_dead_letter(
    message_id: int = Path(..., description="Message ID"),
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Delete a dead letter message.
    
    Args:
        message_id: Message ID
        current_user: Current admin user
        db: Database session
        
    Returns:
        Success message
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Delete the dead letter message
        success = await task_repository.delete_dead_letter_message(db, message_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dead letter message not found: {message_id}"
            )
        
        return DeadLetterActionResponse(
            message=f"Dead letter message {message_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting dead letter message: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting dead letter message: {str(e)}"
        )

@router.post("/cleanup", response_model=CleanupResponse)
async def cleanup_old_tasks(
    days: int = Query(30, description="Age in days for tasks to be considered old"),
    current_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Clean up old tasks and dead letter messages.
    
    Args:
        days: Age in days for tasks to be considered old
        current_user: Current admin user
        db: Database session
        
    Returns:
        Cleanup results
    """
    # Get the task repository
    task_repository = TaskRepository()
    
    try:
        # Clean up old tasks
        tasks_deleted = await task_repository.cleanup_old_tasks(db, days)
        
        # Clean up old dead letter messages
        messages_deleted = await task_repository.cleanup_old_dead_letters(db, days)
        
        return CleanupResponse(
            tasks_deleted=tasks_deleted,
            messages_deleted=messages_deleted,
            message=f"Cleaned up {tasks_deleted} tasks and {messages_deleted} dead letter messages older than {days} days"
        )
    except Exception as e:
        logger.error(f"Error cleaning up old tasks: {str(e)}", exc_info=e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up old tasks: {str(e)}"
        )
