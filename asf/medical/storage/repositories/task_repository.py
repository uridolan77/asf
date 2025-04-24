"""
Task repository for the Medical Research Synthesizer.
This module provides a repository for task-related database operations.
"""
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import uuid
from sqlalchemy import and_, or_, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from ..models.task import Task, TaskEvent, DeadLetterMessage, TaskStatus
from .enhanced_base_repository import EnhancedBaseRepository
from ...core.exceptions import DatabaseError
from ...core.logging_config import get_logger
logger = get_logger(__name__)
class TaskRepository(EnhancedBaseRepository[Task]):
    """
    Repository for task-related database operations.
    """
    def __init__(self):
        """Initialize the repository with the Task model."""
        super().__init__(Task)
    async def create_task(
        self, 
        db: AsyncSession, 
        task_type: str, 
        params: Dict[str, Any],
        user_id: Optional[int] = None,
        priority: int = 5,
        max_retries: int = 3,
        task_id: Optional[str] = None
    ) -> Task:
        """
        Create a new task.
        Args:
            db: Database session
            task_type: Type of task
            params: Task parameters
            user_id: User ID (optional)
            priority: Task priority (default: 5)
            max_retries: Maximum number of retries (default: 3)
            task_id: Task ID (optional, generated if not provided)
        Returns:
            Created task
        Raises:
            DatabaseError: If there's an error creating the task
        """
        try:
            task_id = task_id or str(uuid.uuid4())
            task = Task(
                id=task_id,
                type=task_type,
                status=TaskStatus.PENDING,
                priority=priority,
                user_id=user_id,
                params=params,
                max_retries=max_retries,
                created_at=datetime.now(datetime.timezone.utc),
                updated_at=datetime.now(datetime.timezone.utc)
            )
            db.add(task)
            await db.commit()
            await db.refresh(task)
            # Create a task event for the creation
            await self.create_task_event(
                db=db,
                task_id=task.id,
                event_type="created",
                event_data={"status": TaskStatus.PENDING.value}
            )
            return task
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating task: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to create task: {str(e)}")
    async def get_task_by_id(self, db: AsyncSession, task_id: str) -> Optional[Task]:
        """
        Get a task by ID.
        Args:
            db: Database session
            task_id: Task ID
        Returns:
            Task or None if not found
        Raises:
            DatabaseError: If there's an error getting the task
        """
        try:
            stmt = select(Task).where(Task.id == task_id)
            result = await db.execute(stmt)
            return result.scalars().first()
        except Exception as e:
            logger.error(f"Error getting task by ID: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get task by ID: {str(e)}")
    async def update_task_status(
        self, 
        db: AsyncSession, 
        task_id: str, 
        status: TaskStatus,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        worker_id: Optional[str] = None
    ) -> Optional[Task]:
        """
        Update a task's status.
        Args:
            db: Database session
            task_id: Task ID
            status: New task status
            progress: Task progress (0-100)
            message: Status message
            result: Task result
            error: Error message
            worker_id: Worker ID
        Returns:
            Updated task or None if not found
        Raises:
            DatabaseError: If there's an error updating the task
        """
        try:
            task = await self.get_task_by_id(db, task_id)
            if not task:
                return None
            # Update task fields
            task.status = status
            task.updated_at = datetime.now(datetime.timezone.utc)
            if progress is not None:
                task.progress = progress
            if message:
                task.message = message
            if result:
                task.result = result
            if error:
                task.error = error
            if worker_id:
                task.worker_id = worker_id
            # Update timestamps based on status
            if status == TaskStatus.RUNNING and not task.started_at:
                task.started_at = datetime.now(datetime.timezone.utc)
            if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED) and not task.completed_at:
                task.completed_at = datetime.now(datetime.timezone.utc)
            await db.commit()
            await db.refresh(task)
            # Create a task event for the status update
            event_data = {
                "status": status.value,
                "progress": task.progress
            }
            if message:
                event_data["message"] = message
            if error:
                event_data["error"] = error
            await self.create_task_event(
                db=db,
                task_id=task.id,
                event_type=f"status_changed_to_{status.value}",
                event_data=event_data
            )
            return task
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating task status: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to update task status: {str(e)}")
    async def update_task_progress(
        self, 
        db: AsyncSession, 
        task_id: str, 
        progress: float,
        message: Optional[str] = None
    ) -> Optional[Task]:
        """
        Update a task's progress.
        Args:
            db: Database session
            task_id: Task ID
            progress: Task progress (0-100)
            message: Progress message
        Returns:
            Updated task or None if not found
        Raises:
            DatabaseError: If there's an error updating the task
        """
        try:
            task = await self.get_task_by_id(db, task_id)
            if not task:
                return None
            # Update task fields
            task.progress = progress
            task.updated_at = datetime.now(datetime.timezone.utc)
            if message:
                task.message = message
            # If the task is pending and progress is reported, set it to running
            if task.status == TaskStatus.PENDING and progress > 0:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now(datetime.timezone.utc)
            await db.commit()
            await db.refresh(task)
            # Create a task event for the progress update
            event_data = {
                "progress": progress
            }
            if message:
                event_data["message"] = message
            await self.create_task_event(
                db=db,
                task_id=task.id,
                event_type="progress_updated",
                event_data=event_data
            )
            return task
        except Exception as e:
            await db.rollback()
            logger.error(f"Error updating task progress: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to update task progress: {str(e)}")
    async def mark_task_for_retry(
        self, 
        db: AsyncSession, 
        task_id: str, 
        error: str,
        retry_delay: int = 60
    ) -> Optional[Task]:
        """
        Mark a task for retry.
        Args:
            db: Database session
            task_id: Task ID
            error: Error message
            retry_delay: Delay before retry in seconds (default: 60)
        Returns:
            Updated task or None if not found or max retries reached
        Raises:
            DatabaseError: If there's an error updating the task
        """
        try:
            task = await self.get_task_by_id(db, task_id)
            if not task:
                return None
            # Check if max retries reached
            if task.retry_count >= task.max_retries:
                # Mark as failed
                return await self.update_task_status(
                    db=db,
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    error=f"Max retries reached. Last error: {error}"
                )
            # Update retry count and schedule next retry
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            task.error = error
            task.next_retry_at = datetime.now(datetime.timezone.utc) + timedelta(seconds=retry_delay)
            task.updated_at = datetime.now(datetime.timezone.utc)
            await db.commit()
            await db.refresh(task)
            # Create a task event for the retry
            await self.create_task_event(
                db=db,
                task_id=task.id,
                event_type="retry_scheduled",
                event_data={
                    "retry_count": task.retry_count,
                    "max_retries": task.max_retries,
                    "next_retry_at": task.next_retry_at.isoformat() if task.next_retry_at else None,
                    "error": error
                }
            )
            return task
        except Exception as e:
            await db.rollback()
            logger.error(f"Error marking task for retry: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to mark task for retry: {str(e)}")
    async def cancel_task(self, db: AsyncSession, task_id: str) -> Optional[Task]:
        """
        Cancel a task.
        Args:
            db: Database session
            task_id: Task ID
        Returns:
            Updated task or None if not found or not cancellable
        Raises:
            DatabaseError: If there's an error cancelling the task
        """
        try:
            task = await self.get_task_by_id(db, task_id)
            if not task:
                return None
            # Check if the task is cancellable
            if not task.cancellable:
                return None
            # Check if the task is already completed or failed
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return None
            # Update task fields
            task.status = TaskStatus.CANCELLED
            task.cancelled = True
            task.completed_at = datetime.now(datetime.timezone.utc)
            task.updated_at = datetime.now(datetime.timezone.utc)
            await db.commit()
            await db.refresh(task)
            # Create a task event for the cancellation
            await self.create_task_event(
                db=db,
                task_id=task.id,
                event_type="cancelled",
                event_data={"status": TaskStatus.CANCELLED.value}
            )
            return task
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cancelling task: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to cancel task: {str(e)}")
    async def get_tasks_by_status(
        self, 
        db: AsyncSession, 
        status: TaskStatus,
        limit: int = 100,
        offset: int = 0
    ) -> List[Task]:
        """
        Get tasks by status.
        Args:
            db: Database session
            status: Task status
            limit: Maximum number of tasks to return
            offset: Offset for pagination
        Returns:
            List of tasks
        Raises:
            DatabaseError: If there's an error getting the tasks
        """
        try:
            stmt = (
                select(Task)
                .where(Task.status == status)
                .order_by(desc(Task.priority), Task.created_at)
                .offset(offset)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting tasks by status: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get tasks by status: {str(e)}")
    async def get_pending_tasks_for_processing(
        self, 
        db: AsyncSession, 
        limit: int = 10
    ) -> List[Task]:
        """
        Get pending tasks that are ready for processing.
        Args:
            db: Database session
            limit: Maximum number of tasks to return
        Returns:
            List of tasks
        Raises:
            DatabaseError: If there's an error getting the tasks
        """
        try:
            now = datetime.now(datetime.timezone.utc)
            # Get tasks that are pending or scheduled for retry
            stmt = (
                select(Task)
                .where(
                    or_(
                        Task.status == TaskStatus.PENDING,
                        and_(
                            Task.status == TaskStatus.RETRYING,
                            or_(
                                Task.next_retry_at.is_(None),
                                Task.next_retry_at <= now
                            )
                        )
                    )
                )
                .order_by(desc(Task.priority), Task.created_at)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting pending tasks: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get pending tasks: {str(e)}")
    async def get_tasks_by_user_id(
        self, 
        db: AsyncSession, 
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[Task]:
        """
        Get tasks by user ID.
        Args:
            db: Database session
            user_id: User ID
            limit: Maximum number of tasks to return
            offset: Offset for pagination
        Returns:
            List of tasks
        Raises:
            DatabaseError: If there's an error getting the tasks
        """
        try:
            stmt = (
                select(Task)
                .where(Task.user_id == user_id)
                .order_by(desc(Task.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting tasks by user ID: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get tasks by user ID: {str(e)}")
    async def get_task_count_by_status(self, db: AsyncSession) -> Dict[str, int]:
        """
        Get task count by status.
        Args:
            db: Database session
        Returns:
            Dictionary mapping status to count
        Raises:
            DatabaseError: If there's an error getting the task count
        """
        try:
            stmt = (
                select(Task.status, func.count(Task.id))
                .group_by(Task.status)
            )
            result = await db.execute(stmt)
            counts = {status.value: count for status, count in result.all()}
            # Ensure all statuses are included
            for status in TaskStatus:
                if status.value not in counts:
                    counts[status.value] = 0
            return counts
        except Exception as e:
            logger.error(f"Error getting task count by status: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get task count by status: {str(e)}")
    async def create_task_event(
        self, 
        db: AsyncSession, 
        task_id: str, 
        event_type: str,
        event_data: Dict[str, Any]
    ) -> TaskEvent:
        """
        Create a task event.
        Args:
            db: Database session
            task_id: Task ID
            event_type: Event type
            event_data: Event data
        Returns:
            Created task event
        Raises:
            DatabaseError: If there's an error creating the task event
        """
        try:
            task_event = TaskEvent(
                task_id=task_id,
                event_type=event_type,
                event_data=event_data,
                created_at=datetime.now(datetime.timezone.utc)
            )
            db.add(task_event)
            await db.commit()
            await db.refresh(task_event)
            return task_event
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating task event: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to create task event: {str(e)}")
    async def get_task_events(
        self, 
        db: AsyncSession, 
        task_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[TaskEvent]:
        """
        Get events for a task.
        Args:
            db: Database session
            task_id: Task ID
            limit: Maximum number of events to return
            offset: Offset for pagination
        Returns:
            List of task events
        Raises:
            DatabaseError: If there's an error getting the task events
        """
        try:
            stmt = (
                select(TaskEvent)
                .where(TaskEvent.task_id == task_id)
                .order_by(desc(TaskEvent.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting task events: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get task events: {str(e)}")
    async def create_dead_letter_message(
        self, 
        db: AsyncSession, 
        exchange: str,
        routing_key: str,
        message: Dict[str, Any],
        headers: Optional[Dict[str, Any]] = None,
        original_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> DeadLetterMessage:
        """
        Create a dead letter message.
        Args:
            db: Database session
            exchange: Exchange name
            routing_key: Routing key
            message: Message content
            headers: Message headers
            original_id: Original message ID
            error: Error message
        Returns:
            Created dead letter message
        Raises:
            DatabaseError: If there's an error creating the dead letter message
        """
        try:
            dead_letter = DeadLetterMessage(
                original_id=original_id,
                exchange=exchange,
                routing_key=routing_key,
                message=message,
                headers=headers,
                error=error,
                created_at=datetime.now(datetime.timezone.utc),
                updated_at=datetime.now(datetime.timezone.utc)
            )
            db.add(dead_letter)
            await db.commit()
            await db.refresh(dead_letter)
            return dead_letter
        except Exception as e:
            await db.rollback()
            logger.error(f"Error creating dead letter message: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to create dead letter message: {str(e)}")
    async def get_dead_letter_messages(
        self, 
        db: AsyncSession, 
        limit: int = 100,
        offset: int = 0,
        reprocessed: Optional[bool] = None
    ) -> List[DeadLetterMessage]:
        """
        Get dead letter messages.
        Args:
            db: Database session
            limit: Maximum number of messages to return
            offset: Offset for pagination
            reprocessed: Filter by reprocessed status
        Returns:
            List of dead letter messages
        Raises:
            DatabaseError: If there's an error getting the dead letter messages
        """
        try:
            stmt = select(DeadLetterMessage)
            if reprocessed is not None:
                stmt = stmt.where(DeadLetterMessage.reprocessed == reprocessed)
            stmt = (
                stmt
                .order_by(desc(DeadLetterMessage.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = await db.execute(stmt)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"Error getting dead letter messages: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get dead letter messages: {str(e)}")
    async def mark_dead_letter_as_reprocessed(
        self, 
        db: AsyncSession, 
        message_id: int
    ) -> Optional[DeadLetterMessage]:
        """
        Mark a dead letter message as reprocessed.
        Args:
            db: Database session
            message_id: Message ID
        Returns:
            Updated dead letter message or None if not found
        Raises:
            DatabaseError: If there's an error updating the dead letter message
        """
        try:
            stmt = select(DeadLetterMessage).where(DeadLetterMessage.id == message_id)
            result = await db.execute(stmt)
            dead_letter = result.scalars().first()
            if not dead_letter:
                return None
            dead_letter.reprocessed = True
            dead_letter.reprocessed_at = datetime.now(datetime.timezone.utc)
            dead_letter.updated_at = datetime.now(datetime.timezone.utc)
            await db.commit()
            await db.refresh(dead_letter)
            return dead_letter
        except Exception as e:
            await db.rollback()
            logger.error(f"Error marking dead letter as reprocessed: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to mark dead letter as reprocessed: {str(e)}")
    async def delete_dead_letter_message(
        self, 
        db: AsyncSession, 
        message_id: int
    ) -> bool:
        """
        Delete a dead letter message.
        Args:
            db: Database session
            message_id: Message ID
        Returns:
            True if deleted, False if not found
        Raises:
            DatabaseError: If there's an error deleting the dead letter message
        """
        try:
            stmt = select(DeadLetterMessage).where(DeadLetterMessage.id == message_id)
            result = await db.execute(stmt)
            dead_letter = result.scalars().first()
            if not dead_letter:
                return False
            await db.delete(dead_letter)
            await db.commit()
            return True
        except Exception as e:
            await db.rollback()
            logger.error(f"Error deleting dead letter message: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to delete dead letter message: {str(e)}")
    async def get_task_with_events(
        self, 
        db: AsyncSession, 
        task_id: str
    ) -> Tuple[Optional[Task], List[TaskEvent]]:
        """
        Get a task with its events.
        Args:
            db: Database session
            task_id: Task ID
        Returns:
            Tuple of (task, events) or (None, []) if not found
        Raises:
            DatabaseError: If there's an error getting the task
        """
        try:
            # Get the task
            task = await self.get_task_by_id(db, task_id)
            if not task:
                return None, []
            # Get the task events
            events = await self.get_task_events(db, task_id)
            return task, events
        except Exception as e:
            logger.error(f"Error getting task with events: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to get task with events: {str(e)}")
    async def cleanup_old_tasks(
        self, 
        db: AsyncSession, 
        days: int = 30
    ) -> int:
        """
        Clean up old completed, failed, or cancelled tasks.
        Args:
            db: Database session
            days: Age in days for tasks to be considered old
        Returns:
            Number of tasks deleted
        Raises:
            DatabaseError: If there's an error cleaning up tasks
        """
        try:
            cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=days)
            # Find tasks to delete
            stmt = (
                select(Task)
                .where(
                    and_(
                        Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]),
                        Task.updated_at < cutoff_date
                    )
                )
            )
            result = await db.execute(stmt)
            tasks_to_delete = list(result.scalars().all())
            # Delete tasks
            for task in tasks_to_delete:
                await db.delete(task)
            await db.commit()
            return len(tasks_to_delete)
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cleaning up old tasks: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to clean up old tasks: {str(e)}")
    async def cleanup_old_dead_letters(
        self, 
        db: AsyncSession, 
        days: int = 30
    ) -> int:
        """
        Clean up old dead letter messages.
        Args:
            db: Database session
            days: Age in days for messages to be considered old
        Returns:
            Number of messages deleted
        Raises:
            DatabaseError: If there's an error cleaning up messages
        """
        try:
            cutoff_date = datetime.now(datetime.timezone.utc) - timedelta(days=days)
            # Find messages to delete
            stmt = (
                select(DeadLetterMessage)
                .where(
                    and_(
                        DeadLetterMessage.reprocessed == True,
                        DeadLetterMessage.updated_at < cutoff_date
                    )
                )
            )
            result = await db.execute(stmt)
            messages_to_delete = list(result.scalars().all())
            # Delete messages
            for message in messages_to_delete:
                await db.delete(message)
            await db.commit()
            return len(messages_to_delete)
        except Exception as e:
            await db.rollback()
            logger.error(f"Error cleaning up old dead letters: {str(e)}", exc_info=e)
            raise DatabaseError(f"Failed to clean up old dead letters: {str(e)}")