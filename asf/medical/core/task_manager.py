"""
Task manager for the Medical Research Synthesizer.

This module provides a task manager for background tasks.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple
from datetime import datetime, timedelta
import traceback

# Set up logging
logger = logging.getLogger(__name__)

class TaskManager:
    """
    Task manager for background tasks.
    
    This class provides a task manager for running background tasks
    and tracking their status.
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        """
        Initialize the task manager.
        
        Args:
            max_concurrent_tasks: Maximum number of concurrent tasks (default: 10)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.running = False
        self.cleanup_task = None
    
    async def initialize(self):
        """Initialize the task manager."""
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_completed_tasks())
        logger.info("Task manager initialized")
    
    async def shutdown(self):
        """Shutdown the task manager."""
        self.running = False
        
        # Cancel cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task_id, task_info in list(self.tasks.items()):
            if not task_info['task'].done():
                task_info['task'].cancel()
                try:
                    await task_info['task']
                except asyncio.CancelledError:
                    pass
        
        logger.info("Task manager shut down")
    
    async def submit_task(
        self,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> str:
        """
        Submit a task for execution.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID
        """
        # Generate a task ID
        task_id = str(uuid.uuid4())
        
        # Create a task info dictionary
        task_info = {
            'id': task_id,
            'status': 'pending',
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
            'result': None,
            'error': None,
            'func_name': func.__name__
        }
        
        # Create and store the task
        task = asyncio.create_task(self._execute_task(task_id, func, *args, **kwargs))
        task_info['task'] = task
        
        # Store the task info
        self.tasks[task_id] = task_info
        
        logger.info(f"Task submitted: {task_id} ({func.__name__})")
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if the task is not found
        """
        if task_id not in self.tasks:
            return None
        
        task_info = self.tasks[task_id].copy()
        
        # Remove the task object
        if 'task' in task_info:
            del task_info['task']
        
        return task_info
    
    async def get_all_tasks(self) -> List[Dict[str, Any]]:
        """
        Get the status of all tasks.
        
        Returns:
            List of task statuses
        """
        task_statuses = []
        
        for task_id, task_info in self.tasks.items():
            # Create a copy of the task info
            task_status = task_info.copy()
            
            # Remove the task object
            if 'task' in task_status:
                del task_status['task']
            
            task_statuses.append(task_status)
        
        return task_statuses
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if the task was cancelled, False otherwise
        """
        if task_id not in self.tasks:
            return False
        
        task_info = self.tasks[task_id]
        
        # Check if the task is already completed
        if task_info['status'] in ['completed', 'failed', 'cancelled']:
            return False
        
        # Cancel the task
        task_info['task'].cancel()
        
        try:
            await task_info['task']
        except asyncio.CancelledError:
            pass
        
        # Update task info
        task_info['status'] = 'cancelled'
        task_info['completed_at'] = datetime.now()
        
        logger.info(f"Task cancelled: {task_id}")
        
        return True
    
    async def _execute_task(
        self,
        task_id: str,
        func: Callable[..., Awaitable[Any]],
        *args,
        **kwargs
    ) -> Any:
        """
        Execute a task with the semaphore.
        
        Args:
            task_id: Task ID
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Function result
        """
        # Get the task info
        task_info = self.tasks[task_id]
        
        # Acquire the semaphore
        async with self.semaphore:
            # Update task info
            task_info['status'] = 'running'
            task_info['started_at'] = datetime.now()
            
            logger.info(f"Task started: {task_id} ({func.__name__})")
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Update task info
                task_info['status'] = 'completed'
                task_info['completed_at'] = datetime.now()
                task_info['result'] = result
                
                logger.info(f"Task completed: {task_id} ({func.__name__})")
                
                return result
            except asyncio.CancelledError:
                # Task was cancelled
                task_info['status'] = 'cancelled'
                task_info['completed_at'] = datetime.now()
                
                logger.info(f"Task cancelled: {task_id} ({func.__name__})")
                
                raise
            except Exception as e:
                # Task failed
                task_info['status'] = 'failed'
                task_info['completed_at'] = datetime.now()
                task_info['error'] = {
                    'type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
                
                logger.error(f"Task failed: {task_id} ({func.__name__}): {str(e)}")
                
                raise
    
    async def _cleanup_completed_tasks(self):
        """
        Periodically clean up completed tasks.
        
        This method runs in the background and removes completed tasks
        that are older than a certain threshold.
        """
        while self.running:
            try:
                # Sleep for a while
                await asyncio.sleep(3600)  # 1 hour
                
                # Get the current time
                now = datetime.now()
                
                # Find tasks to remove
                tasks_to_remove = []
                
                for task_id, task_info in self.tasks.items():
                    # Check if the task is completed and old enough
                    if (task_info['status'] in ['completed', 'failed', 'cancelled'] and
                            task_info['completed_at'] is not None and
                            now - task_info['completed_at'] > timedelta(days=1)):
                        tasks_to_remove.append(task_id)
                
                # Remove the tasks
                for task_id in tasks_to_remove:
                    del self.tasks[task_id]
                
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} completed tasks")
            except asyncio.CancelledError:
                # Task manager is shutting down
                break
            except Exception as e:
                logger.error(f"Error in task cleanup: {str(e)}")
                
                # Sleep for a shorter time before retrying
                await asyncio.sleep(60)  # 1 minute
