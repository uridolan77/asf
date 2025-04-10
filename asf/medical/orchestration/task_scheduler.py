"""
Task scheduler for the Medical Research Synthesizer.

This module provides a scheduler for running tasks using Ray.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import time
import uuid
import threading
from datetime import datetime, timedelta

from asf.medical.orchestration.ray_manager import RayManager
from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class Task:
    """
    Task class for the task scheduler.
    
    This class represents a task to be scheduled and executed.
    """
    
    def __init__(
        self,
        func: callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        task_id: str = None,
        priority: int = 0,
        timeout: int = 3600,
        retry_count: int = 3,
        retry_delay: int = 60
    ):
        """
        Initialize a task.
        
        Args:
            func: Function to run
            args: Function arguments
            kwargs: Function keyword arguments
            task_id: Task ID (default: auto-generated UUID)
            priority: Task priority (default: 0)
            timeout: Task timeout in seconds (default: 3600)
            retry_count: Number of retries (default: 3)
            retry_delay: Delay between retries in seconds (default: 60)
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        self.status = "pending"
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.retries = 0
    
    def __lt__(self, other):
        """
        Compare tasks by priority.
        
        Args:
            other: Other task
            
        Returns:
            True if this task has higher priority than the other task
        """
        return self.priority > other.priority
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary.
        
        Returns:
            Task dictionary
        """
        return {
            "task_id": self.task_id,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "error": str(self.error) if self.error else None
        }

class TaskScheduler:
    """
    Scheduler for running tasks using Ray.
    
    This scheduler provides methods for scheduling and executing tasks.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the task scheduler.
        
        Returns:
            TaskScheduler: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(TaskScheduler, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the task scheduler."""
        self.ray_manager = RayManager()
        self.tasks = {}
        self.task_queue = []
        self.running_tasks = {}
        self.completed_tasks = {}
        self.lock = threading.RLock()
        self.running = False
        self.worker_thread = None
        
        logger.info("Task scheduler initialized")
    
    def start(self) -> None:
        """Start the task scheduler."""
        with self.lock:
            if self.running:
                logger.info("Task scheduler already running")
                return
            
            logger.info("Starting task scheduler")
            
            # Initialize Ray
            if not self.ray_manager.initialize():
                logger.error("Failed to initialize Ray")
                return
            
            # Start worker thread
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            
            logger.info("Task scheduler started")
    
    def stop(self) -> None:
        """Stop the task scheduler."""
        with self.lock:
            if not self.running:
                logger.info("Task scheduler not running")
                return
            
            logger.info("Stopping task scheduler")
            
            # Stop worker thread
            self.running = False
            
            # Wait for worker thread to finish
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
            
            # Shutdown Ray
            self.ray_manager.shutdown()
            
            logger.info("Task scheduler stopped")
    
    def schedule_task(
        self,
        func: callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        task_id: str = None,
        priority: int = 0,
        timeout: int = 3600,
        retry_count: int = 3,
        retry_delay: int = 60
    ) -> str:
        """
        Schedule a task.
        
        Args:
            func: Function to run
            args: Function arguments
            kwargs: Function keyword arguments
            task_id: Task ID (default: auto-generated UUID)
            priority: Task priority (default: 0)
            timeout: Task timeout in seconds (default: 3600)
            retry_count: Number of retries (default: 3)
            retry_delay: Delay between retries in seconds (default: 60)
            
        Returns:
            Task ID
        """
        with self.lock:
            # Create task
            task = Task(
                func=func,
                args=args,
                kwargs=kwargs,
                task_id=task_id,
                priority=priority,
                timeout=timeout,
                retry_count=retry_count,
                retry_delay=retry_delay
            )
            
            # Add task to queue
            self.tasks[task.task_id] = task
            self.task_queue.append(task)
            
            # Sort queue by priority
            self.task_queue.sort()
            
            logger.info(f"Task scheduled: {task.task_id}")
            
            return task.task_id
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status or None if task not found
        """
        with self.lock:
            # Check if task exists
            if task_id in self.tasks:
                return self.tasks[task_id].to_dict()
            elif task_id in self.running_tasks:
                return self.running_tasks[task_id].to_dict()
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].to_dict()
            else:
                return None
    
    def get_task_result(self, task_id: str) -> Optional[Any]:
        """
        Get task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if task not found or not completed
        """
        with self.lock:
            # Check if task exists and is completed
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task.status == "completed":
                    return task.result
            
            return None
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            True if task was cancelled, False otherwise
        """
        with self.lock:
            # Check if task exists and is pending
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                # Remove task from queue
                self.task_queue.remove(task)
                del self.tasks[task_id]
                
                # Add task to completed tasks
                task.status = "cancelled"
                task.completed_at = datetime.now()
                self.completed_tasks[task_id] = task
                
                logger.info(f"Task cancelled: {task_id}")
                
                return True
            
            return False
    
    def _worker_loop(self) -> None:
        """Worker loop for executing tasks."""
        logger.info("Worker loop started")
        
        while self.running:
            # Get next task
            task = None
            
            with self.lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
                    del self.tasks[task.task_id]
                    self.running_tasks[task.task_id] = task
            
            if task:
                # Execute task
                self._execute_task(task)
            else:
                # Sleep if no tasks
                time.sleep(1)
        
        logger.info("Worker loop stopped")
    
    def _execute_task(self, task: Task) -> None:
        """
        Execute a task.
        
        Args:
            task: Task to execute
        """
        logger.info(f"Executing task: {task.task_id}")
        
        # Update task status
        task.status = "running"
        task.started_at = datetime.now()
        
        try:
            # Execute task
            result = self.ray_manager.run_task(task.func, *task.args, **task.kwargs)
            
            # Update task status
            task.status = "completed"
            task.result = result
            task.completed_at = datetime.now()
            
            logger.info(f"Task completed: {task.task_id}")
        except Exception as e:
            logger.error(f"Task failed: {task.task_id}, error: {str(e)}")
            
            # Update task status
            task.error = e
            
            # Retry if retries left
            if task.retries < task.retry_count:
                task.retries += 1
                task.status = "pending"
                
                # Add task back to queue with delay
                def requeue_task():
                    with self.lock:
                        # Add task back to queue
                        self.task_queue.append(task)
                        self.task_queue.sort()
                        
                        # Move task from running to tasks
                        del self.running_tasks[task.task_id]
                        self.tasks[task.task_id] = task
                        
                        logger.info(f"Task requeued: {task.task_id}, retry: {task.retries}")
                
                # Schedule requeue
                threading.Timer(task.retry_delay, requeue_task).start()
            else:
                # Mark task as failed
                task.status = "failed"
                task.completed_at = datetime.now()
                
                logger.info(f"Task failed: {task.task_id}, max retries reached")
        
        # Move task from running to completed
        with self.lock:
            if task.status in ["completed", "failed"]:
                del self.running_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
