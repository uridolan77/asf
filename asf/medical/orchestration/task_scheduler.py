"""
Task scheduler for the Medical Research Synthesizer.
This module provides a scheduler for running tasks using Ray.
"""
import logging
from typing import Dict, Any, Tuple
import uuid
import threading
from datetime import datetime
from asf.medical.orchestration.ray_manager import RayManager
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
        Initialize a new task.
        
        Args:
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            task_id: Unique identifier for the task
            priority: Task priority (higher values = higher priority)
            timeout: Maximum execution time in seconds
            retry_count: Number of times to retry if the task fails
            retry_delay: Delay between retries in seconds
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
            Task dictionary with key attributes
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
        """
        Initialize the task scheduler.
        """
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
        """
        Start the task scheduler.
        
        Initializes Ray and starts the worker thread to process tasks.
        """
        with self.lock:
            if self.running:
                logger.info("Task scheduler already running")
                return
            logger.info("Starting task scheduler")
            if not self.ray_manager.initialize():
                logger.error("Failed to initialize Ray")
                return
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
            logger.info("Task scheduler started")

    def stop(self) -> None:
        """
        Stop the task scheduler.
        
        Stops the worker thread and shuts down Ray.
        """
        with self.lock:
            if not self.running:
                logger.info("Task scheduler not running")
                return
            logger.info("Stopping task scheduler")
            self.running = False
            if self.worker_thread:
                self.worker_thread.join(timeout=5)
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
        Schedule a new task.
        
        Args:
            func: The function to execute
            args: Positional arguments for the function
            kwargs: Keyword arguments for the function
            task_id: Unique identifier for the task
            priority: Task priority (higher values = higher priority)
            timeout: Maximum execution time in seconds
            retry_count: Number of times to retry if the task fails
            retry_delay: Delay between retries in seconds
        
        Returns:
            str: The task ID of the scheduled task
        """
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
        with self.lock:
            self.tasks[task.task_id] = task
            self.task_queue.append(task)
            self.task_queue.sort()
        logger.info(f"Task {task.task_id} scheduled with priority {priority}")
        return task.task_id