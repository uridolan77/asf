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
        """Initialize the task scheduler.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
        """Start the task scheduler.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
        """Stop the task scheduler.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
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
        Execute a task.
        Args:
            task: Task to execute