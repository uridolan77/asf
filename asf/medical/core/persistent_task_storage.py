"""
Persistent Task Storage

This module provides persistent storage for task status and results using Redis.
It includes integration with Dramatiq for tracking task status and results.
"""

import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps

import redis
try:
    import dramatiq
    from dramatiq.middleware import Middleware
    HAS_DRAMATIQ = True
except ImportError:
    HAS_DRAMATIQ = False

from asf.medical.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class PersistentTaskStorage:
    """
    Persistent storage for task status and results using Redis.

    This class provides methods for storing and retrieving task status and results
    in Redis, ensuring that task information persists across application restarts.
    """

    def __init__(self, redis_url: Optional[str] = None, prefix: str = "task:", ttl: int = 86400):
        """
        Initialize the persistent task storage.

        Args:
            redis_url: Redis URL (default: settings.REDIS_URL or "redis://localhost:6379/0")
            prefix: Key prefix for task storage (default: "task:")
            ttl: Default TTL for task results in seconds (default: 86400 = 24 hours)
        """
        self.redis_url = redis_url or settings.REDIS_URL or "redis://localhost:6379/0"
        self.prefix = prefix
        self.ttl = ttl
        self.redis_client = None
        self._connect()

    def _connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.Redis.from_url(self.redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            self.redis_client = None

    def _get_key(self, task_id: str) -> str:
        """
        Get the Redis key for a task.

        Args:
            task_id: Task ID

        Returns:
            Redis key
        """
        return f"{self.prefix}{task_id}"

    def set_task_status(self, task_id: str, status: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set the status of a task.

        Args:
            task_id: Task ID
            status: Task status dictionary
            ttl: TTL in seconds (default: self.ttl)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis client not available, task status not persisted")
            return False

        try:
            # Add timestamp to status
            status["updated_at"] = time.time()

            # Serialize status to JSON
            status_json = json.dumps(status)

            # Set in Redis with TTL
            key = self._get_key(task_id)
            if ttl is None:
                ttl = self.ttl

            self.redis_client.setex(key, ttl, status_json)
            logger.debug(f"Task status set in Redis: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error setting task status in Redis: {str(e)}")
            return False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task status dictionary or None if not found
        """
        if not self.redis_client:
            logger.warning("Redis client not available, cannot get task status")
            return None

        try:
            # Get from Redis
            key = self._get_key(task_id)
            status_json = self.redis_client.get(key)

            if status_json:
                # Deserialize from JSON
                status = json.loads(status_json)
                logger.debug(f"Task status retrieved from Redis: {task_id}")
                return status

            logger.debug(f"Task status not found in Redis: {task_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting task status from Redis: {str(e)}")
            return None

    def delete_task_status(self, task_id: str) -> bool:
        """
        Delete the status of a task.

        Args:
            task_id: Task ID

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis client not available, cannot delete task status")
            return False

        try:
            # Delete from Redis
            key = self._get_key(task_id)
            self.redis_client.delete(key)
            logger.debug(f"Task status deleted from Redis: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting task status from Redis: {str(e)}")
            return False

    def list_tasks(self, pattern: str = "*", limit: int = 100) -> List[Dict[str, Any]]:
        """
        List tasks matching a pattern.

        Args:
            pattern: Pattern to match (default: "*")
            limit: Maximum number of tasks to return (default: 100)

        Returns:
            List of task status dictionaries
        """
        if not self.redis_client:
            logger.warning("Redis client not available, cannot list tasks")
            return []

        try:
            # Get keys matching pattern
            key_pattern = f"{self.prefix}{pattern}"
            keys = self.redis_client.keys(key_pattern)

            # Limit number of keys
            if limit > 0:
                keys = keys[:limit]

            # Get task status for each key
            tasks = []
            for key in keys:
                status_json = self.redis_client.get(key)
                if status_json:
                    status = json.loads(status_json)
                    # Add task_id to status
                    task_id = key[len(self.prefix):]
                    status["task_id"] = task_id
                    tasks.append(status)

            logger.debug(f"Listed {len(tasks)} tasks from Redis")
            return tasks
        except Exception as e:
            logger.error(f"Error listing tasks from Redis: {str(e)}")
            return []

    def update_task_progress(self, task_id: str, progress: int, **kwargs) -> bool:
        """
        Update the progress of a task.

        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            **kwargs: Additional status fields to update

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis client not available, task progress not persisted")
            return False

        try:
            # Get current status
            key = self._get_key(task_id)
            status_json = self.redis_client.get(key)

            if status_json:
                # Deserialize from JSON
                status = json.loads(status_json)

                # Update progress and additional fields
                status["progress"] = progress
                status.update(kwargs)

                # Add timestamp
                status["updated_at"] = time.time()

                # Serialize and save
                status_json = json.dumps(status)
                ttl = self.redis_client.ttl(key)
                if ttl < 0:
                    ttl = self.ttl

                self.redis_client.setex(key, ttl, status_json)
                logger.debug(f"Task progress updated in Redis: {task_id} ({progress}%)")
                return True

            logger.debug(f"Task not found for progress update: {task_id}")
            return False
        except Exception as e:
            logger.error(f"Error updating task progress in Redis: {str(e)}")
            return False

    def complete_task(self, task_id: str, result: Any, ttl: Optional[int] = None) -> bool:
        """
        Mark a task as completed with result.

        Args:
            task_id: Task ID
            result: Task result
            ttl: TTL in seconds (default: self.ttl)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis client not available, task completion not persisted")
            return False

        try:
            # Get current status
            key = self._get_key(task_id)
            status_json = self.redis_client.get(key)

            if status_json:
                # Deserialize from JSON
                status = json.loads(status_json)
            else:
                # Create new status
                status = {}

            # Update status
            status["status"] = "completed"
            status["progress"] = 100
            status["result"] = result
            status["completed_at"] = time.time()
            status["updated_at"] = time.time()

            # Serialize and save
            status_json = json.dumps(status)
            if ttl is None:
                ttl = self.ttl

            self.redis_client.setex(key, ttl, status_json)
            logger.debug(f"Task marked as completed in Redis: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error marking task as completed in Redis: {str(e)}")
            return False

    def fail_task(self, task_id: str, error: str, ttl: Optional[int] = None) -> bool:
        """
        Mark a task as failed with error.

        Args:
            task_id: Task ID
            error: Error message
            ttl: TTL in seconds (default: self.ttl)

        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            logger.warning("Redis client not available, task failure not persisted")
            return False

        try:
            # Get current status
            key = self._get_key(task_id)
            status_json = self.redis_client.get(key)

            if status_json:
                # Deserialize from JSON
                status = json.loads(status_json)
            else:
                # Create new status
                status = {}

            # Update status
            status["status"] = "failed"
            status["error"] = error
            status["failed_at"] = time.time()
            status["updated_at"] = time.time()

            # Serialize and save
            status_json = json.dumps(status)
            if ttl is None:
                ttl = self.ttl

            self.redis_client.setex(key, ttl, status_json)
            logger.debug(f"Task marked as failed in Redis: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Error marking task as failed in Redis: {str(e)}")
            return False

    def cleanup_old_tasks(self, max_age: int = 86400 * 7) -> int:
        """
        Clean up old tasks.

        Args:
            max_age: Maximum age in seconds (default: 7 days)

        Returns:
            Number of tasks cleaned up
        """
        if not self.redis_client:
            logger.warning("Redis client not available, cannot clean up old tasks")
            return 0

        try:
            # Get all task keys
            key_pattern = f"{self.prefix}*"
            keys = self.redis_client.keys(key_pattern)

            # Get current time
            current_time = time.time()

            # Check each task
            cleaned_up = 0
            for key in keys:
                status_json = self.redis_client.get(key)
                if status_json:
                    status = json.loads(status_json)
                    updated_at = status.get("updated_at", 0)

                    # Check if task is old
                    if current_time - updated_at > max_age:
                        # Delete task
                        self.redis_client.delete(key)
                        cleaned_up += 1

            logger.info(f"Cleaned up {cleaned_up} old tasks")
            return cleaned_up
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {str(e)}")
            return 0

# Create a singleton instance
task_storage = PersistentTaskStorage()


class TaskStorageMiddleware(Middleware):
    """
    Dramatiq middleware for storing task status and results in Redis.

    This middleware integrates with PersistentTaskStorage to store task status
    and results in Redis, ensuring that task information persists across application
    restarts.
    """

    def __init__(self, task_storage=None):
        """
        Initialize the middleware.

        Args:
            task_storage: PersistentTaskStorage instance (default: global task_storage)
        """
        self.task_storage = task_storage or globals()["task_storage"]

    def before_process_message(self, broker, message):
        """
        Called before a message is processed.

        Args:
            broker: Dramatiq broker
            message: Dramatiq message
        """
        if not HAS_DRAMATIQ:
            return

        # Store task status
        task_id = message.message_id
        self.task_storage.set_task_status(task_id, {
            "status": "processing",
            "queue": message.queue_name,
            "actor": message.actor_name,
            "args": message.args,
            "kwargs": message.kwargs,
            "options": message.options,
            "started_at": time.time(),
        })

    def after_process_message(self, broker, message, *, result=None, exception=None):
        """
        Called after a message is processed.

        Args:
            broker: Dramatiq broker
            message: Dramatiq message
            result: Result of the message processing
            exception: Exception raised during message processing
        """
        if not HAS_DRAMATIQ:
            return

        # Get task ID
        task_id = message.message_id

        # Update task status
        if exception is None:
            # Task completed successfully
            self.task_storage.complete_task(task_id, result)
        else:
            # Task failed
            self.task_storage.fail_task(task_id, str(exception))


def with_task_storage(task_id=None):
    """
    Decorator for storing task status and results in Redis.

    This decorator can be used with any function, not just Dramatiq actors.
    It stores the task status and result in Redis, ensuring that task information
    persists across application restarts.

    Args:
        task_id: Task ID (default: auto-generated UUID)

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate task ID if not provided
            nonlocal task_id
            if task_id is None:
                task_id = str(uuid.uuid4())

            # Store task status
            task_storage.set_task_status(task_id, {
                "status": "processing",
                "function": func.__name__,
                "args": args,
                "kwargs": kwargs,
                "started_at": time.time(),
            })

            try:
                # Call function
                result = func(*args, **kwargs)

                # Store task result
                task_storage.complete_task(task_id, result)

                return result
            except Exception as e:
                # Store task error
                task_storage.fail_task(task_id, str(e))

                # Re-raise exception
                raise

        # Add task_id attribute to wrapper function
        wrapper.task_id = task_id

        return wrapper

    return decorator


# Export classes and functions
__all__ = ["task_storage", "PersistentTaskStorage", "TaskStorageMiddleware", "with_task_storage"]
