"""
Persistent Task Storage
This module provides persistent storage for task status and results using Redis.
It includes integration with Dramatiq for tracking task status and results.
"""
import logging
from typing import Optional, List
import redis
try:
    import dramatiq
    from dramatiq.middleware import Middleware
    HAS_DRAMATIQ = True
except ImportError:
    HAS_DRAMATIQ = False
from asf.medical.core.config import settings
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
        """Connect to Redis.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
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
        Set the status of a task.
        Args:
            task_id: Task ID
            status: Task status dictionary
            ttl: TTL in seconds (default: self.ttl)
        Returns:
            True if successful, False otherwise
        Get the status of a task.
        Args:
            task_id: Task ID
        Returns:
            Task status dictionary or None if not found
        Delete the status of a task.
        Args:
            task_id: Task ID
        Returns:
            True if successful, False otherwise
        List tasks matching a pattern.
        Args:
            pattern: Pattern to match (default: "*")
            limit: Maximum number of tasks to return (default: 100)
        Returns:
            List of task status dictionaries
        Update the progress of a task.
        Args:
            task_id: Task ID
            progress: Progress percentage (0-100)
            **kwargs: Additional status fields to update
        Returns:
            True if successful, False otherwise
        Mark a task as completed with result.
        Args:
            task_id: Task ID
            result: Task result
            ttl: TTL in seconds (default: self.ttl)
        Returns:
            True if successful, False otherwise
        Mark a task as failed with error.
        Args:
            task_id: Task ID
            error: Error message
            ttl: TTL in seconds (default: self.ttl)
        Returns:
            True if successful, False otherwise
        Clean up old tasks.
        Args:
            max_age: Maximum age in seconds (default: 7 days)
        Returns:
            Number of tasks cleaned up
    Dramatiq middleware for storing task status and results in Redis.
    This middleware integrates with PersistentTaskStorage to store task status
    and results in Redis, ensuring that task information persists across application
    restarts.
        Initialize the middleware.
        Args:
            task_storage: PersistentTaskStorage instance (default: global task_storage)
        Called before a message is processed.
        Args:
            broker: Dramatiq broker
            message: Dramatiq message
        Called after a message is processed.
        Args:
            broker: Dramatiq broker
            message: Dramatiq message
            result: Result of the message processing
            exception: Exception raised during message processing
    Decorator for storing task status and results in Redis.
    This decorator can be used with any function, not just Dramatiq actors.
    It stores the task status and result in Redis, ensuring that task information
    persists across application restarts.
    Args:
        task_id: Task ID (default: auto-generated UUID)
    Returns:
        Decorated function