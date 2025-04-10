"""
Task Queue Configuration

This module configures Dramatiq for asynchronous task processing.
"""

import os
import logging
import json
from typing import Optional, Dict, Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import Middleware, AgeLimit, TimeLimit, Retries

from asf.medical.core.config import settings
from asf.medical.core.persistent_task_storage import task_storage

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Redis broker
redis_url = settings.REDIS_URL or "redis://localhost:6379/0"
broker = RedisBroker(url=redis_url)

# Add middleware for task management
broker.add_middleware(AgeLimit(max_age=600000))  # 10 minutes
broker.add_middleware(TimeLimit(time_limit=300000))  # 5 minutes
broker.add_middleware(Retries(max_retries=3, min_backoff=1000, max_backoff=60000))

# Set as the default broker
dramatiq.set_broker(broker)

# Custom middleware for task tracking
class TaskTrackingMiddleware(Middleware):
    """Middleware to track task status and results."""

    def __init__(self):
        self.tasks = {}  # In-memory cache for fast access

    def after_process_message(self, broker, message, *, result=None, exception=None):
        """Store task result or exception after processing."""
        task_id = message.message_id

        if exception is None:
            # Task completed successfully
            status = {
                "status": "completed",
                "result": result
            }

            # Store in memory
            self.tasks[task_id] = status

            # Store in Redis
            if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                # For simple types, store directly
                task_storage.complete_task(task_id, result)
            else:
                # For complex types, store a success message
                task_storage.complete_task(task_id, {"message": "Task completed successfully"})
        else:
            # Task failed
            status = {
                "status": "failed",
                "error": str(exception)
            }

            # Store in memory
            self.tasks[task_id] = status

            # Store in Redis
            task_storage.fail_task(task_id, str(exception))

    def before_process_message(self, broker, message):
        """Mark task as processing before execution."""
        task_id = message.message_id

        # Extract args and kwargs
        args = message.args
        kwargs = message.kwargs

        # Create status
        status = {
            "status": "processing",
            "message": args,
            "kwargs": kwargs
        }

        # Store in memory
        self.tasks[task_id] = status

        # Store in Redis
        task_storage.set_task_status(task_id, status)

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task by ID."""
        # Try in-memory cache first
        if task_id in self.tasks:
            return self.tasks.get(task_id)

        # Try Redis
        status = task_storage.get_task_status(task_id)
        if status:
            # Update in-memory cache
            self.tasks[task_id] = status
            return status

        return None

# Add task tracking middleware
task_tracker = TaskTrackingMiddleware()
broker.add_middleware(task_tracker)

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the status of a task by ID.

    Args:
        task_id: The ID of the task

    Returns:
        Task status information or None if not found
    """
    return task_tracker.get_task_status(task_id)
