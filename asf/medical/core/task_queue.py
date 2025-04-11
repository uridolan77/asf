"""
Task Queue Configuration

This module configures Dramatiq for asynchronous task processing.
"""

import logging
from typing import Optional, Dict, Any

import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import Middleware, AgeLimit, TimeLimit, Retries

from asf.medical.core.config import settings
from asf.medical.core.persistent_task_storage import task_storage

logger = logging.getLogger(__name__)

redis_url = settings.REDIS_URL or "redis://localhost:6379/0"
broker = RedisBroker(url=redis_url)

broker.add_middleware(AgeLimit(max_age=600000))  # 10 minutes
broker.add_middleware(TimeLimit(time_limit=300000))  # 5 minutes
broker.add_middleware(Retries(max_retries=3, min_backoff=1000, max_backoff=60000))

dramatiq.set_broker(broker)

class TaskTrackingMiddleware(Middleware):
    """Middleware to track task status and results."""

    def __init__(self):
        self.tasks = {}  # In-memory cache for fast access

    def after_process_message(self, broker, message, *, result=None, exception=None):
        """Store task result or exception after processing.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
        if task_id in self.tasks:
            return self.tasks.get(task_id)

        status = task_storage.get_task_status(task_id)
        if status:
            self.tasks[task_id] = status
            return status

        return None

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
