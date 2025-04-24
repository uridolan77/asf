"""
Task Storage module for the Medical Research Synthesizer.

This module provides a unified storage system for task results and metadata,
with support for both synchronous and asynchronous operations, Redis-based
distributed storage, and local in-memory fallback.

Classes:
    TaskStorageError: Base exception for task storage errors.
    TaskNotFoundError: Exception raised when a task is not found.
    TaskStorage: Unified storage for task results and metadata.

Functions:
    get_task_storage: Get the singleton instance of TaskStorage.
"""

import os
import time
import json
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional, Union
from functools import wraps

logger = logging.getLogger(__name__)

class TaskStorageError(Exception):
    """
    Base exception for task storage errors.

    Attributes:
        message (str): The error message.
        details (Dict[str, Any]): Additional details about the error.
    """

    def __init__(self, message: str, details: Dict[str, Any] = None):
        """
        Initialize the TaskStorageError.

        Args:
            message (str): The error message.
            details (Dict[str, Any], optional): Additional details about the error. Defaults to None.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)

class TaskNotFoundError(TaskStorageError):
    """
    Exception raised when a task is not found.

    Attributes:
        task_id (str): The ID of the task that was not found.
    """

    def __init__(self, task_id: str):
        """
        Initialize the TaskNotFoundError.

        Args:
            task_id (str): The ID of the task that was not found.
        """
        message = f"Task with ID {task_id} not found"
        details = {"task_id": task_id}
        super().__init__(message, details)

class TaskStorage:
    """
    Unified storage for task results and metadata.

    This class provides a persistent storage mechanism for task results and metadata,
    with support for both Redis-based distributed storage and local in-memory fallback.
    It offers both synchronous and asynchronous APIs for flexibility in different contexts.

    Attributes:
        redis_url (str): Redis URL for connecting to the Redis server.
        ttl (int): Default time-to-live for task results in seconds.
        namespace (str): Namespace prefix for task storage keys.
        redis (Optional[Any]): Redis async client instance.
        sync_redis (Optional[Any]): Redis synchronous client instance.
        local_storage (Dict): Local in-memory storage for task results.
        local_expiry (Dict): Expiry times for local storage keys.
        async_lock (asyncio.Lock): Lock for thread-safe async local storage access.
        sync_lock (threading.RLock): Lock for thread-safe sync local storage access.
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the task storage.

        Returns:
            TaskStorage: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(TaskStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: int = 86400,
        namespace: str = "asf:medical:tasks:"
    ):
        """
        Initialize the TaskStorage instance.

        Args:
            redis_url (Optional[str], optional): Redis URL for connecting to the Redis server. 
                Defaults to None (from environment).
            ttl (int, optional): Default time-to-live for task results in seconds. Defaults to 86400 (24 hours).
            namespace (str, optional): Namespace prefix for task storage keys. Defaults to "asf:medical:tasks:".
        """
        if self._initialized:
            return

        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        self.ttl = ttl
        self.namespace = namespace
        self.redis = None
        self.sync_redis = None

        # Local storage for async and sync operations
        self.local_storage = {}
        self.local_expiry = {}
        self.async_lock = asyncio.Lock()
        self.sync_lock = threading.RLock()

        # Initialize Redis clients if Redis URL is provided
        if self.redis_url:
            self._initialize_redis()
        else:
            logger.warning("No Redis URL provided. Using local storage only.")

        self._initialized = True
        logger.info(f"Task storage initialized with namespace: {namespace}")

    def _initialize_redis(self):
        """
        Initialize Redis clients for both async and sync operations.
        """
        # Initialize async Redis client
        try:
            import redis.asyncio as aioredis
            self.redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True,
                health_check_interval=30
            )
            logger.info(f"Async Redis task storage initialized: {self.redis_url}")
        except ImportError:
            logger.error("redis-py package not installed for async operations. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to initialize async Redis task storage: {str(e)}")

        # Initialize sync Redis client
        try:
            import redis
            self.sync_redis = redis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                retry_on_timeout=True
            )
            logger.info(f"Sync Redis task storage initialized: {self.redis_url}")
        except ImportError:
            logger.error("redis-py package not installed for sync operations. Install with: pip install redis")
        except Exception as e:
            logger.error(f"Failed to initialize sync Redis task storage: {str(e)}")

    def _get_namespaced_key(self, task_id: str) -> str:
        """
        Get the namespaced storage key for a task ID.

        Args:
            task_id (str): Task ID.

        Returns:
            str: Namespaced storage key.
        """
        return f"{self.namespace}{task_id}"

    #
    # Asynchronous API
    #

    async def set_task_result(
        self,
        task_id: str,
        result: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a task result asynchronously.

        Args:
            task_id (str): Unique identifier for the task.
            result (Any): The result of the task.
            ttl (Optional[int], optional): Time-to-live for the task result. Defaults to None.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the task. Defaults to None.

        Returns:
            bool: True if the task result was successfully stored, False otherwise.
        """
        namespaced_key = self._get_namespaced_key(task_id)
        ttl = ttl or self.ttl

        task_data = {
            "task_id": task_id,
            "result": result,
            "metadata": metadata or {},
            "created_at": time.time(),
            "expires_at": time.time() + ttl
        }

        # Try Redis first if available
        if self.redis:
            try:
                serialized = json.dumps(task_data)
                await self.redis.set(namespaced_key, serialized, ex=ttl)
                logger.debug(f"Set task result in Redis: {task_id}")
                return True
            except Exception as e:
                logger.error(f"Error setting task result in Redis: {str(e)}")
                logger.warning("Falling back to local task storage")

        # Local storage fallback
        async with self.async_lock:
            self.local_storage[namespaced_key] = task_data
            self.local_expiry[namespaced_key] = time.time() + ttl
            logger.debug(f"Set task result in local storage: {task_id}")
            return True

    async def get_task_result(self, task_id: str, raise_if_not_found: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task result asynchronously.

        Args:
            task_id (str): Unique identifier for the task.
            raise_if_not_found (bool, optional): Whether to raise an exception if the task is not found. 
                Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: The task result data, or None if not found and raise_if_not_found is False.

        Raises:
            TaskNotFoundError: If the task is not found and raise_if_not_found is True.
        """
        namespaced_key = self._get_namespaced_key(task_id)

        # Try Redis first if available
        if self.redis:
            try:
                serialized = await self.redis.get(namespaced_key)
                if serialized:
                    task_data = json.loads(serialized)
                    logger.debug(f"Got task result from Redis: {task_id}")
                    return task_data
            except Exception as e:
                logger.error(f"Error getting task result from Redis: {str(e)}")
                logger.warning("Falling back to local task storage")

        # Local storage fallback
        async with self.async_lock:
            if namespaced_key in self.local_storage:
                if self.local_expiry.get(namespaced_key, 0) > time.time():
                    logger.debug(f"Got task result from local storage: {task_id}")
                    return self.local_storage[namespaced_key]
                else:
                    # Remove expired task
                    del self.local_storage[namespaced_key]
                    del self.local_expiry[namespaced_key]

        logger.debug(f"Task result not found: {task_id}")
        if raise_if_not_found:
            raise TaskNotFoundError(task_id)
        return None

    async def delete_task_result(self, task_id: str) -> bool:
        """
        Delete a task result asynchronously.

        Args:
            task_id (str): Unique identifier for the task.

        Returns:
            bool: True if the task result was successfully deleted, False otherwise.
        """
        namespaced_key = self._get_namespaced_key(task_id)
        deleted = False

        # Try Redis first if available
        if self.redis:
            try:
                redis_deleted = await self.redis.delete(namespaced_key) > 0
                if redis_deleted:
                    logger.debug(f"Deleted task result from Redis: {task_id}")
                    deleted = True
            except Exception as e:
                logger.error(f"Error deleting task result from Redis: {str(e)}")

        # Local storage fallback (also delete from local even if Redis succeeded)
        async with self.async_lock:
            if namespaced_key in self.local_storage:
                del self.local_storage[namespaced_key]
                if namespaced_key in self.local_expiry:
                    del self.local_expiry[namespaced_key]
                logger.debug(f"Deleted task result from local storage: {task_id}")
                deleted = True

        return deleted

    async def list_tasks(self, pattern: str = "*", limit: int = 100) -> List[str]:
        """
        List tasks matching a pattern asynchronously.

        Args:
            pattern (str, optional): Pattern to match. Defaults to "*".
            limit (int, optional): Maximum number of tasks to return. Defaults to 100.

        Returns:
            List[str]: List of matching task IDs.
        """
        namespaced_pattern = f"{self.namespace}{pattern}"
        task_ids = []

        # Try Redis first if available
        if self.redis:
            try:
                keys = await self.redis.keys(namespaced_pattern)
                keys = keys[:limit]
                task_ids = [key[len(self.namespace):] for key in keys]
                logger.debug(f"Listed {len(task_ids)} tasks from Redis")
                return task_ids
            except Exception as e:
                logger.error(f"Error listing tasks from Redis: {str(e)}")
                logger.warning("Falling back to local task storage")

        # Local storage fallback
        async with self.async_lock:
            keys = [
                key for key in self.local_storage.keys()
                if key.startswith(namespaced_pattern.replace("*", ""))
            ]
            keys = keys[:limit]
            task_ids = [key[len(self.namespace):] for key in keys]
            logger.debug(f"Listed {len(task_ids)} tasks from local storage")

        return task_ids

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Update the status of a task asynchronously.

        Args:
            task_id (str): Unique identifier for the task.
            status (str): New status for the task.
            progress (Optional[float], optional): Progress percentage. Defaults to None.
            message (Optional[str], optional): Additional status message. Defaults to None.

        Returns:
            bool: True if the task status was successfully updated, False otherwise.
        """
        # Try to get existing task data
        try:
            task_data = await self.get_task_result(task_id)
        except TaskNotFoundError:
            # Create new task data if it doesn't exist
            metadata = {"status": status}
            if progress is not None:
                metadata["progress"] = progress
            if message is not None:
                metadata["message"] = message

            return await self.set_task_result(
                task_id=task_id,
                result=None,
                metadata=metadata
            )

        # Update existing task data
        if task_data is not None:
            metadata = task_data.get("metadata", {})
            metadata["status"] = status

            if progress is not None:
                metadata["progress"] = progress

            if message is not None:
                metadata["message"] = message

            time_left = max(
                int(task_data.get("expires_at", time.time() + self.ttl) - time.time()),
                60  # Minimum 60 seconds TTL
            )

            return await self.set_task_result(
                task_id=task_id,
                result=task_data.get("result"),
                ttl=time_left,
                metadata=metadata
            )
        
        # Fall back to creating a new task
        metadata = {"status": status}
        if progress is not None:
            metadata["progress"] = progress
        if message is not None:
            metadata["message"] = message

        return await self.set_task_result(
            task_id=task_id,
            result=None,
            metadata=metadata
        )

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task asynchronously.

        Args:
            task_id (str): Unique identifier for the task.

        Returns:
            Dict[str, Any]: Task status data.
        """
        try:
            task_data = await self.get_task_result(task_id)
        except TaskNotFoundError:
            return {
                "task_id": task_id,
                "status": "not_found",
                "created_at": None,
                "expires_at": None
            }

        if task_data is None:
            return {
                "task_id": task_id,
                "status": "unknown",
                "created_at": None,
                "expires_at": None
            }

        metadata = task_data.get("metadata", {})
        status = metadata.get("status", "completed")

        return {
            "task_id": task_id,
            "status": status,
            "progress": metadata.get("progress", 0),
            "message": metadata.get("message", ""),
            "created_at": task_data.get("created_at"),
            "expires_at": task_data.get("expires_at"),
            "metadata": metadata
        }

    async def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """
        Get the progress of a task asynchronously.

        Args:
            task_id (str): Unique identifier for the task.

        Returns:
            Dict[str, Any]: Task progress data.
        """
        status = await self.get_task_status(task_id)
        metadata = status.get("metadata", {})

        return {
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "progress": metadata.get("progress", 0),
            "message": metadata.get("message", ""),
            "created_at": status.get("created_at"),
            "expires_at": status.get("expires_at")
        }

    async def clear_expired_tasks(self) -> int:
        """
        Clear expired tasks from storage asynchronously.

        Returns:
            int: Number of expired tasks cleared.
        """
        # Redis automatically clears expired keys
        local_cleared = 0
        
        # Clear expired tasks from local storage
        async with self.async_lock:
            now = time.time()
            expired_keys = [
                key for key, expiry in self.local_expiry.items() 
                if expiry <= now
            ]
            
            for key in expired_keys:
                del self.local_storage[key]
                del self.local_expiry[key]
                local_cleared += 1
                
        if local_cleared > 0:
            logger.debug(f"Cleared {local_cleared} expired tasks from local storage")
            
        return local_cleared

    async def wait_for_task(
        self,
        task_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
        terminal_statuses: List[str] = None
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete asynchronously.

        Args:
            task_id (str): Unique identifier for the task.
            timeout (float, optional): Maximum time to wait in seconds. Defaults to 300.0.
            poll_interval (float, optional): Interval between status checks in seconds. Defaults to 1.0.
            terminal_statuses (List[str], optional): List of statuses that indicate completion. 
                Defaults to ["completed", "failed", "error"].

        Returns:
            Dict[str, Any]: Task result data.

        Raises:
            TimeoutError: If the task does not complete within the timeout period.
            TaskNotFoundError: If the task is not found.
        """
        if terminal_statuses is None:
            terminal_statuses = ["completed", "failed", "error"]

        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = await self.get_task_status(task_id)
            
            if status.get("status") in terminal_statuses:
                try:
                    task_data = await self.get_task_result(task_id)
                    
                    if task_data is None:
                        return {
                            "task_id": task_id,
                            "status": "unknown",
                            "result": None
                        }
                    
                    return {
                        "task_id": task_id,
                        "status": status.get("status"),
                        "result": task_data.get("result"),
                        "metadata": task_data.get("metadata", {})
                    }
                except TaskNotFoundError:
                    return {
                        "task_id": task_id,
                        "status": "not_found",
                        "result": None
                    }
            
            await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

    async def mark_task_completed(
        self,
        task_id: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Mark a task as completed asynchronously.

        Args:
            task_id (str): Unique identifier for the task.
            result (Any): Task result.
            ttl (Optional[int], optional): Time-to-live for the completed task. Defaults to None.

        Returns:
            bool: True if the task was successfully marked as completed, False otherwise.
        """
        # Try to get existing task data
        try:
            task_data = await self.get_task_result(task_id)
            metadata = task_data.get("metadata", {}) if task_data else {}
        except (TaskNotFoundError, Exception):
            metadata = {}

        # Update status
        metadata["status"] = "completed"
        metadata["progress"] = 100
        metadata["completed_at"] = time.time()

        return await self.set_task_result(
            task_id=task_id,
            result=result,
            ttl=ttl,
            metadata=metadata
        )

    async def mark_task_failed(
        self,
        task_id: str,
        error: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Mark a task as failed asynchronously.

        Args:
            task_id (str): Unique identifier for the task.
            error (str): Error message.
            ttl (Optional[int], optional): Time-to-live for the failed task. Defaults to None.

        Returns:
            bool: True if the task was successfully marked as failed, False otherwise.
        """
        # Try to get existing task data
        try:
            task_data = await self.get_task_result(task_id)
            metadata = task_data.get("metadata", {}) if task_data else {}
            result = task_data.get("result") if task_data else None
        except (TaskNotFoundError, Exception):
            metadata = {}
            result = None

        # Update status
        metadata["status"] = "failed"
        metadata["error"] = error
        metadata["failed_at"] = time.time()

        return await self.set_task_result(
            task_id=task_id,
            result=result,
            ttl=ttl,
            metadata=metadata
        )

    #
    # Synchronous API
    #

    def set_task_result_sync(
        self,
        task_id: str,
        result: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a task result synchronously.

        Args:
            task_id (str): Unique identifier for the task.
            result (Any): The result of the task.
            ttl (Optional[int], optional): Time-to-live for the task result. Defaults to None.
            metadata (Optional[Dict[str, Any]], optional): Additional metadata for the task. Defaults to None.

        Returns:
            bool: True if the task result was successfully stored, False otherwise.
        """
        namespaced_key = self._get_namespaced_key(task_id)
        ttl = ttl or self.ttl

        task_data = {
            "task_id": task_id,
            "result": result,
            "metadata": metadata or {},
            "created_at": time.time(),
            "expires_at": time.time() + ttl
        }

        # Try Redis first if available
        if self.sync_redis:
            try:
                serialized = json.dumps(task_data)
                self.sync_redis.set(namespaced_key, serialized, ex=ttl)
                logger.debug(f"Set task result in Redis (sync): {task_id}")
                return True
            except Exception as e:
                logger.error(f"Error setting task result in Redis (sync): {str(e)}")
                logger.warning("Falling back to local task storage")

        # Local storage fallback
        with self.sync_lock:
            self.local_storage[namespaced_key] = task_data
            self.local_expiry[namespaced_key] = time.time() + ttl
            logger.debug(f"Set task result in local storage (sync): {task_id}")
            return True

    def get_task_result_sync(
        self,
        task_id: str,
        raise_if_not_found: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a task result synchronously.

        Args:
            task_id (str): Unique identifier for the task.
            raise_if_not_found (bool, optional): Whether to raise an exception if the task is not found.
                Defaults to False.

        Returns:
            Optional[Dict[str, Any]]: The task result data, or None if not found and raise_if_not_found is False.

        Raises:
            TaskNotFoundError: If the task is not found and raise_if_not_found is True.
        """
        namespaced_key = self._get_namespaced_key(task_id)

        # Try Redis first if available
        if self.sync_redis:
            try:
                serialized = self.sync_redis.get(namespaced_key)
                if serialized:
                    task_data = json.loads(serialized)
                    logger.debug(f"Got task result from Redis (sync): {task_id}")
                    return task_data
            except Exception as e:
                logger.error(f"Error getting task result from Redis (sync): {str(e)}")
                logger.warning("Falling back to local task storage")

        # Local storage fallback
        with self.sync_lock:
            if namespaced_key in self.local_storage:
                if self.local_expiry.get(namespaced_key, 0) > time.time():
                    logger.debug(f"Got task result from local storage (sync): {task_id}")
                    return self.local_storage[namespaced_key]
                else:
                    # Remove expired task
                    del self.local_storage[namespaced_key]
                    del self.local_expiry[namespaced_key]

        logger.debug(f"Task result not found (sync): {task_id}")
        if raise_if_not_found:
            raise TaskNotFoundError(task_id)
        return None

    def delete_task_result_sync(self, task_id: str) -> bool:
        """
        Delete a task result synchronously.

        Args:
            task_id (str): Unique identifier for the task.

        Returns:
            bool: True if the task result was successfully deleted, False otherwise.
        """
        namespaced_key = self._get_namespaced_key(task_id)
        deleted = False

        # Try Redis first if available
        if self.sync_redis:
            try:
                redis_deleted = self.sync_redis.delete(namespaced_key) > 0
                if redis_deleted:
                    logger.debug(f"Deleted task result from Redis (sync): {task_id}")
                    deleted = True
            except Exception as e:
                logger.error(f"Error deleting task result from Redis (sync): {str(e)}")

        # Local storage fallback (also delete from local even if Redis succeeded)
        with self.sync_lock:
            if namespaced_key in self.local_storage:
                del self.local_storage[namespaced_key]
                if namespaced_key in self.local_expiry:
                    del self.local_expiry[namespaced_key]
                logger.debug(f"Deleted task result from local storage (sync): {task_id}")
                deleted = True

        return deleted

    def list_tasks_sync(self, pattern: str = "*", limit: int = 100) -> List[str]:
        """
        List tasks matching a pattern synchronously.

        Args:
            pattern (str, optional): Pattern to match. Defaults to "*".
            limit (int, optional): Maximum number of tasks to return. Defaults to 100.

        Returns:
            List[str]: List of matching task IDs.
        """
        namespaced_pattern = f"{self.namespace}{pattern}"
        task_ids = []

        # Try Redis first if available
        if self.sync_redis:
            try:
                keys = self.sync_redis.keys(namespaced_pattern)
                keys = keys[:limit]
                task_ids = [key[len(self.namespace):] for key in keys]
                logger.debug(f"Listed {len(task_ids)} tasks from Redis (sync)")
                return task_ids
            except Exception as e:
                logger.error(f"Error listing tasks from Redis (sync): {str(e)}")
                logger.warning("Falling back to local task storage")

        # Local storage fallback
        with self.sync_lock:
            keys = [
                key for key in self.local_storage.keys()
                if key.startswith(namespaced_pattern.replace("*", ""))
            ]
            keys = keys[:limit]
            task_ids = [key[len(self.namespace):] for key in keys]
            logger.debug(f"Listed {len(task_ids)} tasks from local storage (sync)")

        return task_ids

    def update_task_status_sync(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Update the status of a task synchronously.

        Args:
            task_id (str): Unique identifier for the task.
            status (str): New status for the task.
            progress (Optional[float], optional): Progress percentage. Defaults to None.
            message (Optional[str], optional): Additional status message. Defaults to None.

        Returns:
            bool: True if the task status was successfully updated, False otherwise.
        """
        # Try to get existing task data
        try:
            task_data = self.get_task_result_sync(task_id)
        except TaskNotFoundError:
            # Create new task data if it doesn't exist
            metadata = {"status": status}
            if progress is not None:
                metadata["progress"] = progress
            if message is not None:
                metadata["message"] = message

            return self.set_task_result_sync(
                task_id=task_id,
                result=None,
                metadata=metadata
            )

        # Update existing task data
        if task_data is not None:
            metadata = task_data.get("metadata", {})
            metadata["status"] = status

            if progress is not None:
                metadata["progress"] = progress

            if message is not None:
                metadata["message"] = message

            time_left = max(
                int(task_data.get("expires_at", time.time() + self.ttl) - time.time()),
                60  # Minimum 60 seconds TTL
            )

            return self.set_task_result_sync(
                task_id=task_id,
                result=task_data.get("result"),
                ttl=time_left,
                metadata=metadata
            )
        
        # Fall back to creating a new task
        metadata = {"status": status}
        if progress is not None:
            metadata["progress"] = progress
        if message is not None:
            metadata["message"] = message

        return self.set_task_result_sync(
            task_id=task_id,
            result=None,
            metadata=metadata
        )

    def get_task_status_sync(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task synchronously.

        Args:
            task_id (str): Unique identifier for the task.

        Returns:
            Dict[str, Any]: Task status data.
        """
        try:
            task_data = self.get_task_result_sync(task_id)
        except TaskNotFoundError:
            return {
                "task_id": task_id,
                "status": "not_found",
                "created_at": None,
                "expires_at": None
            }

        if task_data is None:
            return {
                "task_id": task_id,
                "status": "unknown",
                "created_at": None,
                "expires_at": None
            }

        metadata = task_data.get("metadata", {})
        status = metadata.get("status", "completed")

        return {
            "task_id": task_id,
            "status": status,
            "progress": metadata.get("progress", 0),
            "message": metadata.get("message", ""),
            "created_at": task_data.get("created_at"),
            "expires_at": task_data.get("expires_at"),
            "metadata": metadata
        }

    def mark_task_completed_sync(
        self,
        task_id: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Mark a task as completed synchronously.

        Args:
            task_id (str): Unique identifier for the task.
            result (Any): Task result.
            ttl (Optional[int], optional): Time-to-live for the completed task. Defaults to None.

        Returns:
            bool: True if the task was successfully marked as completed, False otherwise.
        """
        # Try to get existing task data
        try:
            task_data = self.get_task_result_sync(task_id)
            metadata = task_data.get("metadata", {}) if task_data else {}
        except (TaskNotFoundError, Exception):
            metadata = {}

        # Update status
        metadata["status"] = "completed"
        metadata["progress"] = 100
        metadata["completed_at"] = time.time()

        return self.set_task_result_sync(
            task_id=task_id,
            result=result,
            ttl=ttl,
            metadata=metadata
        )

    def mark_task_failed_sync(
        self,
        task_id: str,
        error: str,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Mark a task as failed synchronously.

        Args:
            task_id (str): Unique identifier for the task.
            error (str): Error message.
            ttl (Optional[int], optional): Time-to-live for the failed task. Defaults to None.

        Returns:
            bool: True if the task was successfully marked as failed, False otherwise.
        """
        # Try to get existing task data
        try:
            task_data = self.get_task_result_sync(task_id)
            metadata = task_data.get("metadata", {}) if task_data else {}
            result = task_data.get("result") if task_data else None
        except (TaskNotFoundError, Exception):
            metadata = {}
            result = None

        # Update status
        metadata["status"] = "failed"
        metadata["error"] = error
        metadata["failed_at"] = time.time()

        return self.set_task_result_sync(
            task_id=task_id,
            result=result,
            ttl=ttl,
            metadata=metadata
        )

    def cleanup_old_tasks_sync(self, max_age: int = 7 * 86400) -> int:
        """
        Clean up old tasks synchronously.

        Args:
            max_age (int, optional): Maximum age of tasks to keep in seconds. Defaults to 7 days.

        Returns:
            int: Number of tasks deleted.
        """
        now = time.time()
        deleted_count = 0

        # Get all tasks
        task_ids = self.list_tasks_sync()

        for task_id in task_ids:
            try:
                task_data = self.get_task_result_sync(task_id)
                
                if task_data and 'created_at' in task_data:
                    age = now - task_data['created_at']
                    
                    if age > max_age:
                        if self.delete_task_result_sync(task_id):
                            deleted_count += 1
            except Exception as e:
                logger.warning(f"Error processing task {task_id} during cleanup: {str(e)}")

        return deleted_count

    def clear_expired_tasks_sync(self) -> int:
        """
        Clear expired tasks from storage synchronously.

        Returns:
            int: Number of expired tasks cleared.
        """
        # Redis automatically clears expired keys
        local_cleared = 0
        
        # Clear expired tasks from local storage
        with self.sync_lock:
            now = time.time()
            expired_keys = [
                key for key, expiry in self.local_expiry.items() 
                if expiry <= now
            ]
            
            for key in expired_keys:
                del self.local_storage[key]
                del self.local_expiry[key]
                local_cleared += 1
                
        if local_cleared > 0:
            logger.debug(f"Cleared {local_cleared} expired tasks from local storage (sync)")
            
        return local_cleared

# Singleton instance of the task storage
_task_storage = None

def get_task_storage(**kwargs) -> TaskStorage:
    """
    Get the singleton instance of TaskStorage.

    Args:
        **kwargs: Arguments to pass to the TaskStorage constructor if it hasn't been initialized yet.

    Returns:
        TaskStorage: The singleton instance of TaskStorage.
    """
    global _task_storage
    if _task_storage is None:
        _task_storage = TaskStorage(**kwargs)
    return _task_storage

# Create a default task storage instance
task_storage = get_task_storage()

# Helper functions for easy access
async def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a task result.

    Args:
        task_id: The task ID.

    Returns:
        The task data or None if not found.
    """
    return await task_storage.get_task_result(task_id)

async def set_task(task_id: str, result: Any, **kwargs) -> bool:
    """
    Set a task result.

    Args:
        task_id: The task ID.
        result: The task result.
        **kwargs: Additional arguments to pass to set_task_result.

    Returns:
        True if successful.
    """
    return await task_storage.set_task_result(task_id, result, **kwargs)

async def update_task(task_id: str, status: str, **kwargs) -> bool:
    """
    Update a task status.

    Args:
        task_id: The task ID.
        status: The new task status.
        **kwargs: Additional arguments to pass to update_task_status.

    Returns:
        True if successful.
    """
    return await task_storage.update_task_status(task_id, status, **kwargs)