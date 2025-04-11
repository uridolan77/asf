"""
Enhanced Task Storage for the Medical Research Synthesizer.

This module provides a persistent storage for task results using Redis.
"""

import os
import time
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class EnhancedTaskStorage:
    """
    Enhanced storage for task results.
    
    This class provides a persistent storage for task results using Redis.
    It ensures that task results are available across multiple instances
    of the application.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the enhanced task storage.
        
        Returns:
            EnhancedTaskStorage: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(EnhancedTaskStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: int = 86400,  # 24 hours
        namespace: str = "asf:medical:tasks:"
    ):
        """
        Initialize the enhanced task storage.
        
        Args:
            redis_url: Redis URL for persistent storage (default: from env var REDIS_URL)
            ttl: Time to live in seconds (default: 86400 = 24 hours)
            namespace: Cache namespace prefix (default: "asf:medical:tasks:")
        """
        if self._initialized:
            return
        
        # Get Redis URL from environment variable if not provided
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        
        self.ttl = ttl
        self.namespace = namespace
        self.redis = None
        
        # Local storage (fallback if Redis is not available)
        self.local_storage = {}
        self.local_expiry = {}
        self.lock = asyncio.Lock()
        
        # Initialize Redis if URL is provided
        if self.redis_url:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(
                    self.redis_url,
                    decode_responses=True,  # Automatically decode responses to strings
                    socket_timeout=5.0,     # Socket timeout
                    socket_connect_timeout=5.0,  # Connection timeout
                    retry_on_timeout=True,  # Retry on timeout
                    health_check_interval=30  # Health check interval
                )
                logger.info(f"Redis task storage initialized: {self.redis_url}")
            except ImportError:
                logger.error("redis-py package not installed. Install with: pip install redis")
                logger.warning("Falling back to local task storage")
            except Exception as e:
                logger.error(f"Failed to initialize Redis task storage: {str(e)}")
                logger.warning("Falling back to local task storage")
        else:
            logger.warning("No Redis URL provided. Falling back to local task storage")
        
        self._initialized = True
        logger.info("Enhanced task storage initialized")
    
    async def set_task_result(
        self,
        task_id: str,
        result: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Set a task result.
        
        Args:
            task_id: Task ID
            result: Task result
            ttl: Time to live in seconds (default: self.ttl)
            metadata: Task metadata
            
        Returns:
            bool: True if the result was set, False otherwise
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}{task_id}"
        
        # Use default TTL if not provided
        ttl = ttl or self.ttl
        
        # Create task data
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
                # Serialize task data
                serialized = json.dumps(task_data)
                
                # Set in Redis with TTL
                await self.redis.set(
                    namespaced_key,
                    serialized,
                    ex=ttl
                )
                logger.debug(f"Set task result in Redis: {task_id}")
                return True
            except Exception as e:
                logger.error(f"Error setting task result in Redis: {str(e)}")
                logger.warning("Falling back to local task storage")
        
        # Fall back to local task storage
        async with self.lock:
            self.local_storage[namespaced_key] = task_data
            self.local_expiry[namespaced_key] = time.time() + ttl
            logger.debug(f"Set task result in local storage: {task_id}")
            return True
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict[str, Any]]: Task data or None if not found
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}{task_id}"
        
        # Try Redis first if available
        if self.redis:
            try:
                # Get from Redis
                serialized = await self.redis.get(namespaced_key)
                
                if serialized:
                    # Deserialize task data
                    task_data = json.loads(serialized)
                    logger.debug(f"Got task result from Redis: {task_id}")
                    return task_data
            except Exception as e:
                logger.error(f"Error getting task result from Redis: {str(e)}")
                logger.warning("Falling back to local task storage")
        
        # Fall back to local task storage
        async with self.lock:
            if namespaced_key in self.local_storage:
                # Check if expired
                if self.local_expiry.get(namespaced_key, 0) > time.time():
                    logger.debug(f"Got task result from local storage: {task_id}")
                    return self.local_storage[namespaced_key]
                else:
                    # Remove expired task
                    del self.local_storage[namespaced_key]
                    del self.local_expiry[namespaced_key]
        
        logger.debug(f"Task result not found: {task_id}")
        return None
    
    async def delete_task_result(self, task_id: str) -> bool:
        """
        Delete a task result.
        
        Args:
            task_id: Task ID
            
        Returns:
            bool: True if the result was deleted, False otherwise
        """
        # Apply namespace
        namespaced_key = f"{self.namespace}{task_id}"
        
        # Try Redis first if available
        redis_deleted = False
        if self.redis:
            try:
                # Delete from Redis
                redis_deleted = await self.redis.delete(namespaced_key) > 0
                if redis_deleted:
                    logger.debug(f"Deleted task result from Redis: {task_id}")
            except Exception as e:
                logger.error(f"Error deleting task result from Redis: {str(e)}")
        
        # Also delete from local storage
        local_deleted = False
        async with self.lock:
            if namespaced_key in self.local_storage:
                del self.local_storage[namespaced_key]
                if namespaced_key in self.local_expiry:
                    del self.local_expiry[namespaced_key]
                local_deleted = True
                logger.debug(f"Deleted task result from local storage: {task_id}")
        
        return redis_deleted or local_deleted
    
    async def list_tasks(self, pattern: str = "*") -> List[str]:
        """
        List task IDs matching a pattern.
        
        Args:
            pattern: Pattern to match (default: "*")
            
        Returns:
            List[str]: List of task IDs
        """
        # Apply namespace
        namespaced_pattern = f"{self.namespace}{pattern}"
        
        # Try Redis first if available
        if self.redis:
            try:
                # Get keys from Redis
                keys = await self.redis.keys(namespaced_pattern)
                
                # Remove namespace from keys
                task_ids = [key[len(self.namespace):] for key in keys]
                
                logger.debug(f"Listed {len(task_ids)} tasks from Redis")
                return task_ids
            except Exception as e:
                logger.error(f"Error listing tasks from Redis: {str(e)}")
                logger.warning("Falling back to local task storage")
        
        # Fall back to local task storage
        async with self.lock:
            # Get keys from local storage
            keys = [key for key in self.local_storage.keys() if key.startswith(namespaced_pattern.replace("*", ""))]
            
            # Remove namespace from keys
            task_ids = [key[len(self.namespace):] for key in keys]
            
            logger.debug(f"Listed {len(task_ids)} tasks from local storage")
            return task_ids
    
    async def clear_expired_tasks(self) -> int:
        """
        Clear expired tasks.
        
        Returns:
            int: Number of tasks cleared
        """
        # Try Redis first if available
        redis_cleared = 0
        if self.redis:
            try:
                # Redis automatically removes expired keys
                logger.debug("Redis automatically removes expired keys")
            except Exception as e:
                logger.error(f"Error clearing expired tasks from Redis: {str(e)}")
        
        # Clear expired tasks from local storage
        local_cleared = 0
        async with self.lock:
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
        
        return redis_cleared + local_cleared
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Task status
        """
        # Get task result
        task_data = await self.get_task_result(task_id)
        
        if task_data is None:
            return {
                "task_id": task_id,
                "status": "unknown",
                "created_at": None,
                "expires_at": None
            }
        
        # Extract status from metadata
        status = task_data.get("metadata", {}).get("status", "completed")
        
        return {
            "task_id": task_id,
            "status": status,
            "created_at": task_data.get("created_at"),
            "expires_at": task_data.get("expires_at"),
            "metadata": task_data.get("metadata", {})
        }
    
    async def update_task_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        message: Optional[str] = None
    ) -> bool:
        """
        Update the status of a task.
        
        Args:
            task_id: Task ID
            status: Task status
            progress: Task progress (0.0 to 1.0)
            message: Status message
            
        Returns:
            bool: True if the status was updated, False otherwise
        """
        # Get task result
        task_data = await self.get_task_result(task_id)
        
        if task_data is None:
            # Create new task data
            metadata = {
                "status": status
            }
            
            if progress is not None:
                metadata["progress"] = progress
            
            if message is not None:
                metadata["message"] = message
            
            # Set task result with metadata
            return await self.set_task_result(
                task_id=task_id,
                result=None,
                metadata=metadata
            )
        
        # Update metadata
        metadata = task_data.get("metadata", {})
        metadata["status"] = status
        
        if progress is not None:
            metadata["progress"] = progress
        
        if message is not None:
            metadata["message"] = message
        
        # Update task result
        return await self.set_task_result(
            task_id=task_id,
            result=task_data.get("result"),
            ttl=int(task_data.get("expires_at", time.time() + self.ttl) - time.time()),
            metadata=metadata
        )
    
    async def get_task_progress(self, task_id: str) -> Dict[str, Any]:
        """
        Get the progress of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Task progress
        """
        # Get task status
        status = await self.get_task_status(task_id)
        
        # Extract progress from metadata
        metadata = status.get("metadata", {})
        progress = metadata.get("progress", 0.0)
        message = metadata.get("message", "")
        
        return {
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "progress": progress,
            "message": message,
            "created_at": status.get("created_at"),
            "expires_at": status.get("expires_at")
        }
    
    async def wait_for_task(
        self,
        task_id: str,
        timeout: float = 300.0,
        poll_interval: float = 1.0
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID
            timeout: Timeout in seconds (default: 300.0)
            poll_interval: Poll interval in seconds (default: 1.0)
            
        Returns:
            Dict[str, Any]: Task result
            
        Raises:
            TimeoutError: If the task does not complete within the timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get task status
            status = await self.get_task_status(task_id)
            
            # Check if task is completed
            if status.get("status") in ["completed", "failed", "error"]:
                # Get task result
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
            
            # Wait for poll interval
            await asyncio.sleep(poll_interval)
        
        # Timeout
        raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

# Create a singleton instance
enhanced_task_storage = EnhancedTaskStorage()
