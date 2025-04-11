Unified Task Storage for the Medical Research Synthesizer.

This module provides a persistent storage for task results using Redis,
replacing the in-memory task_results dictionary in export_tasks.py.

import os
import time
import json
import logging
import asyncio

logger = logging.getLogger(__name__)

class UnifiedTaskStorage:
    Unified storage for task results.
    
    This class provides a persistent storage for task results using Redis,
    ensuring that task results are available across multiple instances
    of the application.
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """
        Create a singleton instance of the unified task storage.
        
        Returns:
            UnifiedTaskStorage: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(UnifiedTaskStorage, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                redis_url: Description of redis_url
                ttl: Description of ttl
                namespace: Description of namespace
            """
        ttl: int = 86400,  # 24 hours
        namespace: str = "asf:medical:tasks:"
    ):
        if getattr(self, "_initialized", False):
            return
        
        self.redis_url = redis_url or os.environ.get("REDIS_URL")
        
        self.ttl = ttl
        self.namespace = namespace
        self.redis = None
        
        self.local_storage = {}
        self.local_expiry = {}
        self.lock = asyncio.Lock()
        
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
        logger.info("Unified task storage initialized")
    
    async def set_task_result(
        self,
        task_id: str,
        result: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        namespaced_key = f"{self.namespace}{task_id}"
        
        ttl = ttl or self.ttl
        
        task_data = {
            "task_id": task_id,
            "result": result,
            "metadata": metadata or {},
            "created_at": time.time(),
            "expires_at": time.time() + ttl
        }
        
        if self.redis:
            try:
                serialized = json.dumps(task_data)
                
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
        
        async with self.lock:
            self.local_storage[namespaced_key] = task_data
            self.local_expiry[namespaced_key] = time.time() + ttl
            logger.debug(f"Set task result in local storage: {task_id}")
            return True
    
    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        namespaced_key = f"{self.namespace}{task_id}"
        
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
        
        async with self.lock:
            if namespaced_key in self.local_storage:
                if self.local_expiry.get(namespaced_key, 0) > time.time():
                    logger.debug(f"Got task result from local storage: {task_id}")
                    return self.local_storage[namespaced_key]
                else:
                    del self.local_storage[namespaced_key]
                    del self.local_expiry[namespaced_key]
        
        logger.debug(f"Task result not found: {task_id}")
        return None
    
    async def delete_task_result(self, task_id: str) -> bool:
        namespaced_key = f"{self.namespace}{task_id}"
        
        redis_deleted = False
        if self.redis:
            try:
                redis_deleted = await self.redis.delete(namespaced_key) > 0
                if redis_deleted:
                    logger.debug(f"Deleted task result from Redis: {task_id}")
            except Exception as e:
                logger.error(f"Error deleting task result from Redis: {str(e)}")
        
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
        namespaced_pattern = f"{self.namespace}{pattern}"
        
        if self.redis:
            try:
                keys = await self.redis.keys(namespaced_pattern)
                
                task_ids = [key[len(self.namespace):] for key in keys]
                
                logger.debug(f"Listed {len(task_ids)} tasks from Redis")
                return task_ids
            except Exception as e:
                logger.error(f"Error listing tasks from Redis: {str(e)}")
                logger.warning("Falling back to local task storage")
        
        async with self.lock:
            keys = [key for key in self.local_storage.keys() if key.startswith(namespaced_pattern.replace("*", ""))]
            
            task_ids = [key[len(self.namespace):] for key in keys]
            
            logger.debug(f"Listed {len(task_ids)} tasks from local storage")
            return task_ids
    
    async def clear_expired_tasks(self) -> int:
        redis_cleared = 0
        if self.redis:
            try:
                logger.debug("Redis automatically removes expired keys")
            except Exception as e:
                logger.error(f"Error clearing expired tasks from Redis: {str(e)}")
        
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
        task_data = await self.get_task_result(task_id)
        
        if task_data is None:
            return {
                "task_id": task_id,
                "status": "unknown",
                "created_at": None,
                "expires_at": None
            }
        
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
        task_data = await self.get_task_result(task_id)
        
        if task_data is None:
            metadata = {
                "status": status
            }
            
            if progress is not None:
                metadata["progress"] = progress
            
            if message is not None:
                metadata["message"] = message
            
            return await self.set_task_result(
                task_id=task_id,
                result=None,
                metadata=metadata
            )
        
        metadata = task_data.get("metadata", {})
        metadata["status"] = status
        
        if progress is not None:
            metadata["progress"] = progress
        
        if message is not None:
            metadata["message"] = message
        
        return await self.set_task_result(
            task_id=task_id,
            result=task_data.get("result"),
            ttl=int(task_data.get("expires_at", time.time() + self.ttl) - time.time()),
            metadata=metadata
        )
    
    def get_task_result_sync(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a task result synchronously.
        
        This method is provided for compatibility with synchronous code.
        It should be used only when async/await cannot be used.
        
        Args:
            task_id: Task ID
            
        Returns:
            Optional[Dict[str, Any]]: Task data or None if not found
        """
        namespaced_key = f"{self.namespace}{task_id}"
        
        if self.redis:
            try:
                import redis
                sync_redis = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True
                )
                
                serialized = sync_redis.get(namespaced_key)
                
                if serialized:
                    task_data = json.loads(serialized)
                    logger.debug(f"Got task result from Redis (sync): {task_id}")
                    return task_data
            except Exception as e:
                logger.error(f"Error getting task result from Redis (sync): {str(e)}")
                logger.warning("Falling back to local task storage")
        
        if namespaced_key in self.local_storage:
            if self.local_expiry.get(namespaced_key, 0) > time.time():
                logger.debug(f"Got task result from local storage (sync): {task_id}")
                return self.local_storage[namespaced_key]
        
        logger.debug(f"Task result not found (sync): {task_id}")
        return None
    
    def set_task_result_sync(
        self,
        task_id: str,
            """
            set_task_result_sync function.
            
            This function provides functionality for...
            Args:
                task_id: Description of task_id
                result: Description of result
                ttl: Description of ttl
                metadata: Description of metadata
            
            Returns:
                Description of return value
            """
        result: Any,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        namespaced_key = f"{self.namespace}{task_id}"
        
        ttl = ttl or self.ttl
        
        task_data = {
            "task_id": task_id,
            "result": result,
            "metadata": metadata or {},
            "created_at": time.time(),
            "expires_at": time.time() + ttl
        }
        
        if self.redis:
            try:
                import redis
                sync_redis = redis.from_url(
                    self.redis_url,
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True
                )
                
                serialized = json.dumps(task_data)
                
                sync_redis.set(
                    namespaced_key,
                    serialized,
                    ex=ttl
                )
                logger.debug(f"Set task result in Redis (sync): {task_id}")
                return True
            except Exception as e:
                logger.error(f"Error setting task result in Redis (sync): {str(e)}")
                logger.warning("Falling back to local task storage")
        
        self.local_storage[namespaced_key] = task_data
        self.local_expiry[namespaced_key] = time.time() + ttl
        logger.debug(f"Set task result in local storage (sync): {task_id}")
        return True

unified_task_storage = UnifiedTaskStorage()
