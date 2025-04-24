"""
Connection pooling for LLM API clients.

This module provides a connection pooling implementation for various LLM provider clients
to efficiently manage and reuse connections to external LLM services.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Generic, Union, List
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Type variable for generic client type
T = TypeVar('T')

class LLMConnectionPool(Generic[T]):
    """
    A generic connection pool for LLM provider clients.
    
    This class implements a connection pooling pattern for LLM API clients to:
    - Reduce connection overhead by reusing existing connections
    - Manage connection lifecycle (creation, validation, recycling)
    - Handle connection limits and backpressure
    - Provide monitoring capabilities for connection health
    
    Generic type T represents the client type (e.g., AsyncOpenAI, AnthropicClient)
    """
    
    def __init__(
        self,
        factory: Callable[[], T],
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: float = 300.0,  # 5 minutes
        connection_timeout: float = 10.0,
        connection_validation: Optional[Callable[[T], bool]] = None,
        name: str = "llm_connection_pool",
    ):
        """
        Initialize a new LLM connection pool.
        
        Args:
            factory: Function that creates new client instances
            max_size: Maximum number of connections in the pool
            min_size: Minimum number of connections to maintain
            max_idle_time: Maximum time in seconds a connection can be idle before recycling
            connection_timeout: Timeout in seconds for acquiring a connection
            connection_validation: Optional function to validate connections before use
            name: Name identifier for this connection pool (for logging)
        """
        self.factory = factory
        self.max_size = max_size
        self.min_size = min_size
        self.max_idle_time = max_idle_time
        self.connection_timeout = connection_timeout
        self.validation_fn = connection_validation
        self.name = name
        
        # Pool state
        self._available: List[Dict[str, Any]] = []  # Available connections with metadata
        self._in_use: Dict[int, Dict[str, Any]] = {}  # Connections currently in use
        self._lock = asyncio.Lock()
        self._initialized = False
        self._closed = False
        
        # Metrics
        self.created_count = 0
        self.recycled_count = 0
        self.acquisition_count = 0
        self.peak_connections = 0
        
        logger.info(f"Initialized {self.name} connection pool (max={max_size}, min={min_size})")
    
    async def initialize(self) -> None:
        """Initialize the connection pool with minimum connections."""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            logger.info(f"Initializing {self.name} connection pool with {self.min_size} connections")
            
            # Create minimum number of connections
            for _ in range(self.min_size):
                conn_data = await self._create_connection()
                self._available.append(conn_data)
            
            self._initialized = True
    
    async def _create_connection(self) -> Dict[str, Any]:
        """Create a new connection and return it with metadata."""
        client = self.factory()
        self.created_count += 1
        
        # Update peak connections metric
        current_total = len(self._available) + len(self._in_use) + 1
        self.peak_connections = max(self.peak_connections, current_total)
        
        return {
            "client": client,
            "created_at": time.time(),
            "last_used_at": time.time(),
            "use_count": 0,
        }
    
    async def _get_connection(self) -> Dict[str, Any]:
        """Get an available connection or create a new one if needed."""
        async with self._lock:
            # Check if we have an available connection
            if self._available:
                conn_data = self._available.pop(0)
                
                # Check if connection is too old (idle for too long)
                idle_time = time.time() - conn_data["last_used_at"]
                if idle_time > self.max_idle_time:
                    logger.debug(f"{self.name}: Recycling idle connection (idle for {idle_time:.1f}s)")
                    self.recycled_count += 1
                    # Create a new connection instead
                    conn_data = await self._create_connection()
                
                # Validate connection if validation function exists
                if self.validation_fn and not self.validation_fn(conn_data["client"]):
                    logger.debug(f"{self.name}: Connection validation failed, creating new connection")
                    self.recycled_count += 1
                    conn_data = await self._create_connection()
                    
            # No available connections, create a new one if allowed
            elif len(self._in_use) < self.max_size:
                conn_data = await self._create_connection()
                
            # Otherwise, we've reached the connection limit
            else:
                raise RuntimeError(
                    f"{self.name}: Connection pool exhausted - {len(self._in_use)}/{self.max_size} connections in use"
                )
            
            # Update connection metadata
            conn_data["last_used_at"] = time.time()
            conn_data["use_count"] += 1
            
            # Add to in_use dict with object id as key
            conn_id = id(conn_data["client"])
            self._in_use[conn_id] = conn_data
            
            self.acquisition_count += 1
            logger.debug(
                f"{self.name}: Acquired connection ({len(self._in_use)}/{self.max_size} in use, "
                f"{len(self._available)} available)"
            )
            
            return conn_data
    
    async def _release_connection(self, client: T) -> None:
        """Release a connection back to the pool."""
        conn_id = id(client)
        
        async with self._lock:
            if conn_id not in self._in_use:
                logger.warning(f"{self.name}: Attempted to release a connection not in the in_use pool")
                return
                
            conn_data = self._in_use.pop(conn_id)
            
            # Update last used timestamp
            conn_data["last_used_at"] = time.time()
            
            # Add back to available pool if not closed
            if not self._closed:
                self._available.append(conn_data)
                
            logger.debug(
                f"{self.name}: Released connection ({len(self._in_use)}/{self.max_size} in use, "
                f"{len(self._available)} available)"
            )
    
    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a connection from the pool.
        
        Usage:
        ```
        async with connection_pool.acquire() as client:
            result = await client.some_operation()
        ```
        
        Returns:
            Context manager yielding a connection client
            
        Raises:
            asyncio.TimeoutError: If connection can't be acquired within timeout
            RuntimeError: If pool is exhausted or closed
        """
        if self._closed:
            raise RuntimeError(f"{self.name}: Cannot acquire from closed connection pool")
            
        if not self._initialized:
            await self.initialize()
        
        # Try to get a connection with timeout
        try:
            conn_data = await asyncio.wait_for(
                self._get_connection(),
                timeout=self.connection_timeout
            )
            client = conn_data["client"]
            
            try:
                yield client
            finally:
                # Release the connection back to the pool
                await self._release_connection(client)
                
        except asyncio.TimeoutError:
            logger.error(f"{self.name}: Timed out after {self.connection_timeout}s waiting for connection")
            raise
    
    async def close(self) -> None:
        """Close all connections in the pool."""
        if self._closed:
            return
            
        logger.info(f"Closing {self.name} connection pool")
        
        async with self._lock:
            self._closed = True
            
            # Close each connection properly if they have a close/cleanup method
            for conn_data in self._available:
                client = conn_data["client"]
                try:
                    # Try different close methods as implementations may vary
                    if hasattr(client, "close") and callable(client.close):
                        if asyncio.iscoroutinefunction(client.close):
                            await client.close()
                        else:
                            client.close()
                    elif hasattr(client, "cleanup") and callable(client.cleanup):
                        if asyncio.iscoroutinefunction(client.cleanup):
                            await client.cleanup()
                        else:
                            client.cleanup()
                except Exception as e:
                    logger.warning(f"{self.name}: Error closing connection: {e}", exc_info=True)
            
            # Clear pools
            self._available.clear()
            self._in_use.clear()
            
            logger.info(f"{self.name}: Connection pool closed. Stats: created={self.created_count}, "
                      f"recycled={self.recycled_count}, acquired={self.acquisition_count}, "
                      f"peak={self.peak_connections}")
    
    async def refresh(self) -> None:
        """
        Refresh all available connections in the pool.
        Useful for credential rotation or config changes.
        """
        if self._closed:
            return
        
        logger.info(f"Refreshing all available connections in {self.name} pool")
        
        async with self._lock:
            # Replace all available connections
            old_connections = self._available
            self._available = []
            
            # Create fresh connections to meet minimum size
            for _ in range(min(self.min_size, self.max_size - len(self._in_use))):
                conn_data = await self._create_connection()
                self._available.append(conn_data)
            
            # Close old connections
            for conn_data in old_connections:
                client = conn_data["client"]
                try:
                    if hasattr(client, "close") and callable(client.close):
                        if asyncio.iscoroutinefunction(client.close):
                            await client.close()
                        else:
                            client.close()
                    elif hasattr(client, "cleanup") and callable(client.cleanup):
                        if asyncio.iscoroutinefunction(client.cleanup):
                            await client.cleanup()
                        else:
                            client.cleanup()
                except Exception as e:
                    logger.warning(f"{self.name}: Error closing connection during refresh: {e}", exc_info=True)
            
            logger.info(f"{self.name}: Refreshed {len(old_connections)} connections, created {len(self._available)} new ones")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        return {
            "name": self.name,
            "available": len(self._available),
            "in_use": len(self._in_use),
            "max_size": self.max_size,
            "min_size": self.min_size,
            "created_total": self.created_count,
            "recycled_total": self.recycled_count,
            "acquisition_total": self.acquisition_count,
            "peak_connections": self.peak_connections,
            "is_initialized": self._initialized,
            "is_closed": self._closed,
        }