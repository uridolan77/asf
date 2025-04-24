"""
Connection pool implementation for LLM providers.

This module provides a simple connection pool for managing connections to LLM
provider APIs, ensuring efficient reuse of connections and proper cleanup.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union, AsyncContextManager, AsyncIterator
from contextlib import asynccontextmanager

from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.medical.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type variable for client type

class ConnectionStats:
    """Statistics for a single connection."""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.created_at = datetime.now()
        self.last_used_at = datetime.now()
        self.request_count = 0
        self.error_count = 0
        self.success_count = 0
        self.total_response_time = 0.0
        
    @property
    def avg_response_time(self) -> float:
        """Calculate average response time in seconds."""
        if self.success_count == 0:
            return 0.0
        return self.total_response_time / self.success_count
    
    def record_request(self):
        """Record a new request."""
        self.request_count += 1
        self.last_used_at = datetime.now()
    
    def record_success(self, response_time: float):
        """Record a successful request."""
        self.success_count += 1
        self.total_response_time += response_time
    
    def record_error(self):
        """Record a request error."""
        self.error_count += 1
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "client_id": self.client_id,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "avg_response_time": self.avg_response_time,
            "error_rate": (self.error_count / self.request_count) if self.request_count > 0 else 0.0,
            "age_seconds": (datetime.now() - self.created_at).total_seconds()
        }


class PooledClient(Generic[T]):
    """
    Wrapper for a client with associated metadata and stats.
    
    This class wraps a client instance with additional metadata like
    health status, stats, and circuit breaker.
    """
    
    def __init__(
        self,
        client: T,
        client_id: str,
        circuit_breaker: Optional[CircuitBreaker] = None,
        rate_limiter: Optional[RateLimiter] = None,
        health_check_fn: Optional[Callable[[T], bool]] = None
    ):
        """
        Initialize a pooled client wrapper.
        
        Args:
            client: The client instance to wrap
            client_id: Unique identifier for this client
            circuit_breaker: Optional circuit breaker for this client
            rate_limiter: Optional rate limiter for this client
            health_check_fn: Optional function to check client health
        """
        self.client = client
        self.client_id = client_id
        self.stats = ConnectionStats(client_id)
        self.circuit_breaker = circuit_breaker
        self.rate_limiter = rate_limiter
        self.health_check_fn = health_check_fn
        self.is_healthy = True
        self.last_health_check = datetime.now()
        
    async def check_health(self) -> bool:
        """Check if the client is healthy."""
        try:
            if self.health_check_fn:
                self.is_healthy = self.health_check_fn(self.client)
            else:
                # Default health check assumes client is healthy
                self.is_healthy = True
                
            self.last_health_check = datetime.now()
            return self.is_healthy
        except Exception as e:
            logger.warning(f"Health check failed for client {self.client_id}: {e}")
            self.is_healthy = False
            self.stats.record_error()
            return False


class LLMConnectionPool(Generic[T]):
    """
    A connection pool for LLM providers that manages multiple client instances.
    
    This implementation maintains a pool of client connections that can be
    checked out for use and returned when done, improving performance by
    avoiding repeated connection creation/teardown costs.
    """
    
    def __init__(
        self, 
        create_client_fn: Callable[[], T], 
        max_connections: int = 10,
        min_connections: int = 1,
        max_idle_time_seconds: int = 300,
        health_check_interval_seconds: int = 60,
        health_check_fn: Optional[Callable[[T], bool]] = None,
        circuit_breaker_config: Optional[Dict[str, Any]] = None,
        rate_limit_config: Optional[RateLimitConfig] = None,
        name: str = "default",
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize a new connection pool.
        
        Args:
            create_client_fn: Factory function to create new client instances
            max_connections: Maximum number of connections to maintain in the pool
            min_connections: Minimum number of connections to maintain in the pool
            max_idle_time_seconds: Maximum time in seconds a connection can be idle before removal
            health_check_interval_seconds: Interval between health checks in seconds
            health_check_fn: Optional function to check client health
            circuit_breaker_config: Optional circuit breaker configuration
            rate_limit_config: Optional rate limit configuration
            name: Name identifier for this connection pool (for logging)
            metrics_service: Optional metrics service for recording metrics
            prometheus_exporter: Optional Prometheus exporter
        """
        self._create_client_fn = create_client_fn
        self._max_connections = max(1, max_connections)
        self._min_connections = max(1, min(min_connections, max_connections))
        self._max_idle_time = max_idle_time_seconds
        self._health_check_interval = health_check_interval_seconds
        self._health_check_fn = health_check_fn
        self._name = name
        self._circuit_breaker_config = circuit_breaker_config
        self._rate_limit_config = rate_limit_config
        
        # Metrics and observability
        self._metrics_service = metrics_service or MetricsService()
        self._prometheus = prometheus_exporter or get_prometheus_exporter()
        
        # Connection pool state
        self._available_clients: List[PooledClient[T]] = []
        self._in_use_clients: Dict[str, PooledClient[T]] = {}  # client_id -> client
        
        # Background tasks
        self._maintenance_task = None
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"Initialized LLMConnectionPool '{name}' with max_connections={max_connections}, "
                   f"min_connections={min_connections}, max_idle_time={max_idle_time_seconds}s")
        
        # Start background maintenance task
        self._start_maintenance_task()
    
    def _start_maintenance_task(self):
        """Start background maintenance task for pool health."""
        if self._maintenance_task is None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
            logger.debug(f"[{self._name}] Started connection pool maintenance task")
    
    async def _maintenance_loop(self):
        """Background loop for pool maintenance."""
        try:
            while not self._shutdown_event.is_set():
                # Wait for next maintenance interval
                try:
                    await asyncio.wait_for(self._shutdown_event.wait(), 
                                          timeout=self._health_check_interval)
                    # If we get here, shutdown was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue with maintenance
                    pass
                
                # Perform maintenance
                await self._perform_maintenance()
                
        except asyncio.CancelledError:
            logger.info(f"[{self._name}] Maintenance task cancelled")
        except Exception as e:
            logger.error(f"[{self._name}] Error in maintenance task: {e}", exc_info=True)
    
    async def _perform_maintenance(self):
        """Perform pool maintenance tasks."""
        logger.debug(f"[{self._name}] Performing connection pool maintenance")
        
        # Check health of available connections
        unhealthy_clients = []
        for client in self._available_clients:
            if not await client.check_health():
                unhealthy_clients.append(client)
                logger.warning(f"[{self._name}] Detected unhealthy connection {client.client_id}, marking for removal")
        
        # Remove unhealthy clients from pool
        for client in unhealthy_clients:
            self._available_clients.remove(client)
            try:
                await self._close_client(client.client)
            except Exception as e:
                logger.warning(f"[{self._name}] Error closing unhealthy client {client.client_id}: {e}")
        
        # Clean up idle connections
        await self._cleanup_idle_connections()
        
        # Ensure minimum connections
        await self._ensure_minimum_connections()
        
        # Report metrics
        self._report_metrics()
    
    async def _ensure_minimum_connections(self):
        """Ensure the pool has at least the minimum number of connections."""
        current_count = len(self._available_clients) + len(self._in_use_clients)
        if current_count < self._min_connections:
            to_create = self._min_connections - current_count
            logger.info(f"[{self._name}] Creating {to_create} connections to meet minimum requirement")
            
            for _ in range(to_create):
                try:
                    client = await self._create_new_client()
                    self._available_clients.append(client)
                except Exception as e:
                    logger.error(f"[{self._name}] Failed to create new client: {e}", exc_info=True)
    
    async def _create_new_client(self) -> PooledClient[T]:
        """Create a new client and wrap it."""
        client = self._create_client_fn()
        client_id = f"{self._name}-{id(client)}"
        
        # Create circuit breaker if configured
        circuit_breaker = None
        if self._circuit_breaker_config:
            circuit_breaker = CircuitBreaker(**self._circuit_breaker_config)
        
        # Create rate limiter if configured
        rate_limiter = None
        if self._rate_limit_config:
            rate_limiter = RateLimiter(self._rate_limit_config)
        
        # Create pooled client
        pooled_client = PooledClient(
            client=client,
            client_id=client_id,
            circuit_breaker=circuit_breaker,
            rate_limiter=rate_limiter,
            health_check_fn=self._health_check_fn
        )
        
        return pooled_client
    
    def _report_metrics(self):
        """Report pool metrics to metrics service."""
        try:
            # Basic pool metrics
            metrics = {
                "available_connections": len(self._available_clients),
                "in_use_connections": len(self._in_use_clients),
                "total_connections": len(self._available_clients) + len(self._in_use_clients),
            }
            
            # Add client-specific metrics
            clients_metrics = []
            for client in self._available_clients + list(self._in_use_clients.values()):
                clients_metrics.append(client.stats.to_dict())
            
            # Record metrics
            if self._metrics_service:
                self._metrics_service.record_gauge(f"llm.connection_pool.{self._name}.size", metrics["total_connections"])
                self._metrics_service.record_gauge(f"llm.connection_pool.{self._name}.available", metrics["available_connections"])
                self._metrics_service.record_gauge(f"llm.connection_pool.{self._name}.in_use", metrics["in_use_connections"])
            
            # Record to Prometheus if available
            if self._prometheus:
                self._prometheus.gauge(f"llm_connection_pool_{self._name}_size", metrics["total_connections"])
                self._prometheus.gauge(f"llm_connection_pool_{self._name}_available", metrics["available_connections"])
                self._prometheus.gauge(f"llm_connection_pool_{self._name}_in_use", metrics["in_use_connections"])
        
        except Exception as e:
            logger.error(f"[{self._name}] Error reporting metrics: {e}", exc_info=True)
    
    @asynccontextmanager
    async def connection(self) -> AsyncIterator[T]:
        """
        Get a connection from the pool as an async context manager.
        
        This is the preferred way to get a connection as it ensures proper
        cleanup even if an error occurs.
        
        Example:
            async with pool.connection() as client:
                result = await client.some_operation()
        
        Returns:
            AsyncContextManager yielding a client
        """
        client = None
        try:
            client = await self.get_client()
            yield client
        finally:
            if client is not None:
                await self.release_client(client)
    
    async def get_client(self) -> T:
        """
        Get a client from the pool or create a new one if needed.
        
        Returns:
            A client instance
        """
        # First try to get a healthy available connection
        while self._available_clients:
            pooled_client = self._available_clients.pop(0)
            
            # Verify the client is healthy
            if not await pooled_client.check_health():
                logger.warning(f"[{self._name}] Found unhealthy client in pool: {pooled_client.client_id}")
                try:
                    await self._close_client(pooled_client.client)
                except Exception as e:
                    logger.warning(f"[{self._name}] Error closing unhealthy client: {e}")
                continue
            
            # Client is healthy, use it
            self._in_use_clients[pooled_client.client_id] = pooled_client
            pooled_client.stats.record_request()
            
            logger.debug(f"[{self._name}] Reusing existing connection from pool. "
                        f"Available: {len(self._available_clients)}, In use: {len(self._in_use_clients)}")
            return pooled_client.client
            
        # If we have capacity, create a new connection
        if len(self._in_use_clients) < self._max_connections:
            try:
                pooled_client = await self._create_new_client()
                self._in_use_clients[pooled_client.client_id] = pooled_client
                pooled_client.stats.record_request()
                
                logger.debug(f"[{self._name}] Created new connection. "
                            f"Available: {len(self._available_clients)}, In use: {len(self._in_use_clients)}")
                return pooled_client.client
            except Exception as e:
                logger.error(f"[{self._name}] Failed to create client connection: {e}", exc_info=True)
                raise
                
        # Wait for a connection to become available (this should be rare)
        logger.warning(f"[{self._name}] Connection pool exhausted! "
                      f"Max connections ({self._max_connections}) reached. "
                      f"Consider increasing max_connections for better performance.")
        
        # Wait for a client to be returned to the pool
        while not self._available_clients:
            # Wait a bit then check again
            await asyncio.sleep(0.1)
            
            # If a client becomes available, use it
            if self._available_clients:
                pooled_client = self._available_clients.pop(0)
                self._in_use_clients[pooled_client.client_id] = pooled_client
                pooled_client.stats.record_request()
                return pooled_client.client
            
        # This should never happen since we wait until a client is available
        raise ConnectionError(f"[{self._name}] Connection pool exhausted and no connections available.")
    
    async def release_client(self, client: T):
        """
        Return a client to the pool.
        
        Args:
            client: Client instance to return to the pool
        """
        # Find the pooled client wrapper
        client_id = None
        pooled_client = None
        
        for cid, pc in list(self._in_use_clients.items()):
            if pc.client is client:
                client_id = cid
                pooled_client = pc
                break
        
        if pooled_client:
            self._in_use_clients.pop(client_id)
            self._available_clients.append(pooled_client)
            logger.debug(f"[{self._name}] Connection returned to pool. "
                        f"Available: {len(self._available_clients)}, In use: {len(self._in_use_clients)}")
        else:
            logger.warning(f"[{self._name}] Attempted to release client that wasn't checked out")
            
        # Cleanup idle connections if we have more than needed
        await self._cleanup_idle_connections()
    
    async def _cleanup_idle_connections(self):
        """Remove idle connections that exceed the maximum idle time."""
        current_time = datetime.now()
        idle_cutoff = self._max_idle_time
        
        # Only keep connections if:
        # 1. They're not too old, or
        # 2. We need to maintain min connections
        active_clients = []
        clients_to_close = []
        
        for client in self._available_clients:
            idle_time = (current_time - client.stats.last_used_at).total_seconds()
            
            if idle_time <= idle_cutoff or len(active_clients) < self._min_connections:
                active_clients.append(client)
            else:
                clients_to_close.append(client)
        
        # Close connections that are too idle
        for client in clients_to_close:
            try:
                await self._close_client(client.client)
            except Exception as e:
                logger.warning(f"[{self._name}] Error closing idle client: {e}")
        
        if clients_to_close:
            logger.info(f"[{self._name}] Removed {len(clients_to_close)} idle connections from pool")
            
        self._available_clients = active_clients
    
    async def _close_client(self, client: T):
        """Close a client connection."""
        try:
            if hasattr(client, 'close') and callable(client.close):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
            elif hasattr(client, 'cleanup') and callable(client.cleanup):
                if asyncio.iscoroutinefunction(client.cleanup):
                    await client.cleanup()
                else:
                    client.cleanup()
        except Exception as e:
            logger.warning(f"[{self._name}] Error closing client: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup all connections in the pool."""
        # Signal maintenance task to stop
        self._shutdown_event.set()
        
        # Wait for maintenance task to finish
        if self._maintenance_task:
            try:
                await self._maintenance_task
            except Exception as e:
                logger.warning(f"[{self._name}] Error waiting for maintenance task: {e}")
        
        total_connections = len(self._available_clients) + len(self._in_use_clients)
        logger.info(f"[{self._name}] Cleaning up connection pool. Closing {total_connections} connections.")
        
        # Close available connections
        for pooled_client in self._available_clients:
            try:
                await self._close_client(pooled_client.client)
            except Exception as e:
                logger.warning(f"[{self._name}] Error closing connection: {e}", exc_info=True)
                
        # Also try to close in-use connections
        for pooled_client in list(self._in_use_clients.values()):
            try:
                await self._close_client(pooled_client.client)
            except Exception as e:
                logger.warning(f"[{self._name}] Error closing in-use connection: {e}", exc_info=True)
        
        # Clear the collections
        self._available_clients = []
        self._in_use_clients = {}
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool."""
        stats = {
            "name": self._name,
            "available_connections": len(self._available_clients),
            "in_use_connections": len(self._in_use_clients),
            "total_connections": len(self._available_clients) + len(self._in_use_clients),
            "max_connections": self._max_connections,
            "min_connections": self._min_connections,
            "clients": []
        }
        
        # Add client-specific stats
        for client in self._available_clients:
            client_stats = client.stats.to_dict()
            client_stats["status"] = "available"
            stats["clients"].append(client_stats)
            
        for client in self._in_use_clients.values():
            client_stats = client.stats.to_dict()
            client_stats["status"] = "in_use"
            stats["clients"].append(client_stats)
            
        return stats