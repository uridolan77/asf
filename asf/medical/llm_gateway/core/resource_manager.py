"""
Unified Resource Management Layer for the LLM Gateway.

This module provides a consistent system for managing different types of resources,
including connection pools, client sessions, and model instances. It implements
features like resource pooling, health monitoring, circuit breaking, and adaptive pool sizing.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar, Callable, Awaitable, Union, cast
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from .models import ResourceType, ResourceStatus, ResourceInfo, ResourcePoolStats, ResourcePoolConfig
from .models import ResourceLimits, ResourceMetrics
from .errors import (
    ResourceError, ResourcePoolError, ResourceAcquisitionError, 
    ResourceReleaseError, CircuitBreakerError
)

logger = logging.getLogger(__name__)

# Generic type for resources
T = TypeVar('T')

# Function types for resource operations
ResourceCreator = Callable[[], Awaitable[T]]
ResourceValidator = Callable[[T], Awaitable[bool]]
ResourceInitializer = Callable[[T], Awaitable[None]]
ResourceCleanup = Callable[[T], Awaitable[None]]


class ResourceCircuitBreaker:
    """Circuit breaker pattern implementation for resource pools."""
    
    def __init__(
        self, 
        pool_id: str, 
        failure_threshold: int = 5,
        recovery_timeout_seconds: int = 30
    ):
        self.pool_id = pool_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def record_success(self):
        """Record a successful operation."""
        if self.state == "half-open":
            logger.info(f"Circuit breaker for pool {self.pool_id} closing after successful operation")
            self.state = "closed"
        
        self.failure_count = 0
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == "closed" and self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker for pool {self.pool_id} opening after {self.failure_count} failures"
            )
            self.state = "open"
    
    def allow_request(self) -> bool:
        """Check if a request should be allowed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            # Check if recovery timeout has passed
            if self.last_failure_time and (
                datetime.utcnow() - self.last_failure_time > 
                timedelta(seconds=self.recovery_timeout_seconds)
            ):
                logger.info(
                    f"Circuit breaker for pool {self.pool_id} transitioning to half-open state "
                    f"after {self.recovery_timeout_seconds}s"
                )
                self.state = "half-open"
                return True
            return False
        
        # In half-open state, we allow one request to test if the system is healthy
        return True

    def get_state(self) -> str:
        """Get the current state of the circuit breaker."""
        return self.state


class PooledResource(Generic[T]):
    """Wrapper for resources in a pool."""
    
    def __init__(
        self, 
        resource: T, 
        pool_id: str,
        resource_type: ResourceType,
        resource_id: str = None
    ):
        self.resource = resource
        self.resource_id = resource_id or str(uuid.uuid4())
        self.pool_id = pool_id
        self.resource_type = resource_type
        self.status = ResourceStatus.AVAILABLE
        self.created_at = datetime.utcnow()
        self.last_used_at: Optional[datetime] = None
        self.last_validated_at: Optional[datetime] = None
        self.total_uses = 0
        self.total_errors = 0
        self.error_streak = 0
        self.current_owner: Optional[str] = None
        self.acquisition_start: Optional[float] = None
        self.metrics = ResourceMetrics()
        self.metadata: Dict[str, Any] = {}
    
    def mark_in_use(self, owner_id: Optional[str] = None):
        """Mark resource as in use."""
        self.status = ResourceStatus.IN_USE
        self.current_owner = owner_id
        self.last_used_at = datetime.utcnow()
        self.total_uses += 1
        self.acquisition_start = time.time()
    
    def mark_available(self):
        """Mark resource as available."""
        self.status = ResourceStatus.AVAILABLE
        self.current_owner = None
        if self.acquisition_start:
            # Update average response time
            usage_time = (time.time() - self.acquisition_start) * 1000  # convert to ms
            if self.metrics.avg_response_time_ms is None:
                self.metrics.avg_response_time_ms = usage_time
            else:
                # Simple moving average
                self.metrics.avg_response_time_ms = (
                    self.metrics.avg_response_time_ms * 0.9 + usage_time * 0.1
                )
            self.acquisition_start = None
    
    def mark_error(self):
        """Record an error with this resource."""
        self.total_errors += 1
        self.error_streak += 1
        self.metrics.total_errors += 1
        self.metrics.last_error_time = datetime.utcnow()
        
        # Reduce health score based on error streak
        self.metrics.health_score = max(0.0, self.metrics.health_score - (0.1 * self.error_streak))
    
    def mark_success(self):
        """Record a successful operation with this resource."""
        self.error_streak = 0
        
        # Gradually recover health score with successful operations
        if self.metrics.health_score < 1.0:
            self.metrics.health_score = min(1.0, self.metrics.health_score + 0.05)
    
    def mark_validated(self):
        """Mark resource as validated."""
        self.last_validated_at = datetime.utcnow()
    
    def to_resource_info(self) -> ResourceInfo:
        """Convert to ResourceInfo model for API responses."""
        return ResourceInfo(
            resource_id=self.resource_id,
            pool_id=self.pool_id,
            resource_type=self.resource_type,
            status=self.status,
            created_at=self.created_at,
            last_used_at=self.last_used_at,
            last_validated_at=self.last_validated_at,
            metrics=self.metrics,
            metadata=self.metadata,
            error_count=self.total_errors,
            current_owner=self.current_owner
        )


class ResourceAcquisitionContext(Generic[T]):
    """Context manager for automatically releasing acquired resources."""
    
    def __init__(
        self, 
        pool: 'ResourcePool[T]', 
        resource: PooledResource[T], 
        owner_id: Optional[str] = None
    ):
        self.pool = pool
        self.resource = resource
        self.owner_id = owner_id
        self.released = False
    
    async def __aenter__(self) -> T:
        """Return the actual resource when entering context."""
        return self.resource.resource
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Release the resource when exiting context."""
        if not self.released:
            await self.release()
    
    async def release(self):
        """Release the resource back to the pool."""
        if not self.released:
            await self.pool.release(self.resource, self.owner_id)
            self.released = True
    
    def get_resource_info(self) -> ResourceInfo:
        """Get information about the resource."""
        return self.resource.to_resource_info()


class ResourcePool(Generic[T], ABC):
    """Abstract base class for resource pools."""
    
    def __init__(self, config: ResourcePoolConfig):
        self.config = config
        self.pool_id = config.pool_id
        self.resource_type = config.resource_type
        self.provider_id = config.provider_id
        
        # Resource tracking
        self.resources: Dict[str, PooledResource[T]] = {}
        self.available_resources: Set[str] = set()
        self.in_use_resources: Set[str] = set()
        
        # Waiting queue for resource acquisition
        self.waiting_queue: List[asyncio.Future] = []
        
        # Stats tracking
        self.creation_errors = 0
        self.total_acquisitions = 0
        self.total_timeouts = 0
        self.acquisition_times: List[float] = []
        self.waiting_times: List[float] = []
        
        # Lock for thread safety
        self.lock = asyncio.Lock()
        
        # Circuit breaker
        self.circuit_breaker = ResourceCircuitBreaker(
            pool_id=self.pool_id,
            failure_threshold=config.limits.circuit_breaker_threshold,
            recovery_timeout_seconds=config.limits.circuit_breaker_timeout_seconds
        )
        
        # Pool maintenance task
        self.maintenance_task: Optional[asyncio.Task] = None
        
        # Thread pool for sync operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
    
    @abstractmethod
    async def create_resource(self) -> T:
        """Create a new instance of the resource."""
        pass
    
    async def validate_resource(self, resource: T) -> bool:
        """Validate that a resource is still usable."""
        # Default implementation assumes resources are always valid
        # Override in subclasses for specific validation logic
        return True
    
    async def initialize_resource(self, resource: T) -> None:
        """Perform any necessary initialization on a new resource."""
        # Default implementation does nothing
        # Override in subclasses for specific initialization logic
        pass
    
    async def cleanup_resource(self, resource: T) -> None:
        """Clean up a resource before removing it."""
        # Default implementation does nothing
        # Override in subclasses for specific cleanup logic
        pass
    
    async def start(self):
        """Start the resource pool."""
        logger.info(f"Starting resource pool {self.pool_id} for {self.resource_type.value}")
        
        # Initialize with min_pool_size resources
        async with self.lock:
            for _ in range(self.config.limits.min_pool_size):
                try:
                    await self._create_and_add_resource()
                except Exception as e:
                    logger.error(f"Error creating initial resource in pool {self.pool_id}: {str(e)}")
                    self.creation_errors += 1
        
        # Start maintenance task
        if self.config.enable_health_checks:
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def stop(self):
        """Stop the resource pool and clean up resources."""
        logger.info(f"Stopping resource pool {self.pool_id}")
        
        # Cancel maintenance task
        if self.maintenance_task:
            self.maintenance_task.cancel()
            try:
                await self.maintenance_task
            except asyncio.CancelledError:
                pass
            self.maintenance_task = None
        
        # Clean up all resources
        async with self.lock:
            for resource_id, pooled_resource in list(self.resources.items()):
                try:
                    await self.cleanup_resource(pooled_resource.resource)
                except Exception as e:
                    logger.error(
                        f"Error cleaning up resource {resource_id} in pool {self.pool_id}: {str(e)}"
                    )
            
            # Clear resource tracking
            self.resources.clear()
            self.available_resources.clear()
            self.in_use_resources.clear()
            
            # Reject any waiting futures
            for future in self.waiting_queue:
                if not future.done():
                    future.set_exception(
                        ResourcePoolError(
                            f"Resource pool {self.pool_id} is shutting down",
                            pool_id=self.pool_id,
                            resource_type=self.resource_type.value,
                            operation="acquire"
                        )
                    )
            self.waiting_queue.clear()
        
        # Shut down thread pool
        self.thread_pool.shutdown(wait=True)
    
    async def acquire(
        self, 
        timeout: Optional[float] = None,
        owner_id: Optional[str] = None
    ) -> ResourceAcquisitionContext[T]:
        """Acquire a resource from the pool."""
        if timeout is None:
            timeout = self.config.limits.acquisition_timeout_seconds
        
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise CircuitBreakerError(
                f"Circuit breaker for pool {self.pool_id} is open",
                provider_id=self.provider_id,
                failure_count=self.circuit_breaker.failure_count,
                failure_threshold=self.circuit_breaker.failure_threshold,
                reset_timeout_seconds=self.circuit_breaker.recovery_timeout_seconds
            )
        
        start_time = time.time()
        wait_start_time = start_time
        
        resource_id = None
        waiting_future = None
        
        try:
            # Fast path: Try to get an available resource
            async with self.lock:
                self.total_acquisitions += 1
                
                if self.available_resources:
                    resource_id = next(iter(self.available_resources))
                    self.available_resources.remove(resource_id)
                    self.in_use_resources.add(resource_id)
                    pooled_resource = self.resources[resource_id]
                    pooled_resource.mark_in_use(owner_id)
                    
                    # Track acquisition time
                    acquisition_time = time.time() - start_time
                    self._update_acquisition_time_stats(acquisition_time)
                    
                    return ResourceAcquisitionContext(self, pooled_resource, owner_id)
                
                # Try to create a new resource if pool isn't at max size
                if len(self.resources) < self.config.limits.max_pool_size:
                    try:
                        pooled_resource = await self._create_and_add_resource()
                        resource_id = pooled_resource.resource_id
                        self.available_resources.remove(resource_id)
                        self.in_use_resources.add(resource_id)
                        pooled_resource.mark_in_use(owner_id)
                        
                        # Track acquisition time
                        acquisition_time = time.time() - start_time
                        self._update_acquisition_time_stats(acquisition_time)
                        
                        return ResourceAcquisitionContext(self, pooled_resource, owner_id)
                    except Exception as e:
                        logger.error(f"Error creating resource in pool {self.pool_id}: {str(e)}")
                        self.creation_errors += 1
                        self.circuit_breaker.record_failure()
                        raise ResourceAcquisitionError(
                            f"Failed to create resource: {str(e)}",
                            pool_id=self.pool_id,
                            resource_type=self.resource_type.value,
                            cause=e
                        )
                
                # No resources available, create a waiting future
                waiting_future = asyncio.Future()
                self.waiting_queue.append(waiting_future)
            
            # Wait for a resource to become available
            wait_result = None
            try:
                wait_result = await asyncio.wait_for(waiting_future, timeout)
            except asyncio.TimeoutError:
                async with self.lock:
                    if waiting_future in self.waiting_queue:
                        self.waiting_queue.remove(waiting_future)
                    self.total_timeouts += 1
                
                raise ResourceAcquisitionError(
                    f"Timeout waiting for resource from pool {self.pool_id}",
                    pool_id=self.pool_id,
                    resource_type=self.resource_type.value,
                    timeout_seconds=timeout,
                    retryable=True
                )
            
            # Successfully acquired a resource through waiting queue
            resource_id, pooled_resource = wait_result
            
            # Track waiting time
            waiting_time = time.time() - wait_start_time
            self._update_waiting_time_stats(waiting_time)
            
            # Track acquisition time
            acquisition_time = time.time() - start_time
            self._update_acquisition_time_stats(acquisition_time)
            
            return ResourceAcquisitionContext(self, pooled_resource, owner_id)
        
        except Exception as e:
            # If an exception occurred, clean up
            if waiting_future and not waiting_future.done():
                async with self.lock:
                    if waiting_future in self.waiting_queue:
                        self.waiting_queue.remove(waiting_future)
            
            # Record failure in circuit breaker
            if not isinstance(e, ResourceAcquisitionError):
                self.circuit_breaker.record_failure()
            
            # Re-raise with appropriate error type
            if isinstance(e, (ResourceError, CircuitBreakerError)):
                raise
            
            raise ResourceAcquisitionError(
                f"Error acquiring resource from pool {self.pool_id}: {str(e)}",
                pool_id=self.pool_id,
                resource_type=self.resource_type.value,
                cause=e
            )
    
    async def release(self, pooled_resource: PooledResource[T], owner_id: Optional[str] = None):
        """Release a resource back to the pool."""
        resource_id = pooled_resource.resource_id
        
        try:
            async with self.lock:
                # Check if resource exists and is in use
                if resource_id not in self.resources:
                    raise ResourceReleaseError(
                        f"Resource {resource_id} not found in pool {self.pool_id}",
                        pool_id=self.pool_id,
                        resource_id=resource_id,
                        resource_type=self.resource_type.value
                    )
                
                if resource_id not in self.in_use_resources:
                    # Resource might have been released already
                    logger.warning(
                        f"Resource {resource_id} in pool {self.pool_id} is not marked as in use"
                    )
                    return
                
                # Check owner if specified
                if (owner_id is not None and 
                    pooled_resource.current_owner is not None and 
                    pooled_resource.current_owner != owner_id):
                    raise ResourceReleaseError(
                        f"Resource {resource_id} owned by {pooled_resource.current_owner}, "
                        f"not by {owner_id}",
                        pool_id=self.pool_id,
                        resource_id=resource_id,
                        resource_type=self.resource_type.value
                    )
                
                # Remove from in-use set
                self.in_use_resources.remove(resource_id)
                
                # Check for waiting requests
                if self.waiting_queue:
                    # Give resource directly to a waiting request
                    waiting_future = self.waiting_queue.pop(0)
                    if not waiting_future.done():
                        pooled_resource.mark_in_use()
                        self.in_use_resources.add(resource_id)
                        waiting_future.set_result((resource_id, pooled_resource))
                        return
                
                # Mark as available
                pooled_resource.mark_available()
                self.available_resources.add(resource_id)
                
                # Record success in circuit breaker
                self.circuit_breaker.record_success()
        
        except Exception as e:
            if not isinstance(e, ResourceReleaseError):
                raise ResourceReleaseError(
                    f"Error releasing resource {resource_id} in pool {self.pool_id}: {str(e)}",
                    pool_id=self.pool_id,
                    resource_id=resource_id,
                    resource_type=self.resource_type.value,
                    cause=e
                )
            raise
    
    def get_stats(self) -> ResourcePoolStats:
        """Get statistics about the resource pool."""
        avg_acquisition_time = None
        if self.acquisition_times:
            avg_acquisition_time = sum(self.acquisition_times) / len(self.acquisition_times)
        
        avg_wait_time = None
        if self.waiting_times:
            avg_wait_time = sum(self.waiting_times) / len(self.waiting_times)
        
        return ResourcePoolStats(
            pool_id=self.pool_id,
            resource_type=self.resource_type,
            provider_id=self.provider_id,
            current_size=len(self.resources),
            available_count=len(self.available_resources),
            in_use_count=len(self.in_use_resources),
            waiting_requests=len(self.waiting_queue),
            creation_errors=self.creation_errors,
            circuit_state=self.circuit_breaker.get_state(),
            error_rate=self._calculate_error_rate(),
            avg_acquisition_time_ms=avg_acquisition_time * 1000 if avg_acquisition_time else None,
            avg_wait_time_ms=avg_wait_time * 1000 if avg_wait_time else None,
            total_acquisitions=self.total_acquisitions,
            total_timeouts=self.total_timeouts
        )
    
    def get_resources(self) -> List[ResourceInfo]:
        """Get information about all resources in the pool."""
        return [
            pooled_resource.to_resource_info()
            for pooled_resource in self.resources.values()
        ]
    
    async def _create_and_add_resource(self) -> PooledResource[T]:
        """Create a new resource and add it to the pool."""
        try:
            # Create the resource
            resource = await self.create_resource()
            
            # Initialize the resource
            await self.initialize_resource(resource)
            
            # Create pooled resource wrapper
            pooled_resource = PooledResource(
                resource=resource,
                pool_id=self.pool_id,
                resource_type=self.resource_type
            )
            
            # Add to pool
            self.resources[pooled_resource.resource_id] = pooled_resource
            self.available_resources.add(pooled_resource.resource_id)
            
            logger.debug(
                f"Created new resource {pooled_resource.resource_id} in pool {self.pool_id}"
            )
            
            return pooled_resource
        except Exception as e:
            logger.error(f"Failed to create resource in pool {self.pool_id}: {str(e)}")
            self.creation_errors += 1
            raise
    
    async def _maintenance_loop(self):
        """Background task for pool maintenance."""
        try:
            interval_seconds = self.config.health_check_interval_seconds
            logger.info(
                f"Starting maintenance loop for pool {self.pool_id} with interval {interval_seconds}s"
            )
            
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await self._perform_maintenance()
                except Exception as e:
                    logger.error(
                        f"Error in maintenance loop for pool {self.pool_id}: {str(e)}"
                    )
        except asyncio.CancelledError:
            logger.info(f"Maintenance loop for pool {self.pool_id} cancelled")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error in maintenance loop for pool {self.pool_id}: {str(e)}"
            )
    
    async def _perform_maintenance(self):
        """Perform maintenance tasks:
        - Remove idle resources exceeding max_idle_time
        - Remove resources exceeding max_lifetime
        - Validate available resources
        - Ensure min_pool_size is maintained
        - Resize pool based on load if adaptive_sizing is enabled
        """
        now = datetime.utcnow()
        resources_to_remove = []
        
        async with self.lock:
            # Check for idle resources to remove
            max_idle_time = self.config.limits.max_idle_time_seconds
            if max_idle_time:
                for resource_id, pooled_resource in self.resources.items():
                    if (resource_id in self.available_resources
                            and pooled_resource.last_used_at
                            and (now - pooled_resource.last_used_at).total_seconds() > max_idle_time
                            and len(self.resources) > self.config.limits.min_pool_size):
                        resources_to_remove.append(resource_id)
            
            # Check for resources exceeding max_lifetime
            max_lifetime = self.config.limits.max_lifetime_seconds
            if max_lifetime:
                for resource_id, pooled_resource in self.resources.items():
                    if (resource_id in self.available_resources
                            and (now - pooled_resource.created_at).total_seconds() > max_lifetime
                            and len(self.resources) > self.config.limits.min_pool_size):
                        resources_to_remove.append(resource_id)
            
            # Remove identified resources
            for resource_id in resources_to_remove:
                if resource_id in self.available_resources:
                    pooled_resource = self.resources[resource_id]
                    try:
                        await self.cleanup_resource(pooled_resource.resource)
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up resource {resource_id} in pool {self.pool_id}: {str(e)}"
                        )
                    
                    self.available_resources.remove(resource_id)
                    del self.resources[resource_id]
                    logger.debug(
                        f"Removed resource {resource_id} from pool {self.pool_id} during maintenance"
                    )
            
            # Validate available resources
            validated_resources = []
            invalid_resources = []
            
            for resource_id in list(self.available_resources):
                pooled_resource = self.resources[resource_id]
                try:
                    is_valid = await self.validate_resource(pooled_resource.resource)
                    if is_valid:
                        pooled_resource.mark_validated()
                        validated_resources.append(resource_id)
                    else:
                        invalid_resources.append(resource_id)
                except Exception as e:
                    logger.error(
                        f"Error validating resource {resource_id} in pool {self.pool_id}: {str(e)}"
                    )
                    invalid_resources.append(resource_id)
            
            # Remove invalid resources
            for resource_id in invalid_resources:
                if resource_id in self.available_resources:
                    pooled_resource = self.resources[resource_id]
                    try:
                        await self.cleanup_resource(pooled_resource.resource)
                    except Exception as e:
                        logger.error(
                            f"Error cleaning up invalid resource {resource_id} in pool {self.pool_id}: {str(e)}"
                        )
                    
                    self.available_resources.remove(resource_id)
                    del self.resources[resource_id]
                    logger.debug(
                        f"Removed invalid resource {resource_id} from pool {self.pool_id}"
                    )
            
            # Create new resources to maintain min_pool_size
            while len(self.resources) < self.config.limits.min_pool_size:
                try:
                    await self._create_and_add_resource()
                except Exception as e:
                    logger.error(
                        f"Error creating resource in pool {self.pool_id} during maintenance: {str(e)}"
                    )
                    self.creation_errors += 1
                    break
            
            # Adaptive sizing based on demand
            if self.config.adaptive_sizing:
                # Calculate current utilization
                utilization = len(self.in_use_resources) / max(1, len(self.resources))
                waiting_count = len(self.waiting_queue)
                
                # Scale up if high utilization with waiting requests
                if utilization > 0.8 and waiting_count > 0:
                    capacity_to_add = min(
                        waiting_count,
                        self.config.limits.max_pool_size - len(self.resources)
                    )
                    
                    for _ in range(capacity_to_add):
                        if len(self.resources) >= self.config.limits.max_pool_size:
                            break
                        
                        try:
                            await self._create_and_add_resource()
                        except Exception as e:
                            logger.error(
                                f"Error scaling up pool {self.pool_id}: {str(e)}"
                            )
                            self.creation_errors += 1
                            break
                
                # Scale down if low utilization
                elif (utilization < 0.2 and 
                      waiting_count == 0 and 
                      len(self.resources) > self.config.limits.min_pool_size):
                    # Identify excess resources beyond min_pool_size
                    excess_count = len(self.resources) - self.config.limits.min_pool_size
                    # Remove up to half of excess resources
                    resources_to_remove = list(self.available_resources)[:excess_count // 2]
                    
                    for resource_id in resources_to_remove:
                        if resource_id in self.available_resources:
                            pooled_resource = self.resources[resource_id]
                            try:
                                await self.cleanup_resource(pooled_resource.resource)
                            except Exception as e:
                                logger.error(
                                    f"Error cleaning up resource {resource_id} in pool {self.pool_id}: {str(e)}"
                                )
                            
                            self.available_resources.remove(resource_id)
                            del self.resources[resource_id]
                            logger.debug(
                                f"Removed resource {resource_id} from pool {self.pool_id} during scale down"
                            )
                
    def _update_acquisition_time_stats(self, time_seconds: float):
        """Update acquisition time statistics."""
        self.acquisition_times.append(time_seconds)
        if len(self.acquisition_times) > 100:
            self.acquisition_times = self.acquisition_times[-100:]
    
    def _update_waiting_time_stats(self, time_seconds: float):
        """Update waiting time statistics."""
        self.waiting_times.append(time_seconds)
        if len(self.waiting_times) > 100:
            self.waiting_times = self.waiting_times[-100:]
    
    def _calculate_error_rate(self) -> float:
        """Calculate error rate for the pool."""
        total_ops = self.total_acquisitions
        if total_ops == 0:
            return 0.0
        
        total_errors = self.creation_errors + self.total_timeouts
        return total_errors / total_ops


class GenericResourcePool(ResourcePool[T]):
    """Generic implementation of ResourcePool with callback functions."""
    
    def __init__(
        self,
        config: ResourcePoolConfig,
        creator: ResourceCreator[T],
        validator: Optional[ResourceValidator[T]] = None,
        initializer: Optional[ResourceInitializer[T]] = None,
        cleanup_fn: Optional[ResourceCleanup[T]] = None
    ):
        super().__init__(config)
        self._creator = creator
        self._validator = validator
        self._initializer = initializer
        self._cleanup_fn = cleanup_fn
    
    async def create_resource(self) -> T:
        """Create a new resource using the provided creator function."""
        return await self._creator()
    
    async def validate_resource(self, resource: T) -> bool:
        """Validate resource using the provided validator function."""
        if self._validator:
            return await self._validator(resource)
        return await super().validate_resource(resource)
    
    async def initialize_resource(self, resource: T) -> None:
        """Initialize resource using the provided initializer function."""
        if self._initializer:
            await self._initializer(resource)
        else:
            await super().initialize_resource(resource)
    
    async def cleanup_resource(self, resource: T) -> None:
        """Clean up resource using the provided cleanup function."""
        if self._cleanup_fn:
            await self._cleanup_fn(resource)
        else:
            await super().cleanup_resource(resource)


class ResourceManager:
    """Central manager for all resource pools."""
    
    def __init__(self):
        self.pools: Dict[str, ResourcePool] = {}
        self.configs: Dict[str, ResourcePoolConfig] = {}
        self.lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the resource manager."""
        logger.info("Initializing Resource Manager")
    
    async def shutdown(self):
        """Shutdown all resource pools."""
        logger.info("Shutting down Resource Manager")
        
        async with self.lock:
            for pool_id, pool in list(self.pools.items()):
                try:
                    await pool.stop()
                except Exception as e:
                    logger.error(f"Error stopping pool {pool_id}: {str(e)}")
            
            self.pools.clear()
            self.configs.clear()
    
    async def create_pool(
        self,
        config: ResourcePoolConfig,
        creator: ResourceCreator[T],
        validator: Optional[ResourceValidator[T]] = None,
        initializer: Optional[ResourceInitializer[T]] = None,
        cleanup_fn: Optional[ResourceCleanup[T]] = None
    ) -> str:
        """Create a new resource pool and start it."""
        logger.info(f"Creating resource pool for {config.resource_type.value}")
        
        # Use existing pool_id or generate a new one
        pool_id = config.pool_id
        
        async with self.lock:
            if pool_id in self.pools:
                raise ResourcePoolError(
                    f"Pool {pool_id} already exists",
                    pool_id=pool_id,
                    operation="create"
                )
            
            # Create and start the pool
            pool = GenericResourcePool(
                config=config,
                creator=creator,
                validator=validator,
                initializer=initializer,
                cleanup_fn=cleanup_fn
            )
            
            await pool.start()
            
            # Register pool
            self.pools[pool_id] = pool
            self.configs[pool_id] = config
            
            return pool_id
    
    async def get_pool(self, pool_id: str) -> ResourcePool:
        """Get a resource pool by ID."""
        async with self.lock:
            if pool_id not in self.pools:
                raise ResourcePoolError(
                    f"Pool {pool_id} not found",
                    pool_id=pool_id,
                    operation="get"
                )
            
            return self.pools[pool_id]
    
    async def remove_pool(self, pool_id: str):
        """Remove and stop a resource pool."""
        async with self.lock:
            if pool_id not in self.pools:
                raise ResourcePoolError(
                    f"Pool {pool_id} not found",
                    pool_id=pool_id,
                    operation="remove"
                )
            
            # Stop pool
            pool = self.pools[pool_id]
            await pool.stop()
            
            # Remove from registry
            del self.pools[pool_id]
            if pool_id in self.configs:
                del self.configs[pool_id]
    
    async def get_all_pools(self) -> Dict[str, ResourcePool]:
        """Get all registered resource pools."""
        async with self.lock:
            return dict(self.pools)
    
    async def get_stats(self) -> Dict[str, ResourcePoolStats]:
        """Get statistics for all resource pools."""
        result = {}
        async with self.lock:
            for pool_id, pool in self.pools.items():
                result[pool_id] = pool.get_stats()
        return result
    
    async def get_resources(self, pool_id: str) -> List[ResourceInfo]:
        """Get information about all resources in a pool."""
        pool = await self.get_pool(pool_id)
        return pool.get_resources()


# Singleton resource manager instance
resource_manager = ResourceManager()


async def initialize_resource_manager():
    """Initialize the global resource manager."""
    await resource_manager.initialize()


async def shutdown_resource_manager():
    """Shut down the global resource manager."""
    await resource_manager.shutdown()