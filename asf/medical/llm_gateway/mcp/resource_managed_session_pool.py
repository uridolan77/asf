"""
MCP Session Pool implementation using the resource management layer.

This module provides a specialized resource pool for managing MCP sessions
with features like health checking, prioritization, and adaptive sizing,
building on the generic resource management framework.
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, Optional, Set, Callable, Awaitable, List, TypeVar
from datetime import datetime
from enum import Enum

from ..core.resource_manager import ResourcePool, ResourceAcquisitionContext, ResourceType
from ..core.models import ResourcePoolConfig, ResourceLimits, ResourceInfo
from ..core.errors import (
    ResourceError, ResourcePoolError, ResourceAcquisitionError, 
    MCPSessionError, CircuitBreakerError
)

# Try to import observability components with fallbacks
try:
    from ..observability.metrics import MetricsService
except ImportError:
    class MetricsService:
        """Dummy metrics service."""
        def __init__(self): pass

try:
    from ..observability.prometheus import get_prometheus_exporter
    prometheus_exporter = get_prometheus_exporter()
except ImportError:
    prometheus_exporter = None

# Try to import the event system with fallbacks
try:
    from ..events.event_bus import EventBus
    from ..events.events import (
        MCPSessionCreatedEvent,
        MCPSessionReleasedEvent,
        ErrorOccurredEvent
    )
    from ..events import event_bus as global_event_bus
except ImportError:
    EventBus = None
    MCPSessionCreatedEvent = None
    MCPSessionReleasedEvent = None
    ErrorOccurredEvent = None
    global_event_bus = None

logger = logging.getLogger(__name__)


class SessionPriority(int, Enum):
    """Session priority levels."""
    CRITICAL = 0  # High-priority sessions for critical services
    HIGH = 1      # Higher priority for important workloads
    NORMAL = 2    # Default priority
    LOW = 3       # Lower priority, can be evicted first
    BACKGROUND = 4  # Lowest priority, for background tasks


class MCPSessionMetadata:
    """Metadata for MCP sessions."""
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        provider_id: str = "",
        transport_type: str = "",
        tags: Optional[Set[str]] = None,
        capabilities: Optional[Set[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        priority: SessionPriority = SessionPriority.NORMAL
    ):
        self.model_id = model_id
        self.provider_id = provider_id
        self.transport_type = transport_type
        self.tags = tags or set()
        self.capabilities = capabilities or set()
        self.properties = properties or {}
        self.priority = priority
        self.created_at = datetime.utcnow()
        self.success_count = 0
        self.failure_count = 0
        self.consecutive_failures = 0
        self.last_error = None
        self.last_ping_time = None
        self.ping_latency_ms = 0.0
    
    def record_success(self) -> None:
        """Record a successful operation."""
        self.success_count += 1
        self.consecutive_failures = 0
    
    def record_failure(self, error: Optional[str] = None) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.consecutive_failures += 1
        self.last_error = error
    
    def matches_tags(self, tags: Set[str]) -> bool:
        """Check if session matches all specified tags."""
        if not tags:
            return True  # Empty tag set matches any session
        return tags.issubset(self.tags)
    
    def matches_model(self, model_id: Optional[str]) -> bool:
        """Check if session matches specified model."""
        if model_id is None:
            return True  # No model specified matches any session
        return self.model_id == model_id
    
    def has_capabilities(self, capabilities: Set[str]) -> bool:
        """Check if session has all specified capabilities."""
        if not capabilities:
            return True  # Empty capability set matches any session
        return capabilities.issubset(self.capabilities)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0  # No requests yet, assume good
        return self.success_count / total
    
    @property
    def health_score(self) -> float:
        """Calculate health score between 0.0 and 1.0."""
        # Start with success rate
        score = self.success_rate
        
        # Penalize consecutive failures
        if self.consecutive_failures > 0:
            # Exponential penalty for consecutive failures
            failure_penalty = min(0.9, 0.1 * (2 ** (self.consecutive_failures - 1)))
            score -= failure_penalty
        
        # Ensure score is between 0.0 and 1.0
        return max(0.0, min(1.0, score))


class MCPSessionResource:
    """Wrapper for MCP session objects."""
    
    def __init__(
        self,
        session_id: str,
        session: Any,  # The actual MCP session object
        metadata: MCPSessionMetadata
    ):
        self.session_id = session_id
        self.session = session
        self.metadata = metadata
        self.performance = {
            "total_requests": 0,
            "total_duration_ms": 0.0,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "window_requests": 0,
            "window_latency_ms": 0.0,
            "window_start_time": datetime.utcnow(),
            "token_count": 0,
        }
    
    def record_request(self, duration_ms: float, tokens: int = 0) -> None:
        """Record metrics for a request."""
        self.performance["total_requests"] += 1
        self.performance["total_duration_ms"] += duration_ms
        self.performance["token_count"] += tokens
        self.performance["window_requests"] += 1
        self.performance["window_latency_ms"] += duration_ms
        
        # Update min/max latency
        if (self.performance["min_latency_ms"] is None or 
            duration_ms < self.performance["min_latency_ms"]):
            self.performance["min_latency_ms"] = duration_ms
        
        if (self.performance["max_latency_ms"] is None or 
            duration_ms > self.performance["max_latency_ms"]):
            self.performance["max_latency_ms"] = duration_ms
        
        # Reset window if needed (every 1 minute)
        now = datetime.utcnow()
        if (now - self.performance["window_start_time"]).total_seconds() > 60.0:
            self.performance["window_start_time"] = now
            self.performance["window_requests"] = 1
            self.performance["window_latency_ms"] = duration_ms
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.performance["total_requests"] == 0:
            return 0.0
        return self.performance["total_duration_ms"] / self.performance["total_requests"]
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate current requests per minute."""
        now = datetime.utcnow()
        window_seconds = (now - self.performance["window_start_time"]).total_seconds()
        if window_seconds < 1.0:
            return 0.0
        return (self.performance["window_requests"] / window_seconds) * 60.0
    
    @property
    def score(self) -> float:
        """
        Calculate an overall score for this session (higher is better).
        
        This score considers health, performance, and priority for selection.
        """
        # Start with health score (0.0-1.0)
        base_score = self.metadata.health_score
        
        # Adjust for priority (higher priority gets a bonus)
        priority_factor = 1.0 - (self.metadata.priority.value / 10.0)  # 0.6 to 1.0
        
        # Adjust for performance (lower latency is better)
        latency_factor = 1.0
        avg_latency = self.avg_latency_ms
        if avg_latency > 0:
            # Normalize latency to a factor between 0.5 and 1.0
            normalized_latency = min(5000, avg_latency) / 5000.0
            latency_factor = 1.0 - (normalized_latency * 0.5)  # 0.5 to 1.0
        
        # Combine factors
        return base_score * priority_factor * latency_factor


# Type for session creation function
SessionCreator = Callable[[], Awaitable[Any]]
SessionCloser = Callable[[Any], Awaitable[None]]
SessionPinger = Callable[[Any], Awaitable[float]]


class MCPSessionPool(ResourcePool[MCPSessionResource]):
    """Resource pool for MCP sessions."""
    
    def __init__(
        self,
        config: ResourcePoolConfig,
        provider_id: str,
        create_session_func: SessionCreator,
        close_session_func: SessionCloser,
        ping_session_func: Optional[SessionPinger] = None,
        metrics_service: Optional[Any] = None,
        event_bus: Optional[Any] = None
    ):
        # Override resource type to ensure correct classification
        config.resource_type = ResourceType.MCP_SESSION
        config.provider_id = provider_id
        
        super().__init__(config)
        
        # MCP session configuration
        self.provider_id = provider_id
        self.create_session_func = create_session_func
        self.close_session_func = close_session_func
        self.ping_session_func = ping_session_func
        
        # Metrics and observability
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter
        self.event_bus = event_bus or global_event_bus
        
        # Session tracking by properties
        self.sessions_by_model: Dict[str, Set[str]] = {}
        self.sessions_by_tag: Dict[str, Set[str]] = {}
        self.sessions_by_priority: Dict[SessionPriority, Set[str]] = {}
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.wait_time_total_ms = 0.0
        self.processing_time_total_ms = 0.0
        
        self._session_counter = 0
    
    async def create_resource(self) -> MCPSessionResource:
        """Create a new MCP session."""
        try:
            self._session_counter += 1
            session_id = f"{self.provider_id}_{self._session_counter}"
            
            # Create the session
            session = await self.create_session_func()
            
            # Create metadata
            metadata = MCPSessionMetadata(
                provider_id=self.provider_id
            )
            
            # Create and return the resource
            mcp_resource = MCPSessionResource(
                session_id=session_id,
                session=session,
                metadata=metadata
            )
            
            # Record metrics
            if self.prometheus:
                self.prometheus.record_session_created(
                    provider_id=self.provider_id,
                    session_id=session_id,
                    model_id=metadata.model_id,
                    priority=metadata.priority.name
                )
            
            # Publish event
            if self.event_bus and MCPSessionCreatedEvent:
                event = MCPSessionCreatedEvent(
                    session_id=session_id,
                    model=metadata.model_id or "unknown",
                    session_params={
                        "priority": metadata.priority.name,
                        "tags": list(metadata.tags) if metadata.tags else [],
                        "capabilities": list(metadata.capabilities) if metadata.capabilities else []
                    },
                    metadata={
                        "provider_id": self.provider_id
                    }
                )
                self.event_bus.sync_publish(event)
            
            logger.info(f"Created new MCP session {session_id} for {self.provider_id}")
            
            return mcp_resource
            
        except Exception as e:
            logger.error(f"Failed to create MCP session: {str(e)}", exc_info=True)
            
            # Record metrics
            if self.prometheus:
                self.prometheus.record_session_creation_failed(
                    provider_id=self.provider_id,
                    error=type(e).__name__
                )
            
            # Publish error event
            if self.event_bus and ErrorOccurredEvent:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_creation_failed",
                    error_message=f"Failed to create MCP session: {str(e)}",
                    provider_id=self.provider_id,
                    model="unknown",
                    metadata={"provider_id": self.provider_id}
                )
                self.event_bus.sync_publish(error_event)
            
            raise MCPSessionError(
                f"Failed to create MCP session for {self.provider_id}: {str(e)}",
                provider_id=self.provider_id,
                operation="create",
                cause=e
            )
    
    async def validate_resource(self, resource: MCPSessionResource) -> bool:
        """Validate that the MCP session is still usable."""
        # Skip validation if ping function not provided
        if not self.ping_session_func:
            return True
        
        try:
            # Ping the session
            start_time = time.time()
            await self.ping_session_func(resource.session)
            latency_ms = (time.time() - start_time) * 1000
            
            # Update health metrics
            resource.metadata.last_ping_time = datetime.utcnow()
            resource.metadata.ping_latency_ms = latency_ms
            
            # Count as success
            resource.metadata.record_success()
            
            # Record metrics
            if self.prometheus:
                self.prometheus.record_session_health_check(
                    provider_id=self.provider_id,
                    session_id=resource.session_id,
                    success=True,
                    latency_ms=latency_ms
                )
            
            logger.debug(
                f"Health check passed for MCP session {resource.session_id}, "
                f"latency: {latency_ms:.2f}ms"
            )
            
            return True
            
        except Exception as e:
            # Update health metrics
            resource.metadata.record_failure(str(e))
            
            # Record metrics
            if self.prometheus:
                self.prometheus.record_session_health_check(
                    provider_id=self.provider_id,
                    session_id=resource.session_id,
                    success=False,
                    error=type(e).__name__
                )
            
            # Publish error event
            if self.event_bus and ErrorOccurredEvent:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_health_check_failed",
                    error_message=f"MCP session health check failed: {str(e)}",
                    provider_id=self.provider_id,
                    model=resource.metadata.model_id or "unknown",
                    metadata={"session_id": resource.session_id}
                )
                self.event_bus.sync_publish(error_event)
            
            logger.warning(f"Health check failed for MCP session {resource.session_id}: {str(e)}")
            
            # Determine if session is still usable based on consecutive failures
            return resource.metadata.consecutive_failures < 5
    
    async def cleanup_resource(self, resource: MCPSessionResource) -> None:
        """Clean up an MCP session."""
        try:
            # Calculate session duration
            session_duration_ms = (datetime.utcnow() - resource.metadata.created_at).total_seconds() * 1000
            
            # Close the session
            await self.close_session_func(resource.session)
            
            # Record metrics
            if self.prometheus:
                self.prometheus.record_session_closed(
                    provider_id=self.provider_id,
                    session_id=resource.session_id,
                    reason="cleanup"
                )
            
            # Publish session released event
            if self.event_bus and MCPSessionReleasedEvent:
                event = MCPSessionReleasedEvent(
                    session_id=resource.session_id,
                    model=resource.metadata.model_id or "unknown",
                    duration_ms=session_duration_ms,
                    metadata={
                        "provider_id": self.provider_id,
                        "reason": "cleanup",
                        "success_rate": resource.metadata.success_rate,
                        "total_requests": resource.performance["total_requests"]
                    }
                )
                await self.event_bus.publish(event)
            
            logger.info(f"Closed MCP session {resource.session_id} for {self.provider_id}")
            
        except Exception as e:
            logger.warning(f"Error closing MCP session {resource.session_id}: {str(e)}")
            
            # Record metrics
            if self.prometheus:
                self.prometheus.record_session_close_failed(
                    provider_id=self.provider_id,
                    session_id=resource.session_id,
                    error=type(e).__name__
                )
            
            # Publish error event
            if self.event_bus and ErrorOccurredEvent:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_close_failed",
                    error_message=f"Error closing MCP session: {str(e)}",
                    provider_id=self.provider_id,
                    model=resource.metadata.model_id or "unknown",
                    metadata={"session_id": resource.session_id}
                )
                self.event_bus.sync_publish(error_event)
    
    async def _resource_added(self, resource_id: str, resource: MCPSessionResource) -> None:
        """Track resource by properties for efficient lookup."""
        # Track by model
        if resource.metadata.model_id:
            if resource.metadata.model_id not in self.sessions_by_model:
                self.sessions_by_model[resource.metadata.model_id] = set()
            self.sessions_by_model[resource.metadata.model_id].add(resource_id)
        
        # Track by tags
        for tag in resource.metadata.tags:
            if tag not in self.sessions_by_tag:
                self.sessions_by_tag[tag] = set()
            self.sessions_by_tag[tag].add(resource_id)
        
        # Track by priority
        priority = resource.metadata.priority
        if priority not in self.sessions_by_priority:
            self.sessions_by_priority[priority] = set()
        self.sessions_by_priority[priority].add(resource_id)
    
    async def _resource_removed(self, resource_id: str, resource: MCPSessionResource) -> None:
        """Remove resource from tracking."""
        # Remove from model tracking
        if resource.metadata.model_id and resource.metadata.model_id in self.sessions_by_model:
            self.sessions_by_model[resource.metadata.model_id].discard(resource_id)
            if not self.sessions_by_model[resource.metadata.model_id]:
                del self.sessions_by_model[resource.metadata.model_id]
        
        # Remove from tag tracking
        for tag in resource.metadata.tags:
            if tag in self.sessions_by_tag:
                self.sessions_by_tag[tag].discard(resource_id)
                if not self.sessions_by_tag[tag]:
                    del self.sessions_by_tag[tag]
        
        # Remove from priority tracking
        priority = resource.metadata.priority
        if priority in self.sessions_by_priority:
            self.sessions_by_priority[priority].discard(resource_id)
            if not self.sessions_by_priority[priority]:
                del self.sessions_by_priority[priority]
    
    def _select_best_session(
        self,
        model_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        capabilities: Optional[Set[str]] = None
    ) -> Optional[str]:
        """
        Select the best available session matching the criteria.
        
        Args:
            model_id: Optional model ID to match
            tags: Optional set of tags to match
            capabilities: Required capabilities
            
        Returns:
            Resource ID of the best matching session, or None if no suitable session found
        """
        # Quickly filter by model_id if specified
        candidate_ids = set()
        
        if model_id and model_id in self.sessions_by_model:
            # Start with sessions for this model
            candidate_ids = self.sessions_by_model[model_id].copy()
        else:
            # Start with all available resources
            candidate_ids = self.available_resources.copy()
        
        # Filter by tags if specified
        if tags:
            for tag in tags:
                if tag in self.sessions_by_tag:
                    # Only keep sessions that have this tag
                    candidate_ids &= self.sessions_by_tag[tag]
                    
                    if not candidate_ids:
                        # No sessions left, bail out early
                        return None
        
        # Now check each candidate more thoroughly
        candidates = []
        for resource_id in candidate_ids:
            if resource_id not in self.resources:
                continue
                
            resource = self.resources[resource_id]
            
            # Check model, tags, and capabilities
            if (resource.metadata.matches_model(model_id) and 
                resource.metadata.matches_tags(tags or set()) and 
                resource.metadata.has_capabilities(capabilities or set())):
                candidates.append((resource_id, resource))
        
        if not candidates:
            return None
            
        # Sort by score (highest first)
        candidates.sort(key=lambda x: x[1].score, reverse=True)
        
        # Return the resource ID of the highest scoring session
        return candidates[0][0]
    
    async def get_session(
        self,
        model_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: SessionPriority = SessionPriority.NORMAL,
        timeout: Optional[float] = None,
        capabilities: Optional[Set[str]] = None
    ) -> ResourceAcquisitionContext[MCPSessionResource]:
        """
        Get a session matching the given criteria.
        
        Args:
            model_id: Optional model ID to filter by
            tags: Optional tags to filter by
            priority: Priority for new sessions if one needs to be created
            timeout: Acquisition timeout
            capabilities: Required capabilities
            
        Returns:
            Resource acquisition context for the session
        """
        wait_start = time.time()
        
        # Set owner ID (used for debugging, metrics)
        owner_id = f"{model_id or 'unknown'}_{uuid.uuid4().hex[:6]}"
        
        # Try to find best session or get new one
        session_context = None
        
        # First, try to find an existing session that matches our criteria
        best_session_id = self._select_best_session(model_id, tags, capabilities)
        
        if best_session_id:
            try:
                # Try to acquire this specific session
                session_context = await self.acquire(timeout=timeout, owner_id=owner_id)
            except Exception as e:
                logger.warning(f"Failed to acquire preferred session: {str(e)}")
                best_session_id = None
        
        # If we couldn't get a preferred session, just get any available one
        if not session_context:
            session_context = await self.acquire(timeout=timeout, owner_id=owner_id)
            
            # Set model ID, tags, and capabilities on the session
            if session_context:
                async with session_context as resource:
                    if model_id:
                        resource.metadata.model_id = model_id
                    if tags:
                        resource.metadata.tags.update(tags)
                    if capabilities:
                        resource.metadata.capabilities.update(capabilities)
                    resource.metadata.priority = priority
        
        # Record wait time
        wait_time_ms = (time.time() - wait_start) * 1000
        self.wait_time_total_ms += wait_time_ms
        self.request_count += 1
        
        # Record metrics
        if self.prometheus:
            self.prometheus.record_session_acquisition(
                provider_id=self.provider_id,
                session_id=session_context.resource.session_id if session_context else "unknown",
                model_id=model_id,
                wait_time_ms=wait_time_ms
            )
        
        return session_context
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the session pool.
        
        Returns:
            Dictionary containing statistics
        """
        base_stats = super().get_stats().dict()
        
        # Count sessions by model
        models = {}
        for model, session_ids in self.sessions_by_model.items():
            models[model] = {
                "total": len(session_ids),
                "active": sum(1 for sid in session_ids if sid in self.in_use_resources),
                "ready": sum(1 for sid in session_ids if sid in self.available_resources)
            }
        
        # Count sessions by priority
        priorities = {}
        for priority, session_ids in self.sessions_by_priority.items():
            priorities[priority.name] = {
                "count": len(session_ids),
                "active": sum(1 for sid in session_ids if sid in self.in_use_resources)
            }
        
        # Calculate average latency
        if self.success_count > 0:
            avg_processing_time_ms = self.processing_time_total_ms / self.success_count
        else:
            avg_processing_time_ms = 0.0
        
        if self.request_count > 0:
            avg_wait_time_ms = self.wait_time_total_ms / self.request_count
            success_rate = (self.success_count / self.request_count) * 100
        else:
            avg_wait_time_ms = 0.0
            success_rate = 100.0
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Combine with base stats
        return {
            **base_stats,
            "provider_id": self.provider_id,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "avg_processing_time_ms": avg_processing_time_ms,
            "avg_wait_time_ms": avg_wait_time_ms,
            "uptime_seconds": uptime_seconds,
            "models": models,
            "priorities": priorities,
            # Additional session-specific stats could go here
        }


class ResourceManagedMCPSessionPool:
    """
    MCP Session Pool implementation using the resource management layer.
    
    This class provides a similar interface to the original EnhancedSessionPool
    but delegates to the ResourceManager internally.
    """
    
    def __init__(
        self,
        provider_id: str,
        create_session_func: SessionCreator,
        close_session_func: SessionCloser,
        ping_session_func: Optional[SessionPinger] = None,
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time_seconds: int = 300,
        health_check_interval_seconds: int = 60,
        circuit_breaker_threshold: int = 5,
        acquisition_timeout_seconds: float = 30.0,
        metrics_service: Optional[Any] = None,
        event_bus: Optional[Any] = None
    ):
        """
        Initialize the MCP session pool.
        
        Args:
            provider_id: Provider identifier
            create_session_func: Function to create a new session
            close_session_func: Function to close a session
            ping_session_func: Function to ping a session (returns latency in ms)
            min_size: Minimum pool size
            max_size: Maximum pool size
            max_idle_time_seconds: Maximum idle time for a session
            health_check_interval_seconds: Health check interval
            circuit_breaker_threshold: Failures before circuit breaker opens
            acquisition_timeout_seconds: Default timeout for acquiring sessions
            metrics_service: Metrics service
            event_bus: Event bus for publishing events
        """
        self.provider_id = provider_id
        
        # Create pool configuration
        config = ResourcePoolConfig(
            resource_type=ResourceType.MCP_SESSION,
            pool_id=f"mcp_sessions_{provider_id}",
            provider_id=provider_id,
            limits=ResourceLimits(
                min_pool_size=min_size,
                max_pool_size=max_size,
                max_idle_time_seconds=max_idle_time_seconds,
                circuit_breaker_threshold=circuit_breaker_threshold,
                acquisition_timeout_seconds=acquisition_timeout_seconds
            ),
            enable_health_checks=ping_session_func is not None,
            health_check_interval_seconds=health_check_interval_seconds,
            adaptive_sizing=True
        )
        
        # Create the session pool
        self.session_pool = MCPSessionPool(
            config=config,
            provider_id=provider_id,
            create_session_func=create_session_func,
            close_session_func=close_session_func,
            ping_session_func=ping_session_func,
            metrics_service=metrics_service,
            event_bus=event_bus
        )
        
        logger.info(f"Initialized resource-managed MCP session pool for {provider_id}")
    
    async def start(self) -> None:
        """Start the session pool."""
        await self.session_pool.start()
        logger.info(f"Started resource-managed MCP session pool for {self.provider_id}")
    
    async def stop(self) -> None:
        """Stop the session pool."""
        await self.session_pool.stop()
        logger.info(f"Stopped resource-managed MCP session pool for {self.provider_id}")
    
    @property
    def get_session(self):
        """Get the session acquisition method."""
        return self.session_pool.get_session
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return self.session_pool.get_stats()


# Factory function to create session pools
def create_session_pool(
    provider_id: str,
    create_session_func: SessionCreator,
    close_session_func: SessionCloser,
    ping_session_func: Optional[SessionPinger] = None,
    min_size: int = 2,
    max_size: int = 10,
    max_idle_time_seconds: int = 300,
    health_check_interval_seconds: int = 60,
    use_resource_manager: bool = True,
    **kwargs
) -> Any:
    """
    Create an MCP session pool.
    
    Args:
        provider_id: Provider identifier
        create_session_func: Function to create a new session
        close_session_func: Function to close a session
        ping_session_func: Function to ping a session (returns latency in ms)
        min_size: Minimum pool size
        max_size: Maximum pool size
        max_idle_time_seconds: Maximum idle time for a session
        health_check_interval_seconds: Health check interval
        use_resource_manager: Whether to use the resource management layer (set to False to use legacy implementation)
        **kwargs: Additional arguments passed to the pool constructor
        
    Returns:
        Session pool instance
    """
    if use_resource_manager:
        # Use new resource-managed implementation
        return ResourceManagedMCPSessionPool(
            provider_id=provider_id,
            create_session_func=create_session_func,
            close_session_func=close_session_func,
            ping_session_func=ping_session_func,
            min_size=min_size,
            max_size=max_size,
            max_idle_time_seconds=max_idle_time_seconds,
            health_check_interval_seconds=health_check_interval_seconds,
            **kwargs
        )
    else:
        # Use legacy implementation
        try:
            from .session_pool import EnhancedSessionPool, SessionPoolConfig
            
            # Convert parameters to session pool config
            config = SessionPoolConfig(
                min_size=min_size,
                max_size=max_size,
                max_idle_time_seconds=max_idle_time_seconds,
                health_check_interval_seconds=health_check_interval_seconds,
                warmup_sessions=True,
                create_on_demand=True,
                adaptive_sizing=True,
            )
            
            return EnhancedSessionPool(
                provider_id=provider_id,
                create_session_func=create_session_func,
                close_session_func=close_session_func,
                ping_session_func=ping_session_func,
                config=config,
                **kwargs
            )
        except ImportError:
            logger.warning("Legacy EnhancedSessionPool not available, falling back to new implementation")
            return ResourceManagedMCPSessionPool(
                provider_id=provider_id,
                create_session_func=create_session_func,
                close_session_func=close_session_func,
                ping_session_func=ping_session_func,
                min_size=min_size,
                max_size=max_size,
                max_idle_time_seconds=max_idle_time_seconds,
                health_check_interval_seconds=health_check_interval_seconds,
                **kwargs
            )