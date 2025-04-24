"""
Enhanced session pool for LLM Gateway.

This module provides an enhanced session pool with advanced features like
health checking, prioritization, and session tagging for specific workloads.
"""

import asyncio
from contextlib import asynccontextmanager
import time
import random
import logging
from typing import AsyncGenerator, Dict, List, Any, Optional, Tuple, Set, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import heapq

# Updated imports to use consolidated components
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

# Import event system
from asf.medical.llm_gateway.events.event_bus import EventBus
from asf.medical.llm_gateway.events.events import (
    MCPSessionCreatedEvent,
    MCPSessionReleasedEvent,
    ErrorOccurredEvent
)

# Try to import the singleton event bus, with fallback to None
try:
    from asf.medical.llm_gateway.events import event_bus
except ImportError:
    event_bus = None

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    """Session states."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSED = "closed"


class SessionPriority(int, Enum):
    """Session priority levels."""
    CRITICAL = 0  # High-priority sessions for critical services
    HIGH = 1      # Higher priority for important workloads
    NORMAL = 2    # Default priority
    LOW = 3       # Lower priority, can be evicted first
    BACKGROUND = 4  # Lowest priority, for background tasks


@dataclass
class SessionHealth:
    """Health metrics for a session."""
    last_check_time: datetime = field(default_factory=datetime.utcnow)
    last_ping_time: Optional[datetime] = None
    ping_latency_ms: float = 0.0
    consecutive_failures: int = 0
    total_failures: int = 0
    total_successes: int = 0
    last_error: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.total_successes + self.total_failures
        if total == 0:
            return 1.0  # No requests yet, assume good
        return self.total_successes / total
    
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


@dataclass
class SessionPerformance:
    """Performance metrics for a session."""
    total_requests: int = 0
    total_duration_ms: float = 0.0
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    token_count: int = 0
    window_start_time: datetime = field(default_factory=datetime.utcnow)
    window_requests: int = 0
    window_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.total_requests == 0:
            return 0.0
        return self.total_duration_ms / self.total_requests
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate current requests per minute."""
        now = datetime.utcnow()
        window_seconds = (now - self.window_start_time).total_seconds()
        if window_seconds < 1.0:
            return 0.0
        return (self.window_requests / window_seconds) * 60.0
    
    def record_request(self, duration_ms: float, tokens: int = 0) -> None:
        """Record a request."""
        self.total_requests += 1
        self.total_duration_ms += duration_ms
        self.token_count += tokens
        self.window_requests += 1
        self.window_latency_ms += duration_ms
        self.last_request_time = datetime.utcnow()
        
        # Update min/max latency
        if self.min_latency_ms is None or duration_ms < self.min_latency_ms:
            self.min_latency_ms = duration_ms
        if self.max_latency_ms is None or duration_ms > self.max_latency_ms:
            self.max_latency_ms = duration_ms
        
        # Reset window if needed (every 1 minute)
        now = datetime.utcnow()
        if (now - self.window_start_time).total_seconds() > 60.0:
            self.window_start_time = now
            self.window_requests = 1
            self.window_latency_ms = duration_ms


@dataclass
class SessionMetadata:
    """Metadata for a session."""
    model_id: Optional[str] = None
    provider_id: str = ""
    transport_type: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """
    Enhanced session class with metadata, health, and performance metrics.
    
    This represents a connection to an LLM service with comprehensive tracking
    for adaptive session management.
    """
    session_id: str
    connection: Any  # The actual connection object (type varies by transport)
    state: SessionState = SessionState.INITIALIZING
    priority: SessionPriority = SessionPriority.NORMAL
    metadata: SessionMetadata = field(default_factory=SessionMetadata)
    health: SessionHealth = field(default_factory=SessionHealth)
    performance: SessionPerformance = field(default_factory=SessionPerformance)
    
    # For session pooling
    in_use: bool = False
    last_used_at: datetime = field(default_factory=datetime.utcnow)
    use_count: int = 0
    
    @property
    def score(self) -> float:
        """
        Calculate an overall score for this session (higher is better).
        
        This score considers health, performance, and priority for selection.
        """
        # Start with health score (0.0-1.0)
        base_score = self.health.health_score
        
        # Adjust for priority (higher priority gets a bonus)
        priority_factor = 1.0 - (self.priority.value / 10.0)  # 0.6 to 1.0
        
        # Adjust for performance (lower latency is better)
        latency_factor = 1.0
        if self.performance.avg_latency_ms > 0:
            # Normalize latency to a factor between 0.5 and 1.0
            normalized_latency = min(5000, self.performance.avg_latency_ms) / 5000.0
            latency_factor = 1.0 - (normalized_latency * 0.5)  # 0.5 to 1.0
        
        # Combine factors
        return base_score * priority_factor * latency_factor
    
    def matches_tags(self, tags: Set[str]) -> bool:
        """Check if session matches all specified tags."""
        if not tags:
            return True  # Empty tag set matches any session
        return tags.issubset(self.metadata.tags)
    
    def matches_model(self, model_id: Optional[str]) -> bool:
        """Check if session matches specified model."""
        if model_id is None:
            return True  # No model specified matches any session
        return self.metadata.model_id == model_id
    
    def has_capabilities(self, capabilities: Set[str]) -> bool:
        """Check if session has all specified capabilities."""
        if not capabilities:
            return True  # Empty capability set matches any session
        return capabilities.issubset(self.metadata.capabilities)
    
    def update_state(self) -> None:
        """Update session state based on health and other factors."""
        if self.state == SessionState.CLOSED:
            return  # Don't change closed state
        
        if self.health.consecutive_failures >= 5:
            self.state = SessionState.FAILED
        elif self.health.consecutive_failures >= 2:
            self.state = SessionState.DEGRADED
        elif self.in_use:
            self.state = SessionState.BUSY
        else:
            self.state = SessionState.READY
    
    def record_success(self, duration_ms: float, tokens: int = 0) -> None:
        """Record a successful request."""
        self.health.total_successes += 1
        self.health.consecutive_failures = 0
        self.performance.record_request(duration_ms, tokens)
        self.use_count += 1
        self.last_used_at = datetime.utcnow()
        self.update_state()
    
    def record_failure(self, error: Optional[str] = None) -> None:
        """Record a failed request."""
        self.health.total_failures += 1
        self.health.consecutive_failures += 1
        self.health.last_error = error
        self.use_count += 1
        self.last_used_at = datetime.utcnow()
        self.update_state()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "state": self.state,
            "priority": self.priority,
            "score": self.score,
            "in_use": self.in_use,
            "use_count": self.use_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "created_at": self.metadata.created_at.isoformat() if self.metadata.created_at else None,
            "health": {
                "success_rate": self.health.success_rate,
                "health_score": self.health.health_score,
                "consecutive_failures": self.health.consecutive_failures,
                "total_failures": self.health.total_failures,
                "total_successes": self.health.total_successes,
                "last_error": self.health.last_error,
                "last_check_time": self.health.last_check_time.isoformat() if self.health.last_check_time else None,
                "ping_latency_ms": self.health.ping_latency_ms,
            },
            "performance": {
                "total_requests": self.performance.total_requests,
                "avg_latency_ms": self.performance.avg_latency_ms,
                "min_latency_ms": self.performance.min_latency_ms,
                "max_latency_ms": self.performance.max_latency_ms,
                "requests_per_minute": self.performance.requests_per_minute,
                "token_count": self.performance.token_count,
                "last_request_time": self.performance.last_request_time.isoformat() if self.performance.last_request_time else None,
            },
            "metadata": {
                "model_id": self.metadata.model_id,
                "provider_id": self.metadata.provider_id,
                "transport_type": self.metadata.transport_type,
                "tags": list(self.metadata.tags),
                "capabilities": list(self.metadata.capabilities),
                "properties": self.metadata.properties,
            }
        }


@dataclass
class SessionPoolConfig:
    """Configuration for the enhanced session pool."""
    min_size: int = 2  # Minimum pool size
    max_size: int = 10  # Maximum pool size
    max_idle_time_seconds: int = 300  # Maximum idle time for a session
    health_check_interval_seconds: int = 60  # Health check interval
    warmup_sessions: bool = True  # Whether to pre-warm sessions
    create_on_demand: bool = True  # Whether to create sessions on demand
    adaptive_sizing: bool = True  # Whether to adaptively size the pool
    load_target_percentage: int = 70  # Target load percentage (for adaptive sizing)
    session_timeout_seconds: int = 600  # Maximum session lifetime
    failure_threshold: int = 5  # Failures before marking session as failed
    maintain_priority_balance: bool = True  # Maintain balance of session priorities
    priority_percentages: Dict[SessionPriority, int] = field(default_factory=lambda: {
        SessionPriority.CRITICAL: 20,
        SessionPriority.HIGH: 30,
        SessionPriority.NORMAL: 40,
        SessionPriority.LOW: 10,
        SessionPriority.BACKGROUND: 0
    })
    model_specific_allocation: Dict[str, int] = field(default_factory=dict)
    tag_specific_allocation: Dict[str, int] = field(default_factory=dict)


class EnhancedSessionPool:
    """
    Enhanced session pool with advanced features.
    
    This pool manages a collection of sessions with support for:
    - Health checking and automatic recovery
    - Session prioritization based on importance
    - Adaptive pool sizing based on load
    - Session tagging for specific workloads
    - Performance monitoring for optimal selection
    - Model-specific session allocation
    - Warm-up and preemptive session creation
    - Event-driven architecture for observability and notifications
    """
    
    def __init__(
        self,
        provider_id: str,
        create_session_func: Callable[[], Awaitable[Any]],
        close_session_func: Callable[[Any], Awaitable[None]],
        ping_session_func: Optional[Callable[[Any], Awaitable[float]]] = None,
        config: Optional[SessionPoolConfig] = None,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the enhanced session pool.
        
        Args:
            provider_id: Provider identifier
            create_session_func: Function to create a new session
            close_session_func: Function to close a session
            ping_session_func: Function to ping a session (returns latency in ms)
            config: Pool configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
            event_bus: Event bus for publishing events
        """
        self.provider_id = provider_id
        self.create_session_func = create_session_func
        self.close_session_func = close_session_func
        self.ping_session_func = ping_session_func
        self.config = config or SessionPoolConfig()
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        self.event_bus = event_bus or globals().get('event_bus')
        
        # Session storage
        self.sessions: Dict[str, Session] = {}
        self._session_counter = 0
        self._pool_lock = asyncio.Lock()
        
        # Background tasks
        self._health_check_task = None
        self._cleanup_task = None
        self._adaptive_sizing_task = None
        
        # Performance tracking
        self.start_time = datetime.utcnow()
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        self.wait_time_total_ms = 0.0
        self.processing_time_total_ms = 0.0
        
        # Session availability tracking
        self._session_waiters: List[asyncio.Future] = []
        
        logger.info(f"Initialized enhanced session pool for {provider_id}")
    
    async def start(self) -> None:
        """Start the session pool and initialize minimum sessions."""
        # Start background tasks
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.adaptive_sizing:
            self._adaptive_sizing_task = asyncio.create_task(self._adaptive_sizing_loop())
        
        # Initialize minimum sessions
        if self.config.warmup_sessions:
            async with self._pool_lock:
                await self._ensure_minimum_sessions()
        
        logger.info(f"Started session pool for {self.provider_id} with {len(self.sessions)} initial sessions")
    
    async def stop(self) -> None:
        """Stop the session pool and close all sessions."""
        # Cancel background tasks
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._adaptive_sizing_task and not self._adaptive_sizing_task.done():
            self._adaptive_sizing_task.cancel()
            try:
                await self._adaptive_sizing_task
            except asyncio.CancelledError:
                pass
        
        # Close all sessions
        async with self._pool_lock:
            close_tasks = []
            for session_id, session in list(self.sessions.items()):
                close_tasks.append(self._close_session(session_id))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            self.sessions.clear()
        
        # Cancel all waiters
        for waiter in self._session_waiters:
            if not waiter.done():
                waiter.cancel()
        self._session_waiters.clear()
        
        logger.info(f"Stopped session pool for {self.provider_id}")
    
    @asynccontextmanager
    async def get_session(
        self,
        model_id: Optional[str] = None,
        tags: Optional[Set[str]] = None,
        priority: SessionPriority = SessionPriority.NORMAL,
        timeout: Optional[float] = None,
        capabilities: Optional[Set[str]] = None
    ) -> AsyncGenerator[Any, None]:
        """
        Get a session from the pool.
        
        This will select the best available session matching the criteria
        or create a new one if needed and allowed by configuration.
        
        Args:
            model_id: Optional model ID to match
            tags: Optional set of tags to match
            priority: Session priority level
            timeout: Timeout in seconds
            capabilities: Required capabilities
            
        Returns:
            The session connection object
            
        Raises:
            asyncio.TimeoutError: If a session cannot be acquired within timeout
            RuntimeError: If no suitable session is available
        """
        session_id = None
        tags_set = set(tags) if tags else set()
        capabilities_set = set(capabilities) if capabilities else set()
        wait_start = time.time()
        
        try:
            # Try to get an existing session or create a new one
            session, session_id = await self._acquire_session(
                model_id, tags_set, priority, timeout, capabilities_set
            )
            
            # Mark session as in use
            session.in_use = True
            session.update_state()
            
            # Record wait time
            wait_time_ms = (time.time() - wait_start) * 1000
            self.wait_time_total_ms += wait_time_ms
            
            # Record metrics
            self.prometheus.record_session_acquisition(
                provider_id=self.provider_id,
                session_id=session_id,
                model_id=model_id,
                wait_time_ms=wait_time_ms
            )
            
            # Yield the connection object
            processing_start = time.time()
            try:
                yield session.connection
                
                # Record success
                self.success_count += 1
                
                # Calculate processing time
                processing_time_ms = (time.time() - processing_start) * 1000
                self.processing_time_total_ms += processing_time_ms
                
                # Record success in session
                session.record_success(processing_time_ms)
                
            except Exception as e:
                # Record failure
                self.error_count += 1
                session.record_failure(str(e))
                
                # Publish error event
                if self.event_bus:
                    error_event = ErrorOccurredEvent(
                        request_id=None,
                        error_type="session_operation_failed",
                        error_message=f"Error during session operation: {str(e)}",
                        provider_id=self.provider_id,
                        model=model_id or "unknown",
                        metadata={
                            "session_id": session_id,
                            "priority": priority.name
                        }
                    )
                    self.event_bus.sync_publish(error_event)
                
                raise
            
        finally:
            # Release session if acquired
            if session_id:
                await self._release_session(session_id)
    
    async def _acquire_session(
        self,
        model_id: Optional[str],
        tags: Set[str],
        priority: SessionPriority,
        timeout: Optional[float],
        capabilities: Set[str]
    ) -> Tuple[Session, str]:
        """
        Acquire a session matching the given criteria.
        
        Args:
            model_id: Optional model ID to match
            tags: Set of tags to match
            priority: Session priority level
            timeout: Timeout in seconds
            capabilities: Required capabilities
            
        Returns:
            Tuple of (session, session_id)
            
        Raises:
            asyncio.TimeoutError: If a session cannot be acquired within timeout
            RuntimeError: If no suitable session is available
        """
        start_time = time.time()
        
        while True:
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                raise asyncio.TimeoutError(f"Timeout acquiring session after {timeout}s")
            
            # Try to find a suitable session
            async with self._pool_lock:
                session, session_id = self._select_best_session(model_id, tags, capabilities)
                
                if session:
                    # Found a suitable session
                    return session, session_id
                
                # No suitable session found, try to create one if allowed
                if (self.config.create_on_demand and len(self.sessions) < self.config.max_size):
                    session = await self._create_new_session(model_id, tags, priority, capabilities)
                    return session, session.session_id
            
            # No session available and can't create one, wait for one to become available
            future = asyncio.Future()
            self._session_waiters.append(future)
            
            try:
                # Calculate remaining timeout
                remaining_timeout = None
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0.1, timeout - elapsed)
                
                # Wait for a session to become available
                try:
                    await asyncio.wait_for(future, timeout=remaining_timeout)
                except asyncio.TimeoutError:
                    # Remove waiter and re-raise
                    if future in self._session_waiters:
                        self._session_waiters.remove(future)
                    raise
                
                # A session might be available now, loop and try again
            finally:
                # Clean up waiter if it wasn't already removed
                if future in self._session_waiters:
                    self._session_waiters.remove(future)
    
    async def _release_session(self, session_id: str) -> None:
        """
        Release a session back to the pool.
        
        Args:
            session_id: Session ID to release
        """
        async with self._pool_lock:
            if session_id in self.sessions:
                session = self.sessions[session_id]
                
                # Mark session as not in use
                session.in_use = False
                session.last_used_at = datetime.utcnow()
                session.update_state()
                
                # Update metrics
                self.prometheus.record_session_release(
                    provider_id=self.provider_id,
                    session_id=session_id,
                    state=session.state
                )
                
                # Notify waiters that a session is available
                self._notify_waiters()
    
    def _select_best_session(
        self,
        model_id: Optional[str],
        tags: Set[str],
        capabilities: Set[str]
    ) -> Tuple[Optional[Session], Optional[str]]:
        """
        Select the best available session matching the criteria.
        
        Args:
            model_id: Optional model ID to match
            tags: Set of tags to match
            capabilities: Required capabilities
            
        Returns:
            Tuple of (session, session_id) or (None, None) if no suitable session found
        """
        # Filter sessions that match criteria and are available
        matching_sessions = []
        
        for session_id, session in self.sessions.items():
            # Skip sessions that are in use or failed
            if session.in_use or session.state in (SessionState.FAILED, SessionState.CLOSED):
                continue
            
            # Check if session matches criteria
            if (session.matches_model(model_id) and 
                session.matches_tags(tags) and
                session.has_capabilities(capabilities)):
                matching_sessions.append((session, session_id))
        
        if not matching_sessions:
            return None, None
        
        # Sort sessions by score (highest first)
        matching_sessions.sort(key=lambda x: x[0].score, reverse=True)
        
        # Return the best session
        return matching_sessions[0]
    
    async def _create_new_session(
        self,
        model_id: Optional[str],
        tags: Set[str],
        priority: SessionPriority,
        capabilities: Set[str]
    ) -> Session:
        """
        Create a new session.
        
        Args:
            model_id: Optional model ID
            tags: Set of tags
            priority: Priority level
            capabilities: Required capabilities
            
        Returns:
            Newly created session
            
        Raises:
            RuntimeError: If session creation fails
        """
        self._session_counter += 1
        session_id = f"{self.provider_id}_{self._session_counter}"
        
        try:
            # Create new connection
            connection = await self.create_session_func()
            
            # Create session object
            session = Session(
                session_id=session_id,
                connection=connection,
                state=SessionState.READY,
                priority=priority,
                metadata=SessionMetadata(
                    model_id=model_id,
                    provider_id=self.provider_id,
                    tags=tags,
                    capabilities=capabilities
                )
            )
            
            # Add to pool
            self.sessions[session_id] = session
            
            # Record metrics
            self.prometheus.record_session_created(
                provider_id=self.provider_id,
                session_id=session_id,
                model_id=model_id,
                priority=priority.name
            )
            
            # Publish session created event
            if self.event_bus:
                event = MCPSessionCreatedEvent(
                    session_id=session_id,
                    model=model_id or "unknown",
                    session_params={
                        "priority": priority.name,
                        "tags": list(tags) if tags else [],
                        "capabilities": list(capabilities) if capabilities else []
                    },
                    metadata={
                        "provider_id": self.provider_id
                    }
                )
                self.event_bus.sync_publish(event)
            
            logger.info(f"Created new session {session_id} for {self.provider_id}")
            
            return session
        
        except Exception as e:
            logger.error(f"Failed to create session {session_id}: {str(e)}", exc_info=True)
            
            # Record metrics
            self.prometheus.record_session_creation_failed(
                provider_id=self.provider_id,
                error=type(e).__name__
            )
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_creation_failed",
                    error_message=f"Failed to create session: {str(e)}",
                    provider_id=self.provider_id,
                    model=model_id or "unknown",
                    metadata={
                        "session_id": session_id,
                        "priority": priority.name
                    }
                )
                self.event_bus.sync_publish(error_event)
            
            raise RuntimeError(f"Failed to create session: {str(e)}")
    
    async def _close_session(self, session_id: str) -> None:
        """
        Close a session.
        
        Args:
            session_id: Session ID to close
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Skip if already closed
        if session.state == SessionState.CLOSED:
            return
        
        try:
            # Calculate session duration
            session_duration_ms = (datetime.utcnow() - session.metadata.created_at).total_seconds() * 1000
            
            # Close the connection
            await self.close_session_func(session.connection)
            
            # Update session state
            session.state = SessionState.CLOSED
            
            # Record metrics
            self.prometheus.record_session_closed(
                provider_id=self.provider_id,
                session_id=session_id,
                reason="explicit_close"
            )
            
            # Publish session released event
            if self.event_bus:
                event = MCPSessionReleasedEvent(
                    session_id=session_id,
                    model=session.metadata.model_id or "unknown",
                    duration_ms=session_duration_ms,
                    metadata={
                        "provider_id": self.provider_id,
                        "reason": "explicit_close",
                        "success_rate": session.health.success_rate,
                        "total_requests": session.performance.total_requests
                    }
                )
                await self.event_bus.publish(event)
            
            logger.info(f"Closed session {session_id} for {self.provider_id}")
        
        except Exception as e:
            logger.warning(f"Error closing session {session_id}: {str(e)}")
            
            # Record metrics
            self.prometheus.record_session_close_failed(
                provider_id=self.provider_id,
                session_id=session_id,
                error=type(e).__name__
            )
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_close_failed",
                    error_message=f"Error closing session: {str(e)}",
                    provider_id=self.provider_id,
                    model=session.metadata.model_id or "unknown",
                    metadata={
                        "session_id": session_id
                    }
                )
                self.event_bus.sync_publish(error_event)
        
        # Remove from pool
        self.sessions.pop(session_id, None)
    
    async def _ensure_minimum_sessions(self) -> None:
        """Ensure the pool has the minimum number of sessions."""
        current_size = len([s for s in self.sessions.values() if s.state != SessionState.CLOSED])
        
        if current_size < self.config.min_size:
            # Create sessions to reach minimum size
            tasks = []
            for _ in range(self.config.min_size - current_size):
                tasks.append(self._create_new_session(None, set(), SessionPriority.NORMAL, set()))
            
            # Wait for all sessions to be created
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successful creations
                successful = sum(1 for r in results if not isinstance(r, Exception))
                logger.info(f"Created {successful}/{len(tasks)} sessions to maintain minimum pool size")
    
    async def _health_check_loop(self) -> None:
        """Periodically check the health of all sessions."""
        try:
            while True:
                await asyncio.sleep(self.config.health_check_interval_seconds)
                await self._perform_health_check()
        except asyncio.CancelledError:
            logger.debug("Health check task cancelled")
        except Exception as e:
            logger.error(f"Error in health check task: {str(e)}", exc_info=True)
    
    async def _perform_health_check(self) -> None:
        """Perform health check on all sessions."""
        logger.debug(f"Performing health check for {len(self.sessions)} sessions")
        
        health_check_tasks = []
        
        async with self._pool_lock:
            for session_id, session in list(self.sessions.items()):
                # Skip sessions that are in use or already closed
                if session.in_use or session.state == SessionState.CLOSED:
                    continue
                
                # Add to health check tasks
                health_check_tasks.append(self._check_session_health(session_id))
        
        # Wait for all health checks to complete
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_session_health(self, session_id: str) -> None:
        """
        Check the health of a session.
        
        Args:
            session_id: Session ID to check
        """
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        
        # Update last check time
        session.health.last_check_time = datetime.utcnow()
        
        # Skip ping if ping function not provided
        if self.ping_session_func is None:
            return
        
        try:
            # Ping the session
            start_time = time.time()
            await self.ping_session_func(session.connection)
            latency_ms = (time.time() - start_time) * 1000
            
            # Update health metrics
            session.health.last_ping_time = datetime.utcnow()
            session.health.ping_latency_ms = latency_ms
            
            # Count as success
            session.health.total_successes += 1
            session.health.consecutive_failures = 0
            
            # Record metrics
            self.prometheus.record_session_health_check(
                provider_id=self.provider_id,
                session_id=session_id,
                success=True,
                latency_ms=latency_ms
            )
            
            logger.debug(f"Health check passed for session {session_id}, latency: {latency_ms:.2f}ms")
        
        except Exception as e:
            # Update health metrics
            session.health.total_failures += 1
            session.health.consecutive_failures += 1
            session.health.last_error = str(e)
            
            # Record metrics
            self.prometheus.record_session_health_check(
                provider_id=self.provider_id,
                session_id=session_id,
                success=False,
                error=type(e).__name__
            )
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_health_check_failed",
                    error_message=f"Health check failed: {str(e)}",
                    provider_id=self.provider_id,
                    model=session.metadata.model_id or "unknown",
                    metadata={
                        "session_id": session_id
                    }
                )
                self.event_bus.sync_publish(error_event)
            
            logger.warning(f"Health check failed for session {session_id}: {str(e)}")
            
            # Close session if too many consecutive failures
            if session.health.consecutive_failures >= self.config.failure_threshold:
                logger.warning(f"Closing unhealthy session {session_id} after {session.health.consecutive_failures} failures")
                await self._close_session(session_id)
        
        # Update session state
        session.update_state()
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up idle and expired sessions."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_sessions()
        except asyncio.CancelledError:
            logger.debug("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}", exc_info=True)
    
    async def _cleanup_sessions(self) -> None:
        """Clean up idle and expired sessions."""
        now = datetime.utcnow()
        
        async with self._pool_lock:
            # Identify sessions to close
            sessions_to_close = []
            
            for session_id, session in list(self.sessions.items()):
                # Skip sessions that are in use
                if session.in_use:
                    continue
                
                # Check for idle timeout
                idle_time = (now - session.last_used_at).total_seconds()
                if idle_time > self.config.max_idle_time_seconds:
                    logger.info(f"Closing idle session {session_id} after {idle_time:.1f}s")
                    sessions_to_close.append(session_id)
                    continue
                
                # Check for session lifetime
                session_age = (now - session.metadata.created_at).total_seconds()
                if session_age > self.config.session_timeout_seconds:
                    logger.info(f"Closing expired session {session_id} after {session_age:.1f}s")
                    sessions_to_close.append(session_id)
                    continue
                
                # Check for failed sessions
                if session.state == SessionState.FAILED:
                    logger.info(f"Closing failed session {session_id}")
                    sessions_to_close.append(session_id)
                    continue
            
            # Close identified sessions
            close_tasks = []
            for session_id in sessions_to_close:
                close_tasks.append(self._close_session(session_id))
            
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)
            
            # Ensure minimum pool size
            await self._ensure_minimum_sessions()
    
    async def _adaptive_sizing_loop(self) -> None:
        """Periodically adjust pool size based on load."""
        try:
            while True:
                await asyncio.sleep(30)  # Adjust every 30 seconds
                await self._adjust_pool_size()
        except asyncio.CancelledError:
            logger.debug("Adaptive sizing task cancelled")
        except Exception as e:
            logger.error(f"Error in adaptive sizing task: {str(e)}", exc_info=True)
    
    async def _adjust_pool_size(self) -> None:
        """Adjust pool size based on load."""
        async with self._pool_lock:
            # Count active sessions
            total_sessions = len(self.sessions)
            active_sessions = sum(1 for s in self.sessions.values() if s.in_use)
            
            if total_sessions == 0:
                load_percentage = 0
            else:
                load_percentage = (active_sessions / total_sessions) * 100
            
            logger.debug(f"Current pool load: {load_percentage:.1f}% ({active_sessions}/{total_sessions})")
            
            # Adjust pool size based on load
            if load_percentage > self.config.load_target_percentage + 10:
                # Load is high, increase pool size
                target_size = min(
                    self.config.max_size,
                    total_sessions + 1
                )
                
                if target_size > total_sessions:
                    logger.info(f"Increasing pool size due to high load: {total_sessions} -> {target_size}")
                    
                    # Create new sessions
                    try:
                        await self._create_new_session(None, set(), SessionPriority.NORMAL, set())
                    except Exception as e:
                        logger.warning(f"Failed to increase pool size: {str(e)}")
            
            elif load_percentage < self.config.load_target_percentage - 20 and total_sessions > self.config.min_size:
                # Load is low, decrease pool size
                target_size = max(
                    self.config.min_size,
                    total_sessions - 1
                )
                
                if target_size < total_sessions:
                    logger.info(f"Decreasing pool size due to low load: {total_sessions} -> {target_size}")
                    
                    # Find an idle session to close
                    idle_sessions = [
                        (session_id, session) for session_id, session in self.sessions.items()
                        if not session.in_use and session.state == SessionState.READY
                    ]
                    
                    if idle_sessions:
                        # Close the one with lowest score
                        idle_sessions.sort(key=lambda x: x[1].score)
                        session_id, _ = idle_sessions[0]
                        await self._close_session(session_id)
    
    def _notify_waiters(self) -> None:
        """Notify waiters that a session is available."""
        # Find a waiter to notify
        for waiter in self._session_waiters:
            if not waiter.done():
                waiter.set_result(None)
                return
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        total_sessions = len(self.sessions)
        active_sessions = sum(1 for s in self.sessions.values() if s.in_use)
        ready_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.READY)
        degraded_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.DEGRADED)
        failed_sessions = sum(1 for s in self.sessions.values() if s.state == SessionState.FAILED)
        
        # Calculate average latency
        if self.success_count > 0:
            avg_processing_time_ms = self.processing_time_total_ms / self.success_count
        else:
            avg_processing_time_ms = 0.0
        
        if self.request_count > 0:
            avg_wait_time_ms = self.wait_time_total_ms / self.request_count
        else:
            avg_wait_time_ms = 0.0
        
        # Group sessions by model
        models = {}
        for session in self.sessions.values():
            model_id = session.metadata.model_id or "default"
            if model_id not in models:
                models[model_id] = {"total": 0, "active": 0, "ready": 0}
            
            models[model_id]["total"] += 1
            if session.in_use:
                models[model_id]["active"] += 1
            if session.state == SessionState.READY:
                models[model_id]["ready"] += 1
        
        # Group sessions by priority
        priorities = {}
        for priority in SessionPriority:
            priorities[priority.name] = {
                "count": sum(1 for s in self.sessions.values() if s.priority == priority),
                "active": sum(1 for s in self.sessions.values() if s.priority == priority and s.in_use)
            }
        
        # Calculate uptime
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "provider_id": self.provider_id,
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "ready_sessions": ready_sessions,
            "degraded_sessions": degraded_sessions,
            "failed_sessions": failed_sessions,
            "waiters": len(self._session_waiters),
            "min_size": self.config.min_size,
            "max_size": self.config.max_size,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": (self.success_count / self.request_count * 100) if self.request_count > 0 else 100.0,
            "avg_processing_time_ms": avg_processing_time_ms,
            "avg_wait_time_ms": avg_wait_time_ms,
            "uptime_seconds": uptime_seconds,
            "models": models,
            "priorities": priorities,
            "sessions": [session.to_dict() for session in self.sessions.values()]
        }