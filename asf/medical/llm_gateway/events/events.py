"""
Event classes for LLM Gateway.

This module defines the event classes that can be published and subscribed to
through the event bus. These events represent important occurrences in the
system that may be of interest to multiple components.
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Union


@dataclass
class Event:
    """Base class for all events in the system."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestEvent(Event):
    """Base class for request-related events."""
    request_id: str = field(default="")
    provider_id: str = field(default="")
    model: Optional[str] = None


@dataclass
class RequestReceivedEvent(RequestEvent):
    """Event emitted when a request is received by the gateway."""
    request_type: str = field(default="")
    parameters: Dict[str, Any] = field(default_factory=dict)
    client_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestRoutedEvent(RequestEvent):
    """Event emitted when a request has been routed to a provider."""
    route_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderCalledEvent(RequestEvent):
    """Event emitted when a provider is called."""
    input_tokens: Optional[int] = None
    request_parameters: Dict[str, Any] = field(default_factory=dict)
    operation_type: str = "completion"


@dataclass
class ResponseStartedEvent(RequestEvent):
    """Event emitted when a response starts streaming."""
    streaming: bool = False
    latency_ms: float = 0.0


@dataclass
class ResponseChunkReceivedEvent(RequestEvent):
    """Event emitted when a chunk of response is received."""
    chunk_index: int = field(default=0)
    chunk_size: int = field(default=0)
    is_final: bool = False
    tokens: Optional[int] = None
    token_rate: Optional[float] = None


@dataclass
class ResponseSentEvent(RequestEvent):
    """Event emitted when a response is completed and sent to the client."""
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    duration_ms: float = 0.0
    success: bool = True
    streaming: bool = False


@dataclass
class ErrorOccurredEvent(RequestEvent):
    """Event emitted when an error occurs."""
    error_type: str = field(default="")
    error_message: str = field(default="")
    operation_type: Optional[str] = None
    stacktrace: Optional[str] = None


@dataclass
class SessionEvent(Event):
    """Base class for session-related events."""
    session_id: str = field(default="")
    model: Optional[str] = None


@dataclass
class MCPSessionCreatedEvent(SessionEvent):
    """Event emitted when an MCP session is created."""
    session_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPSessionReleasedEvent(SessionEvent):
    """Event emitted when an MCP session is released."""
    duration_ms: float = 0.0


@dataclass
class ObservabilityEvent(Event):
    """Base class for observability-related events."""
    component_id: str = field(default="")
    message: str = field(default="")
    level: str = "info"


@dataclass
class MetricCollectedEvent(ObservabilityEvent):
    """Event emitted when a metric is collected."""
    metric_name: str = field(default="")
    value: float = field(default=0.0)
    unit: str = field(default="")
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class LogEntryEvent(ObservabilityEvent):
    """Event emitted when a log entry is created."""
    log_level: str = field(default="info")
    context: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class TraceEvent(ObservabilityEvent):
    """Event emitted for tracing purposes."""
    trace_id: str = field(default="")
    span_id: str = field(default="")
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "OK"


@dataclass
class InterventionEvent(Event):
    """Base class for intervention-related events."""
    intervention_id: str = field(default="")
    request_id: Optional[str] = None
    source: str = ""
    intervention_type: str = ""


@dataclass
class InterventionTriggeredEvent(InterventionEvent):
    """Event emitted when an intervention is triggered."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    severity: str = "medium"


@dataclass
class InterventionCompletedEvent(InterventionEvent):
    """Event emitted when an intervention is completed."""
    result: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class LongRunningOperationEvent(Event):
    """Base class for long-running operation events."""
    operation_id: str = field(default="")
    request_id: Optional[str] = None
    operation_type: str = ""
    user_id: Optional[str] = None


@dataclass
class OperationStartedEvent(LongRunningOperationEvent):
    """Event emitted when a long-running operation starts."""
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_duration_ms: Optional[float] = None


@dataclass
class OperationProgressEvent(LongRunningOperationEvent):
    """Event emitted to report progress of a long-running operation."""
    progress_percentage: float = field(default=0.0)
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationCompletedEvent(LongRunningOperationEvent):
    """Event emitted when a long-running operation completes."""
    result: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    success: bool = True


@dataclass
class GatewayEvent(Event):
    """Base class for gateway lifecycle events."""
    gateway_id: str = field(default="")
    component_id: Optional[str] = None


@dataclass
class GatewayStartedEvent(GatewayEvent):
    """Event emitted when the gateway starts."""
    config: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)


@dataclass
class GatewayShuttingDownEvent(GatewayEvent):
    """Event emitted when the gateway is shutting down."""
    reason: str = "normal"
    uptime_seconds: float = 0.0


@dataclass
class ProviderEvent(Event):
    """Base class for provider-related events."""
    provider_id: str = field(default="")
    provider_type: str = ""
    model: Optional[str] = None


@dataclass
class ProviderRegisteredEvent(ProviderEvent):
    """Event emitted when a provider is registered."""
    capabilities: Dict[str, Any] = field(default_factory=dict)
    models: List[str] = field(default_factory=list)


@dataclass
class ProviderUnregisteredEvent(ProviderEvent):
    """Event emitted when a provider is unregistered."""
    reason: str = ""


@dataclass
class ProviderStatusChangedEvent(ProviderEvent):
    """Event emitted when provider status changes."""
    previous_status: str = field(default="")
    new_status: str = field(default="")
    reason: str = ""


@dataclass
class ProviderThrottledEvent(ProviderEvent):
    """Event emitted when a provider is throttled."""
    reason: str = ""
    duration_ms: Optional[float] = None
    request_id: Optional[str] = None


@dataclass
class MCPConnectionEvent(Event):
    """Base class for MCP connection events."""
    connection_id: str = field(default="")
    transport_type: str = ""
    endpoint: str = ""


@dataclass
class MCPConnectionEstablishedEvent(MCPConnectionEvent):
    """Event emitted when an MCP connection is established."""
    session_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPConnectionClosedEvent(MCPConnectionEvent):
    """Event emitted when an MCP connection is closed."""
    reason: str = ""
    duration_ms: float = 0.0
    success: bool = True