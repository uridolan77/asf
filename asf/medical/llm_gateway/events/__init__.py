"""
Event system for LLM Gateway.

This package provides an event-driven architecture for decoupling components
in the LLM Gateway. It includes an event bus for publishing and subscribing
to events, as well as event classes for various aspects of the system.
"""

from asf.medical.llm_gateway.events.event_bus import (
    EventBus,
    get_event_bus,
    set_event_bus,
)
from asf.medical.llm_gateway.events.events import (
    Event,
    RequestEvent,
    RequestReceivedEvent,
    RequestRoutedEvent,
    ProviderCalledEvent,
    ResponseStartedEvent,
    ResponseChunkReceivedEvent,
    ResponseSentEvent,
    ErrorOccurredEvent,
    SessionEvent,
    MCPSessionCreatedEvent,
    MCPSessionReleasedEvent,
    ObservabilityEvent,
    MetricCollectedEvent,
    LogEntryEvent,
    TraceEvent,
    InterventionEvent,
    InterventionTriggeredEvent,
    InterventionCompletedEvent,
    LongRunningOperationEvent,
    OperationStartedEvent,
    OperationProgressEvent,
    OperationCompletedEvent,
    GatewayEvent,
    GatewayStartedEvent,
    GatewayShuttingDownEvent,
    ProviderEvent,
    ProviderRegisteredEvent,
    ProviderUnregisteredEvent,
    ProviderStatusChangedEvent,
    ProviderThrottledEvent,
    MCPConnectionEvent,
    MCPConnectionEstablishedEvent,
    MCPConnectionClosedEvent,
)

# Create the default event bus instance
event_bus = get_event_bus()


__all__ = [
    'EventBus',
    'get_event_bus',
    'set_event_bus',
    'event_bus',
    'Event',
    'RequestEvent',
    'RequestReceivedEvent',
    'RequestRoutedEvent',
    'ProviderCalledEvent',
    'ResponseStartedEvent',
    'ResponseChunkReceivedEvent',
    'ResponseSentEvent',
    'ErrorOccurredEvent',
    'SessionEvent',
    'MCPSessionCreatedEvent',
    'MCPSessionReleasedEvent',
    'ObservabilityEvent',
    'MetricCollectedEvent',
    'LogEntryEvent',
    'TraceEvent',
    'InterventionEvent',
    'InterventionTriggeredEvent',
    'InterventionCompletedEvent',
    'LongRunningOperationEvent',
    'OperationStartedEvent',
    'OperationProgressEvent',
    'OperationCompletedEvent',
    'GatewayEvent',
    'GatewayStartedEvent',
    'GatewayShuttingDownEvent',
    'ProviderEvent',
    'ProviderRegisteredEvent',
    'ProviderUnregisteredEvent',
    'ProviderStatusChangedEvent',
    'ProviderThrottledEvent',
    'MCPConnectionEvent',
    'MCPConnectionEstablishedEvent',
    'MCPConnectionClosedEvent',
]