"""
Event subscribers for LLM Gateway observability.

This module contains event subscribers that handle observability concerns
like metrics, logging, and tracing based on events from the event bus.
"""

import logging
import time
from typing import List, Type, Dict, Any, Optional

from asf.medical.llm_gateway.events.events import (
    Event,
    RequestStartedEvent,
    ResponseCompletedEvent,
    ErrorOccurredEvent,
    MCPSessionCreatedEvent,
    MCPSessionReleasedEvent,
    InterventionTriggeredEvent
)
from asf.medical.llm_gateway.events.subscriber import EventSubscriber
from asf.medical.llm_gateway.events.event_bus import EventBus

# Try to import metrics service, with graceful fallback
try:
    from asf.medical.llm_gateway.observability.metrics import MetricsService
except ImportError:
    # Create a stub if the real metrics service isn't available
    class MetricsService:
        def record_counter_inc(self, *args, **kwargs): pass
        def record_histogram(self, *args, **kwargs): pass
        def record_gauge(self, *args, **kwargs): pass

logger = logging.getLogger(__name__)


class MetricsEventSubscriber(EventSubscriber):
    """
    Event subscriber that records metrics based on LLM Gateway events.
    
    This subscriber listens for events on the event bus and records
    relevant metrics without requiring tight coupling to the event producers.
    """
    
    def __init__(self, event_bus: EventBus, metrics_service: Optional[MetricsService] = None):
        """
        Initialize the metrics event subscriber.
        
        Args:
            event_bus: The event bus to subscribe to
            metrics_service: Optional metrics service for recording metrics
        """
        self.metrics_service = metrics_service or MetricsService()
        super().__init__(event_bus)
    
    def get_event_subscription_types(self) -> List[Type[Event]]:
        """
        Get the list of event types this subscriber is interested in.
        
        Returns:
            List of event types to subscribe to
        """
        return [
            RequestStartedEvent,
            ResponseCompletedEvent,
            ErrorOccurredEvent,
            MCPSessionCreatedEvent,
            MCPSessionReleasedEvent,
            InterventionTriggeredEvent
        ]
    
    async def handle_event(self, event: Event) -> None:
        """
        Handle an event received from the event bus.
        
        Args:
            event: The event to handle
        """
        try:
            if isinstance(event, RequestStartedEvent):
                await self._handle_request_started(event)
            elif isinstance(event, ResponseCompletedEvent):
                await self._handle_response_completed(event)
            elif isinstance(event, ErrorOccurredEvent):
                await self._handle_error_occurred(event)
            elif isinstance(event, MCPSessionCreatedEvent):
                await self._handle_mcp_session_created(event)
            elif isinstance(event, MCPSessionReleasedEvent):
                await self._handle_mcp_session_released(event)
            elif isinstance(event, InterventionTriggeredEvent):
                await self._handle_intervention_triggered(event)
        except Exception as e:
            logger.error(f"Error handling event in MetricsEventSubscriber: {str(e)}", exc_info=True)
    
    async def _handle_request_started(self, event: RequestStartedEvent) -> None:
        """
        Handle a request started event.
        
        Args:
            event: The request started event
        """
        # Increment request counters
        self.metrics_service.record_counter_inc(
            "llm_gateway_requests_total",
            {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
        )
        
        # Set gauge for active requests
        self.metrics_service.record_gauge(
            "llm_gateway_active_requests",
            1,
            {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
        )
    
    async def _handle_response_completed(self, event: ResponseCompletedEvent) -> None:
        """
        Handle a response completed event.
        
        Args:
            event: The response completed event
        """
        # Record latency
        self.metrics_service.record_histogram(
            "llm_gateway_response_latency_ms",
            event.latency_ms,
            {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
        )
        
        # Record token count if available
        if event.token_count is not None:
            self.metrics_service.record_histogram(
                "llm_gateway_token_count",
                event.token_count,
                {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
            )
        
        # Record success/failure
        status_counter_name = "llm_gateway_responses_success" if event.status == "success" else "llm_gateway_responses_failure"
        self.metrics_service.record_counter_inc(
            status_counter_name,
            {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
        )
        
        # Decrement active requests gauge
        self.metrics_service.record_gauge(
            "llm_gateway_active_requests",
            -1,
            {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
        )
    
    async def _handle_error_occurred(self, event: ErrorOccurredEvent) -> None:
        """
        Handle an error occurred event.
        
        Args:
            event: The error occurred event
        """
        # Record error by type
        self.metrics_service.record_counter_inc(
            "llm_gateway_errors_total",
            {
                "error_type": event.error_type,
                "provider": event.provider_id or "unknown",
                "model": event.model or "unknown",
                "operation": event.operation_type or "unknown"
            }
        )
        
        # If this was associated with a request, decrement active requests
        if event.provider_id and event.model and event.operation_type:
            self.metrics_service.record_gauge(
                "llm_gateway_active_requests",
                -1,
                {"model": event.model, "provider": event.provider_id, "operation": event.operation_type}
            )
    
    async def _handle_mcp_session_created(self, event: MCPSessionCreatedEvent) -> None:
        """
        Handle an MCP session created event.
        
        Args:
            event: The MCP session created event
        """
        # Increment session creation counter
        self.metrics_service.record_counter_inc(
            "llm_gateway_mcp_sessions_created",
            {"model": event.model}
        )
        
        # Increment active sessions gauge
        self.metrics_service.record_gauge(
            "llm_gateway_mcp_active_sessions",
            1,
            {"model": event.model}
        )
    
    async def _handle_mcp_session_released(self, event: MCPSessionReleasedEvent) -> None:
        """
        Handle an MCP session released event.
        
        Args:
            event: The MCP session released event
        """
        # Record session duration
        self.metrics_service.record_histogram(
            "llm_gateway_mcp_session_duration_ms",
            event.duration_ms,
            {"model": event.model}
        )
        
        # Decrement active sessions gauge
        self.metrics_service.record_gauge(
            "llm_gateway_mcp_active_sessions",
            -1,
            {"model": event.model}
        )
    
    async def _handle_intervention_triggered(self, event: InterventionTriggeredEvent) -> None:
        """
        Handle an intervention triggered event.
        
        Args:
            event: The intervention triggered event
        """
        # Record intervention by type and result
        self.metrics_service.record_counter_inc(
            "llm_gateway_interventions_total",
            {"type": event.intervention_type, "result": event.intervention_result}
        )


class LoggingEventSubscriber(EventSubscriber):
    """
    Event subscriber that logs information based on LLM Gateway events.
    
    This subscriber listens for events on the event bus and logs
    relevant information without requiring tight coupling to the event producers.
    """
    
    def __init__(self, event_bus: EventBus, log_level: int = logging.INFO):
        """
        Initialize the logging event subscriber.
        
        Args:
            event_bus: The event bus to subscribe to
            log_level: The log level to use (default: INFO)
        """
        self.log_level = log_level
        super().__init__(event_bus)
    
    def get_event_subscription_types(self) -> List[Type[Event]]:
        """
        Get the list of event types this subscriber is interested in.
        
        Returns:
            List of event types to subscribe to
        """
        return [
            RequestStartedEvent,
            ResponseCompletedEvent,
            ErrorOccurredEvent,
            MCPSessionCreatedEvent,
            MCPSessionReleasedEvent,
            InterventionTriggeredEvent
        ]
    
    async def handle_event(self, event: Event) -> None:
        """
        Handle an event received from the event bus.
        
        Args:
            event: The event to handle
        """
        try:
            if isinstance(event, RequestStartedEvent):
                self._log_request_started(event)
            elif isinstance(event, ResponseCompletedEvent):
                self._log_response_completed(event)
            elif isinstance(event, ErrorOccurredEvent):
                self._log_error_occurred(event)
            elif isinstance(event, MCPSessionCreatedEvent):
                self._log_mcp_session_created(event)
            elif isinstance(event, MCPSessionReleasedEvent):
                self._log_mcp_session_released(event)
            elif isinstance(event, InterventionTriggeredEvent):
                self._log_intervention_triggered(event)
        except Exception as e:
            logger.error(f"Error handling event in LoggingEventSubscriber: {str(e)}", exc_info=True)
    
    def _log_request_started(self, event: RequestStartedEvent) -> None:
        """
        Log a request started event.
        
        Args:
            event: The request started event
        """
        logger.log(
            self.log_level,
            f"Request started: id={event.request_id} model={event.model} "
            f"provider={event.provider_id} operation={event.operation_type}"
        )
    
    def _log_response_completed(self, event: ResponseCompletedEvent) -> None:
        """
        Log a response completed event.
        
        Args:
            event: The response completed event
        """
        token_info = f" tokens={event.token_count}" if event.token_count is not None else ""
        logger.log(
            self.log_level,
            f"Response completed: id={event.request_id} model={event.model} "
            f"provider={event.provider_id} operation={event.operation_type} "
            f"status={event.status} latency={event.latency_ms:.2f}ms{token_info}"
        )
    
    def _log_error_occurred(self, event: ErrorOccurredEvent) -> None:
        """
        Log an error occurred event.
        
        Args:
            event: The error occurred event
        """
        model_info = f" model={event.model}" if event.model else ""
        provider_info = f" provider={event.provider_id}" if event.provider_id else ""
        operation_info = f" operation={event.operation_type}" if event.operation_type else ""
        
        logger.log(
            logging.ERROR,
            f"Error occurred: type={event.error_type} "
            f"request_id={event.request_id}{model_info}{provider_info}{operation_info} "
            f"message='{event.error_message}'"
        )
    
    def _log_mcp_session_created(self, event: MCPSessionCreatedEvent) -> None:
        """
        Log an MCP session created event.
        
        Args:
            event: The MCP session created event
        """
        logger.log(
            self.log_level,
            f"MCP session created: id={event.session_id} model={event.model}"
        )
    
    def _log_mcp_session_released(self, event: MCPSessionReleasedEvent) -> None:
        """
        Log an MCP session released event.
        
        Args:
            event: The MCP session released event
        """
        logger.log(
            self.log_level,
            f"MCP session released: id={event.session_id} model={event.model} "
            f"duration={event.duration_ms:.2f}ms"
        )
    
    def _log_intervention_triggered(self, event: InterventionTriggeredEvent) -> None:
        """
        Log an intervention triggered event.
        
        Args:
            event: The intervention triggered event
        """
        reason_info = f" reason='{event.reason}'" if event.reason else ""
        logger.log(
            self.log_level,
            f"Intervention triggered: request_id={event.request_id} "
            f"type={event.intervention_type} result={event.intervention_result}{reason_info}"
        )