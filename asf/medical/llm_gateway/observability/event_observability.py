"""
Event-driven observability for LLM Gateway.

This module provides an event subscriber that centralizes observability concerns
by handling events related to logging, metrics, and tracing.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from asf.medical.llm_gateway.events.subscriber import EventSubscriber, handles_event
from asf.medical.llm_gateway.events.events import (
    RequestReceivedEvent,
    RequestRoutedEvent,
    ProviderCalledEvent,
    ResponseStartedEvent,
    ResponseChunkReceivedEvent,
    ResponseSentEvent,
    ErrorOccurredEvent,
    MCPSessionCreatedEvent,
    MCPSessionReleasedEvent,
    MetricCollectedEvent,
    LogEntryEvent,
    TraceEvent,
    GatewayStartedEvent,
    GatewayShuttingDownEvent,
    ProviderRegisteredEvent,
    ProviderUnregisteredEvent,
    ProviderStatusChangedEvent,
    MCPConnectionEstablishedEvent,
    MCPConnectionClosedEvent,
)

# Try to import metrics and tracing components with fallbacks
try:
    from asf.medical.llm_gateway.observability.metrics import MetricsService
except ImportError:
    MetricsService = None

try:
    from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter
except ImportError:
    get_prometheus_exporter = None

try:
    from asf.medical.llm_gateway.observability.tracer import Tracer
except ImportError:
    Tracer = None

logger = logging.getLogger(__name__)


class ObservabilitySubscriber(EventSubscriber):
    """
    Event subscriber for observability concerns.
    
    This class centralizes the handling of events related to logging,
    metrics, and tracing. It decouples the observability concerns from
    the core business logic.
    """
    
    def __init__(self, metrics_service=None, tracer=None):
        """
        Initialize the observability subscriber.
        
        Args:
            metrics_service: Optional metrics service to use
            tracer: Optional tracer to use
        """
        super().__init__()
        # Set up metrics service
        if metrics_service:
            self.metrics_service = metrics_service
        elif MetricsService:
            self.metrics_service = MetricsService()
        else:
            self.metrics_service = None
            
        # Set up Prometheus exporter
        if get_prometheus_exporter:
            self.prometheus = get_prometheus_exporter()
        else:
            self.prometheus = None
            
        # Set up tracer
        if tracer:
            self.tracer = tracer
        elif Tracer:
            self.tracer = Tracer()
        else:
            self.tracer = None
            
        # Request tracking
        self._active_requests: Dict[str, Dict[str, Any]] = {}
        
    @handles_event(RequestReceivedEvent)
    async def handle_request_received(self, event: RequestReceivedEvent) -> None:
        """
        Handle request received event.
        
        Args:
            event: Request received event
        """
        # Log the request
        logger.info(f"Request received: id={event.request_id}, type={event.request_type}, "
                   f"provider={event.provider_id}, model={event.model}")
        
        # Start tracking the request
        self._active_requests[event.request_id] = {
            "start_time": time.time(),
            "provider_id": event.provider_id,
            "model": event.model,
            "request_type": event.request_type,
        }
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_request_received(
                provider_id=event.provider_id,
                model=event.model,
                request_type=event.request_type
            )
        
        # Start trace
        if self.tracer:
            span = self.tracer.start_span(
                operation_name=f"{event.request_type}_request",
                trace_id=event.request_id,
                tags={
                    "provider_id": event.provider_id,
                    "model": event.model or "unknown",
                    "request_type": event.request_type,
                }
            )
            self._active_requests[event.request_id]["span"] = span
    
    @handles_event(RequestRoutedEvent)
    async def handle_request_routed(self, event: RequestRoutedEvent) -> None:
        """
        Handle request routed event.
        
        Args:
            event: Request routed event
        """
        # Log the routing
        logger.debug(f"Request {event.request_id} routed to provider {event.provider_id}")
        
        # Update request tracking
        if event.request_id in self._active_requests:
            self._active_requests[event.request_id]["provider_id"] = event.provider_id
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_request_routed(
                provider_id=event.provider_id,
                route_info=event.route_info
            )
        
        # Add span event
        if self.tracer and event.request_id in self._active_requests:
            span = self._active_requests[event.request_id].get("span")
            if span:
                self.tracer.add_span_event(
                    span=span,
                    event_name="request_routed",
                    attributes={
                        "provider_id": event.provider_id,
                        "route_info": str(event.route_info),
                    }
                )
    
    @handles_event(ProviderCalledEvent)
    async def handle_provider_called(self, event: ProviderCalledEvent) -> None:
        """
        Handle provider called event.
        
        Args:
            event: Provider called event
        """
        # Log the provider call
        logger.debug(f"Provider {event.provider_id} called for request {event.request_id}")
        
        # Update request tracking
        if event.request_id in self._active_requests:
            self._active_requests[event.request_id]["provider_call_time"] = time.time()
            self._active_requests[event.request_id]["input_tokens"] = event.input_tokens
            
        # Record metric
        if self.prometheus:
            self.prometheus.record_provider_called(
                provider_id=event.provider_id,
                model=event.model,
                operation_type=event.operation_type,
                input_tokens=event.input_tokens
            )
        
        # Add span event
        if self.tracer and event.request_id in self._active_requests:
            span = self._active_requests[event.request_id].get("span")
            if span:
                self.tracer.add_span_event(
                    span=span,
                    event_name="provider_called",
                    attributes={
                        "provider_id": event.provider_id,
                        "operation_type": event.operation_type,
                        "input_tokens": event.input_tokens,
                    }
                )
    
    @handles_event(ResponseStartedEvent)
    async def handle_response_started(self, event: ResponseStartedEvent) -> None:
        """
        Handle response started event.
        
        Args:
            event: Response started event
        """
        # Log the response start
        logger.debug(f"Response started for request {event.request_id}, streaming={event.streaming}")
        
        # Update request tracking
        if event.request_id in self._active_requests:
            request_data = self._active_requests[event.request_id]
            request_data["response_start_time"] = time.time()
            request_data["streaming"] = event.streaming
            
            # Calculate time to first token
            if "provider_call_time" in request_data:
                ttft_ms = (request_data["response_start_time"] - request_data["provider_call_time"]) * 1000.0
                request_data["ttft_ms"] = ttft_ms
            
        # Record metric
        if self.prometheus:
            self.prometheus.record_response_started(
                provider_id=event.provider_id,
                model=event.model,
                streaming=event.streaming,
                latency_ms=event.latency_ms
            )
        
        # Add span event
        if self.tracer and event.request_id in self._active_requests:
            span = self._active_requests[event.request_id].get("span")
            if span:
                self.tracer.add_span_event(
                    span=span, 
                    event_name="response_started",
                    attributes={
                        "streaming": event.streaming,
                        "latency_ms": event.latency_ms,
                    }
                )
    
    @handles_event(ResponseChunkReceivedEvent)
    async def handle_response_chunk(self, event: ResponseChunkReceivedEvent) -> None:
        """
        Handle response chunk received event.
        
        Args:
            event: Response chunk received event
        """
        # Log chunk details for debug only
        logger.debug(
            f"Response chunk {event.chunk_index} for request {event.request_id}, "
            f"size={event.chunk_size}, final={event.is_final}"
        )
        
        # Update request tracking for final chunks
        if event.is_final and event.request_id in self._active_requests:
            self._active_requests[event.request_id]["final_chunk_received"] = True
            self._active_requests[event.request_id]["total_tokens"] = event.tokens
            
        # Record metric
        if self.prometheus:
            self.prometheus.record_response_chunk(
                provider_id=event.provider_id,
                model=event.model,
                chunk_size=event.chunk_size,
                is_final=event.is_final,
                tokens=event.tokens,
                token_rate=event.token_rate
            )
        
        # Add span event for first and final chunks only (to avoid trace bloat)
        if self.tracer and event.request_id in self._active_requests:
            span = self._active_requests[event.request_id].get("span")
            if span and (event.chunk_index == 0 or event.is_final):
                self.tracer.add_span_event(
                    span=span,
                    event_name=f"response_chunk_{'final' if event.is_final else 'first'}",
                    attributes={
                        "chunk_index": event.chunk_index,
                        "chunk_size": event.chunk_size,
                        "tokens": event.tokens,
                        "token_rate": event.token_rate,
                    }
                )
    
    @handles_event(ResponseSentEvent)
    async def handle_response_sent(self, event: ResponseSentEvent) -> None:
        """
        Handle response sent event.
        
        Args:
            event: Response sent event
        """
        # Log the response completion
        logger.info(
            f"Response sent for request {event.request_id}, "
            f"success={event.success}, duration={event.duration_ms:.1f}ms"
        )
        
        # Calculate metrics from request tracking
        request_data = self._active_requests.pop(event.request_id, None)
        if request_data:
            # Calculate total duration
            start_time = request_data.get("start_time")
            if start_time:
                total_duration_ms = (time.time() - start_time) * 1000.0
            else:
                total_duration_ms = event.duration_ms
            
            # Get token counts
            input_tokens = request_data.get("input_tokens", 0)
            output_tokens = event.output_tokens
            
            # Log token details
            logger.info(
                f"Request {event.request_id} completed with input_tokens={input_tokens}, "
                f"output_tokens={output_tokens}, total_tokens={event.total_tokens}"
            )
            
        # Record metric
        if self.prometheus:
            self.prometheus.record_response_sent(
                provider_id=event.provider_id,
                model=event.model,
                output_tokens=event.output_tokens,
                total_tokens=event.total_tokens,
                duration_ms=event.duration_ms,
                success=event.success,
                streaming=event.streaming
            )
        
        # End trace
        if self.tracer and request_data:
            span = request_data.get("span")
            if span:
                self.tracer.end_span(
                    span=span,
                    status="ok" if event.success else "error",
                    attributes={
                        "duration_ms": event.duration_ms,
                        "output_tokens": event.output_tokens,
                        "total_tokens": event.total_tokens,
                    }
                )
    
    @handles_event(ErrorOccurredEvent)
    async def handle_error(self, event: ErrorOccurredEvent) -> None:
        """
        Handle error occurred event.
        
        Args:
            event: Error occurred event
        """
        # Log the error
        logger.error(
            f"Error in request {event.request_id}: {event.error_type} - {event.error_message}"
        )
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_error(
                provider_id=event.provider_id,
                error_type=event.error_type,
                operation_type=event.operation_type or "unknown"
            )
        
        # Add error to trace
        if self.tracer and event.request_id in self._active_requests:
            span = self._active_requests[event.request_id].get("span")
            if span:
                self.tracer.record_exception(
                    span=span,
                    exception_type=event.error_type,
                    exception_message=event.error_message,
                    stacktrace=event.stacktrace
                )
    
    @handles_event(MCPSessionCreatedEvent)
    async def handle_session_created(self, event: MCPSessionCreatedEvent) -> None:
        """
        Handle MCP session created event.
        
        Args:
            event: MCP session created event
        """
        # Log the session creation
        logger.info(f"MCP session {event.session_id} created for model {event.model}")
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_session_created(
                session_id=event.session_id,
                model=event.model,
                **event.session_params
            )
    
    @handles_event(MCPSessionReleasedEvent)
    async def handle_session_released(self, event: MCPSessionReleasedEvent) -> None:
        """
        Handle MCP session released event.
        
        Args:
            event: MCP session released event
        """
        # Log the session release
        logger.info(f"MCP session {event.session_id} released after {event.duration_ms:.1f}ms")
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_session_released(
                session_id=event.session_id,
                model=event.model,
                duration_ms=event.duration_ms
            )
    
    @handles_event(MetricCollectedEvent)
    async def handle_metric(self, event: MetricCollectedEvent) -> None:
        """
        Handle metric collected event.
        
        Args:
            event: Metric collected event
        """
        # Forward to metrics service
        if self.metrics_service:
            self.metrics_service.record_metric(
                name=event.metric_name,
                value=event.value,
                unit=event.unit,
                tags=event.tags
            )
    
    @handles_event(LogEntryEvent)
    async def handle_log_entry(self, event: LogEntryEvent) -> None:
        """
        Handle log entry event.
        
        Args:
            event: Log entry event
        """
        # Map log level to logging method
        log_level = getattr(logging, event.log_level.upper(), logging.INFO)
        
        # Format context as string if present
        context_str = ""
        if event.context:
            context_items = []
            for key, value in event.context.items():
                context_items.append(f"{key}={value}")
            context_str = " ".join(context_items)
        
        # Format complete log message
        if context_str:
            log_message = f"{event.message} [{context_str}]"
        else:
            log_message = event.message
        
        # Add request or session ID if available
        if event.request_id:
            log_message = f"[request_id={event.request_id}] {log_message}"
        if event.session_id:
            log_message = f"[session_id={event.session_id}] {log_message}"
        
        # Log with appropriate level
        logger.log(log_level, f"[{event.component_id}] {log_message}")
    
    @handles_event(TraceEvent)
    async def handle_trace(self, event: TraceEvent) -> None:
        """
        Handle trace event.
        
        Args:
            event: Trace event
        """
        if not self.tracer:
            return
            
        # Check if this is a start or end event
        if event.start_time and not event.end_time:
            # Starting a span
            span = self.tracer.start_span(
                operation_name=event.operation_name,
                trace_id=event.trace_id,
                span_id=event.span_id,
                parent_span_id=event.parent_span_id,
                tags=event.tags,
                start_time=event.start_time
            )
            # Store span if needed
            
        elif event.end_time:
            # Ending a span
            span = self.tracer.get_span(event.span_id)
            if span:
                self.tracer.end_span(
                    span=span,
                    status=event.status,
                    end_time=event.end_time
                )
    
    @handles_event(GatewayStartedEvent)
    async def handle_gateway_started(self, event: GatewayStartedEvent) -> None:
        """
        Handle gateway started event.
        
        Args:
            event: Gateway started event
        """
        features_str = ", ".join(event.features)
        logger.info(
            f"Gateway {event.gateway_id} started with features: {features_str}"
        )
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_gateway_started(
                gateway_id=event.gateway_id,
                features=event.features
            )
    
    @handles_event(GatewayShuttingDownEvent)
    async def handle_gateway_shutdown(self, event: GatewayShuttingDownEvent) -> None:
        """
        Handle gateway shutting down event.
        
        Args:
            event: Gateway shutting down event
        """
        logger.info(
            f"Gateway {event.gateway_id} shutting down after {event.uptime_seconds:.1f}s"
        )
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_gateway_shutdown(
                gateway_id=event.gateway_id,
                uptime_seconds=event.uptime_seconds,
                reason=event.reason
            )
    
    @handles_event(ProviderStatusChangedEvent)
    async def handle_provider_status_changed(self, event: ProviderStatusChangedEvent) -> None:
        """
        Handle provider status changed event.
        
        Args:
            event: Provider status changed event
        """
        logger.info(
            f"Provider {event.provider_id} status changed: {event.previous_status} -> {event.new_status}"
        )
        
        # Record metric
        if self.prometheus:
            self.prometheus.record_provider_status_changed(
                provider_id=event.provider_id,
                previous_status=event.previous_status,
                new_status=event.new_status,
                reason=event.reason
            )


# Singleton instance
_observability_subscriber = None


def get_observability_subscriber() -> ObservabilitySubscriber:
    """
    Get the singleton ObservabilitySubscriber instance.
    
    Returns:
        ObservabilitySubscriber instance
    """
    global _observability_subscriber
    if _observability_subscriber is None:
        _observability_subscriber = ObservabilitySubscriber()
        
    return _observability_subscriber


def initialize_event_observability() -> ObservabilitySubscriber:
    """
    Initialize event-driven observability for the LLM Gateway.
    
    This function creates and subscribes the ObservabilitySubscriber.
    
    Returns:
        ObservabilitySubscriber instance
    """
    subscriber = get_observability_subscriber()
    subscriber.sync_subscribe()
    return subscriber