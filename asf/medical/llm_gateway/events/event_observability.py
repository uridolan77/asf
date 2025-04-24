"""
Observability subscriber for LLM Gateway.

This module provides a subscriber that listens to LLM Gateway events and 
collects metrics and logs for observability purposes.
"""

import logging
import time
from typing import Dict, Any, Optional, Set, Type

from asf.medical.llm_gateway.events.events import (
    Event, RequestReceivedEvent, ProviderCalledEvent, ResponseSentEvent,
    ResponseStartedEvent, ErrorOccurredEvent, GatewayStartedEvent,
    GatewayShuttingDownEvent, ProviderStatusChangedEvent
)
from asf.medical.llm_gateway.events.subscriber import EventSubscriber, handles_event
from asf.medical.llm_gateway.metrics import MetricsService, get_metrics_service

logger = logging.getLogger(__name__)


class ObservabilitySubscriber(EventSubscriber):
    """
    A subscriber that collects metrics and logs from LLM Gateway events.
    
    This subscriber listens to various events in the LLM Gateway and reports
    metrics to the metrics service. It can also log important events for
    debugging and monitoring purposes.
    """
    
    def __init__(self, metrics_service: Optional[MetricsService] = None):
        """
        Initialize the observability subscriber.
        
        Args:
            metrics_service: Optional metrics service to use. If not provided,
                will use the global instance.
        """
        super().__init__()
        self.metrics_service = metrics_service or get_metrics_service()
        self.start_time = time.time()
        self.request_start_times: Dict[str, float] = {}
        self.active_requests = 0
        logger.info("ObservabilitySubscriber initialized")
        
    @handles_event(RequestReceivedEvent)
    async def handle_request_received(self, event: RequestReceivedEvent) -> None:
        """
        Handle request received events.
        
        Args:
            event: The request received event
        """
        # Track request start time for latency calculation
        self.request_start_times[event.request_id] = event.timestamp
        self.active_requests += 1
        
        # Record metrics
        self.metrics_service.increment_counter(
            "llm_gateway_requests_total",
            {"request_type": event.request_type, "model": event.model or "unknown"}
        )
        self.metrics_service.set_gauge(
            "llm_gateway_active_requests",
            self.active_requests
        )
        
        logger.debug(f"Request received: {event.request_id} of type {event.request_type}")
        
    @handles_event(ProviderCalledEvent)
    async def handle_provider_called(self, event: ProviderCalledEvent) -> None:
        """
        Handle provider called events.
        
        Args:
            event: The provider called event
        """
        # Record provider usage
        self.metrics_service.increment_counter(
            "llm_gateway_provider_calls_total",
            {"provider": event.provider_id, "model": event.model or "unknown", 
             "operation": event.operation_type}
        )
        
        # Record input tokens if available
        if event.input_tokens is not None:
            self.metrics_service.add_to_distribution(
                "llm_gateway_input_tokens",
                event.input_tokens,
                {"provider": event.provider_id, "model": event.model or "unknown"}
            )
            
        logger.debug(f"Provider called: {event.provider_id} for request {event.request_id}")
        
    @handles_event(ResponseStartedEvent)
    async def handle_response_started(self, event: ResponseStartedEvent) -> None:
        """
        Handle response started events.
        
        Args:
            event: The response started event
        """
        # Record latency from request to first response
        start_time = self.request_start_times.get(event.request_id)
        if start_time:
            latency = (event.timestamp - start_time) * 1000  # Convert to ms
            
            self.metrics_service.observe_histogram(
                "llm_gateway_response_latency_ms",
                latency,
                {"provider": event.provider_id, "model": event.model or "unknown", 
                 "streaming": str(event.streaming)}
            )
            
        logger.debug(f"Response started for request {event.request_id} with latency {event.latency_ms:.2f}ms")
        
    @handles_event(ResponseSentEvent)
    async def handle_response_sent(self, event: ResponseSentEvent) -> None:
        """
        Handle response sent events.
        
        Args:
            event: The response sent event
        """
        # Clean up tracking for this request
        self.request_start_times.pop(event.request_id, None)
        self.active_requests = max(0, self.active_requests - 1)
        
        # Record metrics
        self.metrics_service.set_gauge(
            "llm_gateway_active_requests",
            self.active_requests
        )
        
        self.metrics_service.observe_histogram(
            "llm_gateway_request_duration_ms",
            event.duration_ms,
            {"provider": event.provider_id, "model": event.model or "unknown", 
             "success": str(event.success), "streaming": str(event.streaming)}
        )
        
        # Record token counts if available
        if event.output_tokens is not None:
            self.metrics_service.add_to_distribution(
                "llm_gateway_output_tokens",
                event.output_tokens,
                {"provider": event.provider_id, "model": event.model or "unknown"}
            )
            
        if event.total_tokens is not None:
            self.metrics_service.add_to_distribution(
                "llm_gateway_total_tokens",
                event.total_tokens,
                {"provider": event.provider_id, "model": event.model or "unknown"}
            )
            
        logger.debug(f"Response completed for request {event.request_id} in {event.duration_ms:.2f}ms")
        
    @handles_event(ErrorOccurredEvent)
    async def handle_error_occurred(self, event: ErrorOccurredEvent) -> None:
        """
        Handle error events.
        
        Args:
            event: The error event
        """
        # Record errors
        self.metrics_service.increment_counter(
            "llm_gateway_errors_total",
            {"provider": event.provider_id, "model": event.model or "unknown", 
             "error_type": event.error_type, "operation": event.operation_type or "unknown"}
        )
        
        # Clean up tracking if this was for a request
        if event.request_id in self.request_start_times:
            self.request_start_times.pop(event.request_id)
            self.active_requests = max(0, self.active_requests - 1)
            self.metrics_service.set_gauge(
                "llm_gateway_active_requests",
                self.active_requests
            )
            
        logger.warning(f"Error in LLM Gateway: {event.error_type}: {event.error_message} "
                      f"for request {event.request_id}")
        
    @handles_event(GatewayStartedEvent)
    async def handle_gateway_started(self, event: GatewayStartedEvent) -> None:
        """
        Handle gateway started events.
        
        Args:
            event: The gateway started event
        """
        self.start_time = event.timestamp
        self.metrics_service.set_gauge(
            "llm_gateway_up",
            1,
            {"gateway_id": event.gateway_id}
        )
        
        # Record enabled features
        for feature in event.features:
            self.metrics_service.set_gauge(
                "llm_gateway_feature_enabled",
                1,
                {"gateway_id": event.gateway_id, "feature": feature}
            )
            
        logger.info(f"Gateway {event.gateway_id} started with {len(event.features)} features")
        
    @handles_event(GatewayShuttingDownEvent)
    async def handle_gateway_shutting_down(self, event: GatewayShuttingDownEvent) -> None:
        """
        Handle gateway shutting down events.
        
        Args:
            event: The gateway shutting down event
        """
        self.metrics_service.set_gauge(
            "llm_gateway_up",
            0,
            {"gateway_id": event.gateway_id}
        )
        
        logger.info(f"Gateway {event.gateway_id} shutting down after "
                   f"{event.uptime_seconds:.1f} seconds, reason: {event.reason}")
        
    @handles_event(ProviderStatusChangedEvent)
    async def handle_provider_status_changed(self, event: ProviderStatusChangedEvent) -> None:
        """
        Handle provider status changed events.
        
        Args:
            event: The provider status changed event
        """
        # Update provider status in metrics
        status_value = 1 if event.new_status.lower() == "available" else 0
        
        self.metrics_service.set_gauge(
            "llm_gateway_provider_available",
            status_value,
            {"provider": event.provider_id, "model": event.model or "unknown"}
        )
        
        logger.info(f"Provider {event.provider_id} status changed from "
                   f"{event.previous_status} to {event.new_status}")
                   
    async def report_periodic_metrics(self) -> None:
        """
        Report periodic metrics that aren't tied to specific events.
        
        This method should be called periodically to update metrics
        like uptime, request rate over time, etc.
        """
        # Report uptime
        uptime_seconds = time.time() - self.start_time
        self.metrics_service.set_gauge("llm_gateway_uptime_seconds", uptime_seconds)
        
        # Report memory usage metrics (if available)
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.metrics_service.set_gauge(
                "llm_gateway_memory_usage_bytes", 
                memory_info.rss
            )
        except (ImportError, Exception):
            # Skip memory metrics if psutil isn't available
            pass


def create_observability_subscriber() -> ObservabilitySubscriber:
    """
    Create and initialize an observability subscriber.
    
    Returns:
        A new ObservabilitySubscriber instance
    """
    subscriber = ObservabilitySubscriber()
    return subscriber