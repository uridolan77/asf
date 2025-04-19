"""
Metrics service for MCP Provider observability.

This module provides a metrics collection and reporting service using
OpenTelemetry as the underlying infrastructure with Prometheus export.

NOTE: This version has been modified to disable metrics functionality.
"""

import os
import time
from typing import Any, Dict, List, Optional, Set

import structlog

# Check if metrics are disabled
METRICS_DISABLED = os.environ.get("DISABLE_METRICS", "0").lower() in ("1", "true", "yes")
PROMETHEUS_DISABLED = os.environ.get("DISABLE_PROMETHEUS", "0").lower() in ("1", "true", "yes")
OBSERVABILITY_DISABLED = os.environ.get("DISABLE_OBSERVABILITY", "0").lower() in ("1", "true", "yes")

logger = structlog.get_logger("mcp_observability.metrics")

# Import dummy implementations if metrics are disabled
if METRICS_DISABLED or PROMETHEUS_DISABLED or OBSERVABILITY_DISABLED:
    from .disable_patch import DummyMetricsService as MetricsService

    logger.info("Metrics service disabled via environment variable")
else:
    # Import real implementations if metrics are enabled
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.metrics import Histogram, Counter, UpDownCounter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from prometheus_client import start_http_server


class MetricsService:
    """
    Metrics collection and reporting service for MCP Provider.

    Features:
    - OpenTelemetry metrics with Prometheus export
    - Standard metrics for requests, responses, errors
    - Circuit breaker and resilience metrics
    - Session and transport metrics
    - Configurable aggregation and dimensions
    """

    def __init__(
        self,
        service_name: str = "mcp_provider",
        namespace: str = "mcp",
        enable_prometheus: bool = True,
        prometheus_port: int = 8000,
        export_interval_ms: int = 30000,
        additional_dimensions: Optional[Dict[str, str]] = None
    ):
        """
        Initialize metrics service.

        Args:
            service_name: Name of the service for metrics context
            namespace: Metrics namespace prefix
            enable_prometheus: Whether to start Prometheus HTTP server
            prometheus_port: Port for Prometheus HTTP server
            export_interval_ms: Milliseconds between metric exports
            additional_dimensions: Static dimensions to add to all metrics
        """
        self.service_name = service_name
        self.namespace = namespace
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        self.export_interval_ms = export_interval_ms
        self.additional_dimensions = additional_dimensions or {}

        # Initialize metrics infrastructure
        self._init_metrics_infrastructure()

        # Create standard metrics
        self._create_standard_metrics()

        self.logger = logger.bind(service=service_name)

        self.logger.info(
            "Initialized metrics service",
            prometheus_enabled=enable_prometheus,
            prometheus_port=prometheus_port if enable_prometheus else None
        )

    def _init_metrics_infrastructure(self) -> None:
        """Initialize OpenTelemetry metrics infrastructure."""
        # Skip initialization if metrics are disabled
        if os.environ.get("DISABLE_METRICS", "0").lower() in ("1", "true", "yes") or \
           os.environ.get("DISABLE_OBSERVABILITY", "0").lower() in ("1", "true", "yes"):
            logger.info("Metrics infrastructure initialization skipped due to environment variables")
            return

        try:
            # Create resource with service info
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                **self.additional_dimensions
            })
        except NameError:
            logger.warning("Resource not defined. Using dummy implementation.")
            # Create a dummy resource
            resource = None

        # Set up metric readers
        readers = []

        # Add Prometheus reader if enabled
        if self.enable_prometheus:
            prometheus_reader = PrometheusMetricReader()
            readers.append(prometheus_reader)

            # Start Prometheus HTTP server
            start_http_server(port=self.prometheus_port)
            logger.info(f"Started Prometheus HTTP server on port {self.prometheus_port}")

        # Create meter provider
        try:
            if resource is None:
                self.meter_provider = MeterProvider(metric_readers=readers)
            else:
                self.meter_provider = MeterProvider(
                    resource=resource,
                    metric_readers=readers
                )
        except Exception as e:
            logger.warning(f"Failed to create meter provider: {str(e)}")
            # Create a dummy meter provider
            class DummyMeterProvider:
                def get_meter(self, *args, **kwargs):
                    return DummyMeter()

            self.meter_provider = DummyMeterProvider()

            # Create a dummy meter
            class DummyMeter:
                def create_counter(self, *args, **kwargs):
                    return DummyMetric()

                def create_histogram(self, *args, **kwargs):
                    return DummyMetric()

                def create_up_down_counter(self, *args, **kwargs):
                    return DummyMetric()

            # Create a dummy metric
            class DummyMetric:
                def add(self, *args, **kwargs):
                    pass

                def record(self, *args, **kwargs):
                    pass

        # Set global meter provider
        metrics.set_meter_provider(self.meter_provider)

        # Create meter for this service
        self.meter = metrics.get_meter(
            f"{self.namespace}.{self.service_name}",
            schema_url=f"https://opentelemetry.io/schemas/1.9.0"
        )

    def _create_standard_metrics(self) -> None:
        """Create standard metrics for MCP provider monitoring."""
        # Request metrics
        self.request_counter = self.meter.create_counter(
            name=f"{self.namespace}_requests_total",
            description="Total number of requests to MCP provider",
            unit="1"
        )

        self.request_duration = self.meter.create_histogram(
            name=f"{self.namespace}_request_duration_seconds",
            description="Duration of MCP requests",
            unit="s"
        )

        self.request_size = self.meter.create_histogram(
            name=f"{self.namespace}_request_size_bytes",
            description="Size of MCP requests in bytes",
            unit="By"
        )

        self.response_size = self.meter.create_histogram(
            name=f"{self.namespace}_response_size_bytes",
            description="Size of MCP responses in bytes",
            unit="By"
        )

        # Error metrics
        self.error_counter = self.meter.create_counter(
            name=f"{self.namespace}_errors_total",
            description="Total number of errors from MCP provider",
            unit="1"
        )

        # Token metrics
        self.prompt_tokens = self.meter.create_counter(
            name=f"{self.namespace}_prompt_tokens_total",
            description="Total number of prompt tokens",
            unit="1"
        )

        self.completion_tokens = self.meter.create_counter(
            name=f"{self.namespace}_completion_tokens_total",
            description="Total number of completion tokens",
            unit="1"
        )

        # Session metrics
        self.session_created = self.meter.create_counter(
            name=f"{self.namespace}_sessions_created_total",
            description="Total number of MCP sessions created",
            unit="1"
        )

        self.session_errors = self.meter.create_counter(
            name=f"{self.namespace}_session_errors_total",
            description="Total number of MCP session errors",
            unit="1"
        )

        self.active_sessions = self.meter.create_up_down_counter(
            name=f"{self.namespace}_active_sessions",
            description="Current number of active MCP sessions",
            unit="1"
        )

        # Circuit breaker metrics
        self.circuit_open = self.meter.create_up_down_counter(
            name=f"{self.namespace}_circuit_breaker_open",
            description="Status of MCP circuit breakers (1=open, 0=closed)",
            unit="1"
        )

        # Retry metrics
        self.retry_counter = self.meter.create_counter(
            name=f"{self.namespace}_retries_total",
            description="Total number of retry attempts",
            unit="1"
        )

        # Streaming metrics
        self.stream_chunks = self.meter.create_counter(
            name=f"{self.namespace}_stream_chunks_total",
            description="Total number of stream chunks",
            unit="1"
        )

        self.stream_errors = self.meter.create_counter(
            name=f"{self.namespace}_stream_errors_total",
            description="Total number of errors during streaming",
            unit="1"
        )

    def record_request_start(
        self,
        provider_id: str,
        model: str,
        request_id: str,
        streaming: bool = False
    ) -> Dict[str, Any]:
        """
        Record the start of an MCP request.

        Args:
            provider_id: ID of the provider
            model: Model identifier
            request_id: Request ID for correlation
            streaming: Whether this is a streaming request

        Returns:
            Context dict with timing info to pass to record_request_end
        """
        # Record request count
        self.request_counter.add(
            1,
            {
                "provider": provider_id,
                "model": model,
                "streaming": str(streaming).lower()
            }
        )

        # Return context with start time
        return {
            "start_time": time.time(),
            "provider_id": provider_id,
            "model": model,
            "request_id": request_id,
            "streaming": streaming
        }

    def record_request_end(
        self,
        context: Dict[str, Any],
        status: str = "success",
        error_code: Optional[str] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        request_size: int = 0,
        response_size: int = 0
    ) -> None:
        """
        Record the completion of an MCP request.

        Args:
            context: Context from record_request_start
            status: Status of the request (success, error)
            error_code: Error code if status is error
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            request_size: Size of request in bytes
            response_size: Size of response in bytes
        """
        # Extract context values
        start_time = context["start_time"]
        provider_id = context["provider_id"]
        model = context["model"]
        request_id = context["request_id"]
        streaming = context["streaming"]

        # Calculate duration
        duration = time.time() - start_time

        # Record duration
        self.request_duration.record(
            duration,
            {
                "provider": provider_id,
                "model": model,
                "status": status,
                "streaming": str(streaming).lower()
            }
        )

        # Record tokens
        if prompt_tokens > 0:
            self.prompt_tokens.add(
                prompt_tokens,
                {
                    "provider": provider_id,
                    "model": model,
                    "request_id": request_id
                }
            )

        if completion_tokens > 0:
            self.completion_tokens.add(
                completion_tokens,
                {
                    "provider": provider_id,
                    "model": model,
                    "request_id": request_id
                }
            )

        # Record request/response size if provided
        if request_size > 0:
            self.request_size.record(
                request_size,
                {
                    "provider": provider_id,
                    "model": model
                }
            )

        if response_size > 0:
            self.response_size.record(
                response_size,
                {
                    "provider": provider_id,
                    "model": model
                }
            )

        # Record error if applicable
        if status == "error" and error_code:
            self.error_counter.add(
                1,
                {
                    "provider": provider_id,
                    "model": model,
                    "error_code": error_code,
                    "streaming": str(streaming).lower()
                }
            )

    def record_session_created(self, provider_id: str, transport_type: str) -> None:
        """
        Record creation of a new MCP session.

        Args:
            provider_id: ID of the provider
            transport_type: Type of transport
        """
        self.session_created.add(
            1,
            {
                "provider": provider_id,
                "transport": transport_type
            }
        )

        # Update active sessions gauge
        self.active_sessions.add(
            1,
            {
                "provider": provider_id,
                "transport": transport_type
            }
        )

    def record_session_closed(self, provider_id: str, transport_type: str) -> None:
        """
        Record closing of an MCP session.

        Args:
            provider_id: ID of the provider
            transport_type: Type of transport
        """
        # Update active sessions gauge
        self.active_sessions.add(
            -1,
            {
                "provider": provider_id,
                "transport": transport_type
            }
        )

    def record_session_error(self, provider_id: str, transport_type: str, error_type: str) -> None:
        """
        Record an MCP session error.

        Args:
            provider_id: ID of the provider
            transport_type: Type of transport
            error_type: Type of error
        """
        self.session_errors.add(
            1,
            {
                "provider": provider_id,
                "transport": transport_type,
                "error_type": error_type
            }
        )

    def record_circuit_breaker_state(self, name: str, provider_id: str, is_open: bool) -> None:
        """
        Record circuit breaker state change.

        Args:
            name: Name of the circuit breaker
            provider_id: ID of the provider
            is_open: Whether the circuit is open
        """
        self.circuit_open.add(
            1 if is_open else 0,
            {
                "name": name,
                "provider": provider_id
            }
        )

    def record_retry(self, provider_id: str, attempt: int, successful: bool, error_code: Optional[str] = None) -> None:
        """
        Record a retry attempt.

        Args:
            provider_id: ID of the provider
            attempt: Attempt number (1-based)
            successful: Whether the retry was successful
            error_code: Error code that triggered the retry
        """
        self.retry_counter.add(
            1,
            {
                "provider": provider_id,
                "attempt": str(attempt),
                "successful": str(successful).lower(),
                "error_code": error_code or "unknown"
            }
        )

    def record_stream_chunk(self, provider_id: str, model: str, chunk_index: int) -> None:
        """
        Record a stream chunk.

        Args:
            provider_id: ID of the provider
            model: Model identifier
            chunk_index: Index of the chunk in the stream
        """
        self.stream_chunks.add(
            1,
            {
                "provider": provider_id,
                "model": model,
                "chunk_index": str(chunk_index)
            }
        )

    def record_stream_error(self, provider_id: str, model: str, error_code: str) -> None:
        """
        Record a streaming error.

        Args:
            provider_id: ID of the provider
            model: Model identifier
            error_code: Error code
        """
        self.stream_errors.add(
            1,
            {
                "provider": provider_id,
                "model": model,
                "error_code": error_code
            }
        )