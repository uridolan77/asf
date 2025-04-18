"""
Prometheus metrics integration for LLM Gateway.

This module provides Prometheus metrics collection and export functionality
for monitoring LLM Gateway performance and usage.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Set
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry, push_to_gateway

logger = logging.getLogger(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define metrics
# Request metrics
request_counter = Counter(
    'llm_gateway_requests_total',
    'Total number of requests processed by the LLM Gateway',
    ['provider', 'model', 'status'],
    registry=registry
)

request_duration = Histogram(
    'llm_gateway_request_duration_seconds',
    'Request duration in seconds',
    ['provider', 'model', 'status'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
    registry=registry
)

# Token metrics
token_counter = Counter(
    'llm_gateway_tokens_total',
    'Total number of tokens processed by the LLM Gateway',
    ['provider', 'model', 'type'],  # type: input, output
    registry=registry
)

# Error metrics
error_counter = Counter(
    'llm_gateway_errors_total',
    'Total number of errors encountered by the LLM Gateway',
    ['provider', 'model', 'error_type'],
    registry=registry
)

# Circuit breaker metrics
circuit_breaker_state = Gauge(
    'llm_gateway_circuit_breaker_state',
    'Circuit breaker state (0=closed, 1=open)',
    ['provider'],
    registry=registry
)

circuit_breaker_failures = Counter(
    'llm_gateway_circuit_breaker_failures_total',
    'Total number of failures tracked by circuit breakers',
    ['provider'],
    registry=registry
)

# Provider metrics
provider_info = Info(
    'llm_gateway_provider_info',
    'Information about LLM providers',
    ['provider'],
    registry=registry
)

provider_models = Gauge(
    'llm_gateway_provider_models',
    'Number of models available per provider',
    ['provider'],
    registry=registry
)

provider_session_count = Gauge(
    'llm_gateway_provider_sessions',
    'Number of active sessions per provider',
    ['provider'],
    registry=registry
)

# Connection pool metrics
connection_pool_size = Gauge(
    'llm_gateway_connection_pool_size',
    'Size of the connection pool',
    ['provider', 'transport_type'],
    registry=registry
)

connection_pool_active = Gauge(
    'llm_gateway_connection_pool_active',
    'Number of active connections in the pool',
    ['provider', 'transport_type'],
    registry=registry
)

connection_errors = Counter(
    'llm_gateway_connection_errors_total',
    'Total number of connection errors',
    ['provider', 'transport_type', 'error_type'],
    registry=registry
)

# Cache metrics
cache_hits = Counter(
    'llm_gateway_cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses = Counter(
    'llm_gateway_cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

cache_size = Gauge(
    'llm_gateway_cache_size',
    'Current size of the cache',
    ['cache_type'],
    registry=registry
)

# WebSocket metrics
websocket_connections = Gauge(
    'llm_gateway_websocket_connections',
    'Number of active WebSocket connections',
    [],
    registry=registry
)

websocket_messages = Counter(
    'llm_gateway_websocket_messages_total',
    'Total number of WebSocket messages',
    ['direction', 'message_type'],  # direction: sent, received
    registry=registry
)

websocket_errors = Counter(
    'llm_gateway_websocket_errors_total',
    'Total number of WebSocket errors',
    ['error_type'],
    registry=registry
)


class PrometheusExporter:
    """
    Prometheus metrics exporter for LLM Gateway.

    This class provides methods for recording metrics and pushing them to a Prometheus
    push gateway or exposing them via HTTP endpoint.
    """

    def __init__(
        self,
        push_gateway_url: Optional[str] = None,
        push_interval_seconds: int = 15,
        job_name: str = 'llm_gateway'
    ):
        """
        Initialize the Prometheus exporter.

        Args:
            push_gateway_url: URL of the Prometheus push gateway (optional)
            push_interval_seconds: Interval in seconds for pushing metrics
            job_name: Job name for the push gateway
        """
        self.push_gateway_url = push_gateway_url
        self.push_interval_seconds = push_interval_seconds
        self.job_name = job_name
        self.last_push_time = 0
        self.registry = registry

        # Track providers and models
        self.known_providers: Set[str] = set()
        self.provider_model_counts: Dict[str, int] = {}

        logger.info(
            f"Initialized Prometheus exporter (push_gateway={push_gateway_url is not None}, push_interval={push_interval_seconds})"
        )

    def record_request(
        self,
        provider_id: str,
        model: str,
        status: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int
    ) -> None:
        """
        Record metrics for a request.

        Args:
            provider_id: Provider identifier
            model: Model identifier
            status: Request status (success, error)
            duration_seconds: Request duration in seconds
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        """
        # Update request metrics
        request_counter.labels(provider=provider_id, model=model, status=status).inc()
        request_duration.labels(provider=provider_id, model=model, status=status).observe(duration_seconds)

        # Update token metrics
        token_counter.labels(provider=provider_id, model=model, type='input').inc(input_tokens)
        token_counter.labels(provider=provider_id, model=model, type='output').inc(output_tokens)

        # Track provider
        self._track_provider(provider_id)

        # Push metrics if needed
        self._maybe_push_metrics()

    def record_error(
        self,
        provider_id: str,
        model: str,
        error_type: str
    ) -> None:
        """
        Record an error.

        Args:
            provider_id: Provider identifier
            model: Model identifier
            error_type: Type of error
        """
        error_counter.labels(provider=provider_id, model=model, error_type=error_type).inc()

        # Track provider
        self._track_provider(provider_id)

        # Push metrics if needed
        self._maybe_push_metrics()

    def update_circuit_breaker(
        self,
        provider_id: str,
        is_open: bool,
        failure_count: int
    ) -> None:
        """
        Update circuit breaker metrics.

        Args:
            provider_id: Provider identifier
            is_open: Whether the circuit breaker is open
            failure_count: Number of failures
        """
        circuit_breaker_state.labels(provider=provider_id).set(1 if is_open else 0)
        circuit_breaker_failures.labels(provider=provider_id).inc(failure_count)

        # Track provider
        self._track_provider(provider_id)

        # Push metrics if needed
        self._maybe_push_metrics()

    def update_provider_info(
        self,
        provider_id: str,
        info: Dict[str, Any]
    ) -> None:
        """
        Update provider information.

        Args:
            provider_id: Provider identifier
            info: Provider information
        """
        # Convert dict to key-value pairs
        info_str = {k: str(v) for k, v in info.items() if v is not None}
        provider_info.labels(provider=provider_id).info(info_str)

        # Track provider
        self._track_provider(provider_id)

        # Update model count if available
        if 'models' in info and isinstance(info['models'], list):
            self.provider_model_counts[provider_id] = len(info['models'])
            provider_models.labels(provider=provider_id).set(len(info['models']))

        # Update session count if available
        if 'session_count' in info:
            provider_session_count.labels(provider=provider_id).set(info['session_count'])

        # Push metrics if needed
        self._maybe_push_metrics()

    def update_connection_pool(
        self,
        provider_id: str,
        transport_type: str,
        pool_size: int,
        active_connections: int
    ) -> None:
        """
        Update connection pool metrics.

        Args:
            provider_id: Provider identifier
            transport_type: Transport type (http, grpc, etc.)
            pool_size: Size of the connection pool
            active_connections: Number of active connections
        """
        connection_pool_size.labels(provider=provider_id, transport_type=transport_type).set(pool_size)
        connection_pool_active.labels(provider=provider_id, transport_type=transport_type).set(active_connections)

        # Track provider
        self._track_provider(provider_id)

        # Push metrics if needed
        self._maybe_push_metrics()

    def record_connection_error(
        self,
        provider_id: str,
        transport_type: str,
        error_type: str
    ) -> None:
        """
        Record a connection error.

        Args:
            provider_id: Provider identifier
            transport_type: Transport type (http, grpc, etc.)
            error_type: Type of error
        """
        connection_errors.labels(provider=provider_id, transport_type=transport_type, error_type=error_type).inc()

        # Track provider
        self._track_provider(provider_id)

        # Push metrics if needed
        self._maybe_push_metrics()

    def update_cache_metrics(
        self,
        cache_type: str,
        hits: int = 0,
        misses: int = 0,
        size: Optional[int] = None
    ) -> None:
        """
        Update cache metrics.

        Args:
            cache_type: Type of cache (response, embedding, etc.)
            hits: Number of cache hits
            misses: Number of cache misses
            size: Current size of the cache
        """
        if hits > 0:
            cache_hits.labels(cache_type=cache_type).inc(hits)

        if misses > 0:
            cache_misses.labels(cache_type=cache_type).inc(misses)

        if size is not None:
            cache_size.labels(cache_type=cache_type).set(size)

        # Push metrics if needed
        self._maybe_push_metrics()

    def update_websocket_metrics(
        self,
        active_connections: int,
        sent_messages: Optional[Dict[str, int]] = None,
        received_messages: Optional[Dict[str, int]] = None,
        errors: Optional[Dict[str, int]] = None
    ) -> None:
        """
        Update WebSocket metrics.

        Args:
            active_connections: Number of active WebSocket connections
            sent_messages: Dictionary of sent message types and counts
            received_messages: Dictionary of received message types and counts
            errors: Dictionary of error types and counts
        """
        websocket_connections.set(active_connections)

        if sent_messages:
            for message_type, count in sent_messages.items():
                websocket_messages.labels(direction='sent', message_type=message_type).inc(count)

        if received_messages:
            for message_type, count in received_messages.items():
                websocket_messages.labels(direction='received', message_type=message_type).inc(count)

        if errors:
            for error_type, count in errors.items():
                websocket_errors.labels(error_type=error_type).inc(count)

        # Push metrics if needed
        self._maybe_push_metrics()

    def _track_provider(self, provider_id: str) -> None:
        """
        Track a provider.

        Args:
            provider_id: Provider identifier
        """
        if provider_id not in self.known_providers:
            self.known_providers.add(provider_id)
            logger.info(f"Tracking new provider: {provider_id}")

    def _maybe_push_metrics(self) -> None:
        """
        Push metrics to the Prometheus push gateway if the interval has elapsed.
        """
        if not self.push_gateway_url:
            return

        now = time.time()
        if now - self.last_push_time >= self.push_interval_seconds:
            try:
                push_to_gateway(
                    self.push_gateway_url,
                    job=self.job_name,
                    registry=self.registry
                )
                self.last_push_time = now
                logger.debug("Pushed metrics to Prometheus push gateway")
            except Exception as e:
                logger.error(
                    f"Failed to push metrics to Prometheus push gateway: {str(e)}",
                    exc_info=True
                )

    def push_metrics(self) -> None:
        """
        Manually push metrics to the Prometheus push gateway.
        """
        if not self.push_gateway_url:
            logger.warning("No push gateway URL configured")
            return

        try:
            push_to_gateway(
                self.push_gateway_url,
                job=self.job_name,
                registry=self.registry
            )
            self.last_push_time = time.time()
            logger.info("Manually pushed metrics to Prometheus push gateway")
        except Exception as e:
            logger.error(
                f"Failed to manually push metrics to Prometheus push gateway: {str(e)}",
                exc_info=True
            )


# Create a global instance
exporter = PrometheusExporter()


def get_prometheus_exporter() -> PrometheusExporter:
    """
    Get the global Prometheus exporter instance.

    Returns:
        PrometheusExporter instance
    """
    return exporter


def configure_prometheus_exporter(
    push_gateway_url: Optional[str] = None,
    push_interval_seconds: int = 15,
    job_name: str = 'llm_gateway'
) -> PrometheusExporter:
    """
    Configure the global Prometheus exporter.

    Args:
        push_gateway_url: URL of the Prometheus push gateway (optional)
        push_interval_seconds: Interval in seconds for pushing metrics
        job_name: Job name for the push gateway

    Returns:
        Configured PrometheusExporter instance
    """
    global exporter
    exporter = PrometheusExporter(
        push_gateway_url=push_gateway_url,
        push_interval_seconds=push_interval_seconds,
        job_name=job_name
    )
    return exporter
