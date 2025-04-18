"""
Observability module for the Medical Research Synthesizer.

This module provides a unified observability system including metrics collection,
distributed tracing, and performance monitoring with support for both local and 
distributed environments.

Classes:
    MetricsRegistry: Registry for application metrics.
    Timer: Context manager for timing code execution.
    Counter: Counter for tracking occurrences of events.
    Gauge: Gauge for tracking current values of metrics.
    Histogram: Histogram for tracking distributions of values.
    Tracer: Interface for distributed tracing.
    OpenTelemetryTracer: OpenTelemetry implementation of the Tracer interface.
    PrometheusMetricsCollector: Prometheus implementation of metrics collection.

Functions:
    init_observability: Initialize the observability system.
    get_metrics: Get the current metrics.
    increment_counter: Increment a counter metric.
    set_gauge: Set a gauge metric.
    record_histogram: Record a histogram metric.
    timed: Decorator for timing function execution.
    async_timed: Async decorator for timing coroutine execution.
    setup_tracing: Set up distributed tracing.
    setup_metrics: Set up metrics collection.
    log_error: Log an error with context.
    log_request: Log an API request with timing information.
    trace: Decorator for adding trace spans to functions.
"""
import os
import time
import logging
import traceback
import threading
import json
import socket
from typing import Dict, Any, Optional, Callable, Generator, Tuple, List, Union
from functools import wraps
from datetime import datetime
from contextlib import contextmanager
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Environment configuration
DEFAULT_LOKI_URL = "http://localhost:3100/loki/api/v1/push"
DEFAULT_TEMPO_URL = "http://localhost:14268/api/traces"
DEFAULT_PROMETHEUS_URL = "http://localhost:9090/api/v1/push"
LOKI_URL = os.environ.get("LOKI_URL", DEFAULT_LOKI_URL)
TEMPO_URL = os.environ.get("TEMPO_URL", DEFAULT_TEMPO_URL)
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
PUSH_GATEWAY_URL = os.environ.get("PUSH_GATEWAY_URL", "localhost:9091")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "medical-research-synthesizer")
HOSTNAME = socket.gethostname()

# Shared metrics storage for simple monitoring
_metrics = {
    "counters": {},
    "gauges": {},
    "histograms": {},
    "timers": {}
}
_metrics_lock = threading.RLock()
_health_checks = {}

#
# Base Classes and Interfaces
#

class MetricType:
    """Metric types enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    TIMER = "timer"

class Tracer(ABC):
    """
    Abstract base class for tracers.

    This class defines the interface for distributed tracing implementations.

    Attributes:
        service_name (str): Name of the service being traced.
    """

    def __init__(self, service_name: str):
        """
        Initialize the Tracer instance.

        Args:
            service_name (str): Name of the service being traced.
        """
        self.service_name = service_name

    @abstractmethod
    def start_span(self, name: str, context: Dict[str, Any] = None) -> Any:
        """
        Start a new trace span.

        Args:
            name (str): Name of the span.
            context (Dict[str, Any], optional): Context information for the span. Defaults to None.

        Returns:
            Any: Span object.
        """
        pass

    @abstractmethod
    def end_span(self, span: Any) -> None:
        """
        End a trace span.

        Args:
            span (Any): Span object to end.
        """
        pass

    @contextmanager
    def span(self, name: str, context: Dict[str, Any] = None) -> Generator:
        """
        Context manager for trace spans.

        Args:
            name (str): Name of the span.
            context (Dict[str, Any], optional): Context information for the span. Defaults to None.

        Yields:
            Any: Span object.
        """
        span = self.start_span(name, context)
        try:
            yield span
        finally:
            self.end_span(span)

class MetricsCollector(ABC):
    """
    Abstract base class for metrics collectors.

    This class defines the interface for metrics collection implementations.

    Attributes:
        service_name (str): Name of the service collecting metrics.
    """

    def __init__(self, service_name: str):
        """
        Initialize the MetricsCollector instance.

        Args:
            service_name (str): Name of the service collecting metrics.
        """
        self.service_name = service_name

    @abstractmethod
    def record_counter(self, name: str, value: int, dimensions: Dict[str, str] = None) -> None:
        """
        Record a counter metric.

        Args:
            name (str): Name of the counter.
            value (int): Value to increment by.
            dimensions (Dict[str, str], optional): Dimensions for the counter. Defaults to None.
        """
        pass

    @abstractmethod
    def record_gauge(self, name: str, value: float, dimensions: Dict[str, str] = None) -> None:
        """
        Record a gauge metric.

        Args:
            name (str): Name of the gauge.
            value (float): Value to set.
            dimensions (Dict[str, str], optional): Dimensions for the gauge. Defaults to None.
        """
        pass

    @abstractmethod
    def record_histogram(self, name: str, value: float, dimensions: Dict[str, str] = None) -> None:
        """
        Record a histogram metric.

        Args:
            name (str): Name of the histogram.
            value (float): Value to record.
            dimensions (Dict[str, str], optional): Dimensions for the histogram. Defaults to None.
        """
        pass

#
# Simple Local Metrics Implementation
#

class MetricsRegistry:
    """
    Registry for application metrics.

    This class maintains a registry of metrics for the application,
    including counters, gauges, timers, and histograms.

    Attributes:
        metrics (Dict[str, Any]): Dictionary of metrics by name.
    """

    def __init__(self):
        """Initialize the MetricsRegistry instance."""
        self.metrics = {}

    def register_counter(self, name: str, description: str = "", tags: Dict[str, str] = None) -> 'Counter':
        """
        Register a counter metric.

        Args:
            name (str): Name of the counter.
            description (str, optional): Description of the counter. Defaults to "".
            tags (Dict[str, str], optional): Tags for the counter. Defaults to None.

        Returns:
            Counter: The registered counter.
        """
        counter = Counter(name, description, tags, self)
        self.metrics[name] = counter
        return counter

    def register_gauge(self, name: str, description: str = "", tags: Dict[str, str] = None) -> 'Gauge':
        """
        Register a gauge metric.

        Args:
            name (str): Name of the gauge.
            description (str, optional): Description of the gauge. Defaults to "".
            tags (Dict[str, str], optional): Tags for the gauge. Defaults to None.

        Returns:
            Gauge: The registered gauge.
        """
        gauge = Gauge(name, description, tags, self)
        self.metrics[name] = gauge
        return gauge

    def register_histogram(self, name: str, description: str = "", tags: Dict[str, str] = None,
                           buckets: List[float] = None) -> 'Histogram':
        """
        Register a histogram metric.

        Args:
            name (str): Name of the histogram.
            description (str, optional): Description of the histogram. Defaults to "".
            tags (Dict[str, str], optional): Tags for the histogram. Defaults to None.
            buckets (List[float], optional): Bucket boundaries. Defaults to None.

        Returns:
            Histogram: The registered histogram.
        """
        histogram = Histogram(name, description, tags, self, buckets)
        self.metrics[name] = histogram
        return histogram

    def register_timer(self, name: str, description: str = "", tags: Dict[str, str] = None) -> 'Timer':
        """
        Register a timer metric.

        Args:
            name (str): Name of the timer.
            description (str, optional): Description of the timer. Defaults to "".
            tags (Dict[str, str], optional): Tags for the timer. Defaults to None.

        Returns:
            Timer: The registered timer.
        """
        timer = Timer(name, description, tags, self)
        self.metrics[name] = timer
        return timer

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all registered metrics.

        Returns:
            Dict[str, Any]: Dictionary of metrics.
        """
        return self.metrics
        
    def get_metric(self, name: str) -> Any:
        """
        Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            The requested metric or None if not found
        """
        return self.metrics.get(name)

class Timer:
    """
    Context manager for timing code execution.

    This class provides a way to time code execution using a context manager
    or a decorator.

    Attributes:
        name (str): Name of the timer.
        description (str): Description of the timer.
        tags (Dict[str, str]): Tags for the timer.
        registry (MetricsRegistry): Registry for the timer.
    """

    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None, registry: MetricsRegistry = None):
        """
        Initialize the Timer instance.

        Args:
            name (str): Name of the timer.
            description (str, optional): Description of the timer. Defaults to "".
            tags (Dict[str, str], optional): Tags for the timer. Defaults to None.
            registry (MetricsRegistry, optional): Registry for the timer. Defaults to None.
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.registry = registry
        self.start_time = None
        self.elapsed_time = None

    def __enter__(self) -> 'Timer':
        """
        Enter the context manager, starting the timer.

        Returns:
            Timer: The timer instance.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        Exit the context manager, stopping the timer.

        Args:
            exc_type: Exception type if an exception occurred, None otherwise.
            exc_value: Exception value if an exception occurred, None otherwise.
            traceback: Traceback if an exception occurred, None otherwise.
        """
        self.stop()

    def start(self) -> None:
        """
        Start the timer.
        """
        self.start_time = time.time()

    def stop(self) -> float:
        """
        Stop the timer and record the elapsed time.

        Returns:
            float: Elapsed time in seconds.
        """
        if self.start_time is None:
            return 0.0
        
        self.elapsed_time = time.time() - self.start_time
        
        # Record to shared metrics if not in a registry
        if self.registry is None and _metrics is not None:
            with _metrics_lock:
                if self.name not in _metrics["timers"]:
                    _metrics["timers"][self.name] = {"timers": {}, "tags": self.tags}
                timer_id = str(int(time.time() * 1000))  # Unique ID based on timestamp
                _metrics["timers"][self.name]["timers"][timer_id] = {
                    "start": self.start_time,
                    "end": time.time(),
                    "elapsed": self.elapsed_time
                }
                
        return self.elapsed_time

class Counter:
    """
    Counter for tracking occurrences of events.

    This class provides a way to count occurrences of events.

    Attributes:
        name (str): Name of the counter.
        description (str): Description of the counter.
        tags (Dict[str, str]): Tags for the counter.
        registry (MetricsRegistry): Registry for the counter.
        value (int): Current value of the counter.
    """

    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None, registry: MetricsRegistry = None):
        """
        Initialize the Counter instance.

        Args:
            name (str): Name of the counter.
            description (str, optional): Description of the counter. Defaults to "".
            tags (Dict[str, str], optional): Tags for the counter. Defaults to None.
            registry (MetricsRegistry, optional): Registry for the counter. Defaults to None.
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.registry = registry
        self.value = 0

    def increment(self, amount: int = 1) -> None:
        """
        Increment the counter.

        Args:
            amount (int, optional): Amount to increment by. Defaults to 1.
        """
        self.value += amount
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            increment_counter(self.name, amount, self.tags)

    def decrement(self, amount: int = 1) -> None:
        """
        Decrement the counter.

        Args:
            amount (int, optional): Amount to decrement by. Defaults to 1.
        """
        self.value -= amount
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            increment_counter(self.name, -amount, self.tags)

    def reset(self) -> None:
        """
        Reset the counter to zero.
        """
        old_value = self.value
        self.value = 0
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            with _metrics_lock:
                if self.name in _metrics["counters"]:
                    _metrics["counters"][self.name]["value"] = 0

class Gauge:
    """
    Gauge for tracking current values of metrics.

    This class provides a way to track current values of metrics.

    Attributes:
        name (str): Name of the gauge.
        description (str): Description of the gauge.
        tags (Dict[str, str]): Tags for the gauge.
        registry (MetricsRegistry): Registry for the gauge.
        value (float): Current value of the gauge.
    """

    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None, registry: MetricsRegistry = None):
        """
        Initialize the Gauge instance.

        Args:
            name (str): Name of the gauge.
            description (str, optional): Description of the gauge. Defaults to "".
            tags (Dict[str, str], optional): Tags for the gauge. Defaults to None.
            registry (MetricsRegistry, optional): Registry for the gauge. Defaults to None.
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.registry = registry
        self.value = 0.0

    def set(self, value: float) -> None:
        """
        Set the gauge value.

        Args:
            value (float): Value to set.
        """
        self.value = value
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            with _metrics_lock:
                if self.name not in _metrics["gauges"]:
                    _metrics["gauges"][self.name] = {"value": 0.0, "tags": self.tags}
                _metrics["gauges"][self.name]["value"] = value

    def increment(self, amount: float = 1) -> None:
        """
        Increment the gauge value.

        Args:
            amount (float, optional): Amount to increment by. Defaults to 1.
        """
        self.value += amount
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            with _metrics_lock:
                if self.name not in _metrics["gauges"]:
                    _metrics["gauges"][self.name] = {"value": 0.0, "tags": self.tags}
                _metrics["gauges"][self.name]["value"] += amount

    def decrement(self, amount: float = 1) -> None:
        """
        Decrement the gauge value.

        Args:
            amount (float, optional): Amount to decrement by. Defaults to 1.
        """
        self.value -= amount
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            with _metrics_lock:
                if self.name not in _metrics["gauges"]:
                    _metrics["gauges"][self.name] = {"value": 0.0, "tags": self.tags}
                _metrics["gauges"][self.name]["value"] -= amount

class Histogram:
    """
    Histogram for tracking distributions of values.
    
    This class provides a way to track distributions of values.
    
    Attributes:
        name (str): Name of the histogram.
        description (str): Description of the histogram.
        tags (Dict[str, str]): Tags for the histogram.
        registry (MetricsRegistry): Registry for the histogram.
        buckets (List[float]): Bucket boundaries.
        values (List[float]): Recorded values.
    """
    
    def __init__(self, name: str, description: str = "", tags: Dict[str, str] = None, 
                 registry: MetricsRegistry = None, buckets: List[float] = None):
        """
        Initialize the Histogram instance.
        
        Args:
            name (str): Name of the histogram.
            description (str, optional): Description of the histogram. Defaults to "".
            tags (Dict[str, str], optional): Tags for the histogram. Defaults to None.
            registry (MetricsRegistry, optional): Registry for the histogram. Defaults to None.
            buckets (List[float], optional): Bucket boundaries. Defaults to None.
        """
        self.name = name
        self.description = description
        self.tags = tags or {}
        self.registry = registry
        self.buckets = buckets or [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self.values = []
        
    def observe(self, value: float) -> None:
        """
        Record a value in the histogram.
        
        Args:
            value (float): Value to record.
        """
        self.values.append(value)
        
        # If not in a registry, update shared metrics
        if self.registry is None and _metrics is not None:
            with _metrics_lock:
                if self.name not in _metrics["histograms"]:
                    _metrics["histograms"][self.name] = {"values": [], "tags": self.tags}
                _metrics["histograms"][self.name]["values"].append(value)

    def get_count(self) -> int:
        """
        Get the count of recorded values.
        
        Returns:
            int: Count of recorded values.
        """
        return len(self.values)
        
    def get_sum(self) -> float:
        """
        Get the sum of recorded values.
        
        Returns:
            float: Sum of recorded values.
        """
        return sum(self.values)
        
    def get_average(self) -> float:
        """
        Get the average of recorded values.
        
        Returns:
            float: Average of recorded values.
        """
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)

#
# Prometheus Integration for Distributed Metrics
#

class PrometheusMetricsCollector(MetricsCollector):
    """
    Prometheus implementation of the MetricsCollector interface.
    
    Attributes:
        service_name (str): Name of the service collecting metrics.
        registry (CollectorRegistry): Prometheus registry.
        counters (Dict[str, Counter]): Dictionary of Prometheus counters.
        gauges (Dict[str, Gauge]): Dictionary of Prometheus gauges.
        histograms (Dict[str, Histogram]): Dictionary of Prometheus histograms.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize the PrometheusMetricsCollector instance.
        
        Args:
            service_name (str): Name of the service collecting metrics.
        """
        super().__init__(service_name)
        try:
            from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry
            self.registry = CollectorRegistry()
            self.counters = {}
            self.gauges = {}
            self.histograms = {}
            self._prometheus_available = True
            logger.info(f"Initialized Prometheus metrics collector for {service_name}")
        except ImportError:
            self._prometheus_available = False
            logger.warning("Prometheus client library not available. Using local metrics only.")
        
    def record_counter(self, name: str, value: int = 1, dimensions: Dict[str, str] = None) -> None:
        """
        Record a counter metric.
        
        Args:
            name (str): Name of the counter.
            value (int): Value to increment by. Defaults to 1.
            dimensions (Dict[str, str], optional): Dimensions for the counter. Defaults to None.
        """
        if not self._prometheus_available:
            increment_counter(name, value, dimensions)
            return
            
        dimensions = dimensions or {}
        metric_key = f"{name}_{sorted(dimensions.items())}"
        
        if metric_key not in self.counters:
            from prometheus_client import Counter
            self.counters[metric_key] = Counter(
                name,
                f"{name} counter",
                list(dimensions.keys()),
                registry=self.registry
            )
        
        self.counters[metric_key].labels(**dimensions).inc(value)
        
        # Also update local metrics
        increment_counter(name, value, dimensions)
    
    def record_gauge(self, name: str, value: float, dimensions: Dict[str, str] = None) -> None:
        """
        Record a gauge metric.
        
        Args:
            name (str): Name of the gauge.
            value (float): Value to set.
            dimensions (Dict[str, str], optional): Dimensions for the gauge. Defaults to None.
        """
        if not self._prometheus_available:
            set_gauge(name, value, dimensions)
            return
            
        dimensions = dimensions or {}
        metric_key = f"{name}_{sorted(dimensions.items())}"
        
        if metric_key not in self.gauges:
            from prometheus_client import Gauge
            self.gauges[metric_key] = Gauge(
                name,
                f"{name} gauge",
                list(dimensions.keys()),
                registry=self.registry
            )
        
        self.gauges[metric_key].labels(**dimensions).set(value)
        
        # Also update local metrics
        set_gauge(name, value, dimensions)
    
    def record_histogram(self, name: str, value: float, dimensions: Dict[str, str] = None) -> None:
        """
        Record a histogram metric.
        
        Args:
            name (str): Name of the histogram.
            value (float): Value to record.
            dimensions (Dict[str, str], optional): Dimensions for the histogram. Defaults to None.
        """
        if not self._prometheus_available:
            record_histogram(name, value, dimensions)
            return
            
        dimensions = dimensions or {}
        metric_key = f"{name}_{sorted(dimensions.items())}"
        
        if metric_key not in self.histograms:
            from prometheus_client import Histogram
            self.histograms[metric_key] = Histogram(
                name,
                f"{name} histogram",
                list(dimensions.keys()),
                registry=self.registry
            )
        
        self.histograms[metric_key].labels(**dimensions).observe(value)
        
        # Also update local metrics
        record_histogram(name, value, dimensions)
    
    def push_to_gateway(self, job: str = None, gateway: str = None) -> bool:
        """
        Push metrics to Prometheus Push Gateway.
        
        Args:
            job (str, optional): Job name. Defaults to service_name.
            gateway (str, optional): Gateway URL. Defaults to PUSH_GATEWAY_URL.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        if not self._prometheus_available:
            return False
            
        try:
            from prometheus_client import push_to_gateway
            push_to_gateway(
                gateway or PUSH_GATEWAY_URL,
                job=job or self.service_name,
                registry=self.registry
            )
            logger.debug("Metrics pushed to Prometheus Push Gateway")
            return True
        except Exception as e:
            logger.error(f"Error pushing metrics to Prometheus Push Gateway: {str(e)}")
            return False

#
# OpenTelemetry Integration for Distributed Tracing
#

class OpenTelemetryTracer(Tracer):
    """
    OpenTelemetry implementation of the Tracer interface.

    Attributes:
        service_name (str): Name of the service being traced.
        tracer (opentelemetry.trace.Tracer): OpenTelemetry tracer instance.
    """

    def __init__(self, service_name: str, endpoint: str = None):
        """
        Initialize the OpenTelemetryTracer instance.

        Args:
            service_name (str): Name of the service being traced.
            endpoint (str, optional): Endpoint for the trace exporter. Defaults to None.
        """
        super().__init__(service_name)
        self.endpoint = endpoint or TEMPO_URL
        self._initialize_tracer()

    def _initialize_tracer(self):
        """
        Initialize OpenTelemetry tracer.
        """
        try:
            import opentelemetry.trace as trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            trace.set_tracer_provider(TracerProvider(resource=resource))
            
            # Configure exporter
            span_exporter = OTLPSpanExporter(endpoint=self.endpoint)
            span_processor = BatchSpanProcessor(span_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(self.service_name)
            self._otel_available = True
            logger.info(f"OpenTelemetry tracing initialized for {self.service_name}")
        except ImportError:
            logger.warning("OpenTelemetry not available. Traces will not be exported.")
            self._otel_available = False
            self.tracer = None

    def start_span(self, name: str, context: Dict[str, Any] = None) -> Any:
        """
        Start a new trace span.

        Args:
            name (str): Name of the span.
            context (Dict[str, Any], optional): Context information for the span. Defaults to None.

        Returns:
            Any: Span object.
        """
        if not self._otel_available or not self.tracer:
            # Return a simple local span representation
            return {
                "name": name, 
                "start_time": time.time(),
                "context": context or {}
            }
            
        try:
            import opentelemetry.trace as trace
            attrs = {}
            if context:
                for k, v in context.items():
                    if isinstance(v, (str, int, float, bool)):
                        attrs[k] = v
            
            span = self.tracer.start_span(name, attributes=attrs)
            return span
        except Exception as e:
            logger.error(f"Error starting OpenTelemetry span: {str(e)}")
            return {
                "name": name, 
                "start_time": time.time(),
                "context": context or {}
            }

    def end_span(self, span: Any) -> None:
        """
        End a trace span.

        Args:
            span (Any): Span object to end.
        """
        if not self._otel_available or not self.tracer:
            # Just record to local metrics for simple span dicts
            if isinstance(span, dict) and "start_time" in span:
                elapsed = time.time() - span["start_time"]
                with _metrics_lock:
                    span_name = span.get("name", "unknown")
                    if span_name not in _metrics["timers"]:
                        _metrics["timers"][span_name] = {
                            "timers": {}, 
                            "tags": span.get("context", {})
                        }
                    timer_id = str(int(time.time() * 1000))
                    _metrics["timers"][span_name]["timers"][timer_id] = {
                        "start": span["start_time"],
                        "end": time.time(),
                        "elapsed": elapsed
                    }
            return
            
        try:
            span.end()
        except Exception as e:
            logger.error(f"Error ending OpenTelemetry span: {str(e)}")

#
# Helper Functions for Metrics Collection
#

def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Increment a counter metric.

    Args:
        name: Metric name
        value: Value to increment by
        tags: Tags to associate with the metric
    """
    with _metrics_lock:
        if name not in _metrics["counters"]:
            _metrics["counters"][name] = {"value": 0, "tags": tags or {}}
        _metrics["counters"][name]["value"] += value

def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Set a gauge metric value.
    
    Args:
        name: Metric name
        value: Value to set
        tags: Tags to associate with the metric
    """
    with _metrics_lock:
        if name not in _metrics["gauges"]:
            _metrics["gauges"][name] = {"value": 0.0, "tags": tags or {}}
        _metrics["gauges"][name]["value"] = value

def record_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a value in a histogram.
    
    Args:
        name: Metric name
        value: Value to record
        tags: Tags to associate with the metric
    """
    with _metrics_lock:
        if name not in _metrics["histograms"]:
            _metrics["histograms"][name] = {"values": [], "tags": tags or {}}
        _metrics["histograms"][name]["values"].append(value)

def get_metrics() -> Dict[str, Any]:
    """
    Get the current metrics.

    Returns:
        Dict[str, Any]: Dictionary of metrics.
    """
    return _metrics

def export_metrics_to_json(file_path: str) -> None:
    """
    Export metrics to a JSON file.

    Args:
        file_path: Path to the JSON file
    """
    metrics = get_metrics()
    serializable_metrics = {
        "counters": {},
        "gauges": {},
        "histograms": {},
        "timers": {}
    }
    for name, data in metrics["counters"].items():
        serializable_metrics["counters"][name] = {
            "value": data["value"],
            "tags": data["tags"]
        }
    for name, data in metrics["gauges"].items():
        serializable_metrics["gauges"][name] = {
            "value": data["value"],
            "tags": data["tags"]
        }
    for name, data in metrics["histograms"].items():
        serializable_metrics["histograms"][name] = {
            "values": data["values"],
            "tags": data["tags"]
        }
    for name, data in metrics["timers"].items():
        serializable_metrics["timers"][name] = {
            "timers": [],
            "tags": data["tags"]
        }
        for timer_id, timer_data in data["timers"].items():
            if "elapsed" in timer_data:
                serializable_metrics["timers"][name]["timers"].append({
                    "id": timer_id,
                    "start": timer_data["start"],
                    "end": timer_data["end"],
                    "elapsed": timer_data["elapsed"]
                })
    serializable_metrics["timestamp"] = datetime.now().isoformat()
    with open(file_path, "w") as f:
        json.dump(serializable_metrics, f, indent=2)

def register_health_check(name: str, health_check_fn: Callable[[], Dict[str, Any]]) -> None:
    """
    Register a health check function.
    
    Args:
        name: Health check name
        health_check_fn: Health check function
    """
    _health_checks[name] = health_check_fn

def get_health_checks() -> Dict[str, Any]:
    """
    Run all health checks and return results.
    
    Returns:
        Dictionary of health check results
    """
    results = {}
    for name, check_fn in _health_checks.items():
        try:
            results[name] = check_fn()
        except Exception as e:
            results[name] = {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    return results

#
# Logging and Error Reporting Functions
#

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log an error with context.

    Args:
        error: Exception to log
        context: Additional context
    """
    error_data = {
        "error": str(error),
        "type": type(error).__name__,
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat(),
        "context": context or {}
    }
    logger.error(f"Error: {error_data['error']}", extra={"error_data": error_data})
    increment_counter("errors", 1, {"type": type(error).__name__})

def log_request(method: str, path: str, status_code: int, duration: float, user_id: Optional[str] = None) -> None:
    """
    Log an API request.

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in seconds
        user_id: User ID
    """
    increment_counter("requests", 1, {"method": method, "path": path, "status_code": status_code})
    logger.info(
        f"Request: {method} {path} {status_code} {duration:.4f}s",
        extra={
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration": duration,
            "user_id": user_id
        }
    )

def log_service_call(service: str, method: str, duration: float, success: bool, **kwargs) -> None:
    """
    Log a service call with timing information.

    Args:
        service (str): Name of the service.
        method (str): Method of the service that was called.
        duration (float): Duration of the call in seconds.
        success (bool): Whether the call succeeded.
        **kwargs: Additional context information.
    """
    logger.info(
        f"Service call: {service}.{method} {'succeeded' if success else 'failed'} in {duration:.4f}s", 
        extra={"service": service, "method": method, "duration": duration, "success": success, **kwargs}
    )

def log_ml_event(model: str, operation: str, status: str, duration: float = None, **kwargs) -> None:
    """
    Log a machine learning event.

    Args:
        model (str): Name of the machine learning model.
        operation (str): Operation performed with the model.
        status (str): Status of the operation.
        duration (float, optional): Duration of the operation in seconds. Defaults to None.
        **kwargs: Additional context information.
    """
    logger.info(
        f"ML event: {model} {operation} {status}", 
        extra={"model": model, "operation": operation, "status": status, "duration": duration, **kwargs}
    )

#
# Decorators and Context Managers
#

def timed(name: str, description: str = "", tags: Dict[str, str] = None):
    """
    Decorator for timing function execution.

    Args:
        name (str): Name of the timer.
        description (str, optional): Description of the timer. Defaults to "".
        tags (Dict[str, str], optional): Tags for the timer. Defaults to None.

    Returns:
        Callable: Decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with Timer(name, description, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def async_timed(name: str, description: str = "", tags: Dict[str, str] = None):
    """
    Async decorator for timing coroutine execution.

    Args:
        name (str): Name of the timer.
        description (str, optional): Description of the timer. Defaults to "".
        tags (Dict[str, str], optional): Tags for the timer. Defaults to None.

    Returns:
        Callable: Decorated coroutine function.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = time.time() - start_time
                with _metrics_lock:
                    if name not in _metrics["timers"]:
                        _metrics["timers"][name] = {"timers": {}, "tags": tags or {}}
                    timer_id = str(int(time.time() * 1000))
                    _metrics["timers"][name]["timers"][timer_id] = {
                        "start": start_time,
                        "end": time.time(),
                        "elapsed": elapsed
                    }
        return wrapper
    return decorator

def trace(name: str = None):
    """
    Decorator for adding trace spans to functions.

    Args:
        name (str, optional): Name of the span. Defaults to None.

    Returns:
        Callable: Decorated function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = _default_tracer
            if not tracer:
                # Fall back to simple timer
                with Timer(name or func.__name__):
                    return func(*args, **kwargs)
            
            with tracer.span(name or func.__name__):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def async_trace(name: str = None):
    """
    Decorator for adding trace spans to async functions.

    Args:
        name (str, optional): Name of the span. Defaults to None.

    Returns:
        Callable: Decorated function.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = _default_tracer
            if not tracer:
                # Fall back to simple timer
                start_time = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start_time
                    span_name = name or func.__name__
                    with _metrics_lock:
                        if span_name not in _metrics["timers"]:
                            _metrics["timers"][span_name] = {"timers": {}, "tags": {}}
                        timer_id = str(int(time.time() * 1000))
                        _metrics["timers"][span_name]["timers"][timer_id] = {
                            "start": start_time,
                            "end": time.time(),
                            "elapsed": elapsed
                        }
            
            with tracer.span(name or func.__name__):
                return await func(*args, **kwargs)
        return wrapper
    return decorator

#
# Initialization and Setup Functions
#

# Module-level singletons
_default_metrics_registry = MetricsRegistry()
_default_tracer = None
_default_metrics_collector = None

def setup_tracing(service_name: str = SERVICE_NAME, exporter_endpoint: str = None) -> Tracer:
    """
    Set up distributed tracing.

    Args:
        service_name (str): Name of the service being traced.
        exporter_endpoint (str, optional): Endpoint for the trace exporter. Defaults to None.

    Returns:
        Tracer: Tracer instance.
    """
    global _default_tracer
    _default_tracer = OpenTelemetryTracer(service_name, exporter_endpoint)
    return _default_tracer

def setup_metrics(service_name: str = SERVICE_NAME, exporter_endpoint: str = None) -> MetricsCollector:
    """
    Set up metrics collection.

    Args:
        service_name (str): Name of the service collecting metrics.
        exporter_endpoint (str, optional): Endpoint for the metrics exporter. Defaults to None.

    Returns:
        MetricsCollector: MetricsCollector instance.
    """
    global _default_metrics_collector
    _default_metrics_collector = PrometheusMetricsCollector(service_name)
    return _default_metrics_collector

def init_observability(
    service_name: str = SERVICE_NAME,
    tracing_endpoint: str = None,
    metrics_endpoint: str = None,
    enable_prometheus: bool = True,
    enable_opentelemetry: bool = True
) -> Tuple[Optional[Tracer], Optional[MetricsCollector], MetricsRegistry]:
    """
    Initialize the observability system.

    Args:
        service_name (str, optional): Name of the service. Defaults to SERVICE_NAME.
        tracing_endpoint (str, optional): Endpoint for the trace exporter. Defaults to None.
        metrics_endpoint (str, optional): Endpoint for the metrics exporter. Defaults to None.
        enable_prometheus (bool, optional): Whether to enable Prometheus metrics. Defaults to True.
        enable_opentelemetry (bool, optional): Whether to enable OpenTelemetry tracing. Defaults to True.

    Returns:
        Tuple[Optional[Tracer], Optional[MetricsCollector], MetricsRegistry]: 
            Tuple of Tracer, MetricsCollector, and MetricsRegistry instances.
    """
    tracer = None
    metrics_collector = None
    
    if enable_opentelemetry:
        tracer = setup_tracing(service_name, tracing_endpoint)
    
    if enable_prometheus:
        metrics_collector = setup_metrics(service_name, metrics_endpoint)
    
    # Register system health check
    register_health_check("system", lambda: {
        "status": "ok",
        "cpu_usage": os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0,
        "memory_usage": os.popen(f"ps -o %mem -p {os.getpid()} | tail -n 1").read().strip() if os.name != 'nt' else "N/A",
        "timestamp": datetime.now().isoformat()
    })
    
    # Set up logging directory
    os.makedirs("logs", exist_ok=True)
    
    logger.info(f"Observability initialized for {service_name}")
    
    return tracer, metrics_collector, _default_metrics_registry

def setup_monitoring():
    """
    Set up basic monitoring (backward compatibility).
    """
    init_observability(enable_prometheus=False, enable_opentelemetry=False)
    logger.info("Basic monitoring initialized")

# Initialize default registry if not done already
_default_metrics_registry = MetricsRegistry()