"""
Metrics and observability for the Conexus LLM Gateway.

This module provides functionality for collecting and exposing metrics
about the LLM Gateway's performance and operations.
"""

import logging
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Union, Callable

import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary

logger = logging.getLogger(__name__)

# Global metrics registries
_counters: Dict[str, Counter] = {}
_gauges: Dict[str, Gauge] = {}
_histograms: Dict[str, Histogram] = {}
_summaries: Dict[str, Summary] = {}

# Global settings
_metrics_enabled = False
_prometheus_enabled = False
_service_name = "conexus-llm-gateway"
_prometheus_port = 8001

# Constants for metric naming
NAMESPACE = "conexus"
SUBSYSTEM = "llm_gateway"

# Thread for prometheus server
_prometheus_server_thread = None


def configure_metrics(
    enable_metrics: bool = True,
    enable_prometheus: bool = False,
    prometheus_port: int = 8001,
    service_name: str = "conexus-llm-gateway"
) -> None:
    """
    Configure metrics collection and exposure.
    
    Args:
        enable_metrics: Whether to collect metrics
        enable_prometheus: Whether to expose Prometheus metrics
        prometheus_port: Port for Prometheus metrics server
        service_name: Name of the service for metric labels
    """
    global _metrics_enabled, _prometheus_enabled, _service_name, _prometheus_port
    global _prometheus_server_thread
    
    _metrics_enabled = enable_metrics
    _prometheus_enabled = enable_prometheus
    _service_name = service_name
    _prometheus_port = prometheus_port
    
    if not enable_metrics:
        logger.info("Metrics collection disabled")
        return
    
    logger.info(f"Configured metrics collection for service {service_name}")
    
    if enable_prometheus:
        # Only start Prometheus server if not already running
        if _prometheus_server_thread is None or not _prometheus_server_thread.is_alive():
            logger.info(f"Starting Prometheus metrics server on port {prometheus_port}")
            
            # Start Prometheus server in a separate thread
            def start_prometheus_server():
                try:
                    prometheus_client.start_http_server(prometheus_port)
                    logger.info(f"Prometheus metrics server started on port {prometheus_port}")
                except Exception as e:
                    logger.error(f"Failed to start Prometheus server: {e}")
            
            _prometheus_server_thread = threading.Thread(
                target=start_prometheus_server,
                daemon=True
            )
            _prometheus_server_thread.start()


def _format_metric_name(name: str) -> str:
    """
    Format a metric name to follow Prometheus naming conventions.
    
    Args:
        name: Raw metric name
        
    Returns:
        Formatted metric name
    """
    # Replace non-alphanumeric characters with underscores
    formatted = "".join([c if c.isalnum() else "_" for c in name])
    
    # Ensure name starts with a letter
    if not formatted[0].isalpha():
        formatted = "m_" + formatted
        
    return formatted


def get_or_create_counter(
    name: str, 
    description: str = "", 
    labels: Optional[List[str]] = None
) -> Counter:
    """
    Get or create a counter metric.
    
    Args:
        name: Metric name
        description: Metric description
        labels: Label names
        
    Returns:
        Counter metric
    """
    if not _metrics_enabled:
        # Return a no-op counter if metrics are disabled
        class NoOpCounter:
            def inc(self, amount=1, **kwargs):
                pass
        return NoOpCounter()
    
    global _counters
    
    formatted_name = _format_metric_name(name)
    
    if formatted_name not in _counters:
        counter = Counter(
            name=formatted_name,
            documentation=description,
            namespace=NAMESPACE,
            subsystem=SUBSYSTEM,
            labelnames=labels or ["service"]
        )
        _counters[formatted_name] = counter
    
    return _counters[formatted_name]


def get_or_create_gauge(
    name: str, 
    description: str = "", 
    labels: Optional[List[str]] = None
) -> Gauge:
    """
    Get or create a gauge metric.
    
    Args:
        name: Metric name
        description: Metric description
        labels: Label names
        
    Returns:
        Gauge metric
    """
    if not _metrics_enabled:
        # Return a no-op gauge if metrics are disabled
        class NoOpGauge:
            def set(self, value, **kwargs):
                pass
        return NoOpGauge()
    
    global _gauges
    
    formatted_name = _format_metric_name(name)
    
    if formatted_name not in _gauges:
        gauge = Gauge(
            name=formatted_name,
            documentation=description,
            namespace=NAMESPACE,
            subsystem=SUBSYSTEM,
            labelnames=labels or ["service"]
        )
        _gauges[formatted_name] = gauge
    
    return _gauges[formatted_name]


def get_or_create_histogram(
    name: str, 
    description: str = "", 
    labels: Optional[List[str]] = None,
    buckets: Optional[List[float]] = None
) -> Histogram:
    """
    Get or create a histogram metric.
    
    Args:
        name: Metric name
        description: Metric description
        labels: Label names
        buckets: Histogram buckets
        
    Returns:
        Histogram metric
    """
    if not _metrics_enabled:
        # Return a no-op histogram if metrics are disabled
        class NoOpHistogram:
            def observe(self, value, **kwargs):
                pass
        return NoOpHistogram()
    
    global _histograms
    
    formatted_name = _format_metric_name(name)
    
    if formatted_name not in _histograms:
        histogram = Histogram(
            name=formatted_name,
            documentation=description,
            namespace=NAMESPACE,
            subsystem=SUBSYSTEM,
            labelnames=labels or ["service"],
            buckets=buckets
        )
        _histograms[formatted_name] = histogram
    
    return _histograms[formatted_name]


def get_or_create_summary(
    name: str, 
    description: str = "", 
    labels: Optional[List[str]] = None
) -> Summary:
    """
    Get or create a summary metric.
    
    Args:
        name: Metric name
        description: Metric description
        labels: Label names
        
    Returns:
        Summary metric
    """
    if not _metrics_enabled:
        # Return a no-op summary if metrics are disabled
        class NoOpSummary:
            def observe(self, value, **kwargs):
                pass
        return NoOpSummary()
    
    global _summaries
    
    formatted_name = _format_metric_name(name)
    
    if formatted_name not in _summaries:
        summary = Summary(
            name=formatted_name,
            documentation=description,
            namespace=NAMESPACE,
            subsystem=SUBSYSTEM,
            labelnames=labels or ["service"]
        )
        _summaries[formatted_name] = summary
    
    return _summaries[formatted_name]


# --- Helper functions for common metric operations ---

def counter_inc(name: str, amount: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
    """
    Increment a counter metric.
    
    Args:
        name: Metric name
        amount: Amount to increment by
        labels: Label values
    """
    if not _metrics_enabled:
        return
        
    counter = get_or_create_counter(name)
    
    # Prepare labels
    label_values = labels or {}
    if "service" not in label_values:
        label_values["service"] = _service_name
        
    counter.inc(amount, **label_values)


def gauge_set(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """
    Set a gauge metric value.
    
    Args:
        name: Metric name
        value: Value to set
        labels: Label values
    """
    if not _metrics_enabled:
        return
        
    gauge = get_or_create_gauge(name)
    
    # Prepare labels
    label_values = labels or {}
    if "service" not in label_values:
        label_values["service"] = _service_name
        
    gauge.set(value, **label_values)


def histogram_observe(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """
    Observe a value for a histogram metric.
    
    Args:
        name: Metric name
        value: Value to observe
        labels: Label values
    """
    if not _metrics_enabled:
        return
        
    histogram = get_or_create_histogram(name)
    
    # Prepare labels
    label_values = labels or {}
    if "service" not in label_values:
        label_values["service"] = _service_name
        
    histogram.observe(value, **label_values)


def summary_observe(name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
    """
    Observe a value for a summary metric.
    
    Args:
        name: Metric name
        value: Value to observe
        labels: Label values
    """
    if not _metrics_enabled:
        return
        
    summary = get_or_create_summary(name)
    
    # Prepare labels
    label_values = labels or {}
    if "service" not in label_values:
        label_values["service"] = _service_name
        
    summary.observe(value, **label_values)


# --- Timer context manager ---

class Timer:
    """
    Context manager for timing operations.
    
    Usage:
        with Timer("my_operation_time", labels={"operation": "foo"}):
            # Do some operation
    """
    
    def __init__(self, metric_name: str, labels: Optional[Dict[str, str]] = None):
        """
        Initialize the timer.
        
        Args:
            metric_name: Name of the timer metric
            labels: Labels for the metric
        """
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self) -> 'Timer':
        """Enter the context manager and start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager and record the time."""
        if not _metrics_enabled or self.start_time is None:
            return
            
        elapsed_time = time.time() - self.start_time
        
        # Record timing in both histogram and summary
        histogram_observe(f"{self.metric_name}_seconds", elapsed_time, self.labels)
        
        # Also increment a counter for the number of calls
        counter_inc(f"{self.metric_name}_total", labels=self.labels)
        
        # If there was an exception, increment an error counter
        if exc_type is not None:
            counter_inc(f"{self.metric_name}_errors", labels=self.labels)