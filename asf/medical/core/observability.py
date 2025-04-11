"""
Observability module for the Medical Research Synthesizer.
This module provides integration with the Grafana LGTM stack (Loki, Grafana, Tempo, Mimir)
for comprehensive observability.
"""
import os
import logging
import socket
from typing import Dict, Any
from prometheus_client import Counter, Histogram, Gauge, push_to_gateway, CollectorRegistry
logger = logging.getLogger(__name__)
DEFAULT_LOKI_URL = "http://localhost:3100/loki/api/v1/push"
DEFAULT_TEMPO_URL = "http://localhost:14268/api/traces"
DEFAULT_PROMETHEUS_URL = "http://localhost:9090/api/v1/push"
LOKI_URL = os.environ.get("LOKI_URL", DEFAULT_LOKI_URL)
TEMPO_URL = os.environ.get("TEMPO_URL", DEFAULT_TEMPO_URL)
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
PUSH_GATEWAY_URL = os.environ.get("PUSH_GATEWAY_URL", "localhost:9091")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "medical-research-synthesizer")
HOSTNAME = socket.gethostname()
registry = CollectorRegistry()
ml_inference_counter = Counter(
    "ml_inference_total", 
    "Total number of ML inference operations",
    ["model", "operation"],
    registry=registry
)
ml_inference_error_counter = Counter(
    "ml_inference_errors_total", 
    "Total number of ML inference errors",
    ["model", "operation", "error_type"],
    registry=registry
)
ml_inference_duration = Histogram(
    "ml_inference_duration_seconds", 
    "Duration of ML inference operations in seconds",
    ["model", "operation"],
    registry=registry
)
ml_inference_queue_size = Gauge(
    "ml_inference_queue_size", 
    "Number of ML inference operations in queue",
    ["model"],
    registry=registry
)
ml_model_memory_usage = Gauge(
    "ml_model_memory_usage_bytes", 
    "Memory usage of ML models in bytes",
    ["model"],
    registry=registry
)
def push_metrics():
    """Push metrics to Prometheus Push Gateway.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    try:
        push_to_gateway(PUSH_GATEWAY_URL, job=SERVICE_NAME, registry=registry)
        logger.debug("Metrics pushed to Prometheus Push Gateway")
    except Exception as e:
        logger.error(f"Error pushing metrics to Prometheus Push Gateway: {str(e)}")
def send_log_to_loki(log_data: Dict[str, Any]):
    """
    Send log data to Loki.
    Args:
        log_data: Log data to send
    Send trace data to Tempo.
    Args:
        trace_data: Trace data to send
    Context manager for tracing ML operations.
    Args:
        model: Name of the model
        operation: Name of the operation
    Update the queue size for a model.
    Args:
        model: Name of the model
        size: Queue size
    Update the memory usage for a model.
    Args:
        model: Name of the model
        memory_bytes: Memory usage in bytes
    Log an ML event.
    Args:
        model: Name of the model
        operation: Name of the operation
        event_type: Type of event (e.g., "load", "unload", "inference")
        details: Additional details
    logger.info("Setting up observability components")
    log_data = {
        "message": "Observability components initialized",
        "service": SERVICE_NAME,
        "hostname": HOSTNAME,
        "loki_url": LOKI_URL,
        "tempo_url": TEMPO_URL,
        "prometheus_url": PROMETHEUS_URL,
        "level": "info"
    }
    send_log_to_loki(log_data)
    push_metrics()
    logger.info("Observability components initialized")