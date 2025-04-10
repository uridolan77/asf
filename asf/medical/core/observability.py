"""
Observability module for the Medical Research Synthesizer.

This module provides integration with the Grafana LGTM stack (Loki, Grafana, Tempo, Mimir)
for comprehensive observability.
"""

import os
import logging
import json
import time
import uuid
import socket
from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager

import requests
from prometheus_client import Counter, Histogram, Gauge, push_to_gateway, CollectorRegistry, REGISTRY

from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_LOKI_URL = "http://localhost:3100/loki/api/v1/push"
DEFAULT_TEMPO_URL = "http://localhost:14268/api/traces"
DEFAULT_PROMETHEUS_URL = "http://localhost:9090/api/v1/push"

# Configuration
LOKI_URL = os.environ.get("LOKI_URL", DEFAULT_LOKI_URL)
TEMPO_URL = os.environ.get("TEMPO_URL", DEFAULT_TEMPO_URL)
PROMETHEUS_URL = os.environ.get("PROMETHEUS_URL", DEFAULT_PROMETHEUS_URL)
PUSH_GATEWAY_URL = os.environ.get("PUSH_GATEWAY_URL", "localhost:9091")
SERVICE_NAME = os.environ.get("SERVICE_NAME", "medical-research-synthesizer")
HOSTNAME = socket.gethostname()

# Prometheus metrics
registry = CollectorRegistry()

# Counters
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

# Histograms
ml_inference_duration = Histogram(
    "ml_inference_duration_seconds", 
    "Duration of ML inference operations in seconds",
    ["model", "operation"],
    registry=registry
)

# Gauges
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
    """Push metrics to Prometheus Push Gateway."""
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
    """
    try:
        # Format log data for Loki
        timestamp_ns = int(time.time() * 1e9)
        
        # Extract log message
        message = log_data.pop("message", "")
        
        # Convert remaining data to JSON string
        log_json = json.dumps(log_data)
        
        # Create Loki payload
        payload = {
            "streams": [
                {
                    "stream": {
                        "service": SERVICE_NAME,
                        "hostname": HOSTNAME,
                        "level": log_data.get("level", "info")
                    },
                    "values": [
                        [str(timestamp_ns), f"{message} {log_json}"]
                    ]
                }
            ]
        }
        
        # Send to Loki
        response = requests.post(
            LOKI_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=1.0  # Short timeout to avoid blocking
        )
        
        if response.status_code >= 400:
            logger.error(f"Error sending log to Loki: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error sending log to Loki: {str(e)}")

def send_trace_to_tempo(trace_data: Dict[str, Any]):
    """
    Send trace data to Tempo.
    
    Args:
        trace_data: Trace data to send
    """
    try:
        # Send to Tempo
        response = requests.post(
            TEMPO_URL,
            json=trace_data,
            headers={"Content-Type": "application/json"},
            timeout=1.0  # Short timeout to avoid blocking
        )
        
        if response.status_code >= 400:
            logger.error(f"Error sending trace to Tempo: {response.status_code} {response.text}")
    except Exception as e:
        logger.error(f"Error sending trace to Tempo: {str(e)}")

@contextmanager
def trace_ml_operation(model: str, operation: str):
    """
    Context manager for tracing ML operations.
    
    Args:
        model: Name of the model
        operation: Name of the operation
    """
    # Generate trace ID
    trace_id = str(uuid.uuid4())
    
    # Start time
    start_time = time.time()
    
    # Increment counter
    ml_inference_counter.labels(model=model, operation=operation).inc()
    
    # Log start
    log_data = {
        "message": f"Starting ML operation: {operation}",
        "model": model,
        "operation": operation,
        "trace_id": trace_id,
        "level": "info"
    }
    send_log_to_loki(log_data)
    
    try:
        # Yield control
        yield trace_id
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record duration
        ml_inference_duration.labels(model=model, operation=operation).observe(duration)
        
        # Log completion
        log_data = {
            "message": f"Completed ML operation: {operation}",
            "model": model,
            "operation": operation,
            "trace_id": trace_id,
            "duration": duration,
            "level": "info"
        }
        send_log_to_loki(log_data)
        
        # Send trace to Tempo
        trace_data = {
            "traceId": trace_id,
            "spans": [
                {
                    "traceId": trace_id,
                    "spanId": str(uuid.uuid4()),
                    "name": f"{model}_{operation}",
                    "startTime": int(start_time * 1e6),
                    "endTime": int(time.time() * 1e6),
                    "tags": [
                        {"key": "model", "value": model},
                        {"key": "operation", "value": operation},
                        {"key": "service", "value": SERVICE_NAME},
                        {"key": "hostname", "value": HOSTNAME}
                    ]
                }
            ]
        }
        send_trace_to_tempo(trace_data)
        
    except Exception as e:
        # Record error
        ml_inference_error_counter.labels(
            model=model, 
            operation=operation, 
            error_type=type(e).__name__
        ).inc()
        
        # Log error
        log_data = {
            "message": f"Error in ML operation: {operation}",
            "model": model,
            "operation": operation,
            "trace_id": trace_id,
            "error": str(e),
            "error_type": type(e).__name__,
            "level": "error"
        }
        send_log_to_loki(log_data)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record duration
        ml_inference_duration.labels(model=model, operation=operation).observe(duration)
        
        # Send trace to Tempo
        trace_data = {
            "traceId": trace_id,
            "spans": [
                {
                    "traceId": trace_id,
                    "spanId": str(uuid.uuid4()),
                    "name": f"{model}_{operation}",
                    "startTime": int(start_time * 1e6),
                    "endTime": int(time.time() * 1e6),
                    "tags": [
                        {"key": "model", "value": model},
                        {"key": "operation", "value": operation},
                        {"key": "service", "value": SERVICE_NAME},
                        {"key": "hostname", "value": HOSTNAME},
                        {"key": "error", "value": str(e)},
                        {"key": "error_type", "value": type(e).__name__}
                    ]
                }
            ]
        }
        send_trace_to_tempo(trace_data)
        
        # Re-raise the exception
        raise

def update_queue_size(model: str, size: int):
    """
    Update the queue size for a model.
    
    Args:
        model: Name of the model
        size: Queue size
    """
    ml_inference_queue_size.labels(model=model).set(size)

def update_model_memory_usage(model: str, memory_bytes: int):
    """
    Update the memory usage for a model.
    
    Args:
        model: Name of the model
        memory_bytes: Memory usage in bytes
    """
    ml_model_memory_usage.labels(model=model).set(memory_bytes)

def log_ml_event(model: str, operation: str, event_type: str, details: Dict[str, Any] = None):
    """
    Log an ML event.
    
    Args:
        model: Name of the model
        operation: Name of the operation
        event_type: Type of event (e.g., "load", "unload", "inference")
        details: Additional details
    """
    log_data = {
        "message": f"ML event: {event_type}",
        "model": model,
        "operation": operation,
        "event_type": event_type,
        "level": "info"
    }
    
    if details:
        log_data.update(details)
    
    send_log_to_loki(log_data)

def setup_observability():
    """Set up observability components."""
    logger.info("Setting up observability components")
    
    # Log setup
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
    
    # Push initial metrics
    push_metrics()
    
    logger.info("Observability components initialized")
