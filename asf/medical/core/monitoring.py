"""
Monitoring and observability module for the Medical Research Synthesizer.
This module provides monitoring and observability functionality.
"""
import time
import logging
import traceback
from typing import Dict, Any, Optional
from datetime import datetime
import threading
import json
import os
logger = logging.getLogger(__name__)
_metrics = {
    "counters": {},
    "gauges": {},
    "histograms": {},
    "timers": {}
}
_metrics_lock = threading.RLock()
_health_checks = {}
class MetricType:
    """Metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
def increment_counter(name: str, value: int = 1, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Increment a counter metric.
    Args:
        name: Metric name
        value: Value to increment by
        tags: Tags to associate with the metric
    Set a gauge metric.
    Args:
        name: Metric name
        value: Value to set
        tags: Tags to associate with the metric
    Record a value in a histogram metric.
    Args:
        name: Metric name
        value: Value to record
        tags: Tags to associate with the metric
    Start a timer.
    Args:
        name: Timer name
        tags: Tags to associate with the timer
    Returns:
        Timer ID
    Stop a timer.
    Args:
        name: Timer name
        timer_id: Timer ID
    Returns:
        Elapsed time in seconds
    Context manager for timing a block of code.
    Args:
        name: Timer name
        tags: Tags to associate with the timer
    Yields:
        None
    Decorator for timing a function.
    Args:
        name: Timer name (defaults to function name)
        tags: Tags to associate with the timer
    Returns:
        Decorated function
    Decorator for timing an async function.
    Args:
        name: Timer name (defaults to function name)
        tags: Tags to associate with the timer
    Returns:
        Decorated function
    Register a health check function.
    Args:
        name: Health check name
        check_func: Health check function
    Run all registered health checks.
    Returns:
        Health check results
    Get all metrics.
    Returns:
        All metrics
    with _metrics_lock:
        _metrics["counters"] = {}
        _metrics["gauges"] = {}
        _metrics["histograms"] = {}
        _metrics["timers"] = {}
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
    record_histogram("request_duration", duration, {"method": method, "path": path})
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
def setup_monitoring():
    """Set up monitoring.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    register_health_check("system", lambda: {
        "status": "ok",
        "cpu_usage": os.getloadavg()[0],
        "memory_usage": os.popen("ps -o %mem -p " + str(os.getpid()) + " | tail -n 1").read().strip(),
        "timestamp": datetime.now().isoformat()
    })
    os.makedirs("logs", exist_ok=True)
    logger.info("Monitoring initialized")