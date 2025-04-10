"""
Monitoring and observability module for the Medical Research Synthesizer.

This module provides monitoring and observability functionality.
"""

import time
import logging
import functools
import traceback
from typing import Dict, Any, Optional, Callable, List, Union
from datetime import datetime
import threading
import asyncio
import json
import os
from contextlib import contextmanager

# Configure logging
logger = logging.getLogger(__name__)

# Metrics storage
_metrics = {
    "counters": {},
    "gauges": {},
    "histograms": {},
    "timers": {}
}

# Locks for thread safety
_metrics_lock = threading.RLock()

# Health check registry
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
    """
    with _metrics_lock:
        if name not in _metrics["counters"]:
            _metrics["counters"][name] = {"value": 0, "tags": tags or {}}
        
        _metrics["counters"][name]["value"] += value

def set_gauge(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Set a gauge metric.
    
    Args:
        name: Metric name
        value: Value to set
        tags: Tags to associate with the metric
    """
    with _metrics_lock:
        _metrics["gauges"][name] = {"value": value, "tags": tags or {}}

def record_histogram(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """
    Record a value in a histogram metric.
    
    Args:
        name: Metric name
        value: Value to record
        tags: Tags to associate with the metric
    """
    with _metrics_lock:
        if name not in _metrics["histograms"]:
            _metrics["histograms"][name] = {"values": [], "tags": tags or {}}
        
        _metrics["histograms"][name]["values"].append(value)
        
        # Keep only the last 100 values to avoid memory issues
        if len(_metrics["histograms"][name]["values"]) > 100:
            _metrics["histograms"][name]["values"] = _metrics["histograms"][name]["values"][-100:]

def start_timer(name: str, tags: Optional[Dict[str, str]] = None) -> int:
    """
    Start a timer.
    
    Args:
        name: Timer name
        tags: Tags to associate with the timer
        
    Returns:
        Timer ID
    """
    timer_id = int(time.time() * 1000000)
    
    with _metrics_lock:
        if name not in _metrics["timers"]:
            _metrics["timers"][name] = {"timers": {}, "tags": tags or {}}
        
        _metrics["timers"][name]["timers"][timer_id] = {"start": time.time(), "end": None}
    
    return timer_id

def stop_timer(name: str, timer_id: int) -> float:
    """
    Stop a timer.
    
    Args:
        name: Timer name
        timer_id: Timer ID
        
    Returns:
        Elapsed time in seconds
    """
    end_time = time.time()
    
    with _metrics_lock:
        if name not in _metrics["timers"] or timer_id not in _metrics["timers"][name]["timers"]:
            logger.warning(f"Timer {name} with ID {timer_id} not found")
            return 0.0
        
        timer = _metrics["timers"][name]["timers"][timer_id]
        timer["end"] = end_time
        elapsed = end_time - timer["start"]
        
        # Keep only the last 100 timers to avoid memory issues
        timers = _metrics["timers"][name]["timers"]
        if len(timers) > 100:
            oldest_id = min(timers.keys())
            del timers[oldest_id]
        
        return elapsed

@contextmanager
def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """
    Context manager for timing a block of code.
    
    Args:
        name: Timer name
        tags: Tags to associate with the timer
        
    Yields:
        None
    """
    timer_id = start_timer(name, tags)
    try:
        yield
    finally:
        stop_timer(name, timer_id)

def timed(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Decorator for timing a function.
    
    Args:
        name: Timer name (defaults to function name)
        tags: Tags to associate with the timer
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            with timer(timer_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def async_timed(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Decorator for timing an async function.
    
    Args:
        name: Timer name (defaults to function name)
        tags: Tags to associate with the timer
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            timer_name = name or f"{func.__module__}.{func.__name__}"
            timer_id = start_timer(timer_name, tags)
            try:
                return await func(*args, **kwargs)
            finally:
                stop_timer(timer_name, timer_id)
        return wrapper
    return decorator

def register_health_check(name: str, check_func: Callable[[], Dict[str, Any]]) -> None:
    """
    Register a health check function.
    
    Args:
        name: Health check name
        check_func: Health check function
    """
    _health_checks[name] = check_func

def run_health_checks() -> Dict[str, Any]:
    """
    Run all registered health checks.
    
    Returns:
        Health check results
    """
    results = {}
    
    for name, check_func in _health_checks.items():
        try:
            result = check_func()
            results[name] = result
        except Exception as e:
            logger.error(f"Error running health check {name}: {str(e)}")
            results[name] = {
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    return results

def get_metrics() -> Dict[str, Any]:
    """
    Get all metrics.
    
    Returns:
        All metrics
    """
    with _metrics_lock:
        # Make a copy to avoid race conditions
        metrics_copy = {
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {}
        }
        
        # Copy counters
        for name, data in _metrics["counters"].items():
            metrics_copy["counters"][name] = {
                "value": data["value"],
                "tags": data["tags"].copy() if data["tags"] else {}
            }
        
        # Copy gauges
        for name, data in _metrics["gauges"].items():
            metrics_copy["gauges"][name] = {
                "value": data["value"],
                "tags": data["tags"].copy() if data["tags"] else {}
            }
        
        # Copy histograms
        for name, data in _metrics["histograms"].items():
            metrics_copy["histograms"][name] = {
                "values": data["values"].copy() if data["values"] else [],
                "tags": data["tags"].copy() if data["tags"] else {}
            }
        
        # Copy timers
        for name, data in _metrics["timers"].items():
            metrics_copy["timers"][name] = {
                "timers": {},
                "tags": data["tags"].copy() if data["tags"] else {}
            }
            
            for timer_id, timer_data in data["timers"].items():
                if timer_data["end"] is not None:
                    metrics_copy["timers"][name]["timers"][timer_id] = {
                        "start": timer_data["start"],
                        "end": timer_data["end"],
                        "elapsed": timer_data["end"] - timer_data["start"]
                    }
        
        return metrics_copy

def reset_metrics() -> None:
    """Reset all metrics."""
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
    
    # Convert to serializable format
    serializable_metrics = {
        "counters": {},
        "gauges": {},
        "histograms": {},
        "timers": {}
    }
    
    # Convert counters
    for name, data in metrics["counters"].items():
        serializable_metrics["counters"][name] = {
            "value": data["value"],
            "tags": data["tags"]
        }
    
    # Convert gauges
    for name, data in metrics["gauges"].items():
        serializable_metrics["gauges"][name] = {
            "value": data["value"],
            "tags": data["tags"]
        }
    
    # Convert histograms
    for name, data in metrics["histograms"].items():
        serializable_metrics["histograms"][name] = {
            "values": data["values"],
            "tags": data["tags"]
        }
    
    # Convert timers
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
    
    # Add timestamp
    serializable_metrics["timestamp"] = datetime.now().isoformat()
    
    # Write to file
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
    
    # Increment error counter
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
    # Increment request counter
    increment_counter("requests", 1, {"method": method, "path": path, "status_code": status_code})
    
    # Record request duration
    record_histogram("request_duration", duration, {"method": method, "path": path})
    
    # Log request
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
    """Set up monitoring."""
    # Register health checks
    register_health_check("system", lambda: {
        "status": "ok",
        "cpu_usage": os.getloadavg()[0],
        "memory_usage": os.popen("ps -o %mem -p " + str(os.getpid()) + " | tail -n 1").read().strip(),
        "timestamp": datetime.now().isoformat()
    })
    
    # Set up metrics export
    os.makedirs("logs", exist_ok=True)
    
    # Log initial message
    logger.info("Monitoring initialized")
