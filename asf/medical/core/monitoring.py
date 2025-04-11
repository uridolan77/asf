"""
Monitoring module for the Medical Research Synthesizer.

This module is maintained for backward compatibility and imports
from the unified observability module.

Classes:
    MetricsRegistry: Registry for application metrics.
    Timer: Context manager for timing code execution.
    Counter: Counter for tracking occurrences of events.
    Gauge: Gauge for tracking current values of metrics.

Functions:
    init_monitoring: Initialize the monitoring system.
    get_metrics: Get the current metrics.
    timed: Decorator for timing function execution.
    async_timed: Async decorator for timing coroutine execution.
"""
import warnings
from asf.medical.core.observability import (
    MetricsRegistry,
    Timer,
    Counter,
    Gauge,
    Histogram,
    get_metrics,
    increment_counter,
    set_gauge,
    record_histogram,
    export_metrics_to_json,
    log_error,
    log_request,
    register_health_check,
    get_health_checks,
    timed,
    async_timed,
    setup_monitoring as _setup_monitoring
)

# Display a deprecation warning
warnings.warn(
    "The monitoring module is deprecated. Use observability module instead.",
    DeprecationWarning,
    stacklevel=2
)

# Alias for backward compatibility
def init_monitoring():
    """
    Initialize the monitoring system.
    
    This function is maintained for backward compatibility.
    It calls setup_monitoring from the observability module.
    """
    return _setup_monitoring()

# Ensure basic monitoring is initialized
setup_monitoring = init_monitoring