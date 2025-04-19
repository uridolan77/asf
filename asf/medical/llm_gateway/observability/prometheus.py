"""
Prometheus metrics integration for LLM Gateway.

This module provides Prometheus metrics collection and export functionality
for monitoring LLM Gateway performance and usage.

NOTE: This version has been completely disabled - no metrics functionality is active.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Set

logger = logging.getLogger(__name__)
logger.info("Prometheus metrics completely disabled")

# Define an empty registry - no metrics will be collected
class Registry:
    """Minimal no-op implementation of Registry."""
    def __init__(self):
        pass

registry = Registry()

# Define a no-op PrometheusExporter class with no initialization logging
class PrometheusExporter:
    """
    Prometheus metrics exporter for LLM Gateway - completely disabled version.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the disabled Prometheus exporter."""
        # No logging of initialization to avoid misleading log messages
        pass

    def record_request(self, *args, **kwargs): pass
    def record_error(self, *args, **kwargs): pass
    def record_tokens(self, *args, **kwargs): pass
    def update_circuit_breaker(self, *args, **kwargs): pass
    def update_provider_info(self, *args, **kwargs): pass
    def update_connection_pool(self, *args, **kwargs): pass
    def record_connection_error(self, *args, **kwargs): pass
    def update_cache_metrics(self, *args, **kwargs): pass
    def update_websocket_metrics(self, *args, **kwargs): pass
    def push_metrics(self, *args, **kwargs): pass

# Empty exporter instance - doesn't log its creation
exporter = PrometheusExporter()

# Define dummy metric functions that do nothing
def Counter(*args, **kwargs): return None
def Histogram(*args, **kwargs): return None
def Gauge(*args, **kwargs): return None
def Info(*args, **kwargs): return None

# Define publicly accessible functions with no logging
def get_prometheus_exporter() -> PrometheusExporter:
    """Get the global Prometheus exporter instance."""
    return exporter

def configure_prometheus_exporter(*args, **kwargs) -> PrometheusExporter:
    """Configure the global Prometheus exporter - disabled version."""
    return exporter
