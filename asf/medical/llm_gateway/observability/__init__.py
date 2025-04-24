"""
Observability package for LLM Gateway.

This package has been completely disabled to prevent any tracing or metrics functionality.
All module imports are intercepted and replaced with no-op implementations.
"""

import sys
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("mcp_observability")
logger.info("Observability package completely disabled at import level")

# Create minimal no-op classes
class DummyClass:
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Create no-op implementations of all modules
class DummyMetricsService(DummyClass):
    def record_request_start(self, *args, **kwargs):
        return {"start_time": 0, "provider_id": "", "model": "", "request_id": "", "streaming": False}

class DummyTracingService(DummyClass):
    def __init__(self, *args, **kwargs):
        self.tracer = DummyClass()

class DummyPrometheusExporter(DummyClass):
    pass

# Export these classes directly from the package
MetricsService = DummyMetricsService
TracingService = DummyTracingService
PrometheusExporter = DummyPrometheusExporter

# Define no-op functions
def get_prometheus_exporter(*args, **kwargs):
    return DummyPrometheusExporter()

def configure_prometheus_exporter(*args, **kwargs):
    return DummyPrometheusExporter()

# Override the module imports - this is the most aggressive approach
# Any import of a module from this package will return our dummy implementation
class ObservabilityModuleOverride:
    def __init__(self):
        self.MetricsService = DummyMetricsService
        self.TracingService = DummyTracingService
        self.PrometheusExporter = DummyPrometheusExporter
        self.get_prometheus_exporter = get_prometheus_exporter
        self.configure_prometheus_exporter = configure_prometheus_exporter
    
    def __getattr__(self, name):
        # Return a dummy class for any attribute
        return DummyClass

# Inject our override into sys.modules to intercept imports
sys.modules['asf.medical.llm_gateway.observability.metrics'] = ObservabilityModuleOverride()
sys.modules['asf.medical.llm_gateway.observability.tracing'] = ObservabilityModuleOverride()
sys.modules['asf.medical.llm_gateway.observability.prometheus'] = ObservabilityModuleOverride()
