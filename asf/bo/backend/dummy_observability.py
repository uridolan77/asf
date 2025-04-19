"""
Dummy implementations of observability components.

This module provides dummy implementations of observability components
to replace the real implementations when observability is disabled.
"""

import logging

logger = logging.getLogger(__name__)

# Dummy metrics service
class DummyMetricsService:
    """Dummy implementation of MetricsService."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy metrics service."""
        pass
    
    def increment(self, *args, **kwargs):
        """Dummy implementation of increment."""
        pass
    
    def observe(self, *args, **kwargs):
        """Dummy implementation of observe."""
        pass
    
    def gauge(self, *args, **kwargs):
        """Dummy implementation of gauge."""
        pass
    
    def start_timer(self, *args, **kwargs):
        """Dummy implementation of start_timer."""
        return DummyTimer()
    
    def record_request(self, *args, **kwargs):
        """Dummy implementation of record_request."""
        pass
    
    def record_error(self, *args, **kwargs):
        """Dummy implementation of record_error."""
        pass
    
    def record_tokens(self, *args, **kwargs):
        """Dummy implementation of record_tokens."""
        pass

# Dummy timer
class DummyTimer:
    """Dummy implementation of Timer."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy timer."""
        pass
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, *args):
        """Exit the context manager."""
        pass
    
    def stop(self, *args, **kwargs):
        """Dummy implementation of stop."""
        return 0.0

# Dummy tracing service
class DummyTracingService:
    """Dummy implementation of TracingService."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy tracing service."""
        self.tracer = DummyTracer()
    
    def start_span(self, *args, **kwargs):
        """Dummy implementation of start_span."""
        return DummySpan()
    
    def start_as_current_span(self, *args, **kwargs):
        """Dummy implementation of start_as_current_span."""
        return DummySpan()

# Dummy tracer
class DummyTracer:
    """Dummy implementation of Tracer."""
    
    def start_span(self, *args, **kwargs):
        """Dummy implementation of start_span."""
        return DummySpan()
    
    def start_as_current_span(self, *args, **kwargs):
        """Dummy implementation of start_as_current_span."""
        return DummySpan()

# Dummy span
class DummySpan:
    """Dummy implementation of Span."""
    
    def __enter__(self):
        """Enter the context manager."""
        return self
    
    def __exit__(self, *args):
        """Exit the context manager."""
        pass
    
    def add_event(self, *args, **kwargs):
        """Dummy implementation of add_event."""
        pass
    
    def set_attribute(self, *args, **kwargs):
        """Dummy implementation of set_attribute."""
        pass
    
    def set_status(self, *args, **kwargs):
        """Dummy implementation of set_status."""
        pass
    
    def end(self, *args, **kwargs):
        """Dummy implementation of end."""
        pass

# Dummy prometheus exporter
class DummyPrometheusExporter:
    """Dummy implementation of PrometheusExporter."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy prometheus exporter."""
        pass
    
    def record_request(self, *args, **kwargs):
        """Dummy implementation of record_request."""
        pass
    
    def record_error(self, *args, **kwargs):
        """Dummy implementation of record_error."""
        pass
    
    def record_tokens(self, *args, **kwargs):
        """Dummy implementation of record_tokens."""
        pass
    
    def push_metrics(self, *args, **kwargs):
        """Dummy implementation of push_metrics."""
        pass

# Dummy resilience tracing
class DummyResilienceTracing:
    """Dummy implementation of ResilienceTracing."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy resilience tracing."""
        pass
    
    def circuit_breaker_span(self, *args, **kwargs):
        """Dummy implementation of circuit_breaker_span."""
        return DummySpan()
    
    def retry_span(self, *args, **kwargs):
        """Dummy implementation of retry_span."""
        return DummySpan()
    
    def timeout_span(self, *args, **kwargs):
        """Dummy implementation of timeout_span."""
        return DummySpan()

# Dummy functions to replace real functions
def get_dummy_prometheus_exporter():
    """Get a dummy prometheus exporter."""
    return DummyPrometheusExporter()

def configure_dummy_prometheus_exporter(*args, **kwargs):
    """Configure a dummy prometheus exporter."""
    return DummyPrometheusExporter()

def get_dummy_resilience_tracing():
    """Get a dummy resilience tracing."""
    return DummyResilienceTracing()

def setup_dummy_tracing(*args, **kwargs):
    """Set up dummy tracing."""
    return DummyTracingService()

def init_dummy_observability(*args, **kwargs):
    """Initialize dummy observability."""
    return DummyTracingService(), DummyMetricsService(), None

def setup_dummy_monitoring(*args, **kwargs):
    """Set up dummy monitoring."""
    pass
