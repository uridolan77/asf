"""
Utility module to disable all observability components.

This module provides functions to disable all observability components
including tracing, metrics, and Prometheus exporters.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

def disable_all_observability():
    """
    Disable all observability components by setting environment variables.
    """
    # Set environment variables to disable observability
    os.environ["DISABLE_PROMETHEUS"] = "1"
    os.environ["DISABLE_TRACING"] = "1"
    os.environ["DISABLE_METRICS"] = "1"
    os.environ["DISABLE_OTLP"] = "1"
    os.environ["DISABLE_OBSERVABILITY"] = "1"

    logger.info("All observability components disabled via environment variables")

    # Try to monkey patch key observability modules
    try:
        _monkey_patch_observability()
        logger.info("Successfully monkey patched observability modules")
    except Exception as e:
        logger.warning(f"Failed to monkey patch observability modules: {str(e)}")

def _monkey_patch_observability():
    """
    Monkey patch observability modules to prevent them from initializing.
    """
    # Import dummy implementations
    from dummy_observability import (
        DummyMetricsService, DummyTracingService, DummyPrometheusExporter,
        DummyResilienceTracing, get_dummy_prometheus_exporter,
        configure_dummy_prometheus_exporter, get_dummy_resilience_tracing,
        setup_dummy_tracing, init_dummy_observability, setup_dummy_monitoring
    )

    # Patch observability modules
    modules_to_patch = {
        'asf.medical.llm_gateway.observability.metrics': {
            'MetricsService': DummyMetricsService,
        },
        'asf.medical.llm_gateway.observability.tracing': {
            'TracingService': DummyTracingService,
        },
        'asf.medical.llm_gateway.observability.prometheus': {
            'PrometheusExporter': DummyPrometheusExporter,
            'get_prometheus_exporter': get_dummy_prometheus_exporter,
            'configure_prometheus_exporter': configure_dummy_prometheus_exporter,
        },
        'asf.medical.llm_gateway.resilience.tracing': {
            'ResilienceTracing': DummyResilienceTracing,
            'get_resilience_tracing': get_dummy_resilience_tracing,
        },
        'asf.medical.core.observability': {
            'setup_tracing': setup_dummy_tracing,
            'init_observability': init_dummy_observability,
            'setup_monitoring': setup_dummy_monitoring,
        },
    }

    # Apply patches
    for module_name, patches in modules_to_patch.items():
        if module_name in sys.modules:
            for attr_name, replacement in patches.items():
                setattr(sys.modules[module_name], attr_name, replacement)
                logger.info(f"Patched {module_name}.{attr_name}")

    # Patch OpenTelemetry if it's imported
    if 'opentelemetry' in sys.modules and 'opentelemetry.trace' in sys.modules:
        # Patch the trace module with dummy implementations
        sys.modules['opentelemetry.trace'].get_tracer = lambda *args, **kwargs: DummyTracingService().tracer
        sys.modules['opentelemetry.trace'].set_tracer_provider = lambda *args, **kwargs: None
        logger.info("Patched OpenTelemetry trace module")
