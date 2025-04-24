"""
Metrics service for the LLM Gateway.

This module provides metrics collection and export functionality for the LLM Gateway,
supporting Prometheus export format for integration with monitoring systems.
"""

import logging
import threading
from typing import Dict, Any, Optional, List, Union, Set

try:
    import prometheus_client as prom
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Global metrics service instance
_METRICS_SERVICE: Optional['MetricsService'] = None


class MetricsService:
    """
    Service for collecting and exporting metrics from the LLM Gateway.
    
    This service provides methods for tracking counters, gauges, histograms,
    and distributions. It can export metrics in Prometheus format when the
    prometheus_client library is available.
    """
    
    def __init__(self, use_prometheus: bool = True):
        """
        Initialize the metrics service.
        
        Args:
            use_prometheus: Whether to use Prometheus for metrics export.
                Requires prometheus_client to be installed.
        """
        self._use_prometheus = use_prometheus and PROMETHEUS_AVAILABLE
        self._lock = threading.RLock()
        
        # Prometheus metrics collections
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
        
        # Fallback metrics collections when prometheus is not available
        self._fallback_counters: Dict[str, Dict[str, float]] = {}
        self._fallback_gauges: Dict[str, Dict[str, float]] = {}
        self._fallback_histograms: Dict[str, Dict[str, List[float]]] = {}
        self._fallback_distributions: Dict[str, Dict[str, List[float]]] = {}
        
        # Plugin integration flag
        self._plugin_integration_enabled = False
        
        logger.info(f"MetricsService initialized (using Prometheus: {self._use_prometheus})")
    
    def enable_plugin_integration(self) -> None:
        """Enable integration with the plugin system if available."""
        try:
            # Use a conditional import to avoid requiring the plugin system
            from asf.medical.llm_gateway.core.plugins import get_registry
            self._plugin_integration_enabled = True
            logger.info("MetricsService plugin integration enabled")
        except ImportError:
            logger.warning("Failed to enable metrics plugin integration: plugin system not available")
            self._plugin_integration_enabled = False
    
    def _label_dict_to_tuple(self, labels: Dict[str, str]) -> tuple:
        """
        Convert a dictionary of labels to a sorted tuple of values.
        
        Args:
            labels: Dictionary of label names and values
            
        Returns:
            Tuple of label values in consistent order
        """
        # Use the sorted label names to ensure consistent ordering
        return tuple(str(labels[key]) for key in sorted(labels.keys()))
    
    def _get_prometheus_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Get or create a Prometheus counter.
        
        Args:
            name: The name of the counter
            labels: Dictionary of label names and values
            
        Returns:
            Prometheus Counter object
        """
        if not self._use_prometheus:
            return None
            
        with self._lock:
            if name not in self._counters:
                label_names = sorted(labels.keys()) if labels else []
                self._counters[name] = prom.Counter(
                    name,
                    f"{name} counter",
                    label_names
                )
            return self._counters[name]
    
    def _get_prometheus_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Get or create a Prometheus gauge.
        
        Args:
            name: The name of the gauge
            labels: Dictionary of label names and values
            
        Returns:
            Prometheus Gauge object
        """
        if not self._use_prometheus:
            return None
            
        with self._lock:
            if name not in self._gauges:
                label_names = sorted(labels.keys()) if labels else []
                self._gauges[name] = prom.Gauge(
                    name,
                    f"{name} gauge",
                    label_names
                )
            return self._gauges[name]
    
    def _get_prometheus_histogram(self, name: str, labels: Optional[Dict[str, str]] = None) -> Any:
        """
        Get or create a Prometheus histogram.
        
        Args:
            name: The name of the histogram
            labels: Dictionary of label names and values
            
        Returns:
            Prometheus Histogram object
        """
        if not self._use_prometheus:
            return None
            
        with self._lock:
            if name not in self._histograms:
                label_names = sorted(labels.keys()) if labels else []
                # Customize buckets based on the metric name
                if "latency" in name or "duration" in name:
                    # Latency buckets in ms: 10ms to 60s
                    buckets = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000]
                elif "tokens" in name:
                    # Token count buckets
                    buckets = [10, 50, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000]
                else:
                    # Default buckets
                    buckets = prom.Histogram.DEFAULT_BUCKETS
                    
                self._histograms[name] = prom.Histogram(
                    name,
                    f"{name} histogram",
                    label_names,
                    buckets=buckets
                )
            return self._histograms[name]
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: The name of the counter
            labels: Dictionary of labels for this counter
            value: The amount to increment by (default 1.0)
        """
        labels = labels or {}
        
        if self._use_prometheus:
            counter = self._get_prometheus_counter(name, labels)
            if len(labels) > 0:
                counter.labels(*[labels.get(key) for key in sorted(labels.keys())]).inc(value)
            else:
                counter.inc(value)
        else:
            with self._lock:
                if name not in self._fallback_counters:
                    self._fallback_counters[name] = {}
                    
                label_key = self._label_dict_to_tuple(labels)
                if label_key not in self._fallback_counters[name]:
                    self._fallback_counters[name][label_key] = 0.0
                    
                self._fallback_counters[name][label_key] += value
                
        # Forward to plugin system if enabled
        self._forward_to_plugins("counter_incremented", {
            "name": name,
            "labels": labels,
            "value": value
        })
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric.
        
        Args:
            name: The name of the gauge
            value: The value to set
            labels: Dictionary of labels for this gauge
        """
        labels = labels or {}
        
        if self._use_prometheus:
            gauge = self._get_prometheus_gauge(name, labels)
            if len(labels) > 0:
                gauge.labels(*[labels.get(key) for key in sorted(labels.keys())]).set(value)
            else:
                gauge.set(value)
        else:
            with self._lock:
                if name not in self._fallback_gauges:
                    self._fallback_gauges[name] = {}
                    
                label_key = self._label_dict_to_tuple(labels)
                self._fallback_gauges[name][label_key] = value
                
        # Forward to plugin system if enabled
        self._forward_to_plugins("gauge_set", {
            "name": name,
            "labels": labels,
            "value": value
        })
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Observe a value for a histogram metric.
        
        Args:
            name: The name of the histogram
            value: The value to observe
            labels: Dictionary of labels for this observation
        """
        labels = labels or {}
        
        if self._use_prometheus:
            histogram = self._get_prometheus_histogram(name, labels)
            if len(labels) > 0:
                histogram.labels(*[labels.get(key) for key in sorted(labels.keys())]).observe(value)
            else:
                histogram.observe(value)
        else:
            with self._lock:
                if name not in self._fallback_histograms:
                    self._fallback_histograms[name] = {}
                    
                label_key = self._label_dict_to_tuple(labels)
                if label_key not in self._fallback_histograms[name]:
                    self._fallback_histograms[name][label_key] = []
                    
                self._fallback_histograms[name][label_key].append(value)
                
        # Forward to plugin system if enabled
        self._forward_to_plugins("histogram_observed", {
            "name": name,
            "labels": labels,
            "value": value
        })
    
    def add_to_distribution(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """
        Add a value to a distribution.
        
        This is similar to observe_histogram but focuses on collecting
        the raw values rather than binning them.
        
        Args:
            name: The name of the distribution
            value: The value to add
            labels: Dictionary of labels for this value
        """
        # For Prometheus, we use histograms
        if self._use_prometheus:
            self.observe_histogram(name, value, labels)
        else:
            labels = labels or {}
            with self._lock:
                if name not in self._fallback_distributions:
                    self._fallback_distributions[name] = {}
                    
                label_key = self._label_dict_to_tuple(labels)
                if label_key not in self._fallback_distributions[name]:
                    self._fallback_distributions[name][label_key] = []
                    
                self._fallback_distributions[name][label_key].append(value)
                
        # Forward to plugin system if enabled
        self._forward_to_plugins("distribution_added", {
            "name": name,
            "labels": labels,
            "value": value
        })
    
    def get_metrics_text(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            String containing metrics in Prometheus text format
        """
        if self._use_prometheus:
            return prom.generate_latest().decode('utf-8')
        else:
            # Simple fallback format similar to Prometheus
            lines = []
            
            # Format counters
            for counter_name, counter_data in self._fallback_counters.items():
                for label_key, value in counter_data.items():
                    label_str = '{' + ','.join([f'{k}="{v}"' for k, v in zip(sorted(label_key), label_key)]) + '}'
                    lines.append(f"{counter_name}{label_str} {value}")
            
            # Format gauges
            for gauge_name, gauge_data in self._fallback_gauges.items():
                for label_key, value in gauge_data.items():
                    label_str = '{' + ','.join([f'{k}="{v}"' for k, v in zip(sorted(label_key), label_key)]) + '}'
                    lines.append(f"{gauge_name}{label_str} {value}")
            
            # We skip histograms in the fallback mode as they're more complex
            
            return '\n'.join(lines)
    
    def start_http_server(self, port: int = 9090) -> bool:
        """
        Start a HTTP server to expose metrics.
        
        Args:
            port: The port to listen on
            
        Returns:
            True if server started successfully, False otherwise
        """
        if not self._use_prometheus:
            logger.warning("Cannot start metrics HTTP server without prometheus_client")
            return False
        
        try:
            prom.start_http_server(port)
            logger.info(f"Started metrics HTTP server on port {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to start metrics HTTP server: {e}")
            return False
    
    def _forward_to_plugins(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Forward metrics events to the plugin system if enabled.
        
        Args:
            event_type: The type of metrics event
            data: The event data
        """
        if not self._plugin_integration_enabled:
            return
            
        try:
            # Import here to avoid circular imports
            from asf.medical.llm_gateway.core.plugins import get_registry
            
            # Get the plugin registry
            registry = get_registry()
            
            # Dispatch the event asynchronously
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a task if we're in an async context
                loop.create_task(registry.dispatch_event("custom", {
                    "source": "metrics_service",
                    "event": event_type,
                    "data": data
                }, "metric"))
            else:
                # Run the coroutine directly if we're not in an async context
                # This is not ideal but works for basic integration
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(registry.dispatch_event("custom", {
                        "source": "metrics_service",
                        "event": event_type,
                        "data": data
                    }, "metric"))
                finally:
                    loop.close()
                    
        except Exception as e:
            # Log but don't fail if plugin integration fails
            logger.debug(f"Failed to forward metrics event to plugins: {e}")


def init_metrics_service(use_prometheus: bool = True) -> MetricsService:
    """
    Initialize the global metrics service.
    
    Args:
        use_prometheus: Whether to use Prometheus for metrics export
        
    Returns:
        The initialized MetricsService instance
    """
    global _METRICS_SERVICE
    if _METRICS_SERVICE is None:
        _METRICS_SERVICE = MetricsService(use_prometheus=use_prometheus)
        
        # Try to enable plugin integration
        try:
            _METRICS_SERVICE.enable_plugin_integration()
        except Exception as e:
            logger.warning(f"Failed to enable metrics plugin integration: {e}")
            
    return _METRICS_SERVICE


def get_metrics_service() -> MetricsService:
    """
    Get the global metrics service, initializing if needed.
    
    Returns:
        The global MetricsService instance
    """
    global _METRICS_SERVICE
    if _METRICS_SERVICE is None:
        return init_metrics_service()
    return _METRICS_SERVICE