"""
Metrics service for LLM Gateway.

This module provides centralized collection and processing of metrics in the LLM Gateway.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Union, Callable

logger = logging.getLogger(__name__)

# Type aliases
MetricValue = Union[int, float, str]
MetricTags = Dict[str, str]


class MetricsService:
    """
    Service for collecting and processing metrics.
    
    This service provides a centralized way to collect metrics from various
    components of the LLM Gateway. It supports forwarding metrics to various
    backends and implementing custom metric handlers.
    """
    
    def __init__(self):
        """Initialize the metrics service."""
        self._handlers: List[Callable] = []
        self._metrics_store: Dict[str, Dict[str, Any]] = {}
    
    def add_handler(self, handler: Callable) -> None:
        """
        Add a metrics handler.
        
        Args:
            handler: Callable that takes metric_name, value, unit, and tags
        """
        self._handlers.append(handler)
    
    def record_metric(
        self, name: str, value: MetricValue, unit: Optional[str] = None, 
        tags: Optional[MetricTags] = None
    ) -> None:
        """
        Record a metric value.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            unit: Optional unit of the metric
            tags: Optional tags for the metric
        """
        # Store the metric
        metric_key = self._get_metric_key(name, tags)
        self._metrics_store[metric_key] = {
            "name": name,
            "value": value,
            "unit": unit,
            "tags": tags or {},
            "timestamp": time.time()
        }
        
        # Forward to handlers
        for handler in self._handlers:
            try:
                handler(name=name, value=value, unit=unit, tags=tags)
            except Exception as e:
                logger.error(f"Error in metrics handler: {e}")
    
    def get_metric(self, name: str, tags: Optional[MetricTags] = None) -> Optional[Dict[str, Any]]:
        """
        Get the current value of a metric.
        
        Args:
            name: Name of the metric
            tags: Optional tags for the metric
            
        Returns:
            Dictionary with metric details or None if not found
        """
        metric_key = self._get_metric_key(name, tags)
        return self._metrics_store.get(metric_key)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all currently stored metrics.
        
        Returns:
            Dictionary of all metrics
        """
        return self._metrics_store.copy()
    
    def _get_metric_key(self, name: str, tags: Optional[MetricTags] = None) -> str:
        """
        Get a unique key for a metric based on name and tags.
        
        Args:
            name: Name of the metric
            tags: Optional tags for the metric
            
        Returns:
            String key
        """
        if not tags:
            return name
            
        # Sort tags by key for consistency
        tag_str = ";".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    # Convenience methods for common metric types
    def increment_counter(self, name: str, tags: Optional[MetricTags] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Name of the metric
            tags: Optional tags for the metric
        """
        current = self.get_metric(name, tags)
        current_value = current["value"] if current else 0
        self.record_metric(name, current_value + 1, "count", tags)
    
    def record_gauge(self, name: str, value: float, unit: str, tags: Optional[MetricTags] = None) -> None:
        """
        Record a gauge metric.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            unit: Unit of the metric
            tags: Optional tags for the metric
        """
        self.record_metric(name, value, unit, tags)
    
    def record_duration(self, name: str, duration_ms: float, tags: Optional[MetricTags] = None) -> None:
        """
        Record a duration metric.
        
        Args:
            name: Name of the metric
            duration_ms: Duration in milliseconds
            tags: Optional tags for the metric
        """
        self.record_metric(name, duration_ms, "ms", tags)
    
    def record_size(self, name: str, size_bytes: int, tags: Optional[MetricTags] = None) -> None:
        """
        Record a size metric.
        
        Args:
            name: Name of the metric
            size_bytes: Size in bytes
            tags: Optional tags for the metric
        """
        self.record_metric(name, size_bytes, "bytes", tags)
    
    def record_tokens(
        self, name: str, tokens: int, token_type: str, 
        provider_id: Optional[str] = None, model: Optional[str] = None
    ) -> None:
        """
        Record a token count metric.
        
        Args:
            name: Name of the metric
            tokens: Number of tokens
            token_type: Type of tokens (input, output, total)
            provider_id: Optional provider ID
            model: Optional model name
        """
        tags = {
            "token_type": token_type,
        }
        if provider_id:
            tags["provider_id"] = provider_id
        if model:
            tags["model"] = model
            
        self.record_metric(name, tokens, "tokens", tags)


# Singleton instance
_metrics_service = None


def get_metrics_service() -> MetricsService:
    """
    Get the singleton MetricsService instance.
    
    Returns:
        MetricsService instance
    """
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    
    return _metrics_service