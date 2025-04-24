"""
Observability component for the Enhanced LLM Service.

This module provides observability functionality for the Enhanced LLM Service,
including metrics recording and tracing.
"""

import logging
from typing import Any, Dict, Optional, Union

from asf.medical.llm_gateway.observability.metrics import MetricsService

logger = logging.getLogger(__name__)

class ObservabilityComponent:
    """
    Observability component for the Enhanced LLM Service.
    
    This class provides observability functionality for the Enhanced LLM Service,
    including metrics recording and tracing.
    """
    
    def __init__(self, metrics_service: Optional[MetricsService] = None, enabled: bool = True):
        """
        Initialize the observability component.
        
        Args:
            metrics_service: Optional metrics service to use
            enabled: Whether observability is enabled
        """
        self.metrics_service = metrics_service
        self.enabled = enabled
        
        # Initialize metrics counters
        self._metrics: Dict[str, Dict[str, float]] = {
            "llm.requests": {},
            "llm.tokens.prompt": {},
            "llm.tokens.completion": {},
            "llm.tokens.total": {},
            "llm.latency": {},
            "llm.errors": {},
            "llm.cache.hit": {},
            "llm.cache.miss": {},
            "llm.stream.chunks": {},
            "llm.stream.chunk_size": {}
        }
    
    def record_metric(self, 
                     name: str, 
                     value: Union[int, float, str], 
                     tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        if not self.enabled:
            return
        
        try:
            # Convert value to float if it's a number
            if isinstance(value, (int, float)):
                float_value = float(value)
            else:
                # Skip non-numeric values
                return
            
            # Create tag string for indexing
            tag_str = ""
            if tags:
                tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            
            # Initialize metric if needed
            if name not in self._metrics:
                self._metrics[name] = {}
            
            # Update metric value
            if tag_str in self._metrics[name]:
                self._metrics[name][tag_str] += float_value
            else:
                self._metrics[name][tag_str] = float_value
            
            # Log metric
            logger.debug(f"Recorded metric {name}={float_value} tags={tag_str}")
            
            # Forward to metrics service if available
            if self.metrics_service:
                self.metrics_service.record_metric(name, float_value, tags)
        except Exception as e:
            logger.error(f"Error recording metric: {str(e)}")
    
    def start_span(self, 
                  name: str, 
                  parent_span: Optional[Any] = None,
                  attributes: Optional[Dict[str, str]] = None) -> Any:
        """
        Start a new tracing span.
        
        Args:
            name: Span name
            parent_span: Optional parent span
            attributes: Optional span attributes
            
        Returns:
            Span object
        """
        if not self.enabled:
            return None
        
        try:
            # TODO: Implement proper span creation
            # For now, return a simple dictionary as a placeholder
            import time
            span = {
                "name": name,
                "start_time": time.time(),
                "attributes": attributes or {},
                "parent_span": parent_span
            }
            
            # Log span start
            logger.debug(f"Started span {name} with attributes {attributes}")
            
            return span
        except Exception as e:
            logger.error(f"Error starting span: {str(e)}")
            return None
    
    def end_span(self, span: Any) -> None:
        """
        End a tracing span.
        
        Args:
            span: Span to end
        """
        if not self.enabled or span is None:
            return
        
        try:
            # TODO: Implement proper span ending
            # For now, just log the span duration
            import time
            if isinstance(span, dict) and "start_time" in span:
                duration = time.time() - span["start_time"]
                logger.debug(f"Ended span {span.get('name', 'unknown')} after {duration:.3f}s")
                
                # Record span duration as a metric
                self.record_metric(
                    f"span.duration.{span.get('name', 'unknown')}",
                    duration,
                    span.get("attributes")
                )
        except Exception as e:
            logger.error(f"Error ending span: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get all recorded metrics.
        
        Returns:
            Dictionary containing all recorded metrics
        """
        if not self.enabled:
            return {}
        
        return self._metrics
    
    def reset_metrics(self) -> None:
        """
        Reset all metrics.
        """
        if not self.enabled:
            return
        
        for metric_name in self._metrics:
            self._metrics[metric_name] = {}
        
        logger.debug("Reset all metrics")
