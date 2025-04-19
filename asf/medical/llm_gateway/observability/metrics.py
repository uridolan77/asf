"""
Metrics service for MCP Provider observability.

This module provides a metrics collection and reporting service.
NOTE: This version has been completely disabled - no metrics functionality is active.
All imports and initializations are bypassed to prevent server hanging.
"""

from typing import Any, Dict

import structlog

# Create logger but don't output any messages
logger = structlog.get_logger("mcp_observability.metrics")

class MetricsService:
    """
    Metrics collection and reporting service - completely disabled.
    No initialization code is executed to prevent server hanging.
    """

    def __init__(self, *args, **kwargs):
        """Initialize metrics service with absolute minimal implementation."""
        self.meter = None
        # No logger output during initialization
        
    # Empty implementations for all methods
    def _init_metrics_infrastructure(self) -> None: pass
    def _create_standard_metrics(self) -> None: pass
    
    def record_request_start(self, *args, **kwargs) -> Dict[str, Any]:
        return {"start_time": 0, "provider_id": "", "model": "", "request_id": "", "streaming": False}
        
    def record_request_end(self, *args, **kwargs) -> None: pass
    def record_session_created(self, *args, **kwargs) -> None: pass
    def record_session_closed(self, *args, **kwargs) -> None: pass
    def record_session_error(self, *args, **kwargs) -> None: pass
    def record_circuit_breaker_state(self, *args, **kwargs) -> None: pass
    def record_retry(self, *args, **kwargs) -> None: pass
    def record_stream_chunk(self, *args, **kwargs) -> None: pass
    def record_stream_error(self, *args, **kwargs) -> None: pass
    def increment(self, *args, **kwargs) -> None: pass
    def observe(self, *args, **kwargs) -> None: pass
    def gauge(self, *args, **kwargs) -> None: pass