"""
Metrics Plugin for LLM Gateway.

This plugin integrates with the metrics service to track request/response
metrics and provides additional monitoring capabilities.
"""

import logging
import time
from typing import Any, Dict, Optional

from asf.medical.llm_gateway.core.plugins import BasePlugin, PluginCategory, PluginEventType
from asf.medical.llm_gateway.core.models import LLMRequest, LLMResponse, StreamChunk
from asf.medical.llm_gateway.metrics import get_metrics_service

logger = logging.getLogger(__name__)

class MetricsPlugin(BasePlugin):
    """
    Plugin for collecting and reporting metrics from the LLM Gateway.
    
    This plugin hooks into various gateway events to collect metrics
    about request processing, token usage, latency, etc.
    """
    
    name = "metrics_plugin"
    display_name = "Metrics Plugin"
    category: PluginCategory = "metric"
    priority = 10  # Run early to capture accurate timings
    description = "Collects and reports metrics from the LLM Gateway"
    version = "1.0.0"
    author = "ASF Medical Research"
    tags = {"metrics", "monitoring", "observability"}
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the metrics plugin."""
        super().__init__(config)
        self.metrics_service = get_metrics_service()
        self.request_timestamps: Dict[str, float] = {}
        logger.info("Metrics plugin initialized")
    
    async def on_event(self, event_type: PluginEventType, payload: Any) -> Any:
        """
        Process gateway events and collect metrics.
        
        Args:
            event_type: The type of event
            payload: Event data
            
        Returns:
            The payload, unmodified
        """
        if event_type == "request_start" and isinstance(payload, dict) and "request" in payload:
            await self._handle_request_start(payload["request"])
            
        elif event_type == "response_end" and isinstance(payload, dict) and "response" in payload and "request" in payload:
            await self._handle_response_end(payload["request"], payload["response"])
            
        elif event_type == "stream_chunk" and isinstance(payload, dict) and "chunk" in payload and "request" in payload:
            await self._handle_stream_chunk(payload["request"], payload["chunk"])
            
        elif event_type == "error" and isinstance(payload, dict) and "request" in payload and "error" in payload:
            await self._handle_error(payload["request"], payload["error"])
            
        # Pass through payload unmodified
        return payload
    
    async def _handle_request_start(self, request: LLMRequest) -> None:
        """Record the start of a request."""
        request_id = request.request_id
        self.request_timestamps[request_id] = time.time()
        
        # Increment request counter with labels
        self.metrics_service.increment_counter(
            "llm_requests_total",
            labels={
                "model": request.config.model_identifier,
                "streaming": str(request.config.stream).lower(),
            }
        )
        
        # Track prompt tokens if available
        if request.metadata and request.metadata.token_counts and request.metadata.token_counts.prompt_tokens:
            prompt_tokens = request.metadata.token_counts.prompt_tokens
            self.metrics_service.observe_histogram(
                "llm_prompt_tokens",
                prompt_tokens,
                labels={
                    "model": request.config.model_identifier,
                }
            )
        
        logger.debug(f"Recorded request start for {request_id}")
    
    async def _handle_response_end(self, request: LLMRequest, response: LLMResponse) -> None:
        """Record the completion of a request."""
        request_id = request.request_id
        start_time = self.request_timestamps.pop(request_id, None)
        
        if start_time:
            # Calculate duration in milliseconds
            duration_ms = (time.time() - start_time) * 1000
            
            # Record latency
            self.metrics_service.observe_histogram(
                "llm_request_duration_ms",
                duration_ms,
                labels={
                    "model": request.config.model_identifier,
                    "streaming": str(request.config.stream).lower(),
                    "status": "success" if not response.error_details else "error",
                }
            )
            
            # Track completion tokens if available
            if response.metadata and response.metadata.token_counts and response.metadata.token_counts.completion_tokens:
                completion_tokens = response.metadata.token_counts.completion_tokens
                self.metrics_service.observe_histogram(
                    "llm_completion_tokens",
                    completion_tokens,
                    labels={
                        "model": request.config.model_identifier,
                    }
                )
            
            # Track total tokens if available
            if response.metadata and response.metadata.token_counts and response.metadata.token_counts.total_tokens:
                total_tokens = response.metadata.token_counts.total_tokens
                self.metrics_service.observe_histogram(
                    "llm_total_tokens",
                    total_tokens,
                    labels={
                        "model": request.config.model_identifier,
                    }
                )
            
            logger.debug(f"Recorded response metrics for {request_id}: {duration_ms:.2f}ms")
    
    async def _handle_stream_chunk(self, request: LLMRequest, chunk: StreamChunk) -> None:
        """Record metrics for a streaming chunk."""
        # Increment stream chunk counter
        self.metrics_service.increment_counter(
            "llm_stream_chunks_total",
            labels={
                "model": request.config.model_identifier,
            }
        )
        
        # Track chunk size if text content is available
        if chunk.delta and chunk.delta.text_content:
            chunk_length = len(chunk.delta.text_content)
            self.metrics_service.observe_histogram(
                "llm_stream_chunk_size_chars",
                chunk_length,
                labels={
                    "model": request.config.model_identifier,
                }
            )
    
    async def _handle_error(self, request: LLMRequest, error: Any) -> None:
        """Record metrics for an error."""
        # Increment error counter
        error_code = getattr(error, "code", "unknown_error")
        self.metrics_service.increment_counter(
            "llm_errors_total",
            labels={
                "model": request.config.model_identifier,
                "error_code": error_code,
            }
        )
        
        # Clean up request timestamp if needed
        request_id = request.request_id
        if request_id in self.request_timestamps:
            del self.request_timestamps[request_id]