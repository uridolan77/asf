"""
Plugin Adapters for LLM Gateway.

This module provides adapters that convert existing components
(interventions, providers, etc.) to the new plugin architecture.
"""

import logging
from typing import Any, Dict, List, Optional, Type, cast

from asf.medical.llm_gateway.interventions.base import BaseIntervention, InterventionHookType
from asf.medical.llm_gateway.core.models import (
    LLMRequest, LLMResponse, StreamChunk, InterventionContext
)

from .base import BasePlugin, PluginCategory, PluginEventType

logger = logging.getLogger(__name__)


class InterventionPluginAdapter(BasePlugin):
    """
    Adapter that converts a BaseIntervention to a BasePlugin.
    
    This allows existing interventions to be used with the new plugin architecture
    without modifying their code.
    """
    
    category: PluginCategory = "intervention"
    
    def __init__(self, intervention: BaseIntervention):
        """
        Initialize the adapter with an existing intervention.
        
        Args:
            intervention: The intervention to adapt
        """
        self.intervention = intervention
        self.name = f"{intervention.name}_adapter"
        self.display_name = f"{intervention.name} (Adapted)"
        self.priority = getattr(intervention, 'priority', 100)
        self.description = f"Adapter for intervention: {intervention.name}"
        self.hook_type = intervention.hook_type
        
        # Pass through configuration
        super().__init__(getattr(intervention, 'config', {}))
    
    async def initialize(self) -> None:
        """Initialize the adapted intervention if needed."""
        if hasattr(self.intervention, 'initialize_async'):
            await self.intervention.initialize_async()
    
    async def on_event(self, event_type: PluginEventType, payload: Any) -> Any:
        """
        Process an event by routing to the appropriate intervention method.
        
        Args:
            event_type: The type of event being processed
            payload: The event payload (request, response, chunk, etc.)
            
        Returns:
            The result of the intervention processing
        """
        # Map event types to intervention methods
        if event_type == "request_start" and hasattr(self.intervention, "process_request"):
            if "pre" in self.hook_type or "pre_post" in self.hook_type or "pre_stream" in self.hook_type:
                if isinstance(payload, dict) and "request" in payload and "context" in payload:
                    request = cast(LLMRequest, payload["request"])
                    context = cast(InterventionContext, payload["context"])
                    return await self.intervention.process_request(request, context)
        
        elif event_type == "response_end" and hasattr(self.intervention, "process_response"):
            if "post" in self.hook_type or "pre_post" in self.hook_type or "post_stream" in self.hook_type:
                if isinstance(payload, dict) and "response" in payload and "context" in payload:
                    response = cast(LLMResponse, payload["response"])
                    context = cast(InterventionContext, payload["context"])
                    return await self.intervention.process_response(response, context)
        
        elif event_type == "stream_chunk" and hasattr(self.intervention, "process_stream_chunk"):
            if "stream" in self.hook_type or "pre_stream" in self.hook_type or "post_stream" in self.hook_type:
                if isinstance(payload, dict) and "chunk" in payload and "context" in payload:
                    chunk = cast(StreamChunk, payload["chunk"])
                    context = cast(InterventionContext, payload["context"])
                    return await self.intervention.process_stream_chunk(chunk, context)
        
        # Return payload unchanged if no matching method or hook type
        return payload
    
    async def shutdown(self) -> None:
        """Shutdown the adapted intervention if needed."""
        if hasattr(self.intervention, 'cleanup'):
            await self.intervention.cleanup()


def adapt_intervention(intervention: BaseIntervention) -> BasePlugin:
    """
    Create a plugin adapter for an existing intervention.
    
    Args:
        intervention: The intervention to adapt
        
    Returns:
        A plugin adapter wrapping the intervention
    """
    return InterventionPluginAdapter(intervention)


def adapt_interventions(interventions: List[BaseIntervention]) -> List[BasePlugin]:
    """
    Create plugin adapters for multiple interventions.
    
    Args:
        interventions: List of interventions to adapt
        
    Returns:
        List of plugin adapters
    """
    return [adapt_intervention(intervention) for intervention in interventions]