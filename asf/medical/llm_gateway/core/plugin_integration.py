"""
Plugin Integration for LLM Gateway.

This module demonstrates how to integrate the plugin system with the existing
LLM Gateway components, such as the intervention manager and provider factory.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

from asf.medical.llm_gateway.core.models import (
    LLMRequest, LLMResponse, StreamChunk, InterventionContext, GatewayConfig
)
from asf.medical.llm_gateway.interventions.base import BaseIntervention
from asf.medical.llm_gateway.interventions.factory import InterventionFactory
from asf.medical.llm_gateway.core.plugins import (
    get_registry, discover_plugins, adapt_intervention, adapt_interventions,
    BasePlugin, PluginEventType
)

logger = logging.getLogger(__name__)


async def initialize_plugin_system(config_path: Optional[str] = None) -> None:
    """
    Initialize the plugin system and discover plugins.
    
    Args:
        config_path: Path to the plugin configuration file (YAML)
    """
    # Use default config path if not provided
    if config_path is None:
        config_dir = Path(__file__).parent.parent / "config"
        config_path = config_dir / "plugins.yaml"
        
    logger.info(f"Initializing plugin system with config: {config_path}")
    
    # Discover plugins from all sources
    plugins = await discover_plugins(config_path)
    logger.info(f"Discovered {len(plugins)} plugins")
    
    # Additional initialization can happen here
    # ...
    
    logger.info("Plugin system initialized successfully")


async def register_existing_interventions(intervention_factory: InterventionFactory) -> None:
    """
    Register existing interventions as plugins.
    
    This allows existing interventions to be used with the new plugin architecture.
    
    Args:
        intervention_factory: The existing intervention factory
    """
    registry = get_registry()
    
    # Get all intervention names from the factory
    intervention_names = intervention_factory.get_registered_intervention_names()
    logger.info(f"Found {len(intervention_names)} existing interventions to adapt as plugins")
    
    # Adapt each intervention
    for name in intervention_names:
        try:
            # Get the intervention instance
            intervention = await intervention_factory.get_intervention(name)
            
            # Adapt it to a plugin
            plugin = adapt_intervention(intervention)
            
            # Register it in the registry
            logger.info(f"Adapting intervention '{name}' as plugin")
            await registry.register_plugin(plugin.__class__, plugin.config)
            
        except Exception as e:
            logger.error(f"Failed to adapt intervention '{name}' as plugin: {str(e)}")


class PluginAwareInterventionManager:
    """
    Example of how to integrate the plugin system with the intervention manager.
    
    This is a conceptual class to show how the existing manager could be enhanced
    to use plugins. In practice, you might modify the existing InterventionManager
    rather than creating a new class.
    """
    
    def __init__(self, gateway_config: GatewayConfig):
        """
        Initialize the plugin-aware intervention manager.
        
        Args:
            gateway_config: Gateway configuration
        """
        self.gateway_config = gateway_config
        self.registry = get_registry()
        
    async def process_request(self, request: LLMRequest) -> LLMResponse:
        """
        Process a request using the plugin system.
        
        Args:
            request: The LLM request to process
            
        Returns:
            The processed LLM response
        """
        context = request.initial_context
        modified_request = request
        response = None
        
        try:
            # Dispatch request_start event to plugins
            logger.debug(f"Dispatching request_start event for {request.request_id}")
            results = await self.registry.dispatch_event(
                "request_start",
                {"request": request, "context": context},
                "intervention"
            )
            
            # Check for early interception
            for result in results:
                if isinstance(result, dict) and result.get("skip_provider", False):
                    logger.info(f"Request {request.request_id} intercepted by plugin")
                    response = result.get("response")
                    break
            
            # If no plugin intercepted, proceed with normal provider call
            if response is None:
                # Here you would call a provider with the request
                # response = await provider.generate(modified_request)
                logger.info(f"Would call provider for {request.request_id}")
                
                # For demonstration, create a sample response
                from asf.medical.llm_gateway.core.models import ContentBlock, ContentItem
                response = LLMResponse(
                    request_id=request.request_id,
                    content_blocks=[
                        ContentBlock(
                            items=[
                                ContentItem.from_text("This is a sample response from the provider.")
                            ]
                        )
                    ]
                )
            
            # Dispatch response_end event to plugins
            logger.debug(f"Dispatching response_end event for {request.request_id}")
            results = await self.registry.dispatch_event(
                "response_end",
                {"request": request, "response": response, "context": context},
                "intervention"
            )
            
            # Process plugin modifications to response
            for result in results:
                if isinstance(result, LLMResponse):
                    response = result
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {str(e)}", exc_info=True)
            
            # Dispatch error event to plugins
            await self.registry.dispatch_event(
                "error",
                {"request": request, "error": e, "context": context},
                "intervention"
            )
            
            # Create error response
            from asf.medical.llm_gateway.core.models import ErrorDetails, ErrorLevel, FinishReason
            error_details = ErrorDetails(
                code="PLUGIN_PROCESSING_ERROR",
                message=f"Error during plugin processing: {str(e)}",
                level=ErrorLevel.ERROR
            )
            return LLMResponse(
                request_id=request.request_id,
                error_details=error_details,
                finish_reason=FinishReason.ERROR
            )
    
    async def process_stream(self, request: LLMRequest) -> asyncio.Queue:
        """
        Process a streaming request using the plugin system.
        
        Args:
            request: The LLM request to process
            
        Returns:
            Queue of StreamChunk objects
        """
        context = request.initial_context
        stream_queue = asyncio.Queue()
        
        try:
            # Dispatch request_start event to plugins
            await self.registry.dispatch_event(
                "request_start",
                {"request": request, "context": context},
                "intervention"
            )
            
            # Create a task to process the stream
            asyncio.create_task(self._process_stream_chunks(request, context, stream_queue))
            
            return stream_queue
            
        except Exception as e:
            logger.error(f"Error setting up streaming for {request.request_id}: {str(e)}", exc_info=True)
            
            # Put an error on the queue
            from asf.medical.llm_gateway.core.models import ErrorDetails, ErrorLevel
            error_details = ErrorDetails(
                code="STREAM_SETUP_ERROR",
                message=f"Error during stream setup: {str(e)}",
                level=ErrorLevel.ERROR
            )
            await stream_queue.put(StreamChunk(error_details=error_details))
            
            return stream_queue
    
    async def _process_stream_chunks(
        self, request: LLMRequest, context: InterventionContext, queue: asyncio.Queue
    ) -> None:
        """
        Process stream chunks through plugins and put them on the queue.
        
        Args:
            request: The original request
            context: The intervention context
            queue: Queue to put processed chunks on
        """
        try:
            # Here you would get a stream from a provider
            # For demonstration, we'll create some sample chunks
            sample_chunks = [
                StreamChunk(delta=ContentItem.from_text("This is ")),
                StreamChunk(delta=ContentItem.from_text("a sample ")),
                StreamChunk(delta=ContentItem.from_text("streaming ")),
                StreamChunk(delta=ContentItem.from_text("response.")),
            ]
            
            # Process each chunk through plugins
            for chunk in sample_chunks:
                # Dispatch stream_chunk event to plugins
                results = await self.registry.dispatch_event(
                    "stream_chunk",
                    {"request": request, "chunk": chunk, "context": context},
                    "intervention"
                )
                
                # Process plugin modifications
                modified_chunk = chunk
                for result in results:
                    if isinstance(result, StreamChunk):
                        modified_chunk = result
                
                # Put the (possibly modified) chunk on the queue
                await queue.put(modified_chunk)
                
                # Add a short delay to simulate realistic streaming
                await asyncio.sleep(0.1)
            
            # Dispatch response_end event for final processing
            final_content = "This is a sample streaming response."
            from asf.medical.llm_gateway.core.models import ContentBlock
            response = LLMResponse(
                request_id=request.request_id,
                content_blocks=[
                    ContentBlock(
                        items=[ContentItem.from_text(final_content)]
                    )
                ]
            )
            
            await self.registry.dispatch_event(
                "response_end",
                {"request": request, "response": response, "context": context},
                "intervention"
            )
            
        except Exception as e:
            logger.error(f"Error processing stream for {request.request_id}: {str(e)}", exc_info=True)
            
            # Put an error on the queue
            from asf.medical.llm_gateway.core.models import ErrorDetails, ErrorLevel
            error_details = ErrorDetails(
                code="STREAM_PROCESSING_ERROR",
                message=f"Error during stream processing: {str(e)}",
                level=ErrorLevel.ERROR
            )
            await queue.put(StreamChunk(error_details=error_details))
            
        finally:
            # Mark the end of the stream
            await queue.put(None)