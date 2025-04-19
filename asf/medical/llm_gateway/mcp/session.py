"""
MCP Session

This module provides a session abstraction for interacting with MCP servers.
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

# Updated imports to use the consolidated transport layer
from asf.medical.llm_gateway.transport.base import Transport, TransportError
from .errors import McpError, McpTransportError

# Import event system
from asf.medical.llm_gateway.events.event_bus import EventBus
from asf.medical.llm_gateway.events.events import (
    MCPSessionCreatedEvent,
    MCPSessionReleasedEvent,
    ErrorOccurredEvent
)

# Try to import the singleton event bus, with fallback to None
try:
    from asf.medical.llm_gateway.events import event_bus
except ImportError:
    event_bus = None

logger = logging.getLogger(__name__)


class MCPSession:
    """
    Session for interacting with an MCP server.
    
    This class provides a high-level interface for sending messages to an MCP server
    and receiving responses, abstracting away the details of the transport layer.
    """
    
    def __init__(self, transport: Transport, event_bus: Optional[EventBus] = None):
        """
        Initialize the MCP session.
        
        Args:
            transport: Transport for communicating with the MCP server
            event_bus: Optional event bus for publishing events
        """
        self.transport = transport
        self._valid = False
        self.session_id = str(uuid.uuid4())
        self.model = None
        self.created_at = time.time()
        self.event_bus = event_bus or globals().get('event_bus')
    
    async def initialize(self, model: Optional[str] = None, **kwargs) -> None:
        """
        Initialize the session.
        
        Args:
            model: Optional model identifier
            **kwargs: Additional initialization parameters
        """
        try:
            await self.transport.initialize()
            self._valid = True
            self.model = model
            
            # Publish session created event
            if self.event_bus:
                session_params = kwargs.copy()
                if model:
                    session_params['model'] = model
                    
                event = MCPSessionCreatedEvent(
                    session_id=self.session_id,
                    model=model or "unknown",
                    session_params=session_params
                )
                
                # Use sync_publish in case this is called from a synchronous context
                self.event_bus.sync_publish(event)
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {str(e)}")
            self._valid = False
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_initialization_failed",
                    error_message=f"Failed to initialize MCP session: {str(e)}",
                    provider_id="mcp",
                    model=model or "unknown"
                )
                self.event_bus.sync_publish(error_event)
                
            raise McpTransportError(f"Failed to initialize MCP session: {str(e)}") from e
    
    def is_valid(self) -> bool:
        """Check if the session is still valid and usable."""
        return self._valid and self.transport.is_valid()
    
    async def create_message(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Any:
        """
        Create a message using the MCP server.
        
        Args:
            messages: List of messages in the conversation
            model: Model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of stop sequences
            tools: List of tools available to the model
            **kwargs: Additional parameters to pass to the MCP server
            
        Returns:
            Response from the MCP server
        """
        if not self.is_valid():
            raise McpError("Session is not initialized or has been closed")
        
        # Update the model if it was provided during initialization
        if not model and self.model:
            model = self.model
        
        # Prepare request
        request = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens
        }
        
        # Add optional parameters
        if temperature is not None:
            request["temperature"] = temperature
        
        if top_p is not None:
            request["top_p"] = top_p
        
        if top_k is not None:
            request["top_k"] = top_k
        
        if stop_sequences:
            request["stop_sequences"] = stop_sequences
        
        if tools:
            request["tools"] = tools
        
        # Add any additional parameters
        request.update(kwargs)
        
        try:
            # Send request
            response = await self.transport.send_message(request)
            return response
        except Exception as e:
            logger.error(f"Error creating message: {str(e)}")
            self._valid = False
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=kwargs.get('request_id'),
                    error_type="message_creation_failed",
                    error_message=f"Error creating message: {str(e)}",
                    provider_id="mcp",
                    model=model,
                    operation_type="create_message"
                )
                self.event_bus.sync_publish(error_event)
                
            raise McpError(f"Error creating message: {str(e)}") from e
    
    async def stream_message(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> AsyncGenerator[Any, None]:
        """
        Stream a message from the MCP server.
        
        Args:
            messages: List of messages in the conversation
            model: Model to use for generation
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of stop sequences
            tools: List of tools available to the model
            **kwargs: Additional parameters to pass to the MCP server
            
        Yields:
            Chunks of the response from the MCP server
        """
        if not self.is_valid():
            raise McpError("Session is not initialized or has been closed")
        
        # Update the model if it was provided during initialization
        if not model and self.model:
            model = self.model
        
        # Prepare request
        request = {
            "messages": messages,
            "model": model,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        # Add optional parameters
        if temperature is not None:
            request["temperature"] = temperature
        
        if top_p is not None:
            request["top_p"] = top_p
        
        if top_k is not None:
            request["top_k"] = top_k
        
        if stop_sequences:
            request["stop_sequences"] = stop_sequences
        
        if tools:
            request["tools"] = tools
        
        # Add any additional parameters
        request.update(kwargs)
        
        try:
            # Send request and stream response
            async for chunk in self.transport.send_message_stream(request):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming message: {str(e)}")
            self._valid = False
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=kwargs.get('request_id'),
                    error_type="message_streaming_failed",
                    error_message=f"Error streaming message: {str(e)}",
                    provider_id="mcp",
                    model=model,
                    operation_type="stream_message"
                )
                self.event_bus.sync_publish(error_event)
                
            raise McpError(f"Error streaming message: {str(e)}") from e
    
    async def close(self) -> None:
        """Close the session and release resources."""
        try:
            # Calculate session duration
            session_duration_ms = (time.time() - self.created_at) * 1000
            
            # Close transport
            await self.transport.close()
            logger.info(f"Closed MCP session: {self.session_id}")
            
            # Publish session released event
            if self.event_bus and self._valid:
                event = MCPSessionReleasedEvent(
                    session_id=self.session_id,
                    model=self.model or "unknown",
                    duration_ms=session_duration_ms
                )
                await self.event_bus.publish(event)
                
        except Exception as e:
            logger.error(f"Error closing MCP session: {str(e)}")
            
            # Publish error event
            if self.event_bus:
                error_event = ErrorOccurredEvent(
                    request_id=None,
                    error_type="session_close_failed",
                    error_message=f"Error closing MCP session: {str(e)}",
                    provider_id="mcp",
                    model=self.model or "unknown"
                )
                self.event_bus.sync_publish(error_event)
                
        finally:
            self._valid = False
