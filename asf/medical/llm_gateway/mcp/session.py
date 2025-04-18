"""
MCP Session

This module provides a session abstraction for interacting with MCP servers.
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, AsyncGenerator, Union

# Updated imports to use the consolidated transport layer
from asf.medical.llm_gateway.transport.base import Transport, TransportError
from .errors import McpError, McpTransportError

logger = logging.getLogger(__name__)


class MCPSession:
    """
    Session for interacting with an MCP server.
    
    This class provides a high-level interface for sending messages to an MCP server
    and receiving responses, abstracting away the details of the transport layer.
    """
    
    def __init__(self, transport: Transport):
        """
        Initialize the MCP session.
        
        Args:
            transport: Transport for communicating with the MCP server
        """
        self.transport = transport
        self._valid = False
    
    async def initialize(self) -> None:
        """Initialize the session."""
        try:
            await self.transport.initialize()
            self._valid = True
        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {str(e)}")
            self._valid = False
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
            raise McpError(f"Error streaming message: {str(e)}") from e
    
    async def close(self) -> None:
        """Close the session and release resources."""
        try:
            await self.transport.close()
            logger.info("Closed MCP session")
        except Exception as e:
            logger.error(f"Error closing MCP session: {str(e)}")
        finally:
            self._valid = False
