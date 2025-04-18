"""
MCP Transport Layer

This module provides transport implementations for the MCP protocol,
supporting different communication methods (stdio, gRPC, HTTP).
"""

import asyncio
import json
import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, AsyncGenerator, Union

from pydantic import BaseModel, Field

# Import the base Transport and TransportError from the consolidated transport layer
from asf.medical.llm_gateway.transport.base import Transport as BaseTransport, TransportError

logger = logging.getLogger(__name__)


class TransportConfig(BaseModel):
    """Base class for transport configuration."""
    transport_type: str


class StdioConfig(TransportConfig):
    """Configuration for stdio transport."""
    transport_type: str = "stdio"
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    cwd: Optional[str] = None


class GrpcConfig(TransportConfig):
    """Configuration for gRPC transport."""
    transport_type: str = "grpc"
    endpoint: str
    use_tls: bool = False
    ca_cert: Optional[str] = None
    client_cert: Optional[str] = None
    client_key: Optional[str] = None
    metadata: Dict[str, str] = Field(default_factory=dict)


class HttpConfig(TransportConfig):
    """Configuration for HTTP transport."""
    transport_type: str = "http"
    base_url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    verify_ssl: bool = True


class Transport(ABC):
    """
    Abstract base class for MCP transports.
    
    A transport is responsible for sending and receiving messages
    to/from an MCP server using a specific communication protocol.
    
    Note: This is an MCP-specific transport interface that wraps
    the core Transport implementations from the consolidated transport layer.
    """
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the transport."""
        pass
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the MCP server and receive a response.
        
        Args:
            message: The message to send
            
        Returns:
            The response from the MCP server
        """
        pass
    
    @abstractmethod
    async def send_message_stream(self, message: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message to the MCP server and receive a streaming response.
        
        Args:
            message: The message to send
            
        Yields:
            Chunks of the response from the MCP server
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the transport and release resources."""
        pass
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if the transport is still valid and usable."""
        pass


class StdioTransport(Transport):
    """
    Transport implementation using stdio for communication.
    
    This transport spawns a subprocess and communicates with it
    using standard input/output.
    """
    
    def __init__(self, config: StdioConfig):
        """
        Initialize the stdio transport.
        
        Args:
            config: Configuration for the stdio transport
        """
        self.config = config
        self.process = None
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self._valid = False
    
    async def initialize(self) -> None:
        """Initialize the transport by starting the subprocess."""
        try:
            # Prepare environment
            env = os.environ.copy()
            env.update(self.config.env)
            
            # Start process
            self.process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.config.cwd
            )
            
            self.stdin = self.process.stdin
            self.stdout = self.process.stdout
            self.stderr = self.process.stderr
            
            self._valid = True
            
            logger.info(f"Started MCP subprocess: {self.config.command} {' '.join(self.config.args)}")
        except Exception as e:
            logger.error(f"Failed to start MCP subprocess: {str(e)}")
            self._valid = False
            raise
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the MCP server and receive a response.
        
        Args:
            message: The message to send
            
        Returns:
            The response from the MCP server
        """
        if not self._valid or not self.process or not self.stdin or not self.stdout:
            raise RuntimeError("Transport is not initialized or has been closed")
        
        try:
            # Send message
            message_json = json.dumps(message) + "\n"
            self.stdin.write(message_json.encode())
            await self.stdin.drain()
            
            # Read response
            response_line = await self.stdout.readline()
            if not response_line:
                raise RuntimeError("MCP subprocess closed unexpectedly")
            
            # Parse response
            response = json.loads(response_line.decode())
            
            return response
        except Exception as e:
            logger.error(f"Error communicating with MCP subprocess: {str(e)}")
            self._valid = False
            raise
    
    async def send_message_stream(self, message: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message to the MCP server and receive a streaming response.
        
        Args:
            message: The message to send
            
        Yields:
            Chunks of the response from the MCP server
        """
        if not self._valid or not self.process or not self.stdin or not self.stdout:
            raise RuntimeError("Transport is not initialized or has been closed")
        
        try:
            # Send message
            message_json = json.dumps(message) + "\n"
            self.stdin.write(message_json.encode())
            await self.stdin.drain()
            
            # Read streaming response
            while True:
                response_line = await self.stdout.readline()
                if not response_line:
                    break
                
                # Parse response chunk
                try:
                    response_chunk = json.loads(response_line.decode())
                    yield response_chunk
                    
                    # Check if this is the last chunk
                    if "stop_reason" in response_chunk:
                        break
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding MCP response: {str(e)}")
                    continue
        except Exception as e:
            logger.error(f"Error in streaming communication with MCP subprocess: {str(e)}")
            self._valid = False
            raise
    
    async def close(self) -> None:
        """Close the transport and terminate the subprocess."""
        if self.process:
            try:
                # Close stdin to signal EOF
                if self.stdin:
                    self.stdin.close()
                
                # Terminate process
                self.process.terminate()
                
                # Wait for process to terminate with timeout
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("MCP subprocess did not terminate, killing it")
                    self.process.kill()
                    await self.process.wait()
                
                logger.info("MCP subprocess terminated")
            except Exception as e:
                logger.error(f"Error closing MCP subprocess: {str(e)}")
            finally:
                self.process = None
                self.stdin = None
                self.stdout = None
                self.stderr = None
                self._valid = False
    
    def is_valid(self) -> bool:
        """Check if the transport is still valid and usable."""
        return self._valid and self.process is not None and self.process.returncode is None


class GrpcTransport(Transport):
    """
    Transport implementation using gRPC for communication.
    
    This transport uses gRPC to communicate with an MCP server.
    """
    
    def __init__(self, config: GrpcConfig):
        """
        Initialize the gRPC transport.
        
        Args:
            config: Configuration for the gRPC transport
        """
        self.config = config
        self._valid = False
        self._client = None
    
    async def initialize(self) -> None:
        """Initialize the transport by creating a gRPC client."""
        try:
            # Import grpc here to avoid dependency if not used
            import grpc
            
            # This is a placeholder for actual gRPC client initialization
            # In a real implementation, you would create a gRPC channel and client
            
            # For now, just set valid to True
            self._valid = True
            
            logger.info(f"Initialized gRPC transport to {self.config.endpoint}")
        except ImportError:
            logger.error("grpcio package is required for gRPC transport")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize gRPC transport: {str(e)}")
            self._valid = False
            raise
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the MCP server and receive a response.
        
        Args:
            message: The message to send
            
        Returns:
            The response from the MCP server
        """
        if not self._valid:
            raise RuntimeError("Transport is not initialized or has been closed")
        
        # This is a placeholder for actual gRPC client call
        # In a real implementation, you would call the appropriate gRPC method
        
        # For now, just return a mock response
        return {
            "id": "mock-response",
            "model": message.get("model", "unknown"),
            "content": "This is a mock response from the gRPC transport",
            "stop_reason": "mock"
        }
    
    async def send_message_stream(self, message: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message to the MCP server and receive a streaming response.
        
        Args:
            message: The message to send
            
        Yields:
            Chunks of the response from the MCP server
        """
        if not self._valid:
            raise RuntimeError("Transport is not initialized or has been closed")
        
        # This is a placeholder for actual gRPC streaming call
        # In a real implementation, you would call the appropriate gRPC streaming method
        
        # For now, just yield a mock response
        yield {
            "id": "mock-chunk-1",
            "model": message.get("model", "unknown"),
            "delta": {"content": "This is a mock "}
        }
        
        yield {
            "id": "mock-chunk-2",
            "model": message.get("model", "unknown"),
            "delta": {"content": "response from the "}
        }
        
        yield {
            "id": "mock-chunk-3",
            "model": message.get("model", "unknown"),
            "delta": {"content": "gRPC transport"},
            "stop_reason": "mock"
        }
    
    async def close(self) -> None:
        """Close the transport and release resources."""
        try:
            # This is a placeholder for actual gRPC client cleanup
            # In a real implementation, you would close the gRPC channel
            
            self._valid = False
            
            logger.info("Closed gRPC transport")
        except Exception as e:
            logger.error(f"Error closing gRPC transport: {str(e)}")
    
    def is_valid(self) -> bool:
        """Check if the transport is still valid and usable."""
        return self._valid


class HttpTransport(Transport):
    """
    Transport implementation using HTTP for communication.
    
    This transport uses HTTP to communicate with an MCP server.
    """
    
    def __init__(self, config: HttpConfig):
        """
        Initialize the HTTP transport.
        
        Args:
            config: Configuration for the HTTP transport
        """
        self.config = config
        self._valid = False
        self._session = None
    
    async def initialize(self) -> None:
        """Initialize the transport by creating an HTTP session."""
        try:
            # Import aiohttp here to avoid dependency if not used
            import aiohttp
            
            # Create session
            self._session = aiohttp.ClientSession(
                headers=self.config.headers
            )
            
            self._valid = True
            
            logger.info(f"Initialized HTTP transport to {self.config.base_url}")
        except ImportError:
            logger.error("aiohttp package is required for HTTP transport")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HTTP transport: {str(e)}")
            self._valid = False
            raise
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the MCP server and receive a response.
        
        Args:
            message: The message to send
            
        Returns:
            The response from the MCP server
        """
        if not self._valid or not self._session:
            raise RuntimeError("Transport is not initialized or has been closed")
        
        try:
            # Send request
            async with self._session.post(
                f"{self.config.base_url}/v1/messages",
                json=message,
                verify_ssl=self.config.verify_ssl
            ) as response:
                # Check response status
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")
                
                # Parse response
                return await response.json()
        except Exception as e:
            logger.error(f"Error in HTTP communication: {str(e)}")
            raise
    
    async def send_message_stream(self, message: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a message to the MCP server and receive a streaming response.
        
        Args:
            message: The message to send
            
        Yields:
            Chunks of the response from the MCP server
        """
        if not self._valid or not self._session:
            raise RuntimeError("Transport is not initialized or has been closed")
        
        try:
            # Send request
            async with self._session.post(
                f"{self.config.base_url}/v1/messages",
                json=message,
                headers={"Accept": "text/event-stream"},
                verify_ssl=self.config.verify_ssl
            ) as response:
                # Check response status
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"HTTP error {response.status}: {error_text}")
                
                # Process streaming response
                async for line in response.content:
                    line = line.decode().strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith(":"):
                        continue
                    
                    # Parse SSE data
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError as e:
                            logger.error(f"Error decoding SSE data: {str(e)}")
                            continue
        except Exception as e:
            logger.error(f"Error in HTTP streaming communication: {str(e)}")
            raise
    
    async def close(self) -> None:
        """Close the transport and release resources."""
        if self._session:
            try:
                await self._session.close()
                logger.info("Closed HTTP transport")
            except Exception as e:
                logger.error(f"Error closing HTTP transport: {str(e)}")
            finally:
                self._session = None
                self._valid = False
    
    def is_valid(self) -> bool:
        """Check if the transport is still valid and usable."""
        return self._valid and self._session is not None


class TransportFactory:
    """
    Factory for creating transport instances.
    
    This factory creates the appropriate transport based on the configuration.
    """
    
    def create_transport(self, config: TransportConfig) -> Transport:
        """
        Create a transport instance.
        
        Args:
            config: Transport configuration
            
        Returns:
            Transport instance
            
        Raises:
            ValueError: If the transport type is not supported
        """
        if config.transport_type == "stdio":
            if not isinstance(config, StdioConfig):
                raise ValueError("Invalid configuration for stdio transport")
            return StdioTransport(config)
        elif config.transport_type == "grpc":
            if not isinstance(config, GrpcConfig):
                raise ValueError("Invalid configuration for gRPC transport")
            return GrpcTransport(config)
        elif config.transport_type == "http":
            if not isinstance(config, HttpConfig):
                raise ValueError("Invalid configuration for HTTP transport")
            return HttpTransport(config)
        else:
            raise ValueError(f"Unsupported transport type: {config.transport_type}")
