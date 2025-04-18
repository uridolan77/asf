"""
HTTP transport implementation for MCP communication.

This module provides a transport that connects to an MCP server
using HTTP/REST for communication, suitable for cloud-based MCP services.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union

import structlog

from asf.medical.llm_gateway.transport.base import BaseTransport, ConnectionError, CommunicationError

# Conditional imports to handle missing httpx package gracefully
try:
    import httpx
    import ssl
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = structlog.get_logger("mcp_transport.http")


class HttpClient:
    """
    Wrapper around httpx.AsyncClient to represent our HTTP transport.
    
    Provides standard read/write methods expected by MCP session,
    translating between HTTP requests and the MCP protocol.
    """
    
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        auth_headers: Dict[str, str],
        request_timeout: float = 30.0
    ):
        """
        Initialize HTTP client wrapper.
        
        Args:
            client: httpx AsyncClient instance
            base_url: Base URL for API requests
            auth_headers: Authentication headers
            request_timeout: Default request timeout in seconds
        """
        self.client = client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers
        self.request_timeout = request_timeout
        
        # Queues for MCP protocol simulation over HTTP
        self._read_queue = asyncio.Queue()
        self._initialized = False
        
        # Track last activity
        self.last_activity = datetime.utcnow()
        
        self.logger = logger.bind(base_url=base_url)
    
    async def initialize(self) -> None:
        """Initialize by checking the server is available."""
        try:
            # Make a simple health check request
            response = await self.client.get(
                f"{self.base_url}/api/v1/health",
                headers=self.auth_headers,
                timeout=self.request_timeout
            )
            
            if response.status_code != 200:
                raise ConnectionError(
                    f"MCP server returned status {response.status_code}",
                    transport_type="http"
                )
            
            self._initialized = True
            self.last_activity = datetime.utcnow()
            
            # Push initial message to read queue to indicate successful connection
            await self._read_queue.put({"type": "connection_established"})
            
        except httpx.RequestError as e:
            raise ConnectionError(
                f"Failed to connect to MCP server: {str(e)}",
                transport_type="http",
                original_error=e
            )
    
    async def read(self) -> bytes:
        """
        Read a message from the server (simulated for HTTP).
        
        This method reads from an internal queue that's populated
        by write/response handling.
        
        Returns:
            JSON-encoded message from the server
        """
        try:
            message = await self._read_queue.get()
            self.last_activity = datetime.utcnow()
            
            # Convert to JSON bytes
            return json.dumps(message).encode('utf-8')
        
        except Exception as e:
            raise CommunicationError(
                f"Error reading from HTTP transport: {str(e)}",
                transport_type="http",
                original_error=e
            )
    
    async def write(self, data: bytes) -> None:
        """
        Write a message to the server.
        
        Translates MCP protocol messages to HTTP requests.
        
        Args:
            data: JSON-encoded message to send
        """
        try:
            # Parse the message
            message = json.loads(data.decode('utf-8'))
            self.last_activity = datetime.utcnow()
            
            # Handle different message types
            if "type" not in message:
                raise ValueError(f"Message has no type field: {message}")
            
            message_type = message["type"]
            
            if message_type == "initialize":
                # Already handled in initialize()
                # Just respond with success
                await self._read_queue.put({
                    "type": "initialize_response",
                    "protocol_version": "v1"
                })
            
            elif message_type == "create_message":
                # Map to POST /api/v1/messages
                response = await self.client.post(
                    f"{self.base_url}/api/v1/messages",
                    headers=self.auth_headers,
                    json=message,
                    timeout=self.request_timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                # Queue the response
                await self._read_queue.put({
                    "type": "create_message_response",
                    "request_id": message.get("request_id"),
                    **response_data
                })
            
            elif message_type == "stream_message":
                # Start streaming response in background
                asyncio.create_task(self._handle_streaming(message))
                
            elif message_type == "call_tool":
                # Map to POST /api/v1/tools
                response = await self.client.post(
                    f"{self.base_url}/api/v1/tools",
                    headers=self.auth_headers,
                    json=message,
                    timeout=self.request_timeout
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                # Queue the response
                await self._read_queue.put({
                    "type": "call_tool_response",
                    "request_id": message.get("request_id"),
                    **response_data
                })
            
            else:
                # Unhandled message type
                self.logger.warning(
                    "Unhandled message type",
                    message_type=message_type
                )
                
                # Return an error response
                await self._read_queue.put({
                    "type": "error",
                    "request_id": message.get("request_id"),
                    "error": {
                        "code": "UNHANDLED_MESSAGE_TYPE",
                        "message": f"Unhandled message type: {message_type}"
                    }
                })
        
        except httpx.HTTPStatusError as e:
            # HTTP error with response
            error_code = "HTTP_ERROR"
            error_message = f"HTTP error: {e.response.status_code}"
            
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_code = error_data["error"].get("code", error_code)
                    error_message = error_data["error"].get("message", error_message)
            except:
                pass
            
            # Queue error response
            await self._read_queue.put({
                "type": "error",
                "request_id": message.get("request_id", None),
                "error": {
                    "code": error_code,
                    "message": error_message
                }
            })
        
        except httpx.RequestError as e:
            # Network or timeout error
            await self._read_queue.put({
                "type": "error",
                "request_id": message.get("request_id", None),
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": f"Connection error: {str(e)}"
                }
            })
        
        except Exception as e:
            # Other errors
            await self._read_queue.put({
                "type": "error",
                "request_id": message.get("request_id", None),
                "error": {
                    "code": "UNEXPECTED_ERROR",
                    "message": f"Unexpected error: {str(e)}"
                }
            })
    
    async def _handle_streaming(self, message: Dict[str, Any]) -> None:
        """
        Handle streaming messages via HTTP.
        
        Args:
            message: Stream message request
        """
        request_id = message.get("request_id")
        
        try:
            # Send stream request
            async with self.client.stream(
                "POST",
                f"{self.base_url}/api/v1/stream/messages",
                headers=self.auth_headers,
                json=message,
                timeout=None  # No timeout for streaming
            ) as response:
                response.raise_for_status()
                
                # Process stream chunks
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    # Parse chunk
                    try:
                        chunk = json.loads(line)
                        
                        # Add request ID if missing
                        if "request_id" not in chunk:
                            chunk["request_id"] = request_id
                        
                        # Wrap in stream_message_delta response format
                        delta_response = {
                            "type": "stream_message_delta",
                            **chunk
                        }
                        
                        # Queue for reading
                        await self._read_queue.put(delta_response)
                        self.last_activity = datetime.utcnow()
                        
                    except json.JSONDecodeError:
                        self.logger.warning(
                            "Invalid JSON in stream chunk",
                            chunk=line
                        )
                
                # Add final message if needed
                await self._read_queue.put({
                    "type": "stream_message_end",
                    "request_id": request_id
                })
        
        except httpx.HTTPStatusError as e:
            # HTTP error with response
            error_code = "HTTP_ERROR"
            error_message = f"HTTP error: {e.response.status_code}"
            
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_code = error_data["error"].get("code", error_code)
                    error_message = error_data["error"].get("message", error_message)
            except:
                pass
            
            # Queue error response
            await self._read_queue.put({
                "type": "error",
                "request_id": request_id,
                "error": {
                    "code": error_code,
                    "message": error_message
                }
            })
        
        except httpx.RequestError as e:
            # Network or timeout error
            await self._read_queue.put({
                "type": "error",
                "request_id": request_id,
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": f"Connection error: {str(e)}"
                }
            })
        
        except Exception as e:
            # Other errors
            await self._read_queue.put({
                "type": "error",
                "request_id": request_id,
                "error": {
                    "code": "UNEXPECTED_ERROR",
                    "message": f"Unexpected error: {str(e)}"
                }
            })
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class HttpTransport(BaseTransport):
    """
    Transport implementation that communicates with an MCP server
    via HTTP for services that expose an HTTP/REST interface.
    
    Features:
    - Connection pooling via httpx
    - TLS/mTLS support
    - Authentication via headers/API keys
    - Built-in retries and timeout handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HTTP transport.
        
        Args:
            config: Configuration dictionary with keys:
                - base_url: Base URL for HTTP requests
                - headers: HTTP headers
                - verify_ssl: Whether to verify SSL certificates
                - ca_cert: CA certificate file for SSL verification
                - client_cert: Client certificate file for mutual TLS
                - client_key: Client key file for mutual TLS
                - timeout_seconds: Request timeout
                - pool_maxsize: Maximum number of connections
                - keepalive_timeout: Seconds to keep connections alive
                - api_key_env_var: Environment variable name for API key
        """
        super().__init__(config)
        
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package is required for HttpTransport")
        
        # Extract configuration
        self.base_url = config.get("base_url")
        if not self.base_url:
            raise ValueError("base_url is required for HTTP transport")
        
        self.headers = config.get("headers", {})
        self.verify_ssl = config.get("verify_ssl", True)
        self.ca_cert = config.get("ca_cert")
        self.client_cert = config.get("client_cert")
        self.client_key = config.get("client_key")
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.pool_maxsize = config.get("pool_maxsize", 10)
        self.keepalive_timeout = config.get("keepalive_timeout", 5)
        self.api_key_env_var = config.get("api_key_env_var")
        
        # Client initialization is deferred to connect()
        self.client = None
        
        self.logger = logger.bind(
            base_url=self.base_url,
            transport_type="http"
        )
    
    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[Any, None]:
        """
        Connect to the HTTP MCP server.
        
        Returns:
            HttpClient object that implements the read/write interface
            
        Raises:
            ConnectionError: If connection fails
        """
        # Create SSL context if needed
        ssl_context = None
        if self.verify_ssl or self.client_cert:
            ssl_context = self._create_ssl_context()
        
        # Get authentication headers
        auth_headers = self._get_auth_headers()
        
        # Create client if not exists
        if self.client is None:
            try:
                # Create httpx client
                httpx_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={**self.headers},
                    verify=ssl_context if ssl_context else self.verify_ssl,
                    limits=httpx.Limits(
                        max_connections=self.pool_maxsize,
                        keep_alive_expiry=self.keepalive_timeout
                    ),
                    timeout=self.timeout_seconds
                )
                
                # Create our wrapper client
                self.client = HttpClient(
                    client=httpx_client,
                    base_url=self.base_url,
                    auth_headers=auth_headers,
                    request_timeout=self.timeout_seconds
                )
                
                # Initialize the client
                await self.client.initialize()
                
                self.logger.info(
                    "Connected to HTTP MCP server",
                    base_url=self.base_url
                )
            
            except Exception as e:
                if isinstance(e, ConnectionError):
                    raise
                
                self.logger.error(
                    "Failed to connect to HTTP MCP server",
                    base_url=self.base_url,
                    error=str(e),
                    exc_info=True
                )
                
                raise ConnectionError(
                    f"Failed to connect to HTTP MCP server: {str(e)}",
                    transport_type="http",
                    original_error=e
                )
        
        try:
            # Yield the client
            yield self.client
        
        finally:
            # No cleanup here - we keep the client for reuse
            pass
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Create SSL context for HTTPS connections.
        
        Returns:
            Configured SSL context
        """
        ssl_context = ssl.create_default_context()
        
        # Set up CA certificate for server verification
        if self.ca_cert:
            ssl_context.load_verify_locations(cafile=self.ca_cert)
        
        # Set up client certificate for mTLS
        if self.client_cert and self.client_key:
            ssl_context.load_cert_chain(
                certfile=self.client_cert,
                keyfile=self.client_key
            )
        
        return ssl_context
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for requests.
        
        Returns:
            Dict of authentication headers
        """
        auth_headers = {}
        
        # Add API key if configured
        if self.api_key_env_var:
            api_key = os.environ.get(self.api_key_env_var)
            if api_key:
                auth_headers["Authorization"] = f"Bearer {api_key}"
                self.logger.debug(
                    "Using API key from environment variable",
                    env_var=self.api_key_env_var
                )
        
        # Fall back to generic key
        elif "MCP_SERVER_API_KEY" in os.environ:
            api_key = os.environ.get("MCP_SERVER_API_KEY")
            auth_headers["Authorization"] = f"Bearer {api_key}"
            self.logger.debug("Using generic MCP_SERVER_API_KEY for auth")
        
        return auth_headers
    
    async def cleanup(self) -> None:
        """Close the HTTP client."""
        if self.client:
            try:
                await self.client.close()
                self.logger.info("Closed HTTP client")
            except Exception as e:
                self.logger.warning(
                    "Error closing HTTP client",
                    error=str(e)
                )
            finally:
                self.client = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the transport."""
        status = "unknown"
        message = "No connection attempt yet"
        
        if self.client:
            try:
                # Check if client is initialized
                if not getattr(self.client, "_initialized", False):
                    status = "initializing"
                    message = "Client not initialized"
                else:
                    # Check age of last activity
                    last_activity = getattr(self.client, "last_activity", None)
                    if last_activity:
                        age = (datetime.utcnow() - last_activity).total_seconds()
                        if age > 300:  # 5 minutes
                            status = "stale"
                            message = f"Client inactive for {age:.1f} seconds"
                        else:
                            status = "available"
                            message = f"Client active, last activity {age:.1f} seconds ago"
                    else:
                        status = "unknown"
                        message = "Client initialized but no activity timestamp"
            except Exception as e:
                status = "error"
                message = f"Error checking client: {str(e)}"
        
        return {
            "status": status,
            "transport_type": "http",
            "message": message,
            "base_url": self.base_url,
            "verify_ssl": self.verify_ssl
        }