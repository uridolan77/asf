"""
HTTP transport implementation for MCP communication.

This module provides a transport that connects to an MCP server
using HTTP/REST for communication, suitable for cloud-based MCP services.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Union, List
from dataclasses import dataclass

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


@dataclass
class ConnectionStats:
    """Statistics for a connection in the pool."""
    created_at: datetime
    last_used_at: datetime
    request_count: int = 0
    error_count: int = 0
    success_rate: float = 1.0
    avg_response_time_ms: float = 0.0

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.last_used_at = datetime.utcnow()

        # Update average response time with exponential moving average
        if self.avg_response_time_ms == 0:
            self.avg_response_time_ms = response_time_ms
        else:
            # Use a weight of 0.2 for the new value
            self.avg_response_time_ms = (0.8 * self.avg_response_time_ms) + (0.2 * response_time_ms)

        # Update success rate
        self.success_rate = (self.request_count - self.error_count) / self.request_count

    def record_error(self) -> None:
        """Record a failed request."""
        self.request_count += 1
        self.error_count += 1
        self.last_used_at = datetime.utcnow()

        # Update success rate
        self.success_rate = (self.request_count - self.error_count) / self.request_count


class ConnectionPool:
    """Enhanced connection pool for HTTP clients."""

    def __init__(self, max_size: int = 10, max_idle_seconds: int = 300):
        """Initialize connection pool.

        Args:
            max_size: Maximum number of connections in the pool
            max_idle_seconds: Maximum time in seconds a connection can be idle
        """
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self.pool: Dict[str, Tuple[HttpClient, ConnectionStats]] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task = None
        self.logger = logger.bind(component="connection_pool")

    async def start_cleanup_task(self) -> None:
        """Start background task to clean up idle connections."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.debug("Started connection pool cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self.cleanup_idle_connections()
        except asyncio.CancelledError:
            self.logger.debug("Connection pool cleanup task cancelled")
        except Exception as e:
            self.logger.error(
                "Error in connection pool cleanup task",
                error=str(e),
                exc_info=True
            )

    async def cleanup_idle_connections(self) -> None:
        """Clean up idle connections from the pool."""
        now = datetime.utcnow()
        async with self.lock:
            # Find idle connections
            idle_keys = []
            for key, (client, stats) in self.pool.items():
                idle_seconds = (now - stats.last_used_at).total_seconds()
                if idle_seconds > self.max_idle_seconds:
                    idle_keys.append(key)

            # Close and remove idle connections
            for key in idle_keys:
                client, _ = self.pool.pop(key)
                try:
                    await client.close()
                    self.logger.debug(
                        "Closed idle connection",
                        connection_id=key,
                        idle_time=f"{(now - stats.last_used_at).total_seconds():.1f}s"
                    )
                except Exception as e:
                    self.logger.warning(
                        "Error closing idle connection",
                        connection_id=key,
                        error=str(e)
                    )

    async def get_connection(self, create_fn) -> Tuple[HttpClient, str]:
        """Get a connection from the pool or create a new one.

        Args:
            create_fn: Async function that creates a new connection

        Returns:
            Tuple of (connection, connection_id)
        """
        async with self.lock:
            # Find the best available connection
            best_connection = None
            best_key = None
            best_score = -1

            for key, (client, stats) in self.pool.items():
                # Skip connections with high error rates
                if stats.success_rate < 0.5 and stats.request_count > 5:
                    continue

                # Calculate a score based on recency and performance
                recency_score = 1.0 / (1.0 + (datetime.utcnow() - stats.last_used_at).total_seconds())
                performance_score = stats.success_rate
                score = (0.7 * recency_score) + (0.3 * performance_score)

                if score > best_score:
                    best_score = score
                    best_connection = client
                    best_key = key

            if best_connection is not None:
                # Update stats
                _, stats = self.pool[best_key]
                stats.last_used_at = datetime.utcnow()
                return best_connection, best_key

            # Create a new connection if pool is not full
            if len(self.pool) < self.max_size:
                try:
                    new_connection = await create_fn()
                    connection_id = f"conn_{len(self.pool)}_{int(time.time())}"

                    # Add to pool
                    self.pool[connection_id] = (
                        new_connection,
                        ConnectionStats(
                            created_at=datetime.utcnow(),
                            last_used_at=datetime.utcnow()
                        )
                    )

                    self.logger.debug(
                        "Created new connection",
                        connection_id=connection_id,
                        pool_size=len(self.pool)
                    )

                    return new_connection, connection_id
                except Exception as e:
                    self.logger.error(
                        "Error creating new connection",
                        error=str(e),
                        exc_info=True
                    )
                    raise

            # If pool is full, find the least recently used connection
            lru_key = min(
                self.pool.keys(),
                key=lambda k: self.pool[k][1].last_used_at
            )

            # Close and replace it
            old_client, _ = self.pool.pop(lru_key)
            try:
                await old_client.close()
            except Exception as e:
                self.logger.warning(
                    "Error closing connection",
                    connection_id=lru_key,
                    error=str(e)
                )

            # Create new connection
            try:
                new_connection = await create_fn()
                connection_id = f"conn_{lru_key.split('_')[1]}_{int(time.time())}"

                # Add to pool
                self.pool[connection_id] = (
                    new_connection,
                    ConnectionStats(
                        created_at=datetime.utcnow(),
                        last_used_at=datetime.utcnow()
                    )
                )

                self.logger.debug(
                    "Replaced connection in pool",
                    old_id=lru_key,
                    new_id=connection_id
                )

                return new_connection, connection_id
            except Exception as e:
                self.logger.error(
                    "Error creating replacement connection",
                    error=str(e),
                    exc_info=True
                )
                raise

    async def release_connection(self, connection_id: str, success: bool = True, response_time_ms: Optional[float] = None) -> None:
        """Release a connection back to the pool.

        Args:
            connection_id: ID of the connection to release
            success: Whether the operation was successful
            response_time_ms: Response time in milliseconds
        """
        async with self.lock:
            if connection_id not in self.pool:
                return

            _, stats = self.pool[connection_id]

            if success:
                if response_time_ms is not None:
                    stats.record_success(response_time_ms)
                else:
                    stats.record_success(0)
            else:
                stats.record_error()

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self.lock:
            for key, (client, _) in list(self.pool.items()):
                try:
                    await client.close()
                    self.logger.debug(
                        "Closed connection during shutdown",
                        connection_id=key
                    )
                except Exception as e:
                    self.logger.warning(
                        "Error closing connection during shutdown",
                        connection_id=key,
                        error=str(e)
                    )

            self.pool.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool."""
        stats = {
            "pool_size": len(self.pool),
            "max_size": self.max_size,
            "connections": []
        }

        for key, (_, conn_stats) in self.pool.items():
            stats["connections"].append({
                "id": key,
                "created_at": conn_stats.created_at.isoformat(),
                "last_used_at": conn_stats.last_used_at.isoformat(),
                "request_count": conn_stats.request_count,
                "error_count": conn_stats.error_count,
                "success_rate": conn_stats.success_rate,
                "avg_response_time_ms": conn_stats.avg_response_time_ms
            })

        return stats


class HttpTransport(BaseTransport):
    """
    Transport implementation that communicates with an MCP server
    via HTTP for services that expose an HTTP/REST interface.

    Features:
    - Enhanced connection pooling with adaptive selection
    - Connection health monitoring and automatic pruning
    - TLS/mTLS support
    - Authentication via headers/API keys
    - Built-in retries and timeout handling
    - Performance metrics collection
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
        self.max_idle_seconds = config.get("max_idle_seconds", 300)
        self.api_key_env_var = config.get("api_key_env_var")

        # Initialize connection pool
        self.connection_pool = ConnectionPool(
            max_size=self.pool_maxsize,
            max_idle_seconds=self.max_idle_seconds
        )

        # Start connection pool cleanup task
        asyncio.create_task(self.connection_pool.start_cleanup_task())

        # Track active connections
        self.active_connections: Dict[str, HttpClient] = {}

        self.logger = logger.bind(
            base_url=self.base_url,
            transport_type="http"
        )

    async def create_client(self) -> HttpClient:
        """
        Create a new HTTP client.

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
            client = HttpClient(
                client=httpx_client,
                base_url=self.base_url,
                auth_headers=auth_headers,
                request_timeout=self.timeout_seconds
            )

            # Initialize the client
            await client.initialize()

            self.logger.info(
                "Connected to HTTP MCP server",
                base_url=self.base_url
            )

            return client

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

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[Any, None]:
        """
        Connect to the HTTP MCP server using the connection pool.

        Returns:
            HttpClient object that implements the read/write interface

        Raises:
            ConnectionError: If connection fails
        """
        connection_id = None
        start_time = time.time()

        try:
            # Get a connection from the pool
            client, connection_id = await self.connection_pool.get_connection(self.create_client)

            # Track active connection
            self.active_connections[connection_id] = client

            # Yield the client
            yield client

            # Record success and response time
            response_time_ms = (time.time() - start_time) * 1000
            await self.connection_pool.release_connection(
                connection_id,
                success=True,
                response_time_ms=response_time_ms
            )

        except Exception as e:
            # Record failure
            if connection_id:
                await self.connection_pool.release_connection(
                    connection_id,
                    success=False
                )
            raise

        finally:
            # Remove from active connections
            if connection_id and connection_id in self.active_connections:
                del self.active_connections[connection_id]

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

    async def close(self) -> None:
        """Close the HTTP transport."""
        # Close all connections in the pool
        await self.connection_pool.close_all()

        # Close any active connections
        for client in self.active_connections.values():
            try:
                await client.close()
            except Exception as e:
                self.logger.warning(
                    "Error closing active connection",
                    error=str(e)
                )

        self.active_connections.clear()

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool."""
        return self.connection_pool.get_stats()

    async def cleanup(self) -> None:
        """Close the HTTP client (legacy method)."""
        await self.close()

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