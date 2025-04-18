"""
HTTP transport implementation for LLM Gateway.

This module provides a transport that connects to LLM services
using HTTP/REST for communication, suitable for cloud-based LLM services.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Optional, Tuple, Union, List
from dataclasses import dataclass, field

import logging

from asf.medical.llm_gateway.transport.base import (
    Transport, TransportConfig, TransportResponse, TransportError,
    CircuitBreakerOpenError, RateLimitExceededError
)
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter
from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.medical.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig

# Conditional imports to handle missing httpx package gracefully
try:
    import httpx
    import ssl
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


class HTTPTransportConfig(TransportConfig):
    """Configuration for HTTP transport."""
    
    transport_type: str = "http"
    base_url: str  # Base URL for HTTP API
    api_key_env_var: Optional[str] = None  # Environment variable name for API key
    api_key_header: str = "Authorization"  # Header name for API key
    api_key_prefix: str = "Bearer"  # Prefix for API key in header
    headers: Optional[Dict[str, str]] = None  # Additional headers
    verify_ssl: bool = True  # Whether to verify SSL certificates
    ca_cert: Optional[str] = None  # CA certificate for SSL verification
    client_cert: Optional[str] = None  # Client certificate for mTLS
    client_key: Optional[str] = None  # Client key for mTLS
    timeout_seconds: float = 30.0  # Request timeout in seconds
    pool_maxsize: int = 10  # Maximum number of connections in pool
    keepalive_timeout: int = 5  # Keep-alive timeout in seconds
    max_idle_seconds: int = 300  # Maximum idle time for connections
    enable_streaming: bool = True  # Whether to enable streaming
    stream_chunk_size: int = 1024  # Chunk size for streaming
    enable_retries: bool = True  # Whether to enable retries
    retry_max_attempts: int = 3  # Maximum number of retry attempts
    retry_base_delay: float = 1.0  # Base delay for retries
    retry_max_delay: float = 30.0  # Maximum delay for retries
    retry_jitter_factor: float = 0.2  # Jitter factor for retries
    retry_status_codes: List[int] = [408, 429, 500, 502, 503, 504]  # HTTP status codes to retry


class HttpClient:
    """
    Wrapper around httpx.AsyncClient for LLM Gateway.

    Provides standard read/write methods expected by LLM Gateway,
    translating between HTTP requests and LLM Gateway protocols.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        auth_headers: Dict[str, str],
        request_timeout: float = 30.0,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize HTTP client wrapper.

        Args:
            client: httpx AsyncClient instance
            base_url: Base URL for API requests
            auth_headers: Authentication headers
            request_timeout: Default request timeout in seconds
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.client = client
        self.base_url = base_url.rstrip('/')
        self.auth_headers = auth_headers
        self.request_timeout = request_timeout
        self.metrics_service = metrics_service
        self.prometheus = prometheus_exporter

        # Queues for protocol simulation over HTTP
        self._read_queue = asyncio.Queue()
        self._initialized = False

        # Track last activity
        self.last_activity = datetime.utcnow()

        logger.debug(f"Initialized HTTP client for {base_url}")

    async def initialize(self) -> None:
        """Initialize by checking the server is available."""
        try:
            # Make a simple health check request
            start_time = time.time()
            response = await self.client.get(
                f"{self.base_url}/health",
                headers=self.auth_headers,
                timeout=self.request_timeout
            )

            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            if self.metrics_service:
                self.metrics_service.record_transport_request(
                    transport_type="http",
                    endpoint="health",
                    status_code=response.status_code,
                    latency_ms=latency_ms
                )

            if response.status_code != 200:
                raise TransportError(
                    message=f"LLM service returned status {response.status_code}",
                    code=f"HTTP_{response.status_code}",
                    details={"status_code": response.status_code}
                )

            self._initialized = True
            self.last_activity = datetime.utcnow()

            # Push initial message to read queue to indicate successful connection
            await self._read_queue.put({"type": "connection_established"})

        except httpx.RequestError as e:
            logger.error(f"Failed to connect to LLM service: {str(e)}", exc_info=True)
            raise TransportError(
                message=f"Failed to connect to LLM service: {str(e)}",
                code="CONNECTION_ERROR",
                details={"error_type": type(e).__name__}
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
            raise TransportError(
                message=f"Error reading from HTTP transport: {str(e)}",
                code="READ_ERROR",
                details={"error_type": type(e).__name__}
            )

    async def write(self, data: bytes) -> None:
        """
        Write a message to the server.

        Translates LLM Gateway protocol messages to HTTP requests.

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
            start_time = time.time()

            if message_type == "completions":
                # Map to POST /v1/completions
                response = await self.client.post(
                    f"{self.base_url}/v1/completions",
                    headers=self.auth_headers,
                    json=message,
                    timeout=self.request_timeout
                )

                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                if self.metrics_service:
                    self.metrics_service.record_transport_request(
                        transport_type="http",
                        endpoint="completions",
                        status_code=response.status_code,
                        latency_ms=latency_ms
                    )

                response.raise_for_status()
                response_data = response.json()

                # Queue the response
                await self._read_queue.put({
                    "type": "completions_response",
                    "request_id": message.get("request_id"),
                    **response_data
                })

            elif message_type == "chat_completions":
                # Map to POST /v1/chat/completions
                response = await self.client.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self.auth_headers,
                    json=message,
                    timeout=self.request_timeout
                )

                # Record metrics
                latency_ms = (time.time() - start_time) * 1000
                if self.metrics_service:
                    self.metrics_service.record_transport_request(
                        transport_type="http",
                        endpoint="chat/completions",
                        status_code=response.status_code,
                        latency_ms=latency_ms
                    )

                response.raise_for_status()
                response_data = response.json()

                # Queue the response
                await self._read_queue.put({
                    "type": "chat_completions_response",
                    "request_id": message.get("request_id"),
                    **response_data
                })

            elif message_type in ["stream_chat_completions", "stream_completions"]:
                # Start streaming response in background
                asyncio.create_task(self._handle_streaming(message))

            else:
                # Unhandled message type
                logger.warning(f"Unhandled message type: {message_type}")

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
            error_code = f"HTTP_{e.response.status_code}"
            error_message = f"HTTP error: {e.response.status_code}"

            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_code = error_data["error"].get("code", error_code)
                    error_message = error_data["error"].get("message", error_message)
            except:
                pass

            # Record metrics
            if self.metrics_service:
                self.metrics_service.record_transport_error(
                    transport_type="http",
                    error_type=error_code,
                    error_message=error_message
                )

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
            error_type = type(e).__name__
            error_message = str(e)

            # Record metrics
            if self.metrics_service:
                self.metrics_service.record_transport_error(
                    transport_type="http",
                    error_type=error_type,
                    error_message=error_message
                )

            await self._read_queue.put({
                "type": "error",
                "request_id": message.get("request_id", None),
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": f"Connection error: {error_message}"
                }
            })

        except Exception as e:
            # Other errors
            error_type = type(e).__name__
            error_message = str(e)

            # Record metrics
            if self.metrics_service:
                self.metrics_service.record_transport_error(
                    transport_type="http",
                    error_type=error_type,
                    error_message=error_message
                )

            await self._read_queue.put({
                "type": "error",
                "request_id": message.get("request_id", None),
                "error": {
                    "code": "UNEXPECTED_ERROR",
                    "message": f"Unexpected error: {error_message}"
                }
            })

    async def _handle_streaming(self, message: Dict[str, Any]) -> None:
        """
        Handle streaming messages via HTTP.

        Args:
            message: Stream message request
        """
        request_id = message.get("request_id")
        start_time = time.time()
        endpoint = "chat/completions" if message["type"] == "stream_chat_completions" else "completions"
        
        try:
            # Send stream request with stream=True
            if "stream" not in message:
                message["stream"] = True

            # Send the streaming request
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/{endpoint}",
                headers=self.auth_headers,
                json=message,
                timeout=None  # No timeout for streaming
            ) as response:
                response.raise_for_status()

                # Record initial response metrics
                if self.metrics_service:
                    self.metrics_service.record_transport_request(
                        transport_type="http",
                        endpoint=f"{endpoint}_stream",
                        status_code=response.status_code,
                        latency_ms=(time.time() - start_time) * 1000
                    )

                chunk_count = 0
                # Process stream chunks
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    # OpenAI format: lines starting with "data: " for SSE
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    # Skip "[DONE]" marker
                    if line.strip() == "[DONE]":
                        continue

                    # Parse chunk
                    try:
                        chunk = json.loads(line)
                        chunk_count += 1

                        # Add request ID if missing
                        if "request_id" not in chunk:
                            chunk["request_id"] = request_id

                        # Map to our streaming format
                        stream_type = "stream_chat_completions_chunk" if message["type"] == "stream_chat_completions" else "stream_completions_chunk"
                        delta_response = {
                            "type": stream_type,
                            **chunk
                        }

                        # Queue for reading
                        await self._read_queue.put(delta_response)
                        self.last_activity = datetime.utcnow()

                        # Record chunk metrics
                        if self.prometheus and chunk_count % 10 == 0:  # Record every 10 chunks to reduce overhead
                            self.prometheus.record_stream_chunk(
                                provider_id="http",
                                chunk_index=chunk_count
                            )

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in stream chunk: {line}")

                # Record streaming complete metrics  
                if self.metrics_service:
                    self.metrics_service.record_streaming_complete(
                        transport_type="http",
                        endpoint=f"{endpoint}_stream",
                        chunks=chunk_count,
                        total_time_ms=(time.time() - start_time) * 1000
                    )

                # Add final message
                await self._read_queue.put({
                    "type": f"{message['type']}_end",
                    "request_id": request_id
                })

        except httpx.HTTPStatusError as e:
            # HTTP error with response
            error_code = f"HTTP_{e.response.status_code}"
            error_message = f"HTTP error: {e.response.status_code}"

            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_code = error_data["error"].get("code", error_code)
                    error_message = error_data["error"].get("message", error_message)
            except:
                pass

            # Record error metrics
            if self.metrics_service:
                self.metrics_service.record_transport_error(
                    transport_type="http",
                    error_type=error_code,
                    error_message=error_message
                )

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
            error_type = type(e).__name__
            error_message = str(e)

            # Record error metrics
            if self.metrics_service:
                self.metrics_service.record_transport_error(
                    transport_type="http",
                    error_type=error_type,
                    error_message=error_message
                )

            await self._read_queue.put({
                "type": "error",
                "request_id": request_id,
                "error": {
                    "code": "CONNECTION_ERROR",
                    "message": f"Connection error: {error_message}"
                }
            })

        except Exception as e:
            # Other errors
            error_type = type(e).__name__
            error_message = str(e)

            # Record error metrics
            if self.metrics_service:
                self.metrics_service.record_transport_error(
                    transport_type="http",
                    error_type=error_type,
                    error_message=error_message
                )

            await self._read_queue.put({
                "type": "error",
                "request_id": request_id,
                "error": {
                    "code": "UNEXPECTED_ERROR",
                    "message": f"Unexpected error: {error_message}"
                }
            })

    async def close(self) -> None:
        """Close the HTTP client."""
        try:
            await self.client.aclose()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {str(e)}")


@dataclass
class ConnectionStats:
    """Statistics for a connection in the pool."""
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used_at: datetime = field(default_factory=datetime.utcnow)
    request_count: int = 0
    error_count: int = 0
    success_count: int = 0
    total_response_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Get success rate."""
        if self.request_count == 0:
            return 1.0
        return self.success_count / self.request_count
    
    @property
    def avg_response_time_ms(self) -> float:
        """Get average response time."""
        if self.success_count == 0:
            return 0.0
        return self.total_response_time_ms / self.success_count

    def record_success(self, response_time_ms: float) -> None:
        """Record a successful request."""
        self.request_count += 1
        self.success_count += 1
        self.last_used_at = datetime.utcnow()
        self.total_response_time_ms += response_time_ms

    def record_error(self) -> None:
        """Record a failed request."""
        self.request_count += 1
        self.error_count += 1
        self.last_used_at = datetime.utcnow()


class ConnectionPool:
    """Enhanced connection pool for HTTP clients."""

    def __init__(
        self,
        max_size: int = 10,
        max_idle_seconds: int = 300,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """Initialize connection pool.

        Args:
            max_size: Maximum number of connections in the pool
            max_idle_seconds: Maximum time in seconds a connection can be idle
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.max_size = max_size
        self.max_idle_seconds = max_idle_seconds
        self.metrics_service = metrics_service
        self.prometheus = prometheus_exporter
        self.pool: Dict[str, Tuple[HttpClient, ConnectionStats]] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task = None

        logger.debug(f"Initialized HTTP connection pool with max size {max_size}")

    async def start_cleanup_task(self) -> None:
        """Start background task to clean up idle connections."""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Started connection pool cleanup task")

    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self.cleanup_idle_connections()
        except asyncio.CancelledError:
            logger.debug("Connection pool cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in connection pool cleanup task: {str(e)}", exc_info=True)

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
                client, stats = self.pool.pop(key)
                try:
                    await client.close()
                    logger.debug(f"Closed idle connection {key} (idle for {(now - stats.last_used_at).total_seconds():.1f}s)")
                except Exception as e:
                    logger.warning(f"Error closing idle connection {key}: {str(e)}")

            # Update pool metrics
            if self.prometheus:
                self.prometheus.update_connection_pool(
                    provider_id="http",
                    transport_type="http",
                    pool_size=self.max_size,
                    active_connections=len(self.pool)
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

                    logger.debug(f"Created new connection {connection_id}, pool size: {len(self.pool)}")

                    # Update pool metrics
                    if self.prometheus:
                        self.prometheus.update_connection_pool(
                            provider_id="http",
                            transport_type="http",
                            pool_size=self.max_size,
                            active_connections=len(self.pool)
                        )

                    return new_connection, connection_id
                except Exception as e:
                    logger.error(f"Error creating new connection: {str(e)}", exc_info=True)
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
                logger.warning(f"Error closing connection {lru_key}: {str(e)}")

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

                logger.debug(f"Replaced connection in pool: {lru_key} -> {connection_id}")

                return new_connection, connection_id
            except Exception as e:
                logger.error(f"Error creating replacement connection: {str(e)}", exc_info=True)
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
                    logger.debug(f"Closed connection {key} during shutdown")
                except Exception as e:
                    logger.warning(f"Error closing connection {key} during shutdown: {str(e)}")

            self.pool.clear()

            # Update pool metrics
            if self.prometheus:
                self.prometheus.update_connection_pool(
                    provider_id="http",
                    transport_type="http",
                    pool_size=self.max_size,
                    active_connections=0
                )

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
                "success_count": conn_stats.success_count,
                "success_rate": conn_stats.success_rate,
                "avg_response_time_ms": conn_stats.avg_response_time_ms
            })

        return stats


class HTTPTransport(Transport):
    """
    Transport implementation that communicates with LLM services
    via HTTP for services that expose an HTTP/REST interface.

    Features:
    - Enhanced connection pooling with adaptive selection
    - Connection health monitoring and automatic pruning
    - TLS/mTLS support
    - Authentication via headers/API keys
    - Built-in retries and timeout handling
    - Performance metrics collection
    """

    def __init__(
        self,
        provider_id: str,
        config: Dict[str, Any],
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize HTTP transport.

        Args:
            provider_id: Provider ID
            config: Configuration dictionary
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.provider_id = provider_id
        
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx package is required for HTTPTransport")

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
        self.api_key_header = config.get("api_key_header", "Authorization")
        self.api_key_prefix = config.get("api_key_prefix", "Bearer")
        self.transport_type = "http"

        # Retry configuration
        self.retry_policy = RetryPolicy(
            max_retries=config.get("retry_max_attempts", 3),
            base_delay=config.get("retry_base_delay", 1.0),
            max_delay=config.get("retry_max_delay", 30.0),
            jitter_factor=config.get("retry_jitter_factor", 0.2),
            retry_codes=set(config.get("retry_status_codes", [408, 429, 500, 502, 503, 504]))
        )

        # Circuit breaker configuration  
        self.circuit_breaker = None
        if config.get("enable_circuit_breaker", True):
            self.circuit_breaker = CircuitBreaker(
                name=f"http_{self.provider_id}",
                failure_threshold=config.get("circuit_breaker_threshold", 5),
                recovery_timeout=config.get("circuit_breaker_recovery_timeout", 30),
                half_open_max_calls=config.get("circuit_breaker_half_open_max_calls", 1),
                reset_timeout=config.get("circuit_breaker_reset_timeout", 600)
            )

        # Rate limiter configuration
        self.rate_limiter = None
        if config.get("enable_rate_limiting", True):
            rate_limit_config = RateLimitConfig(
                strategy=config.get("rate_limit_strategy", "token_bucket"),
                requests_per_minute=config.get("rate_limit_rpm", 600),
                burst_size=config.get("rate_limit_burst_size", 100),
                window_size_seconds=config.get("rate_limit_window_size", 60),
                adaptive_factor=config.get("rate_limit_adaptive_factor", 0.5)
            )
            self.rate_limiter = RateLimiter(
                provider_id=provider_id,
                config=rate_limit_config
            )
        
        # Set up metrics and monitoring
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()

        # Initialize connection pool
        self.connection_pool = ConnectionPool(
            max_size=self.pool_maxsize,
            max_idle_seconds=self.max_idle_seconds,
            metrics_service=self.metrics_service,
            prometheus_exporter=self.prometheus
        )

        # Start connection pool cleanup task
        asyncio.create_task(self.connection_pool.start_cleanup_task())

        # Track active connections
        self.active_connections: Dict[str, HttpClient] = {}

        logger.info(f"Initialized HTTP transport for {self.base_url} (provider: {provider_id})")

    async def create_client(self) -> HttpClient:
        """
        Create a new HTTP client.

        Returns:
            HttpClient object that implements the read/write interface

        Raises:
            TransportError: If connection fails
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
                request_timeout=self.timeout_seconds,
                metrics_service=self.metrics_service,
                prometheus_exporter=self.prometheus
            )

            # Initialize the client
            await client.initialize()

            logger.info(f"Connected to LLM service via HTTP: {self.base_url}")

            return client

        except Exception as e:
            if isinstance(e, TransportError):
                raise

            logger.error(f"Failed to connect to LLM service: {str(e)}", exc_info=True)

            raise TransportError(
                message=f"Failed to connect to LLM service: {str(e)}",
                code="CONNECTION_ERROR",
                details={"error_type": type(e).__name__}
            )

    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[HttpClient, None]:
        """
        Connect to the LLM service using the connection pool.

        Returns:
            HttpClient object that implements the read/write interface

        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            RateLimitExceededError: If rate limit is exceeded
            TransportError: If connection fails
        """
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker open for {self.provider_id}, failing fast")
            if self.prometheus:
                self.prometheus.record_circuit_breaker_event(
                    provider_id=self.provider_id,
                    state="open",
                    event="rejected_request"
                )
            raise CircuitBreakerOpenError(
                message=f"Circuit breaker open for {self.provider_id}",
                details={"transport_type": "http"}
            )
        
        # Check rate limit
        if self.rate_limiter:
            success, wait_time = await self.rate_limiter.acquire()
            if not success:
                logger.warning(f"Rate limit exceeded for {self.provider_id}, retry after {wait_time:.2f}s")
                if self.prometheus:
                    self.prometheus.record_rate_limit_event(
                        provider_id=self.provider_id,
                        wait_time=wait_time
                    )
                raise RateLimitExceededError(
                    message=f"Rate limit exceeded, retry after {wait_time:.2f}s",
                    details={"retry_after": wait_time}
                )
        
        connection_id = None
        start_time = time.time()

        try:
            # Get a connection from the pool
            client, connection_id = await self.connection_pool.get_connection(self.create_client)

            # Track active connection
            self.active_connections[connection_id] = client

            # Yield the client
            yield client

            # Record success
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
            
            if self.rate_limiter:
                await self.rate_limiter.record_success()

            # Record success and response time
            response_time_ms = (time.time() - start_time) * 1000
            await self.connection_pool.release_connection(
                connection_id,
                success=True,
                response_time_ms=response_time_ms
            )

        except Exception as e:
            # Record failure
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            if self.rate_limiter:
                await self.rate_limiter.record_failure()

            # Record connection failure 
            if connection_id:
                await self.connection_pool.release_connection(
                    connection_id,
                    success=False
                )
            
            # Re-raise the error
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
                auth_headers[self.api_key_header] = f"{self.api_key_prefix} {api_key}"
                logger.debug(f"Using API key from environment variable {self.api_key_env_var}")

        # Add any headers from the config
        if self.headers:
            auth_headers.update(self.headers)

        return auth_headers

    async def send_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> TransportResponse:
        """
        Send a unary request.
        
        Args:
            method: Method name
            request: Request data
            metadata: Request metadata
            timeout: Request timeout
            
        Returns:
            Transport response
        """
        # Record metrics for the request
        start_time = time.time()
        timeout = timeout or self.timeout_seconds
        
        # Prepare message
        message = {
            "type": method,
            "request_id": metadata.get("request_id") if metadata else None,
            **request
        }
        
        # Serialize the message
        data = json.dumps(message).encode('utf-8')
        
        # Apply retry policy
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                # Get connection from pool
                async with self.connect() as client:
                    # Send the request
                    await client.write(data)
                    
                    # Get the response
                    response_data = await client.read()
                    response = json.loads(response_data.decode('utf-8'))
                    
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Record success metrics
                    self.prometheus.record_request(
                        provider_id=self.provider_id,
                        method=method,
                        status="success",
                        duration=time.time() - start_time
                    )
                    
                    # Return response
                    return TransportResponse(
                        data=response,
                        metadata={},
                        latency_ms=latency_ms
                    )
            
            except TransportError as e:
                # Check if error is retryable
                if e.code in self.retry_policy.retry_codes and attempt < self.retry_policy.max_retries:
                    # Calculate retry delay
                    delay = self.retry_policy.calculate_delay(attempt + 1)
                    logger.info(f"Retrying after error: {e.code} (attempt {attempt+1}/{self.retry_policy.max_retries}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)
                    continue
                raise
            
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=method,
                    status="error",
                    duration=duration,
                    error_type=type(e).__name__
                )
                
                # Convert to TransportError
                raise TransportError(
                    message=f"Unexpected error in HTTP transport: {str(e)}",
                    code="INTERNAL",
                    details={"error_type": type(e).__name__}
                )
    
    async def send_streaming_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> AsyncIterator[TransportResponse]:
        """
        Send a streaming request.
        
        Args:
            method: Method name
            request: Request data
            metadata: Request metadata
            timeout: Request timeout
            
        Returns:
            Iterator of transport responses
        """
        # Record metrics for the request
        start_time = time.time()
        timeout = timeout or self.timeout_seconds
        
        # Prepare message
        stream_method = f"stream_{method}"
        message = {
            "type": stream_method,
            "request_id": metadata.get("request_id") if metadata else None,
            "stream": True,
            **request
        }
        
        # Serialize the message
        data = json.dumps(message).encode('utf-8')
        
        # Apply retry policy
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                # Get connection from pool
                async with self.connect() as client:
                    # Send the request
                    await client.write(data)
                    
                    # Read stream responses
                    chunk_index = 0
                    while True:
                        response_data = await client.read()
                        response = json.loads(response_data.decode('utf-8'))
                        
                        # Check if this is the end message
                        if response.get("type") == f"{stream_method}_end":
                            break
                            
                        # Check for error messages
                        if response.get("type") == "error":
                            error = response.get("error", {})
                            raise TransportError(
                                message=error.get("message", "Unknown error"),
                                code=error.get("code", "UNKNOWN"),
                                details=error
                            )
                        
                        # Calculate latency for this chunk
                        chunk_latency_ms = (time.time() - start_time) * 1000
                        
                        # Record chunk metrics
                        self.prometheus.record_stream_chunk(
                            provider_id=self.provider_id,
                            chunk_index=chunk_index
                        )
                        
                        # Yield response
                        yield TransportResponse(
                            data=response,
                            metadata={},
                            latency_ms=chunk_latency_ms
                        )
                        chunk_index += 1
                    
                    # Record success metrics for the stream
                    duration = time.time() - start_time
                    self.prometheus.record_request(
                        provider_id=self.provider_id,
                        method=stream_method,
                        status="success",
                        duration=duration,
                        chunks=chunk_index
                    )
                    
                    # Break out of retry loop on success
                    break
            
            except TransportError as e:
                # Record failure metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=stream_method,
                    status="error",
                    duration=duration,
                    error_type=e.code
                )
                
                # Check if error is retryable
                if e.code in self.retry_policy.retry_codes and attempt < self.retry_policy.max_retries:
                    # Calculate retry delay
                    delay = self.retry_policy.calculate_delay(attempt + 1)
                    logger.info(f"Retrying stream after error: {e.code} (attempt {attempt+1}/{self.retry_policy.max_retries}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)
                    continue
                raise
            
            except Exception as e:
                # Record failure metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=stream_method,
                    status="error",
                    duration=duration,
                    error_type=type(e).__name__
                )
                
                # Convert to TransportError
                raise TransportError(
                    message=f"Unexpected error in HTTP stream: {str(e)}",
                    code="INTERNAL",
                    details={"error_type": type(e).__name__}
                )

    async def start(self) -> None:
        """Start the transport."""
        pass
    
    async def stop(self) -> None:
        """Stop the transport."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP transport."""
        # Close all connections in the pool
        await self.connection_pool.close_all()

        # Close any active connections
        for client in self.active_connections.values():
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing active connection: {str(e)}")

        self.active_connections.clear()

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        stats = self.connection_pool.get_stats()
        stats["provider_id"] = self.provider_id
        stats["transport_type"] = "http"
        stats["base_url"] = self.base_url
        
        if self.circuit_breaker:
            stats["circuit_breaker"] = {
                "is_open": self.circuit_breaker.is_open(),
                "failure_count": self.circuit_breaker.failure_count,
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "last_failure_time": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None,
                "reset_timeout_seconds": self.circuit_breaker.reset_timeout_seconds
            }
        
        if self.rate_limiter:
            stats["rate_limiter"] = self.rate_limiter.get_stats()
            
        return stats