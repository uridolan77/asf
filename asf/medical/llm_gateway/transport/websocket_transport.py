"""
WebSocket transport implementation for LLM Gateway.

This module provides a WebSocket transport implementation for real-time
bidirectional communication with LLM services that support WebSockets.
"""

import asyncio
import json
import logging
import time
import ssl
import uuid
from typing import AsyncGenerator, Dict, List, Any, Optional, AsyncIterator, Tuple, Union
from contextlib import asynccontextmanager
from datetime import datetime

from asf.medical.llm_gateway.transport.base import (
    Transport, TransportConfig, TransportResponse, TransportError,
    CircuitBreakerOpenError, RateLimitExceededError
)
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.resilience.circuit_breaker import CircuitBreaker
from asf.medical.llm_gateway.resilience.retry import RetryPolicy, DEFAULT_RETRY_POLICY
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter
from asf.medical.llm_gateway.resilience.rate_limiter import RateLimiter, RateLimitConfig

# Import websockets library
try:
    import websockets
    from websockets.client import WebSocketClientProtocol
    from websockets.exceptions import (
        WebSocketException, ConnectionClosed, ConnectionClosedError,
        ConnectionClosedOK, InvalidStatusCode, InvalidHandshake
    )
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    # Create stub classes for type checking
    class WebSocketClientProtocol:
        """Stub for WebSocketClientProtocol."""
        pass
    
    class WebSocketException(Exception):
        """Stub for WebSocketException."""
        pass
    
    class ConnectionClosed(WebSocketException):
        """Stub for ConnectionClosed."""
        pass
    
    class ConnectionClosedError(ConnectionClosed):
        """Stub for ConnectionClosedError."""
        pass
    
    class ConnectionClosedOK(ConnectionClosed):
        """Stub for ConnectionClosedOK."""
        pass
    
    class InvalidStatusCode(WebSocketException):
        """Stub for InvalidStatusCode."""
        pass
    
    class InvalidHandshake(WebSocketException):
        """Stub for InvalidHandshake."""
        pass

logger = logging.getLogger(__name__)


class WebSocketTransportConfig(TransportConfig):
    """Configuration for WebSocket transport."""
    
    transport_type: str = "websocket"
    uri: str  # WebSocket URI (ws:// or wss://)
    headers: Optional[Dict[str, str]] = None  # Additional headers for connection
    subprotocols: Optional[List[str]] = None  # WebSocket subprotocols
    ping_interval: float = 20.0  # Ping interval in seconds
    ping_timeout: float = 10.0  # Ping timeout in seconds
    close_timeout: float = 10.0  # Close timeout in seconds
    max_size: int = 10 * 1024 * 1024  # Maximum message size (10 MB)
    max_queue: int = 32  # Maximum queue size
    compression: Optional[str] = "deflate"  # Compression ("deflate" or None)
    pool_size: int = 5  # WebSocket connection pool size
    pool_max_idle_time_seconds: int = 300  # Maximum idle time for pooled connections
    connect_timeout: float = 30.0  # Connection timeout in seconds
    read_timeout: float = 60.0  # Read timeout in seconds
    ssl_context: Optional[Dict[str, Any]] = None  # SSL context options
    auth_token: Optional[str] = None  # Authentication token


class WebSocketSession:
    """
    WebSocket session for managing a single WebSocket connection.
    
    This class wraps a WebSocket connection and provides methods
    for sending and receiving messages with proper error handling.
    """
    
    def __init__(
        self,
        websocket: WebSocketClientProtocol,
        session_id: str,
        provider_id: str,
        config: WebSocketTransportConfig,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize WebSocket session.
        
        Args:
            websocket: WebSocket connection
            session_id: Session ID
            provider_id: Provider ID
            config: Transport configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.websocket = websocket
        self.session_id = session_id
        self.provider_id = provider_id
        self.config = config
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        # Session state
        self.created_at = datetime.utcnow()
        self.last_used_at = datetime.utcnow()
        self.message_count = 0
        self.error_count = 0
        self.is_closed = False
        
        # Set up ping task
        self._ping_task = None
        self._close_lock = asyncio.Lock()
        
        # Message handling
        self._response_queues: Dict[str, asyncio.Queue] = {}
        self._global_queue = asyncio.Queue()
        self._message_task = None
        
        # Start tasks
        self._start_tasks()
    
    def _start_tasks(self) -> None:
        """Start background tasks for ping and message handling."""
        if self._ping_task is None:
            self._ping_task = asyncio.create_task(self._ping_loop())
        
        if self._message_task is None:
            self._message_task = asyncio.create_task(self._message_loop())
    
    async def _ping_loop(self) -> None:
        """Send regular pings to keep the connection alive."""
        try:
            while True:
                await asyncio.sleep(self.config.ping_interval)
                
                if self.is_closed:
                    break
                
                try:
                    # Send ping
                    ping_waiter = await self.websocket.ping()
                    
                    # Wait for pong with timeout
                    await asyncio.wait_for(ping_waiter, timeout=self.config.ping_timeout)
                    
                    # Update timestamp
                    self.last_used_at = datetime.utcnow()
                except asyncio.TimeoutError:
                    logger.warning(f"WebSocket ping timeout for session {self.session_id}")
                    await self.close()
                    break
                except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
                    logger.warning(f"WebSocket connection closed during ping for session {self.session_id}")
                    await self.close()
                    break
                except Exception as e:
                    logger.error(f"Error in WebSocket ping loop: {str(e)}", exc_info=True)
                    self.error_count += 1
                    
                    if self.error_count > 3:
                        logger.warning(f"Too many ping errors, closing WebSocket session {self.session_id}")
                        await self.close()
                        break
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            await self.close()
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket ping loop: {str(e)}", exc_info=True)
            await self.close()
    
    async def _message_loop(self) -> None:
        """Process incoming messages and dispatch to queues."""
        try:
            while True:
                if self.is_closed:
                    break
                
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.config.read_timeout
                    )
                    
                    # Parse message
                    if isinstance(message, str):
                        data = json.loads(message)
                    else:
                        # Binary message
                        data = {'type': 'binary', 'data': message}
                    
                    # Update timestamp
                    self.last_used_at = datetime.utcnow()
                    
                    # Check if message has a request_id
                    request_id = data.get('request_id')
                    
                    if request_id and request_id in self._response_queues:
                        # Put message in specific queue
                        await self._response_queues[request_id].put(data)
                    else:
                        # Put message in global queue
                        await self._global_queue.put(data)
                except asyncio.TimeoutError:
                    logger.warning(f"WebSocket receive timeout for session {self.session_id}")
                    # Don't close - might get new messages later
                except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
                    logger.warning(f"WebSocket connection closed during receive for session {self.session_id}")
                    await self.close()
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in WebSocket message: {str(e)}")
                    self.error_count += 1
                except Exception as e:
                    logger.error(f"Error in WebSocket message loop: {str(e)}", exc_info=True)
                    self.error_count += 1
                    
                    if self.error_count > 5:
                        logger.warning(f"Too many message errors, closing WebSocket session {self.session_id}")
                        await self.close()
                        break
        except asyncio.CancelledError:
            # Task was cancelled, clean up
            await self.close()
        except Exception as e:
            logger.error(f"Unexpected error in WebSocket message loop: {str(e)}", exc_info=True)
            await self.close()
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message and wait for a response.
        
        Args:
            message: Message to send
            
        Returns:
            Response message
            
        Raises:
            TransportError: If an error occurs
        """
        if self.is_closed:
            raise TransportError(
                message="WebSocket connection is closed",
                code="CONNECTION_CLOSED"
            )
        
        # Update timestamp
        self.last_used_at = datetime.utcnow()
        self.message_count += 1
        
        # Generate request_id if not present
        request_id = message.get('request_id')
        if not request_id:
            request_id = str(uuid.uuid4())
            message['request_id'] = request_id
        
        # Create response queue
        response_queue = asyncio.Queue()
        self._response_queues[request_id] = response_queue
        
        try:
            # Send message
            try:
                json_data = json.dumps(message)
                await self.websocket.send(json_data)
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                raise TransportError(
                    message=f"WebSocket connection closed while sending: {str(e)}",
                    code="CONNECTION_CLOSED",
                    details={"error_type": type(e).__name__}
                )
            except Exception as e:
                raise TransportError(
                    message=f"Error sending WebSocket message: {str(e)}",
                    code="SEND_ERROR",
                    details={"error_type": type(e).__name__}
                )
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=self.config.read_timeout
                )
                return response
            except asyncio.TimeoutError as e:
                raise TransportError(
                    message=f"Timeout waiting for WebSocket response after {self.config.read_timeout}s",
                    code="RESPONSE_TIMEOUT",
                    details={"timeout": self.config.read_timeout}
                )
        finally:
            # Clean up response queue
            self._response_queues.pop(request_id, None)
    
    async def send_message_stream(self, message: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """
        Send a message and stream responses.
        
        Args:
            message: Message to send
            
        Returns:
            Iterator of response messages
            
        Raises:
            TransportError: If an error occurs
        """
        if self.is_closed:
            raise TransportError(
                message="WebSocket connection is closed",
                code="CONNECTION_CLOSED"
            )
        
        # Update timestamp
        self.last_used_at = datetime.utcnow()
        self.message_count += 1
        
        # Generate request_id if not present
        request_id = message.get('request_id')
        if not request_id:
            request_id = str(uuid.uuid4())
            message['request_id'] = request_id
        
        # Mark as streaming request
        message['stream'] = True
        
        # Create response queue
        response_queue = asyncio.Queue()
        self._response_queues[request_id] = response_queue
        
        try:
            # Send message
            try:
                json_data = json.dumps(message)
                await self.websocket.send(json_data)
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as e:
                raise TransportError(
                    message=f"WebSocket connection closed while sending: {str(e)}",
                    code="CONNECTION_CLOSED",
                    details={"error_type": type(e).__name__}
                )
            except Exception as e:
                raise TransportError(
                    message=f"Error sending WebSocket message: {str(e)}",
                    code="SEND_ERROR",
                    details={"error_type": type(e).__name__}
                )
            
            # Stream responses
            stop_reason = None
            while stop_reason is None:
                try:
                    # Get next chunk with timeout
                    chunk = await asyncio.wait_for(
                        response_queue.get(),
                        timeout=self.config.read_timeout
                    )
                    
                    # Check for stop reason
                    stop_reason = chunk.get('stop_reason')
                    
                    # Yield chunk
                    yield chunk
                    
                    # Break if this is the final chunk
                    if stop_reason is not None or chunk.get('type') == 'final':
                        break
                except asyncio.TimeoutError as e:
                    # If we don't receive a chunk for a while, assume the stream is complete
                    logger.warning(f"Timeout waiting for next WebSocket stream chunk after {self.config.read_timeout}s")
                    break
        finally:
            # Clean up response queue
            self._response_queues.pop(request_id, None)
    
    async def close(self) -> None:
        """Close the WebSocket connection."""
        async with self._close_lock:
            if self.is_closed:
                return
            
            self.is_closed = True
            
            # Cancel background tasks
            if self._ping_task and not self._ping_task.done():
                self._ping_task.cancel()
                try:
                    await self._ping_task
                except asyncio.CancelledError:
                    pass
            
            if self._message_task and not self._message_task.done():
                self._message_task.cancel()
                try:
                    await self._message_task
                except asyncio.CancelledError:
                    pass
            
            # Close WebSocket
            try:
                await self.websocket.close(
                    code=1000,  # Normal closure
                    reason="Session closed"
                )
                logger.debug(f"WebSocket session {self.session_id} closed normally")
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {str(e)}")
            
            # Clear response queues
            for queue in self._response_queues.values():
                if not queue.empty():
                    queue.put_nowait(None)
            self._response_queues.clear()
            
            # Add error to global queue if not empty
            if not self._global_queue.empty():
                self._global_queue.put_nowait(None)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Session statistics
        """
        return {
            "session_id": self.session_id,
            "provider_id": self.provider_id,
            "created_at": self.created_at.isoformat(),
            "last_used_at": self.last_used_at.isoformat(),
            "message_count": self.message_count,
            "error_count": self.error_count,
            "is_closed": self.is_closed,
            "active_requests": len(self._response_queues)
        }


class WebSocketConnectionPool:
    """
    Pool of WebSocket connections.
    
    This class manages a pool of WebSocket connections, handling
    connection creation, reuse, and cleanup.
    """
    
    def __init__(
        self,
        provider_id: str,
        config: WebSocketTransportConfig,
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize WebSocket connection pool.
        
        Args:
            provider_id: Provider ID
            config: Transport configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.provider_id = provider_id
        self.config = config
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        # Connection pool
        self.pool: Dict[str, WebSocketSession] = {}
        self._pool_lock = asyncio.Lock()
        self._session_counter = 0
        
        # Cleanup task
        self._cleanup_task = None
    
    async def start(self) -> None:
        """Start the connection pool."""
        # Start cleanup task
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Started WebSocket connection pool for {self.provider_id}")
    
    async def stop(self) -> None:
        """Stop the connection pool."""
        # Cancel cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        async with self._pool_lock:
            for session in list(self.pool.values()):
                await session.close()
            self.pool.clear()
        
        logger.info(f"Stopped WebSocket connection pool for {self.provider_id}")
    
    async def get_session(self) -> WebSocketSession:
        """
        Get a WebSocket session from the pool or create a new one.
        
        Returns:
            WebSocket session
        
        Raises:
            TransportError: If connection fails
        """
        async with self._pool_lock:
            # Find best available session
            best_session = None
            for session in self.pool.values():
                if not session.is_closed and session.error_count < 3:
                    if best_session is None or session.message_count < best_session.message_count:
                        best_session = session
            
            # Create new session if needed
            if best_session is None and len(self.pool) < self.config.pool_size:
                self._session_counter += 1
                session_id = f"{self.provider_id}_{self._session_counter}"
                best_session = await self._create_session(session_id)
                self.pool[session_id] = best_session
            
            # If no session available, wait for one to become available or pool to have space
            if best_session is None:
                # Wait for a session to become available or pool to have space
                for _ in range(10):  # Try 10 times
                    # First try to find an available session again
                    for session in self.pool.values():
                        if not session.is_closed and session.error_count < 3:
                            best_session = session
                            break
                    
                    if best_session is not None:
                        break
                    
                    # Clear out closed sessions
                    closed_sessions = [sid for sid, session in self.pool.items() if session.is_closed]
                    for sid in closed_sessions:
                        del self.pool[sid]
                    
                    # Create new session if there's space now
                    if len(self.pool) < self.config.pool_size:
                        self._session_counter += 1
                        session_id = f"{self.provider_id}_{self._session_counter}"
                        best_session = await self._create_session(session_id)
                        self.pool[session_id] = best_session
                        break
                    
                    # Wait a bit and try again
                    await asyncio.sleep(0.5)
            
            # If still no session available, raise an error
            if best_session is None:
                raise TransportError(
                    message="No WebSocket sessions available and pool is full",
                    code="POOL_FULL"
                )
            
            return best_session
    
    async def _create_session(self, session_id: str) -> WebSocketSession:
        """
        Create a new WebSocket session.
        
        Args:
            session_id: Session ID
            
        Returns:
            WebSocket session
            
        Raises:
            TransportError: If connection fails
        """
        if not WEBSOCKETS_AVAILABLE:
            raise TransportError(
                message="WebSocket transport requires 'websockets' package",
                code="MISSING_DEPENDENCY"
            )
        
        # Create SSL context if needed
        ssl_context = None
        if self.config.uri.startswith("wss://"):
            ssl_context = self._create_ssl_context()
        
        # Prepare headers
        headers = {}
        if self.config.headers:
            headers.update(self.config.headers)
        
        # Add authorization header if token is provided
        if self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        
        # Create WebSocket connection
        try:
            websocket = await asyncio.wait_for(
                websockets.connect(
                    self.config.uri,
                    extra_headers=headers,
                    subprotocols=self.config.subprotocols,
                    max_size=self.config.max_size,
                    max_queue=self.config.max_queue,
                    compression=self.config.compression,
                    ssl=ssl_context
                ),
                timeout=self.config.connect_timeout
            )
            
            # Create session
            session = WebSocketSession(
                websocket=websocket,
                session_id=session_id,
                provider_id=self.provider_id,
                config=self.config,
                metrics_service=self.metrics_service,
                prometheus_exporter=self.prometheus
            )
            
            logger.info(f"Created WebSocket session {session_id} for {self.provider_id}")
            
            # Record metrics
            self.prometheus.record_websocket_connection(
                provider_id=self.provider_id,
                session_id=session_id,
                state="connected"
            )
            
            return session
        except asyncio.TimeoutError as e:
            raise TransportError(
                message=f"Timeout connecting to WebSocket after {self.config.connect_timeout}s",
                code="CONNECTION_TIMEOUT",
                details={"timeout": self.config.connect_timeout}
            )
        except InvalidStatusCode as e:
            raise TransportError(
                message=f"WebSocket connection failed with status code {e.status_code}",
                code="INVALID_STATUS",
                details={"status_code": e.status_code}
            )
        except InvalidHandshake as e:
            raise TransportError(
                message=f"WebSocket handshake failed: {str(e)}",
                code="HANDSHAKE_FAILED",
                details={"error": str(e)}
            )
        except Exception as e:
            raise TransportError(
                message=f"Error creating WebSocket connection: {str(e)}",
                code="CONNECTION_ERROR",
                details={"error_type": type(e).__name__}
            )
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Create SSL context for secure WebSocket connections.
        
        Returns:
            SSL context
        """
        ssl_config = self.config.ssl_context or {}
        
        # Create context
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        
        # Configure context
        if "ca_cert" in ssl_config:
            context.load_verify_locations(cafile=ssl_config["ca_cert"])
        
        if "client_cert" in ssl_config and "client_key" in ssl_config:
            context.load_cert_chain(
                certfile=ssl_config["client_cert"],
                keyfile=ssl_config["client_key"]
            )
        
        # Set verification mode
        verify_mode = ssl_config.get("verify_mode", "CERT_REQUIRED")
        if verify_mode == "CERT_NONE":
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        elif verify_mode == "CERT_OPTIONAL":
            context.check_hostname = False
            context.verify_mode = ssl.CERT_OPTIONAL
        else:
            context.check_hostname = True
            context.verify_mode = ssl.CERT_REQUIRED
        
        return context
    
    async def _cleanup_loop(self) -> None:
        """Clean up idle and closed sessions."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_sessions()
        except asyncio.CancelledError:
            logger.debug("WebSocket cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in WebSocket cleanup task: {str(e)}", exc_info=True)
    
    async def _cleanup_sessions(self) -> None:
        """Clean up idle and closed sessions."""
        now = datetime.utcnow()
        
        async with self._pool_lock:
            # Find sessions to close
            sessions_to_close = []
            for session_id, session in self.pool.items():
                # Close sessions that are already marked as closed
                if session.is_closed:
                    sessions_to_close.append(session_id)
                    continue
                
                # Close idle sessions
                idle_time = (now - session.last_used_at).total_seconds()
                if idle_time > self.config.pool_max_idle_time_seconds:
                    logger.info(f"Closing idle WebSocket session {session_id} after {idle_time:.1f}s")
                    await session.close()
                    sessions_to_close.append(session_id)
                    continue
                
                # Close sessions with too many errors
                if session.error_count > 5:
                    logger.info(f"Closing WebSocket session {session_id} with {session.error_count} errors")
                    await session.close()
                    sessions_to_close.append(session_id)
                    continue
            
            # Remove closed sessions
            for session_id in sessions_to_close:
                self.pool.pop(session_id, None)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        return {
            "provider_id": self.provider_id,
            "transport_type": "websocket",
            "pool_size": len(self.pool),
            "max_pool_size": self.config.pool_size,
            "active_sessions": sum(1 for session in self.pool.values() if not session.is_closed),
            "idle_sessions": sum(1 for session in self.pool.values() if not session.is_closed and session.message_count == 0),
            "sessions": [session.get_stats() for session in self.pool.values()]
        }


class WebSocketTransport(Transport):
    """
    WebSocket transport implementation.
    
    This class provides a WebSocket transport implementation for real-time
    bidirectional communication with LLM services that support WebSockets.
    
    Features:
    - Connection pooling with adaptive session selection
    - Automatic reconnection and health monitoring
    - Session-based message routing
    - Streaming support with backpressure
    - SSL/TLS support with custom verification
    - Comprehensive error handling and metrics
    """
    
    def __init__(
        self,
        provider_id: str,
        config: Dict[str, Any],
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize WebSocket transport.
        
        Args:
            provider_id: Provider ID
            config: Transport configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.provider_id = provider_id
        
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("WebSocket transport requires 'websockets' package")
        
        # Ensure required config
        if "uri" not in config:
            raise ValueError("WebSocket URI is required")
        
        # Create configuration
        self.config = WebSocketTransportConfig(
            transport_type="websocket",
            uri=config["uri"],
            headers=config.get("headers"),
            subprotocols=config.get("subprotocols"),
            ping_interval=config.get("ping_interval", 20.0),
            ping_timeout=config.get("ping_timeout", 10.0),
            close_timeout=config.get("close_timeout", 10.0),
            max_size=config.get("max_size", 10 * 1024 * 1024),
            max_queue=config.get("max_queue", 32),
            compression=config.get("compression", "deflate"),
            pool_size=config.get("pool_size", 5),
            pool_max_idle_time_seconds=config.get("pool_max_idle_time_seconds", 300),
            connect_timeout=config.get("connect_timeout", 30.0),
            read_timeout=config.get("read_timeout", 60.0),
            ssl_context=config.get("ssl_context"),
            auth_token=config.get("auth_token")
        )
        
        # Set up metrics and monitoring
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        # Create connection pool
        self.pool = WebSocketConnectionPool(
            provider_id=provider_id,
            config=self.config,
            metrics_service=self.metrics_service,
            prometheus_exporter=self.prometheus
        )
        
        # Create retry policy
        self.retry_policy = RetryPolicy(
            max_retries=config.get("max_retries", 3),
            base_delay=config.get("retry_base_delay", 1.0),
            max_delay=config.get("retry_max_delay", 30.0),
            jitter_factor=config.get("retry_jitter_factor", 0.2),
            retry_codes=set(config.get("retry_codes", ["CONNECTION_CLOSED", "CONNECTION_ERROR", "SEND_ERROR"]))
        )
        
        # Create circuit breaker if enabled
        self.circuit_breaker = None
        if config.get("enable_circuit_breaker", True):
            self.circuit_breaker = CircuitBreaker(
                name=f"websocket_{provider_id}",
                failure_threshold=config.get("circuit_breaker_threshold", 5),
                recovery_timeout=config.get("circuit_breaker_recovery_timeout", 30),
                half_open_max_calls=config.get("circuit_breaker_half_open_max_calls", 1),
                reset_timeout=config.get("circuit_breaker_reset_timeout", 600)
            )
        
        # Create rate limiter if enabled
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
        
        logger.info(f"Initialized WebSocket transport for {provider_id} with URI {self.config.uri}")
    
    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[WebSocketSession, None]:
        """
        Get a WebSocket session from the pool.
        
        Returns:
            WebSocket session
            
        Raises:
            CircuitBreakerOpenError: If circuit breaker is open
            RateLimitExceededError: If rate limit is exceeded
            TransportError: If connection fails
        """
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker open for {self.provider_id}, failing fast")
            self.prometheus.record_circuit_breaker_event(
                provider_id=self.provider_id,
                state="open",
                event="rejected_request"
            )
            raise CircuitBreakerOpenError(
                message=f"Circuit breaker open for {self.provider_id}",
                details={"transport_type": "websocket"}
            )
        
        # Check rate limit
        if self.rate_limiter:
            success, wait_time = await self.rate_limiter.acquire()
            if not success:
                logger.warning(f"Rate limit exceeded for {self.provider_id}, retry after {wait_time:.2f}s")
                self.prometheus.record_rate_limit_event(
                    provider_id=self.provider_id,
                    wait_time=wait_time
                )
                raise RateLimitExceededError(
                    message=f"Rate limit exceeded, retry after {wait_time:.2f}s",
                    details={"retry_after": wait_time}
                )
        
        try:
            # Get session from pool
            session = await self.pool.get_session()
            
            try:
                # Yield session
                yield session
                
                # Record success
                if self.circuit_breaker:
                    self.circuit_breaker.record_success()
                
                if self.rate_limiter:
                    await self.rate_limiter.record_success()
            
            except Exception:
                # Session will be returned to pool, mark failure if needed
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure()
                
                if self.rate_limiter:
                    await self.rate_limiter.record_failure()
                
                raise
        
        except TransportError:
            # Record metrics and re-raise
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            
            if self.rate_limiter:
                await self.rate_limiter.record_failure()
            
            raise
    
    async def send_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> TransportResponse:
        """
        Send a request and get a response.
        
        Args:
            method: Method name
            request: Request data
            metadata: Request metadata
            timeout: Request timeout
            
        Returns:
            Response data
        """
        # Record metrics for the request
        start_time = time.time()
        
        # Create message
        message = {
            "method": method,
            "request_id": metadata.get("request_id") if metadata else str(uuid.uuid4()),
            "params": request
        }
        
        # Add metadata
        if metadata:
            message["metadata"] = metadata
        
        # Apply retry policy
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                # Get connection from pool
                async with self.connect() as session:
                    # Send message
                    response = await session.send_message(message)
                    
                    # Record metrics
                    duration = time.time() - start_time
                    latency_ms = duration * 1000
                    
                    self.prometheus.record_request(
                        provider_id=self.provider_id,
                        method=method,
                        status="success",
                        duration=duration
                    )
                    
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
                
                # Record metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=method,
                    status="error",
                    duration=duration,
                    error_type=e.code
                )
                
                raise
            
            except Exception as e:
                # Record metrics
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
                    message=f"Unexpected error in WebSocket transport: {str(e)}",
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
            Iterator of response chunks
        """
        # Record metrics for the request
        start_time = time.time()
        
        # Create message
        message = {
            "method": method,
            "request_id": metadata.get("request_id") if metadata else str(uuid.uuid4()),
            "params": request,
            "stream": True
        }
        
        # Add metadata
        if metadata:
            message["metadata"] = metadata
        
        # Apply retry policy
        for attempt in range(self.retry_policy.max_retries + 1):
            try:
                # Get connection from pool
                async with self.connect() as session:
                    # Send message and stream responses
                    chunk_index = 0
                    async for chunk in session.send_message_stream(message):
                        # Calculate latency
                        duration = time.time() - start_time
                        latency_ms = duration * 1000
                        
                        # Record chunk metrics
                        self.prometheus.record_stream_chunk(
                            provider_id=self.provider_id,
                            chunk_index=chunk_index
                        )
                        
                        # Yield response
                        yield TransportResponse(
                            data=chunk,
                            metadata={},
                            latency_ms=latency_ms
                        )
                        
                        chunk_index += 1
                    
                    # Record metrics for the complete stream
                    duration = time.time() - start_time
                    self.prometheus.record_request(
                        provider_id=self.provider_id,
                        method=method,
                        status="success",
                        duration=duration,
                        chunks=chunk_index
                    )
                    
                    # Break out of retry loop on success
                    break
            
            except TransportError as e:
                # Check if error is retryable
                if e.code in self.retry_policy.retry_codes and attempt < self.retry_policy.max_retries:
                    # Calculate retry delay
                    delay = self.retry_policy.calculate_delay(attempt + 1)
                    logger.info(f"Retrying stream after error: {e.code} (attempt {attempt+1}/{self.retry_policy.max_retries}, delay: {delay:.2f}s)")
                    await asyncio.sleep(delay)
                    continue
                
                # Record metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=method,
                    status="error",
                    duration=duration,
                    error_type=e.code
                )
                
                raise
            
            except Exception as e:
                # Record metrics
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
                    message=f"Unexpected error in WebSocket stream: {str(e)}",
                    code="INTERNAL",
                    details={"error_type": type(e).__name__}
                )
    
    async def start(self) -> None:
        """Start the transport."""
        await self.pool.start()
    
    async def stop(self) -> None:
        """Stop the transport."""
        await self.pool.stop()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Pool statistics
        """
        stats = self.pool.get_pool_stats()
        
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