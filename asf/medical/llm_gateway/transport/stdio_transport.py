"""
StdIO transport implementation for LLM Gateway.

This module provides a transport that launches and communicates with
an LLM server process using standard input/output streams.
"""

import asyncio
import os
import shlex
import signal
import shutil
import sys
import time
import json
from asyncio.subprocess import Process
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, AsyncIterator

from asf.medical.llm_gateway.transport.base import (
    Transport, TransportConfig, TransportResponse, TransportError
)
from asf.medical.llm_gateway.observability.metrics import MetricsService
from asf.medical.llm_gateway.observability.prometheus import get_prometheus_exporter

import logging
logger = logging.getLogger(__name__)


class StdioTransportConfig(TransportConfig):
    """Configuration for StdIO transport."""
    
    transport_type: str = "stdio"
    command: str  # Command to run
    args: Optional[List[str]] = None  # Command arguments
    env: Optional[Dict[str, str]] = None  # Environment variables
    cwd: Optional[str] = None  # Working directory
    timeout_seconds: float = 30.0  # Process startup timeout
    termination_timeout: float = 5.0  # Graceful termination timeout
    process_poll_interval: float = 10.0  # Health check interval
    max_process_inactivity_seconds: float = 3600.0  # Max process inactivity before restart
    api_key_env_var: Optional[str] = None  # API key environment variable name


class StdioTransport(Transport):
    """
    Transport implementation that communicates with an LLM server
    via stdin/stdout of a subprocess.
    
    Features:
    - Process lifecycle management
    - Signal handling (graceful termination)
    - Timeouts and health monitoring
    - Environment variable management for auth
    - Streaming support with proper process management
    """
    
    def __init__(
        self,
        provider_id: str,
        config: Dict[str, Any],
        metrics_service: Optional[MetricsService] = None,
        prometheus_exporter: Optional[Any] = None
    ):
        """
        Initialize StdIO transport.
        
        Args:
            provider_id: Provider ID
            config: Transport configuration
            metrics_service: Metrics service
            prometheus_exporter: Prometheus exporter
        """
        self.provider_id = provider_id
        
        # Validate required configuration
        if "command" not in config:
            raise ValueError("Command is required for StdIO transport")
        
        # Extract configuration
        self.command = config["command"]
        self.args = config.get("args", [])
        self.env = config.get("env", {})
        self.cwd = config.get("cwd")
        self.timeout_seconds = config.get("timeout_seconds", 30)
        self.termination_timeout = config.get("termination_timeout", 5)
        self.process_poll_interval = config.get("process_poll_interval", 10)
        self.max_process_inactivity = config.get("max_process_inactivity_seconds", 3600)
        self.api_key_env_var = config.get("api_key_env_var")
        
        # Process state
        self.process: Optional[Process] = None
        self.last_heartbeat = datetime.utcnow()
        self._monitor_task: Optional[asyncio.Task] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._process_lock = asyncio.Lock()
        
        # Set up metrics and monitoring
        self.metrics_service = metrics_service or MetricsService()
        self.prometheus = prometheus_exporter or get_prometheus_exporter()
        
        logger.info(f"Initialized StdIO transport for {provider_id} with command: {self.command}")
    
    @asynccontextmanager
    async def connect(self) -> AsyncGenerator[Tuple[asyncio.StreamReader, asyncio.StreamWriter], None]:
        """
        Start the server process and establish stdin/stdout streams.
        
        Returns:
            Tuple of (reader, writer) for the process
        
        Raises:
            TransportError: If the process fails to start
        """
        async with self._process_lock:
            try:
                # Start the process if needed
                process = await self._start_process()
                
                # Update heartbeat timestamp
                self.last_heartbeat = datetime.utcnow()
                
                # Record connection attempt
                self.prometheus.record_process_connection(
                    provider_id=self.provider_id,
                    pid=process.pid,
                    state="connected"
                )
                
                # Start monitoring task if needed
                await self._ensure_monitor_task()
                
                # Yield the streams for communication
                yield process.stdout, process.stdin
            
            except Exception as e:
                logger.error(f"Error connecting to process: {str(e)}", exc_info=True)
                
                # Record failure
                self.prometheus.record_process_connection(
                    provider_id=self.provider_id,
                    pid=self.process.pid if self.process else None,
                    state="failed"
                )
                
                # Terminate process on connection error
                await self._terminate_process()
                
                # Convert to TransportError
                if isinstance(e, TransportError):
                    raise
                
                raise TransportError(
                    message=f"Failed to connect to process: {str(e)}",
                    code="CONNECTION_ERROR",
                    details={"error_type": type(e).__name__}
                )
    
    async def _start_process(self) -> Process:
        """
        Start the server process.
        
        Returns:
            The Process object
        
        Raises:
            TransportError: If process fails to start
        """
        start_time = time.time()
        
        if self.process is not None and self.process.returncode is None:
            # Process is still running, reuse it
            logger.debug(f"Reusing existing process (PID: {self.process.pid})")
            return self.process
        
        # Resolve the command path
        cmd_path = shutil.which(self.command)
        if cmd_path is None:
            raise TransportError(
                message=f"Command not found: {self.command}",
                code="COMMAND_NOT_FOUND"
            )
        
        # Prepare environment variables
        env = os.environ.copy()
        env.update(self.env)
        
        # Add authentication variables
        auth_env = self._get_auth_env()
        env.update(auth_env)
        
        # Log startup with sensitive info masked
        safe_env = {k: "***" if k.lower().endswith("key") or k.lower().endswith("token") else v 
                   for k, v in env.items() if k not in os.environ or k in auth_env}
        logger.info(
            f"Starting process for {self.provider_id}",
            extra={
                "command": cmd_path,
                "args": self.args,
                "env": safe_env,
                "cwd": self.cwd or os.getcwd()
            }
        )
        
        try:
            # Start the process asynchronously
            self.process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    cmd_path,
                    *self.args,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                    cwd=self.cwd
                ),
                timeout=self.timeout_seconds
            )
            
            # Start logging stderr in the background
            self._start_stderr_logging()
            
            # Record metrics
            self.prometheus.record_process_start(
                provider_id=self.provider_id,
                pid=self.process.pid,
                command=self.command
            )
            
            logger.info(f"Process started for {self.provider_id} (PID: {self.process.pid})")
            
            return self.process
        
        except asyncio.TimeoutError as e:
            raise TransportError(
                message=f"Timeout starting process after {self.timeout_seconds}s",
                code="STARTUP_TIMEOUT",
                details={"timeout": self.timeout_seconds}
            )
        
        except Exception as e:
            logger.error(f"Failed to start process: {str(e)}", exc_info=True)
            
            raise TransportError(
                message=f"Failed to start process: {str(e)}",
                code="STARTUP_ERROR",
                details={"error_type": type(e).__name__}
            )
    
    def _start_stderr_logging(self) -> None:
        """Start logging the stderr output in a background task."""
        if self._stderr_task is not None:
            if not self._stderr_task.done():
                return  # Task is still running
            else:
                # Clean up the done task
                try:
                    self._stderr_task.result()  # Check for exceptions
                except Exception as e:
                    logger.warning(f"Previous stderr logging task failed: {str(e)}")
        
        self._stderr_task = asyncio.create_task(self._log_stderr())
    
    async def _log_stderr(self) -> None:
        """Log the stderr output from the process."""
        if self.process is None or self.process.stderr is None:
            return
        
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                
                stderr_line = line.decode('utf-8', errors='replace').rstrip()
                
                if stderr_line.startswith("ERROR") or stderr_line.startswith("CRITICAL"):
                    logger.error(f"Process stderr: {stderr_line}")
                elif stderr_line.startswith("WARNING"):
                    logger.warning(f"Process stderr: {stderr_line}")
                else:
                    logger.debug(f"Process stderr: {stderr_line}")
                
                # Record stderr output
                self.prometheus.record_process_stderr(
                    provider_id=self.provider_id,
                    pid=self.process.pid,
                    line=stderr_line[:100]  # Limit length for metrics
                )
        
        except asyncio.CancelledError:
            logger.debug(f"Stderr logging task cancelled for PID {self.process.pid}")
            raise
        
        except Exception as e:
            logger.warning(
                f"Error reading stderr: {str(e)}",
                exc_info=True
            )
    
    async def _ensure_monitor_task(self) -> None:
        """Ensure the process monitor task is running."""
        if self._monitor_task is not None:
            if not self._monitor_task.done():
                return  # Task is still running
            else:
                # Clean up the done task
                try:
                    self._monitor_task.result()  # Check for exceptions
                except Exception as e:
                    logger.warning(f"Previous monitor task failed: {str(e)}")
        
        self._monitor_task = asyncio.create_task(self._monitor_process())
    
    async def _monitor_process(self) -> None:
        """
        Monitor the health of the process and restart it if needed.
        
        This runs as a background task to ensure the process stays alive.
        """
        try:
            while True:
                await asyncio.sleep(self.process_poll_interval)
                
                async with self._process_lock:
                    # Check if process is still running
                    if self.process is None:
                        logger.warning("Monitor task found null process reference")
                        break
                    
                    if self.process.returncode is not None:
                        logger.warning(
                            f"Process terminated unexpectedly",
                            extra={
                                "pid": self.process.pid,
                                "returncode": self.process.returncode
                            }
                        )
                        
                        # Record termination
                        self.prometheus.record_process_termination(
                            provider_id=self.provider_id,
                            pid=self.process.pid,
                            returncode=self.process.returncode
                        )
                        
                        # Process will be restarted on next connect() call
                        break
                    
                    # Check for inactivity timeout
                    inactivity_time = (datetime.utcnow() - self.last_heartbeat).total_seconds()
                    
                    if inactivity_time > self.max_process_inactivity:
                        logger.warning(
                            f"Process inactive, terminating",
                            extra={
                                "pid": self.process.pid,
                                "inactivity_seconds": inactivity_time
                            }
                        )
                        
                        await self._terminate_process()
                        break
        
        except asyncio.CancelledError:
            logger.debug(f"Process monitor task cancelled")
            raise
        
        except Exception as e:
            logger.error(
                f"Error in process monitor task: {str(e)}",
                exc_info=True
            )
    
    async def _terminate_process(self) -> None:
        """Gracefully terminate the process."""
        if self.process is None or self.process.returncode is not None:
            return
        
        logger.info(f"Terminating process (PID: {self.process.pid})")
        
        try:
            # Send SIGTERM for graceful shutdown
            self.process.terminate()
            
            # Wait for process to terminate
            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=self.termination_timeout
                )
                logger.info(
                    f"Process terminated gracefully",
                    extra={
                        "pid": self.process.pid,
                        "returncode": self.process.returncode
                    }
                )
                
                # Record termination
                self.prometheus.record_process_termination(
                    provider_id=self.provider_id,
                    pid=self.process.pid,
                    returncode=self.process.returncode,
                    graceful=True
                )
            
            except asyncio.TimeoutError:
                # Force kill if it doesn't terminate gracefully
                logger.warning(
                    f"Process did not terminate gracefully, sending SIGKILL",
                    extra={"pid": self.process.pid}
                )
                
                self.process.kill()
                await self.process.wait()
                
                logger.info(
                    f"Process killed",
                    extra={
                        "pid": self.process.pid,
                        "returncode": self.process.returncode
                    }
                )
                
                # Record forced termination
                self.prometheus.record_process_termination(
                    provider_id=self.provider_id,
                    pid=self.process.pid,
                    returncode=self.process.returncode,
                    graceful=False
                )
        
        except Exception as e:
            logger.error(
                f"Error terminating process: {str(e)}",
                exc_info=True
            )
    
    def _get_auth_env(self) -> Dict[str, str]:
        """Get authentication environment variables for the process."""
        auth_env = {}
        
        # Look for API key in environment
        if self.api_key_env_var:
            api_key = os.environ.get(self.api_key_env_var)
            if api_key:
                # Use configured environment variable for server auth
                auth_env["LLM_API_KEY"] = api_key
                logger.debug(f"Using API key from {self.api_key_env_var}")
                return auth_env
        
        # Check for default API keys
        for key_var in ["LLM_SERVER_API_KEY", "API_KEY", "OPENAI_API_KEY"]:
            key_value = os.environ.get(key_var)
            if key_value:
                auth_env["LLM_API_KEY"] = key_value
                logger.debug(f"Using API key from {key_var}")
                return auth_env
        
        logger.warning("No API key environment variable configured or found")
        return {}
    
    async def send_request(
        self,
        method: str,
        request: Any,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> TransportResponse:
        """
        Send a request to the process.
        
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
        timeout = timeout or self.timeout_seconds
        
        try:
            # Get connection
            async with self.connect() as (reader, writer):
                # Create message
                message = {
                    "method": method,
                    "id": metadata.get("request_id") if metadata else str(time.time()),
                    "params": request
                }
                
                if metadata:
                    message["metadata"] = metadata
                
                # Serialize the message
                json_data = json.dumps(message).encode('utf-8') + b'\n'
                
                # Send the request
                writer.write(json_data)
                await writer.drain()
                
                # Read response
                response_line = await asyncio.wait_for(
                    reader.readline(),
                    timeout=timeout
                )
                
                if not response_line:
                    raise TransportError(
                        message="Process closed connection",
                        code="CONNECTION_CLOSED"
                    )
                
                # Parse response
                try:
                    response_data = json.loads(response_line.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise TransportError(
                        message=f"Invalid JSON response: {str(e)}",
                        code="INVALID_RESPONSE",
                        details={"response": response_line.decode('utf-8', errors='replace')}
                    )
                
                # Check for error
                if "error" in response_data:
                    error = response_data["error"]
                    raise TransportError(
                        message=error.get("message", "Unknown error"),
                        code=error.get("code", "SERVER_ERROR"),
                        details=error
                    )
                
                # Record metrics
                duration = time.time() - start_time
                latency_ms = duration * 1000
                
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=method,
                    status="success",
                    duration=duration
                )
                
                # Update heartbeat
                self.last_heartbeat = datetime.utcnow()
                
                return TransportResponse(
                    data=response_data,
                    metadata={},
                    latency_ms=latency_ms
                )
        
        except asyncio.TimeoutError as e:
            # Record metrics
            duration = time.time() - start_time
            self.prometheus.record_request(
                provider_id=self.provider_id,
                method=method,
                status="error",
                duration=duration,
                error_type="TIMEOUT"
            )
            
            raise TransportError(
                message=f"Request timed out after {timeout}s",
                code="TIMEOUT",
                details={"timeout": timeout}
            )
        
        except TransportError:
            # Record metrics and re-raise
            duration = time.time() - start_time
            self.prometheus.record_request(
                provider_id=self.provider_id,
                method=method,
                status="error",
                duration=duration,
                error_type="TRANSPORT_ERROR"
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
                message=f"Error sending request: {str(e)}",
                code="REQUEST_ERROR",
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
        Send a streaming request to the process.
        
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
        timeout = timeout or self.timeout_seconds
        
        try:
            # Get connection
            async with self.connect() as (reader, writer):
                # Create message
                message = {
                    "method": method,
                    "id": metadata.get("request_id") if metadata else str(time.time()),
                    "params": request,
                    "stream": True
                }
                
                if metadata:
                    message["metadata"] = metadata
                
                # Serialize the message
                json_data = json.dumps(message).encode('utf-8') + b'\n'
                
                # Send the request
                writer.write(json_data)
                await writer.drain()
                
                # Read response stream
                chunk_index = 0
                stream_done = False
                
                while not stream_done:
                    try:
                        # Read next chunk
                        response_line = await asyncio.wait_for(
                            reader.readline(),
                            timeout=timeout
                        )
                        
                        if not response_line:
                            # Connection closed
                            break
                        
                        # Parse response
                        try:
                            chunk_data = json.loads(response_line.decode('utf-8'))
                        except json.JSONDecodeError as e:
                            # Log but don't fail the whole stream
                            logger.warning(f"Invalid JSON in stream chunk: {str(e)}")
                            continue
                        
                        # Check for error
                        if "error" in chunk_data:
                            error = chunk_data["error"]
                            raise TransportError(
                                message=error.get("message", "Unknown error"),
                                code=error.get("code", "SERVER_ERROR"),
                                details=error
                            )
                        
                        # Check for end of stream
                        stream_done = chunk_data.get("done", False) or "stop_reason" in chunk_data
                        
                        # Record metrics for this chunk
                        duration = time.time() - start_time
                        latency_ms = duration * 1000
                        
                        # Record chunk metrics every 10 chunks to reduce overhead
                        if chunk_index % 10 == 0:
                            self.prometheus.record_stream_chunk(
                                provider_id=self.provider_id,
                                chunk_index=chunk_index
                            )
                        
                        # Update heartbeat
                        self.last_heartbeat = datetime.utcnow()
                        
                        # Yield the chunk
                        yield TransportResponse(
                            data=chunk_data,
                            metadata={},
                            latency_ms=latency_ms
                        )
                        
                        chunk_index += 1
                    
                    except asyncio.TimeoutError:
                        # If we haven't received a chunk in a while, assume the stream is done
                        logger.warning(f"Timeout waiting for next stream chunk after {timeout}s")
                        break
                
                # Record final metrics
                duration = time.time() - start_time
                self.prometheus.record_request(
                    provider_id=self.provider_id,
                    method=method,
                    status="success",
                    duration=duration,
                    chunks=chunk_index
                )
        
        except TransportError:
            # Record metrics and re-raise
            duration = time.time() - start_time
            self.prometheus.record_request(
                provider_id=self.provider_id,
                method=method,
                status="error",
                duration=duration,
                error_type="TRANSPORT_ERROR"
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
                message=f"Error in streaming request: {str(e)}",
                code="STREAM_ERROR",
                details={"error_type": type(e).__name__}
            )
    
    async def start(self) -> None:
        """Start the transport by ensuring the process is running."""
        try:
            async with self._process_lock:
                await self._start_process()
                await self._ensure_monitor_task()
            
            logger.info(f"StdIO transport started for {self.provider_id}")
        
        except Exception as e:
            logger.error(f"Error starting StdIO transport: {str(e)}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the transport by terminating the process."""
        await self.cleanup()
    
    async def cleanup(self) -> None:
        """Clean up resources and terminate the process."""
        logger.info(f"Cleaning up StdIO transport for {self.provider_id}")
        
        # Cancel monitor task
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Cancel stderr task
        if self._stderr_task and not self._stderr_task.done():
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
        
        # Terminate the process
        await self._terminate_process()
        
        logger.info(f"StdIO transport stopped for {self.provider_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the transport.
        
        Returns:
            Health check result
        """
        status = "available"
        message = "Process is running"
        
        # Check if process exists and is running
        if self.process is None:
            status = "initializing"
            message = "Process not started yet"
        elif self.process.returncode is not None:
            status = "unavailable"
            message = f"Process terminated with code {self.process.returncode}"
        
        return {
            "provider_id": self.provider_id,
            "transport_type": "stdio",
            "status": status,
            "message": message,
            "pid": self.process.pid if self.process and hasattr(self.process, 'pid') else None,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "command": self.command,
            "args": self.args,
            "inactivity_seconds": (datetime.utcnow() - self.last_heartbeat).total_seconds()
        }