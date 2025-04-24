"""
MCP Provider implementation for the Conexus LLM Gateway.

This module provides an implementation of the LLM Gateway provider interface
that uses the Model Context Protocol (MCP) to communicate with model servers.
"""

import asyncio
import json
import logging
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union, cast

from asf.conexus.llm_gateway.providers.base import BaseProvider
from asf.conexus.llm_gateway.core.models import (
    ContentItem,
    ErrorDetails, 
    ErrorLevel,
    FinishReason,
    LLMRequest, 
    LLMResponse,
    MCPContentType,
    MCPRole, 
    StreamChunk,
    ToolUseRequest,
    UsageStats
)
from asf.conexus.llm_gateway.mcp.config import (
    MCPConnectionConfig,
    TransportType,
    StdioConfig,
    HttpConfig,
    GrpcConfig,
    WebSocketConfig
)
from asf.conexus.llm_gateway.observability.metrics import (
    increment_counter,
    record_latency,
    gauge_set
)
from asf.conexus.llm_gateway.resilience.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)


class MCPTransport(ABC):
    """Abstract base class for MCP transports."""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish a connection to the MCP server."""
        pass
    
    @abstractmethod
    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the MCP server and get a response.
        
        Args:
            request_data: The request data to send
            
        Returns:
            The response data
        """
        pass
    
    @abstractmethod
    async def send_stream_request(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a streaming request to the MCP server.
        
        Args:
            request_data: The request data to send
            
        Yields:
            Response chunks
        """
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the MCP server."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the MCP server.
        
        Returns:
            Health status information
        """
        pass


class StdioTransport(MCPTransport):
    """MCP transport implementation that uses stdio for communication."""
    
    def __init__(self, config: StdioConfig):
        """
        Initialize the stdio transport.
        
        Args:
            config: The stdio configuration
        """
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self._lock = asyncio.Lock()  # Lock for process access
        self._connected = False
    
    async def connect(self) -> None:
        """Start the subprocess and connect to its stdio."""
        if self._connected:
            return
            
        async with self._lock:
            if self._connected:  # Double-check within lock
                return
                
            try:
                logger.info(f"Starting process: {self.config.command} {' '.join(self.config.args)}")
                self.process = subprocess.Popen(
                    [self.config.command, *self.config.args],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=self.config.env or None,
                    cwd=self.config.working_dir,
                    shell=self.config.shell,
                    text=False  # Use binary mode
                )
                
                # Wait a moment for process to initialize
                await asyncio.sleep(0.5)
                
                if self.process.poll() is not None:
                    # Process ended prematurely
                    stderr = self.process.stderr.read() if self.process.stderr else b""
                    raise RuntimeError(f"Process terminated prematurely: {stderr.decode('utf-8', errors='replace')}")
                    
                self._connected = True
                logger.info("Successfully started MCP process")
                    
            except Exception as e:
                logger.error(f"Failed to start MCP process: {e}")
                if self.process is not None:
                    await self.close()
                raise
    
    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a request to the MCP server and get a response.
        
        Args:
            request_data: The request data to send
            
        Returns:
            The response data
        """
        async with self._lock:
            if not self._connected or self.process is None:
                raise RuntimeError("Not connected to MCP process")
                
            if self.process.poll() is not None:
                raise RuntimeError(f"MCP process terminated with exit code {self.process.returncode}")
                
            try:
                # Send the request
                request_json = json.dumps(request_data) + "\n"
                self.process.stdin.write(request_json.encode("utf-8"))
                self.process.stdin.flush()
                
                # Read the response
                response_line = await asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                )
                
                if not response_line:
                    stderr = await asyncio.get_event_loop().run_in_executor(
                        None, self.process.stderr.read
                    )
                    raise RuntimeError(
                        f"Empty response from MCP process. Stderr: {stderr.decode('utf-8', errors='replace')}"
                    )
                
                response_data = json.loads(response_line.decode("utf-8"))
                return response_data
                
            except Exception as e:
                logger.error(f"Error communicating with MCP process: {e}")
                # Try to gracefully close the process
                await self.close()
                raise
    
    async def send_stream_request(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Send a streaming request to the MCP server.
        
        Args:
            request_data: The request data to send
            
        Yields:
            Response chunks
        """
        # Acquire lock for initial request
        async with self._lock:
            if not self._connected or self.process is None:
                raise RuntimeError("Not connected to MCP process")
                
            if self.process.poll() is not None:
                raise RuntimeError(f"MCP process terminated with exit code {self.process.returncode}")
                
            try:
                # Set streaming flag
                request_data["stream"] = True
                
                # Send the request
                request_json = json.dumps(request_data) + "\n"
                self.process.stdin.write(request_json.encode("utf-8"))
                self.process.stdin.flush()
                
                # Release lock after sending initial request
            
            except Exception as e:
                logger.error(f"Error initiating streaming with MCP process: {e}")
                # Try to gracefully close the process
                await self.close()
                raise
        
        # Process the stream (outside the lock to allow concurrent requests)
        try:
            while True:
                # Lock only for reading
                async with self._lock:
                    if not self._connected or self.process is None or self.process.poll() is not None:
                        break
                        
                    response_line = await asyncio.get_event_loop().run_in_executor(
                        None, self.process.stdout.readline
                    )
                
                if not response_line:
                    break
                    
                response_data = json.loads(response_line.decode("utf-8"))
                yield response_data
                
                # Check if this is the final chunk
                if "finish_reason" in response_data and response_data["finish_reason"]:
                    break
                    
        except Exception as e:
            logger.error(f"Error reading stream from MCP process: {e}")
            # Don't close process here, just let the exception propagate
            raise
    
    async def close(self) -> None:
        """Close the connection to the MCP server."""
        async with self._lock:
            if self.process is not None:
                # Try to terminate gracefully first
                try:
                    if self.process.poll() is None:  # Still running
                        # Send termination message if supported
                        try:
                            self.process.stdin.write(b'{"action": "terminate"}\n')
                            self.process.stdin.flush()
                            # Wait a moment for graceful shutdown
                            await asyncio.sleep(0.5)
                        except:
                            pass  # Ignore errors here
                    
                    # If still running, terminate forcefully
                    if self.process.poll() is None:
                        self.process.terminate()
                        # Wait a moment for termination
                        await asyncio.sleep(0.5)
                        
                        # If STILL running, kill
                        if self.process.poll() is None:
                            self.process.kill()
                            
                except Exception as e:
                    logger.warning(f"Error during process termination: {e}")
                    
                finally:
                    self.process = None
                    self._connected = False
                    logger.info("MCP process closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the MCP server.
        
        Returns:
            Health status information
        """
        if not self._connected or self.process is None:
            return {
                "status": "disconnected",
                "error": "Not connected to MCP process"
            }
            
        if self.process.poll() is not None:
            return {
                "status": "error",
                "error": f"Process terminated with exit code {self.process.returncode}"
            }
            
        try:
            response = await self.send_request({
                "action": "health_check",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return {
                "status": "operational",
                "details": response
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }


# You would implement other transports similarly:
class HttpTransport(MCPTransport):
    """MCP transport implementation that uses HTTP for communication."""
    
    def __init__(self, config: HttpConfig):
        """Initialize the HTTP transport."""
        self.config = config
        # Additional initialization here
    
    # Implement abstract methods...
    async def connect(self) -> None:
        # Implementation for HTTP connection setup
        pass
    
    async def send_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation for sending HTTP request
        pass
    
    async def send_stream_request(self, request_data: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        # Implementation for sending HTTP streaming request
        yield {}
    
    async def close(self) -> None:
        # Implementation for closing HTTP connection
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        # Implementation for HTTP health check
        return {"status": "operational"}


class MCPProvider(BaseProvider):
    """
    LLM Gateway provider implementation using the Model Context Protocol.
    
    This provider can connect to any MCP-compatible model server using
    various transport mechanisms (stdio, gRPC, HTTP, WebSocket).
    """
    
    def __init__(
        self,
        provider_config: Dict[str, Any],
        gateway_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the MCP provider.
        
        Args:
            provider_config: Configuration for this provider
            gateway_config: General gateway configuration
        """
        # Initialize base class
        super().__init__(provider_config, gateway_config)
        
        # Extract MCP configuration
        connection_params = self.provider_config.connection_params
        self.mcp_config = MCPConnectionConfig(**connection_params)
        
        # Initialize transport to None (will be created during initialization)
        self.transport: Optional[MCPTransport] = None
        
        # Create circuit breaker
        if self.mcp_config.circuit_breaker.enabled:
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=self.mcp_config.circuit_breaker.failure_threshold,
                reset_timeout_seconds=self.mcp_config.circuit_breaker.reset_timeout_seconds,
                half_open_success_threshold=self.mcp_config.circuit_breaker.half_open_success_threshold
            )
        else:
            self.circuit_breaker = None
        
        # Set up metrics
        self.enable_metrics = self.mcp_config.observability.enable_metrics
        self.metrics_prefix = f"llm_gateway_mcp_{self.provider_id}"
        
        logger.info(f"MCP Provider initialized for {self.provider_id} using {self.mcp_config.transport_type} transport")
    
    async def _initialize_client_async(self) -> None:
        """Initialize the MCP transport based on configuration."""
        # Create appropriate transport based on configuration
        if self.mcp_config.transport_type == TransportType.STDIO:
            if not self.mcp_config.stdio_config:
                raise ValueError("stdio_config is required for stdio transport")
                
            self.transport = StdioTransport(self.mcp_config.stdio_config)
            
        elif self.mcp_config.transport_type == TransportType.HTTP:
            if not self.mcp_config.http_config:
                raise ValueError("http_config is required for HTTP transport")
                
            self.transport = HttpTransport(self.mcp_config.http_config)
            
        # Add other transport types as needed
        
        else:
            raise ValueError(f"Unsupported transport type: {self.mcp_config.transport_type}")
        
        # Connect to the transport
        try:
            await self.transport.connect()
            logger.info(f"Successfully connected to MCP server for provider {self.provider_id}")
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for the given request.
        
        Args:
            request: The LLM request
            
        Returns:
            The LLM response
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize_async()
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker open for provider {self.provider_id}, rejecting request")
            return self._create_circuit_breaker_error_response(request)
        
        start_time = time.time()
        
        try:
            # Convert request to MCP format
            mcp_request = self._convert_to_mcp_request(request)
            
            # Send request to MCP server
            if self.transport is None:
                raise RuntimeError("MCP transport not initialized")
                
            mcp_response = await self.transport.send_request(mcp_request)
            
            # Convert response from MCP format
            response = self._convert_from_mcp_response(mcp_response, request)
            
            # Record metrics
            if self.enable_metrics:
                latency = time.time() - start_time
                record_latency(f"{self.metrics_prefix}_request_latency", latency)
                increment_counter(f"{self.metrics_prefix}_requests_total", {"status": "success"})
                
                if response.usage:
                    increment_counter(
                        f"{self.metrics_prefix}_tokens_total",
                        {"type": "prompt"},
                        response.usage.prompt_tokens
                    )
                    increment_counter(
                        f"{self.metrics_prefix}_tokens_total",
                        {"type": "completion"},
                        response.usage.completion_tokens
                    )
            
            # Update circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_success()
                
            return response
            
        except Exception as e:
            # Record metrics
            if self.enable_metrics:
                increment_counter(f"{self.metrics_prefix}_requests_total", {"status": "error"})
                
            # Update circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
                
            logger.error(f"Error in MCP provider: {e}")
            
            # Create error response
            return LLMResponse(
                request_id=request.request_id,
                generated_content="",
                error_details=ErrorDetails(
                    error_code="mcp_error",
                    message=str(e),
                    level=ErrorLevel.ERROR,
                    source=f"mcp_provider_{self.provider_id}"
                ),
                model_info={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                    "model_identifier": request.config.model_identifier
                }
            )
    
    async def generate_stream(self, request: LLMRequest) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming response for the given request.
        
        Args:
            request: The LLM request
            
        Yields:
            Stream chunks containing partial responses
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize_async()
        
        # Check circuit breaker
        if self.circuit_breaker and self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker open for provider {self.provider_id}, rejecting request")
            yield StreamChunk(
                chunk_id=0,
                request_id=request.request_id,
                error_details=ErrorDetails(
                    error_code="circuit_open",
                    message=f"Circuit breaker open for provider {self.provider_id}",
                    level=ErrorLevel.ERROR,
                    source=f"mcp_provider_{self.provider_id}"
                ),
                finish_reason=FinishReason.ERROR
            )
            return
        
        start_time = time.time()
        success = False
        chunk_index = 0
        
        try:
            # Convert request to MCP format
            mcp_request = self._convert_to_mcp_request(request)
            mcp_request["stream"] = True  # Ensure streaming is enabled
            
            # Send request to MCP server
            if self.transport is None:
                raise RuntimeError("MCP transport not initialized")
                
            # Track accumulated tokens
            total_prompt_tokens = 0
            total_completion_tokens = 0
            
            # Process stream
            async for chunk in self.transport.send_stream_request(mcp_request):
                # Convert chunk from MCP format
                stream_chunk = self._convert_from_mcp_stream_chunk(chunk, request, chunk_index)
                chunk_index += 1
                
                # Update token counts
                if stream_chunk.usage_update:
                    total_prompt_tokens = stream_chunk.usage_update.prompt_tokens
                    total_completion_tokens = stream_chunk.usage_update.completion_tokens
                
                # Yield the chunk
                yield stream_chunk
                
                # Check if this is the last chunk
                if stream_chunk.finish_reason:
                    success = stream_chunk.finish_reason != FinishReason.ERROR
                    break
            
            # Record metrics
            if self.enable_metrics:
                latency = time.time() - start_time
                record_latency(f"{self.metrics_prefix}_stream_latency", latency)
                increment_counter(f"{self.metrics_prefix}_streams_total", {"status": "success" if success else "error"})
                
                if total_prompt_tokens > 0:
                    increment_counter(
                        f"{self.metrics_prefix}_tokens_total",
                        {"type": "prompt"},
                        total_prompt_tokens
                    )
                if total_completion_tokens > 0:
                    increment_counter(
                        f"{self.metrics_prefix}_tokens_total",
                        {"type": "completion"},
                        total_completion_tokens
                    )
            
            # Update circuit breaker
            if self.circuit_breaker:
                if success:
                    self.circuit_breaker.record_success()
                else:
                    self.circuit_breaker.record_failure()
                
        except Exception as e:
            # Record metrics
            if self.enable_metrics:
                increment_counter(f"{self.metrics_prefix}_streams_total", {"status": "error"})
                
            # Update circuit breaker
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
                
            logger.error(f"Error in MCP stream: {e}")
            
            # Yield error chunk
            yield StreamChunk(
                chunk_id=chunk_index,
                request_id=request.request_id,
                error_details=ErrorDetails(
                    error_code="mcp_stream_error",
                    message=str(e),
                    level=ErrorLevel.ERROR,
                    source=f"mcp_provider_{self.provider_id}"
                ),
                finish_reason=FinishReason.ERROR
            )
    
    async def cleanup(self) -> None:
        """Clean up resources used by the MCP provider."""
        if self.transport is not None:
            try:
                await self.transport.close()
                logger.info(f"MCP transport closed for provider {self.provider_id}")
            except Exception as e:
                logger.error(f"Error closing MCP transport: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the MCP provider.
        
        Returns:
            Health status information
        """
        result = {
            "provider_id": self.provider_id,
            "provider_type": self.provider_type,
            "transport_type": self.mcp_config.transport_type,
            "initialized": self._initialized,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.circuit_breaker:
            result["circuit_breaker"] = {
                "state": "open" if self.circuit_breaker.is_open() else "closed",
                "failures": self.circuit_breaker.failure_count,
                "last_failure": self.circuit_breaker.last_failure_time.isoformat() if self.circuit_breaker.last_failure_time else None
            }
        
        if not self._initialized or self.transport is None:
            result["status"] = "not_initialized"
            return result
            
        try:
            transport_health = await self.transport.health_check()
            result.update(transport_health)
            return result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    def _convert_to_mcp_request(self, request: LLMRequest) -> Dict[str, Any]:
        """
        Convert an LLM Gateway request to MCP format.
        
        Args:
            request: The LLM Gateway request
            
        Returns:
            Request in MCP format
        """
        # Basic request data
        mcp_request = {
            "id": request.request_id,
            "model": request.config.model_identifier,
            "max_tokens": request.config.max_tokens,
            "temperature": request.config.temperature,
            "top_p": request.config.top_p,
            "frequency_penalty": request.config.frequency_penalty,
            "presence_penalty": request.config.presence_penalty,
            "stop_sequences": request.config.stop_sequences,
            "stream": request.config.stream
        }
        
        # Handle system prompt
        if request.config.system_prompt:
            mcp_request["system"] = request.config.system_prompt
        
        # Convert prompt content or items to messages
        if request.prompt_items:
            messages = []
            for item in request.prompt_items:
                if item.role is None:
                    continue  # Skip items without a role
                    
                message = {
                    "role": item.role.value,
                    "content": []
                }
                
                # Add content based on type
                if item.type == MCPContentType.TEXT:
                    message["content"].append({
                        "type": "text",
                        "text": item.text or ""
                    })
                elif item.type == MCPContentType.CODE:
                    message["content"].append({
                        "type": "code",
                        "text": item.text or ""
                    })
                elif item.type == MCPContentType.IMAGE:
                    if item.data:
                        message["content"].append({
                            "type": "image",
                            "image_url": item.data.get("url"),
                            "mime_type": item.mime_type
                        })
                
                messages.append(message)
            
            mcp_request["messages"] = messages
        else:
            # Simple text prompt
            mcp_request["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": request.prompt_content
                        }
                    ]
                }
            ]
        
        # Handle tools
        if request.tools:
            tools = []
            for tool in request.tools:
                tool_definition = {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {}
                }
                
                for function in tool.functions:
                    tool_definition["function"] = function.name
                    
                    # Build parameters schema
                    properties = {}
                    required = []
                    
                    for param in function.parameters:
                        param_schema = {"type": param.type, "description": param.description}
                        
                        if param.enum is not None:
                            param_schema["enum"] = param.enum
                            
                        if param.default is not None:
                            param_schema["default"] = param.default
                            
                        properties[param.name] = param_schema
                        
                        if param.required:
                            required.append(param.name)
                    
                    tool_definition["parameters"] = {
                        "type": "object",
                        "properties": properties,
                        "required": required if required else None
                    }
                    
                    # Only support one function per tool for now
                    break
                
                tools.append(tool_definition)
            
            mcp_request["tools"] = tools
        
        # Add additional parameters if specified
        if request.config.additional_params:
            mcp_request.update(request.config.additional_params)
            
        return mcp_request
    
    def _convert_from_mcp_response(self, mcp_response: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        """
        Convert an MCP response to LLM Gateway format.
        
        Args:
            mcp_response: The MCP response
            request: The original LLM request
            
        Returns:
            Response in LLM Gateway format
        """
        # Handle error response
        if "error" in mcp_response:
            return LLMResponse(
                request_id=request.request_id,
                generated_content="",
                error_details=ErrorDetails(
                    error_code=mcp_response.get("error_code", "mcp_error"),
                    message=mcp_response.get("error", "Unknown MCP error"),
                    level=ErrorLevel.ERROR,
                    source=f"mcp_provider_{self.provider_id}"
                ),
                model_info={
                    "provider_id": self.provider_id,
                    "provider_type": self.provider_type,
                    "model_identifier": request.config.model_identifier
                }
            )
        
        # Extract content from response
        content_text = ""
        content_items = []
        
        if "message" in mcp_response:
            message = mcp_response["message"]
            role = message.get("role", "assistant")
            
            # Process message content
            if "content" in message:
                for content in message["content"]:
                    content_type = content.get("type", "text")
                    
                    if content_type == "text":
                        text = content.get("text", "")
                        content_text += text
                        content_items.append(ContentItem(
                            type=MCPContentType.TEXT,
                            text=text,
                            role=MCPRole(role)
                        ))
                    elif content_type == "code":
                        code = content.get("text", "")
                        content_text += code
                        content_items.append(ContentItem(
                            type=MCPContentType.CODE,
                            text=code,
                            role=MCPRole(role)
                        ))
        
        # Extract usage statistics
        usage = None
        if "usage" in mcp_response:
            usage_data = mcp_response["usage"]
            usage = UsageStats(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        # Extract tool calls
        tool_calls = None
        if "tool_calls" in mcp_response:
            tool_calls = []
            for tc in mcp_response["tool_calls"]:
                tool_calls.append(ToolUseRequest(
                    id=tc.get("id", f"tc_{len(tool_calls)}"),
                    tool_id=tc.get("name", "unknown"),
                    function_name=tc.get("function", ""),
                    function_arguments=tc.get("parameters", {})
                ))
        
        # Map finish reason
        finish_reason = None
        if "finish_reason" in mcp_response:
            reason = mcp_response["finish_reason"]
            if reason == "stop" or reason == "end_turn":
                finish_reason = FinishReason.STOP
            elif reason == "length":
                finish_reason = FinishReason.LENGTH
            elif reason == "content_filter":
                finish_reason = FinishReason.CONTENT_FILTER
            elif reason == "tool_calls":
                finish_reason = FinishReason.TOOL_CALLS
            elif reason == "error":
                finish_reason = FinishReason.ERROR
        
        # Create final response
        return LLMResponse(
            request_id=request.request_id,
            generated_content=content_text,
            content_items=content_items,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
            model_info={
                "provider_id": self.provider_id,
                "provider_type": self.provider_type,
                "model_identifier": request.config.model_identifier
            },
            usage=usage,
            additional_info={
                "mcp_id": mcp_response.get("id"),
                "model": mcp_response.get("model")
            }
        )
    
    def _convert_from_mcp_stream_chunk(self, chunk: Dict[str, Any], request: LLMRequest, chunk_index: int) -> StreamChunk:
        """
        Convert an MCP stream chunk to LLM Gateway format.
        
        Args:
            chunk: The MCP stream chunk
            request: The original LLM request
            chunk_index: The index of this chunk
            
        Returns:
            Stream chunk in LLM Gateway format
        """
        # Check for error
        if "error" in chunk:
            return StreamChunk(
                chunk_id=chunk_index,
                request_id=request.request_id,
                error_details=ErrorDetails(
                    error_code=chunk.get("error_code", "mcp_stream_error"),
                    message=chunk.get("error", "Unknown MCP stream error"),
                    level=ErrorLevel.ERROR,
                    source=f"mcp_provider_{self.provider_id}"
                ),
                finish_reason=FinishReason.ERROR
            )
        
        # Extract delta text
        delta_text = None
        delta_content_items = []
        
        if "delta" in chunk:
            delta = chunk["delta"]
            # Process delta content based on type
            if isinstance(delta, str):
                delta_text = delta
            elif isinstance(delta, dict) and "content" in delta:
                content = delta["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            delta_text = item.get("text", "")
                            delta_content_items.append(ContentItem(
                                type=MCPContentType.TEXT,
                                text=delta_text,
                                role=MCPRole.ASSISTANT
                            ))
                        elif item.get("type") == "code":
                            code = item.get("text", "")
                            if delta_text is None:
                                delta_text = code
                            else:
                                delta_text += code
                            delta_content_items.append(ContentItem(
                                type=MCPContentType.CODE,
                                text=code,
                                role=MCPRole.ASSISTANT
                            ))
                else:
                    delta_text = str(content)
            else:
                # Try to extract text from delta
                delta_text = str(delta)
        
        # Extract usage
        usage_update = None
        if "usage" in chunk:
            usage_data = chunk["usage"]
            usage_update = UsageStats(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        # Extract tool calls
        delta_tool_calls = None
        if "tool_calls" in chunk:
            delta_tool_calls = []
            for tc in chunk["tool_calls"]:
                delta_tool_calls.append(ToolUseRequest(
                    id=tc.get("id", f"tc_{len(delta_tool_calls)}"),
                    tool_id=tc.get("name", "unknown"),
                    function_name=tc.get("function", ""),
                    function_arguments=tc.get("parameters", {})
                ))
        
        # Map finish reason
        finish_reason = None
        if "finish_reason" in chunk and chunk["finish_reason"]:
            reason = chunk["finish_reason"]
            if reason == "stop" or reason == "end_turn":
                finish_reason = FinishReason.STOP
            elif reason == "length":
                finish_reason = FinishReason.LENGTH
            elif reason == "content_filter":
                finish_reason = FinishReason.CONTENT_FILTER
            elif reason == "tool_calls":
                finish_reason = FinishReason.TOOL_CALLS
            elif reason == "error":
                finish_reason = FinishReason.ERROR
        
        # Create stream chunk
        return StreamChunk(
            chunk_id=chunk_index,
            request_id=request.request_id,
            delta_text=delta_text,
            delta_content_items=delta_content_items,
            delta_tool_calls=delta_tool_calls,
            finish_reason=finish_reason,
            usage_update=usage_update,
            provider_specific_data={
                "mcp_id": chunk.get("id"),
                "model": chunk.get("model")
            }
        )
    
    def _create_circuit_breaker_error_response(self, request: LLMRequest) -> LLMResponse:
        """
        Create an error response when circuit breaker is open.
        
        Args:
            request: The original request
            
        Returns:
            Error response
        """
        return LLMResponse(
            request_id=request.request_id,
            generated_content="",
            error_details=ErrorDetails(
                error_code="circuit_breaker_open",
                message=f"Circuit breaker is open for provider {self.provider_id}",
                level=ErrorLevel.ERROR,
                source=f"mcp_provider_{self.provider_id}",
                context={
                    "provider_id": self.provider_id,
                    "model": request.config.model_identifier,
                    "retry_after": self.circuit_breaker.retry_after_seconds() if self.circuit_breaker else None
                }
            ),
            model_info={
                "provider_id": self.provider_id,
                "provider_type": self.provider_type,
                "model_identifier": request.config.model_identifier
            }
        )
    
    @property
    def supported_models(self) -> Dict[str, Any]:
        """Get all models supported by this provider."""
        # In a real implementation, you might query the MCP server for supported models
        # For now, provide some reasonable defaults
        return {
            "provider_id": self.provider_id,
            "models": [
                {
                    "model_id": "claude-3-opus",
                    "display_name": "Claude 3 Opus",
                    "provider_id": self.provider_id,
                    "model_type": "text",
                    "context_window": 200000,
                    "capabilities": ["text", "code", "tool_use", "vision"],
                },
                {
                    "model_id": "claude-3-sonnet",
                    "display_name": "Claude 3 Sonnet",
                    "provider_id": self.provider_id,
                    "model_type": "text",
                    "context_window": 180000,
                    "capabilities": ["text", "code", "tool_use", "vision"],
                },
                {
                    "model_id": "claude-3-haiku",
                    "display_name": "Claude 3 Haiku",
                    "provider_id": self.provider_id,
                    "model_type": "text",
                    "context_window": 150000,
                    "capabilities": ["text", "code", "tool_use", "vision"],
                }
            ]
        }