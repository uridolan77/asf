"""
Configuration models for MCP (Model Context Protocol) connections.

This module defines Pydantic models for configuring MCP connections
and related settings.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TransportType(str, Enum):
    """Transport types for MCP connections."""
    STDIO = "stdio"  # Communication via stdin/stdout
    GRPC = "grpc"    # gRPC transport
    HTTP = "http"    # HTTP/REST transport
    WEBSOCKET = "websocket"  # WebSocket transport


class StdioConfig(BaseModel):
    """Configuration for stdio-based MCP transport."""
    command: str
    args: List[str] = Field(default_factory=list)
    env: Dict[str, str] = Field(default_factory=dict)
    working_dir: Optional[str] = None
    shell: bool = False
    timeout_seconds: int = 30


class GrpcConfig(BaseModel):
    """Configuration for gRPC-based MCP transport."""
    host: str
    port: int
    use_tls: bool = False
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None
    timeout_seconds: int = 30
    max_message_size: int = 10 * 1024 * 1024  # 10 MB


class HttpConfig(BaseModel):
    """Configuration for HTTP-based MCP transport."""
    base_url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 30
    verify_ssl: bool = True
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None


class WebSocketConfig(BaseModel):
    """Configuration for WebSocket-based MCP transport."""
    url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 30
    verify_ssl: bool = True
    ca_cert_path: Optional[str] = None
    client_cert_path: Optional[str] = None
    client_key_path: Optional[str] = None


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    jitter_factor: float = 0.1


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker behavior."""
    enabled: bool = True
    failure_threshold: int = 5
    reset_timeout_seconds: int = 30
    half_open_success_threshold: int = 2


class ObservabilityConfig(BaseModel):
    """Configuration for observability features."""
    enable_metrics: bool = True
    enable_tracing: bool = False
    enable_prometheus: bool = False
    prometheus_port: int = 8000
    otlp_endpoint: Optional[str] = None
    service_name: str = "mcp-provider"
    log_level: str = "INFO"
    structured_logging: bool = True


class AuthConfig(BaseModel):
    """Configuration for authentication."""
    api_key: Optional[str] = None
    api_key_header_name: str = "X-API-Key"
    token: Optional[str] = None
    token_header_name: str = "Authorization"
    token_prefix: Optional[str] = "Bearer"


class MCPConnectionConfig(BaseModel):
    """
    Configuration for an MCP connection.
    
    This class represents all the settings needed to establish and
    maintain a connection to an MCP-compatible model server.
    """
    transport_type: TransportType
    stdio_config: Optional[StdioConfig] = None
    grpc_config: Optional[GrpcConfig] = None
    http_config: Optional[HttpConfig] = None
    websocket_config: Optional[WebSocketConfig] = None
    auth: Optional[AuthConfig] = None
    enable_streaming: bool = True
    timeout_seconds: int = 60
    max_retries: int = 3
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    additional_config: Optional[Dict[str, Any]] = None

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True