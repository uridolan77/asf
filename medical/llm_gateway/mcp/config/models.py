"""
Configuration models for MCP Provider.

This module provides Pydantic models for configuration validation
and type safety across the MCP Provider implementation.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, validator


class TransportType(str, Enum):
    """Supported transport types for MCP connection."""
    
    STDIO = "stdio"
    GRPC = "grpc"
    HTTP = "http"
    CUSTOM = "custom"


class CircuitBreakerConfig(BaseModel):
    """Configuration for circuit breaker."""
    
    enabled: bool = Field(default=True, description="Whether circuit breaker is enabled")
    failure_threshold: int = Field(default=5, description="Number of failures before opening circuit", ge=1)
    recovery_timeout: int = Field(default=30, description="Seconds to wait before testing recovery", ge=1)
    half_open_max_calls: int = Field(default=1, description="Max calls to allow in HALF_OPEN before deciding", ge=1)
    reset_timeout: int = Field(default=600, description="Seconds after which to reset failure count", ge=1)


class RetryConfig(BaseModel):
    """Configuration for retry policy."""
    
    max_retries: int = Field(default=3, description="Maximum number of retry attempts", ge=0)
    base_delay: float = Field(default=1.0, description="Initial delay between retries in seconds", ge=0.1)
    max_delay: float = Field(default=60.0, description="Maximum delay between retries in seconds", ge=1.0)
    jitter_factor: float = Field(default=0.2, description="Randomness factor to apply to delays", ge=0.0, le=1.0)
    retry_codes: Set[str] = Field(default_factory=set, description="Set of error codes to retry")
    
    @validator("max_delay")
    def validate_max_delay(cls, v, values):
        """Validate max_delay is greater than base_delay."""
        if "base_delay" in values and v < values["base_delay"]:
            raise ValueError("max_delay must be greater than or equal to base_delay")
        return v


class ObservabilityConfig(BaseModel):
    """Configuration for observability components."""
    
    enable_metrics: bool = Field(default=True, description="Whether to enable metrics collection")
    enable_tracing: bool = Field(default=True, description="Whether to enable distributed tracing")
    enable_prometheus: bool = Field(default=False, description="Whether to start Prometheus HTTP server")
    prometheus_port: int = Field(default=8000, description="Port for Prometheus HTTP server", ge=1024, le=65535)
    otlp_endpoint: Optional[str] = Field(default=None, description="OTLP exporter endpoint for tracing")
    service_name: str = Field(default="mcp_provider", description="Service name for metrics and tracing")
    log_level: str = Field(default="INFO", description="Log level")
    structured_logging: bool = Field(default=True, description="Whether to use structured logging")


class SessionConfig(BaseModel):
    """Configuration for MCP session management."""
    
    max_sessions: int = Field(default=1, description="Maximum number of concurrent sessions", ge=1)
    session_ttl_seconds: int = Field(default=3600, description="Time-to-live for sessions in seconds", ge=60)
    prewarm_sessions: bool = Field(default=False, description="Whether to pre-warm sessions at startup")
    max_process_inactivity_seconds: int = Field(default=3600, description="Maximum inactivity time for a process", ge=60)
    connect_on_init: bool = Field(default=False, description="Whether to connect on initialization")


class StdioTransportConfig(BaseModel):
    """Configuration specific to stdio transport."""
    
    command: str = Field(..., description="Command to run")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    cwd: Optional[str] = Field(default=None, description="Working directory")
    process_poll_interval: int = Field(default=10, description="Seconds between process health checks", ge=1)
    termination_timeout: int = Field(default=5, description="Seconds to wait for graceful termination", ge=1)


class GrpcTransportConfig(BaseModel):
    """Configuration specific to gRPC transport."""
    
    endpoint: str = Field(..., description="gRPC server endpoint (host:port)")
    use_tls: bool = Field(default=False, description="Whether to use TLS")
    ca_cert: Optional[str] = Field(default=None, description="CA certificate file for TLS")
    client_cert: Optional[str] = Field(default=None, description="Client certificate file for mTLS")
    client_key: Optional[str] = Field(default=None, description="Client key file for mTLS")
    max_concurrent_streams: int = Field(default=100, description="Max concurrent streams", ge=1)
    reconnect_interval: int = Field(default=60, description="Seconds between reconnects", ge=1)
    metadata: Dict[str, str] = Field(default_factory=dict, description="Additional metadata for requests")


class HttpTransportConfig(BaseModel):
    """Configuration specific to HTTP transport."""
    
    base_url: str = Field(..., description="Base URL for HTTP requests")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    verify_ssl: bool = Field(default=True, description="Whether to verify SSL certificates")
    ca_cert: Optional[str] = Field(default=None, description="CA certificate file for SSL verification")
    client_cert: Optional[str] = Field(default=None, description="Client certificate file for mutual TLS")
    client_key: Optional[str] = Field(default=None, description="Client key file for mutual TLS")
    pool_maxsize: int = Field(default=10, description="Maximum number of connections in the pool", ge=1)
    keepalive_timeout: int = Field(default=5, description="Seconds to keep idle connections alive", ge=1)


class MCPConnectionConfig(BaseModel):
    """
    Comprehensive configuration for MCP connection.
    
    This model validates and provides defaults for all aspects of
    MCP provider connectivity, resilience, and observability.
    """
    
    # Connection basics
    transport_type: TransportType = Field(default=TransportType.STDIO, description="Transport type")
    timeout_seconds: int = Field(default=30, description="Operation timeout in seconds", ge=1)
    enable_streaming: bool = Field(default=False, description="Whether to enable streaming")
    
    # Authentication
    api_key_env_var: Optional[str] = Field(default=None, description="Environment variable for API key")
    
    # Resilience
    max_retries: Optional[int] = Field(default=None, description="Maximum retry attempts")
    retry_delay_seconds: Optional[float] = Field(default=None, description="Base delay between retries")
    max_jitter_seconds: Optional[float] = Field(default=None, description="Maximum jitter to add to retry delays")
    circuit_breaker_threshold: Optional[int] = Field(default=None, description="Failure threshold for circuit breaker")
    circuit_breaker_recovery_timeout: Optional[int] = Field(default=None, description="Recovery timeout for circuit breaker")
    
    # Session management
    max_sessions: Optional[int] = Field(default=None, description="Maximum number of concurrent sessions")
    session_ttl_seconds: Optional[int] = Field(default=None, description="Time-to-live for sessions in seconds")
    prewarm_sessions: bool = Field(default=False, description="Whether to pre-warm sessions at startup")
    
    # Transport-specific configs
    stdio_config: Optional[StdioTransportConfig] = Field(default=None, description="StdIO transport configuration")
    grpc_config: Optional[GrpcTransportConfig] = Field(default=None, description="gRPC transport configuration")
    http_config: Optional[HttpTransportConfig] = Field(default=None, description="HTTP transport configuration")
    custom_config: Optional[Dict[str, Any]] = Field(default=None, description="Custom transport configuration")
    
    # Observability
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig, description="Observability configuration")
    
    # Advanced retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Advanced retry configuration")
    
    # Advanced circuit breaker configuration
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="Circuit breaker configuration")
    
    # Session management
    session: SessionConfig = Field(default_factory=SessionConfig, description="Session management configuration")
    
    # Additional configuration for extensions
    additional_config: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration options")
    
    @validator("stdio_config")
    def validate_stdio_config(cls, v, values):
        """Validate that stdio_config is provided when transport_type is stdio."""
        if values.get("transport_type") == TransportType.STDIO and v is None:
            raise ValueError("stdio_config is required when transport_type is stdio")
        return v
    
    @validator("grpc_config")
    def validate_grpc_config(cls, v, values):
        """Validate that grpc_config is provided when transport_type is grpc."""
        if values.get("transport_type") == TransportType.GRPC and v is None:
            raise ValueError("grpc_config is required when transport_type is grpc")
        return v
    
    @validator("http_config")
    def validate_http_config(cls, v, values):
        """Validate that http_config is provided when transport_type is http."""
        if values.get("transport_type") == TransportType.HTTP and v is None:
            raise ValueError("http_config is required when transport_type is http")
        return v
    
    @validator("custom_config")
    def validate_custom_config(cls, v, values):
        """Validate that custom_config is provided when transport_type is custom."""
        if values.get("transport_type") == TransportType.CUSTOM and v is None:
            raise ValueError("custom_config is required when transport_type is custom")
        return v
    
    def get_transport_config(self) -> Dict[str, Any]:
        """
        Get the appropriate transport configuration based on transport_type.
        
        Returns:
            Dict with transport configuration
        """
        if self.transport_type == TransportType.STDIO and self.stdio_config:
            return self.stdio_config.dict()
        elif self.transport_type == TransportType.GRPC and self.grpc_config:
            return self.grpc_config.dict()
        elif self.transport_type == TransportType.HTTP and self.http_config:
            return self.http_config.dict()
        elif self.transport_type == TransportType.CUSTOM and self.custom_config:
            return self.custom_config
        else:
            return {}


class ToolConfig(BaseModel):
    """Configuration for tool calling capabilities."""
    
    enabled: bool = Field(default=True, description="Whether to enable tool calling")
    auto_tool_choice: bool = Field(default=True, description="Whether to automatically choose tools")
    tool_choice_strategy: str = Field(default="auto", description="Strategy for tool choice (auto, required, none)")
    max_tool_calls: int = Field(default=10, description="Maximum number of tool calls per interaction", ge=1)
    tools_as_system_prompt: bool = Field(default=False, description="Whether to include tools in system prompt")