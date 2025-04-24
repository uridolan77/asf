"""
Model Context Protocol (MCP) Implementation

This module provides integration with Model Context Protocol (MCP) servers for
the LLM Gateway. It contains specialized session management, transport handling,
and configuration components for MCP compatibility.

The MCP implementation is built on top of the consolidated transport layer
while providing protocol-specific abstractions.
"""

from .errors import (
    McpError,
    McpTransportError,
    McpTimeoutError,
    McpAuthenticationError,
    McpRateLimitError,
    McpInvalidRequestError,
    McpServerError,
)

from .session import MCPSession
from .session_pool import (
    EnhancedSessionPool,
    Session,
    SessionHealth,
    SessionMetadata,
    SessionPerformance,
    SessionPoolConfig,
    SessionPriority,
    SessionState,
)

from .transport import (
    Transport,
    TransportConfig,
    StdioConfig,
    GrpcConfig,
    HttpConfig,
    StdioTransport,
    GrpcTransport,
    HttpTransport,
    TransportFactory,
)

# Import configuration components
from .config.models import (
    TransportType,
    CircuitBreakerConfig,
    RetryConfig,
    ObservabilityConfig,
    SessionConfig,
    StdioTransportConfig,
    GrpcTransportConfig,
    HttpTransportConfig,
    MCPConnectionConfig,
    ToolConfig,
)

from .config.manager import ConfigManager, ConfigurationError

__all__ = [
    # Error classes
    'McpError',
    'McpTransportError',
    'McpTimeoutError',
    'McpAuthenticationError',
    'McpRateLimitError',
    'McpInvalidRequestError',
    'McpServerError',

    # Session components
    'MCPSession',
    'EnhancedSessionPool',
    'Session',
    'SessionHealth',
    'SessionMetadata',
    'SessionPerformance',
    'SessionPoolConfig',
    'SessionPriority',
    'SessionState',

    # Transport components
    'Transport',
    'TransportConfig',
    'StdioConfig',
    'GrpcConfig',
    'HttpConfig',
    'StdioTransport',
    'GrpcTransport',
    'HttpTransport',
    'TransportFactory',

    # Configuration components
    'TransportType',
    'CircuitBreakerConfig',
    'RetryConfig',
    'ObservabilityConfig',
    'SessionConfig',
    'StdioTransportConfig',
    'GrpcTransportConfig',
    'HttpTransportConfig',
    'MCPConnectionConfig',
    'ToolConfig',
    'ConfigManager',
    'ConfigurationError',
]
