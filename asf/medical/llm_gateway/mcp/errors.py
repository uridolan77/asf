"""
MCP Errors

This module defines exceptions for the MCP implementation.
It integrates with the consolidated error handling framework.
"""

from asf.medical.llm_gateway.transport.base import TransportError


class McpError(Exception):
    """Base exception for MCP-related errors."""
    pass


class McpTransportError(McpError):
    """
    Exception for transport-related errors.
    
    This wraps TransportError from the consolidated transport layer
    to provide MCP-specific context.
    """
    def __init__(self, message, original_error=None):
        super().__init__(message)
        self.original_error = original_error
    
    @classmethod
    def from_transport_error(cls, error: TransportError):
        """Create an MCP transport error from a base TransportError."""
        return cls(
            message=f"MCP transport error: {error.message}",
            original_error=error
        )


class McpTimeoutError(McpError):
    """Exception for timeout errors."""
    pass


class McpAuthenticationError(McpError):
    """Exception for authentication errors."""
    pass


class McpRateLimitError(McpError):
    """Exception for rate limit errors."""
    pass


class McpInvalidRequestError(McpError):
    """Exception for invalid request errors."""
    pass


class McpServerError(McpError):
    """Exception for server errors."""
    pass
