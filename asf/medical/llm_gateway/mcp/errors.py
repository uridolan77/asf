"""
MCP Errors

This module defines exceptions for the MCP implementation.
"""


class McpError(Exception):
    """Base exception for MCP-related errors."""
    pass


class McpTransportError(McpError):
    """Exception for transport-related errors."""
    pass


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
