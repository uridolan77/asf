"""
Exceptions for the LLM Gateway service abstraction layer.

This module defines the exceptions that can be raised by the LLM services.
"""

class LLMServiceException(Exception):
    """Base exception for all LLM service errors."""
    pass


class ModelNotAvailableException(LLMServiceException):
    """Raised when a requested model is not available."""
    pass


class ServiceUnavailableException(LLMServiceException):
    """Raised when a service is temporarily unavailable."""
    pass


class AuthenticationException(LLMServiceException):
    """Raised when authentication with the service fails."""
    pass


class InvalidRequestException(LLMServiceException):
    """Raised when the request is invalid."""
    pass


class RateLimitException(LLMServiceException):
    """Raised when rate limits are exceeded."""
    pass


class ContextLengthException(LLMServiceException):
    """Raised when the context length exceeds the model's capacity."""
    pass


class ContentFilterException(LLMServiceException):
    """Raised when content is filtered due to safety concerns."""
    pass


class TransportException(LLMServiceException):
    """Raised when there's an issue with the transport layer."""
    pass


class TimeoutException(LLMServiceException):
    """Raised when a request times out."""
    pass