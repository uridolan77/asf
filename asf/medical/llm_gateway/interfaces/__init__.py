"""
Interfaces for the LLM Gateway service abstraction layer.

This package contains the interfaces that define the contracts
for the LLM Gateway service implementations.
"""

from asf.medical.llm_gateway.interfaces.llm_service import LLMServiceInterface
from asf.medical.llm_gateway.interfaces.exceptions import (
    LLMServiceException,
    ModelNotAvailableException,
    ServiceUnavailableException,
    AuthenticationException,
    InvalidRequestException,
    RateLimitException,
    ContextLengthException,
    ContentFilterException,
    TransportException,
    TimeoutException
)

__all__ = [
    'LLMServiceInterface',
    'LLMServiceException',
    'ModelNotAvailableException',
    'ServiceUnavailableException',
    'AuthenticationException',
    'InvalidRequestException',
    'RateLimitException',
    'ContextLengthException',
    'ContentFilterException',
    'TransportException',
    'TimeoutException'
]