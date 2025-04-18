"""
Error handling for the OpenAI client.

This module provides functions for mapping OpenAI API errors to the Gateway's error model.
"""

import logging
from typing import Dict, Any, Optional
import traceback

from openai import (
    APIError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
)

from asf.medical.llm_gateway.core.models import (
    ErrorDetails,
    ErrorLevel,
    FinishReason,
    StreamChunk,
)

logger = logging.getLogger(__name__)

def map_error(
    error: Exception, 
    provider_id: str, 
    timeout: Optional[float] = None,
    retryable: Optional[bool] = None, 
    retry_after: Optional[int] = None,
) -> ErrorDetails:
    """
    Map various exceptions to the gateway's ErrorDetails model.
    
    Args:
        error: The exception to map
        provider_id: Provider ID for logging
        timeout: Timeout value in seconds
        retryable: Whether the error is retryable
        retry_after: Seconds to wait before retrying
        
    Returns:
        ErrorDetails: Mapped error details
    """
    code = "PROVIDER_ERROR"
    message = f"OpenAI provider '{provider_id}' encountered an error: {str(error)}"
    level = ErrorLevel.ERROR
    provider_details: Optional[Dict[str, Any]] = None
    is_retryable = retryable if retryable is not None else False  # Default to False

    if isinstance(error, APITimeoutError):
        code = "PROVIDER_TIMEOUT"
        message = f"OpenAI request timed out after {timeout or 'unknown'}s."
        is_retryable = True
    elif isinstance(error, RateLimitError):
        code = "PROVIDER_RATE_LIMIT"
        message = f"OpenAI rate limit exceeded. {getattr(error, 'message', '')}"
        is_retryable = True
        provider_details = error.body if hasattr(error, 'body') else None  # Contains details like type, code
    elif isinstance(error, AuthenticationError):
        code = "PROVIDER_AUTH_ERROR"
        message = "OpenAI authentication failed. Check API key or Azure credentials."
        is_retryable = False
    elif isinstance(error, PermissionDeniedError):
        code = "PROVIDER_PERMISSION_ERROR"
        message = "OpenAI permission denied. Check model access or organization settings."
        is_retryable = False
    elif isinstance(error, BadRequestError):  # Includes InvalidRequestError, ContentPolicyViolation
        code = f"PROVIDER_BAD_REQUEST_{getattr(error, 'code', 'UNKNOWN')}"
        message = f"OpenAI rejected the request as invalid ({getattr(error, 'code', 'UNKNOWN')}): {getattr(error, 'message', str(error))}"
        is_retryable = False
        provider_details = error.body if hasattr(error, 'body') else None
        if getattr(error, 'code', None) == 'content_filter':
            code = "PROVIDER_CONTENT_FILTER"
            level = ErrorLevel.WARNING  # Content filter isn't necessarily a system error
    elif isinstance(error, APIStatusError):
        code = f"PROVIDER_HTTP_{error.status_code}"
        message = f"OpenAI API returned status {error.status_code}: {error.message or (error.response.text if error.response else '')}"
        if error.status_code == 429:
            is_retryable = True
            code = "PROVIDER_RATE_LIMIT"
        elif error.status_code >= 500:
            is_retryable = True  # Server errors are potentially retryable
        provider_details = error.body if hasattr(error, 'body') else {"raw_response": error.response.text if error.response else None}
    elif isinstance(error, APIError):  # General API error
        code = "PROVIDER_API_ERROR"
        message = f"OpenAI API Error ({getattr(error, 'code', 'UNKNOWN')}): {str(error)}"
        provider_details = error.body if hasattr(error, 'body') else None
    elif isinstance(error, ConnectionError):  # Includes network errors from httpx
        code = "PROVIDER_CONNECTION_ERROR"
        message = f"Could not connect to OpenAI API: {str(error)}"
        is_retryable = True
    elif isinstance(error, ValueError):  # e.g., mapping errors
        code = "PROVIDER_MAPPING_ERROR"
        message = f"Data mapping error for OpenAI: {str(error)}"
        level = ErrorLevel.WARNING

    # Add body/details if not already captured
    if not provider_details and hasattr(error, 'body'):
        provider_details = getattr(error, 'body', None)
    if not provider_details:
        provider_details = {
            "exception_type": type(error).__name__,
            "details": str(error),
        }

    # Always ensure the 'message' field is included in ErrorDetails
    # This is required for the pydantic model validation
    return ErrorDetails(
        code=code,
        message=message,
        level=level,
        provider_error_details=provider_details,
        retryable=is_retryable,
        retry_after_seconds=retry_after
    )

def create_error_chunk(request_id: str, chunk_id: int, error_details: ErrorDetails) -> StreamChunk:
    """
    Create a StreamChunk representing an error.
    
    Args:
        request_id: The request ID
        chunk_id: The chunk ID
        error_details: Error details
        
    Returns:
        StreamChunk: An error chunk
    """
    return StreamChunk(
        chunk_id=chunk_id,
        request_id=request_id,
        finish_reason=FinishReason.ERROR,
        delta_text="Error: " + error_details.message,  # Include the error message in the delta text
        provider_specific_data={"error": error_details.model_dump()}
    )

def get_retry_after(error: Exception) -> Optional[int]:
    """
    Extract the retry-after header from an error response.
    
    Args:
        error: The exception to extract retry-after from
        
    Returns:
        Optional[int]: The retry-after value in seconds, or None if not found
    """
    # OpenAI SDK v1+ parses headers into the error object sometimes
    headers = getattr(error, 'response', None) and getattr(error.response, 'headers', None)
    if headers and 'retry-after' in headers:
        try:
            return int(headers['retry-after'])
        except (ValueError, TypeError):
            pass
    if headers and 'retry-after-ms' in headers:  # Some APIs use ms
        try:
            return int(headers['retry-after-ms']) // 1000
        except (ValueError, TypeError):
            pass
            
    # Check body for some specific rate limit errors (less common now)
    body = getattr(error, 'body', None)
    if isinstance(body, dict) and 'error' in body and isinstance(body['error'], dict):
        message = body['error'].get('message', '')
        import re
        match = re.search(r"retry after (\d+) seconds", message, re.IGNORECASE)
        if match:
            return int(match.group(1))
            
    return None