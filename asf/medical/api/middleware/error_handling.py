"""
Error handling middleware for the ASF Medical Research Synthesizer API.

This module provides middleware for handling exceptions in the API and converting
them to appropriate HTTP responses.
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exception_handlers import http_exception_handler
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...core.exceptions import (
    ASFError, NotFoundError, ValidationError, AuthenticationError,
    AuthorizationError, RateLimitError, InvalidRequestError, DatabaseError,
    BusinessLogicError, ExternalServiceError, ModelError, ConfigurationError,
    FileError, ExportError, SearchError, AnalysisError, KnowledgeBaseError,
    ContradictionError, TaskError, DuplicateError
)
from ...core.logging_config import get_logger

logger = get_logger(__name__)


async def asf_exception_handler(request: Request, exc: ASFError):
    logger.error(f"Error: {str(e)}")
    """
    Handle ASF-specific exceptions and convert to appropriate HTTP responses.

    Args:
        request: The FastAPI request
        exc: The ASF exception

    Returns:
        JSONResponse with appropriate status code and error details
    """
    # Map exception types to status codes
    status_code_map = {
        # 400 Bad Request
        ValidationError: status.HTTP_400_BAD_REQUEST,
        InvalidRequestError: status.HTTP_400_BAD_REQUEST,
        BusinessLogicError: status.HTTP_400_BAD_REQUEST,

        # 401 Unauthorized
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,

        # 403 Forbidden
        AuthorizationError: status.HTTP_403_FORBIDDEN,

        # 404 Not Found
        NotFoundError: status.HTTP_404_NOT_FOUND,

        # 409 Conflict
        DuplicateError: status.HTTP_409_CONFLICT,

        # 422 Unprocessable Entity
        ExportError: status.HTTP_422_UNPROCESSABLE_ENTITY,

        # 429 Too Many Requests
        RateLimitError: status.HTTP_429_TOO_MANY_REQUESTS,

        # 500 Internal Server Error
        DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ExternalServiceError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ModelError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        FileError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        SearchError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        AnalysisError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        KnowledgeBaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ContradictionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        TaskError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    # Get the appropriate status code or default to 500
    status_code = status_code_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Log the error with context
    log_method = logger.error if status_code >= 500 else logger.warning
    log_method(
        f"Exception handled: {type(exc).__name__}",
        extra={
            "exception_type": type(exc).__name__,
            "status_code": status_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details if hasattr(exc, "details") else {}
        },
        exc_info=exc if status_code >= 500 else False
    )

    # Add headers for certain status codes
    headers = {}
    if status_code == status.HTTP_401_UNAUTHORIZED:
        headers["WWW-Authenticate"] = "Bearer"

    # Create a structured error response
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "type": type(exc).__name__,
                "message": exc.message if hasattr(exc, "message") else str(exc),
                "details": exc.details if hasattr(exc, "details") else {},
                "status_code": status_code
            }
        },
        headers=headers
    )


async def http_exception_handler_with_logging(request: Request, exc: StarletteHTTPException):
    logger.error(f"Error: {str(e)}")
    """
    Handle HTTP exceptions with logging.

    Args:
        request: The FastAPI request
        exc: The HTTP exception

    Returns:
        Response from the default HTTP exception handler
    """
    # Log the error
    log_method = logger.error if exc.status_code >= 500 else logger.warning
    log_method(
        f"HTTP exception handled: {exc.status_code}",
        extra={
            "status_code": exc.status_code,
            "path": request.url.path,
            "method": request.method,
            "detail": exc.detail
        },
        exc_info=exc if exc.status_code >= 500 else False
    )

    # Use the default handler
    return await http_exception_handler(request, exc)


def setup_exception_handlers(app):
    logger.error(f"Error: {str(e)}")
    """
    Set up exception handlers for the FastAPI application.

    Args:
        app: The FastAPI application
    """
    # Register the ASF exception handler
    app.add_exception_handler(ASFError, asf_exception_handler)

    # Register the HTTP exception handler with logging
    app.add_exception_handler(StarletteHTTPException, http_exception_handler_with_logging)

    # Register handlers for specific exception types if needed
    # This allows for more customized handling of certain exceptions
    # app.add_exception_handler(ValidationError, validation_exception_handler)
