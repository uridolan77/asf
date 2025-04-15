"""
Middleware for the Medical Research Synthesizer API.

This module provides middleware components for the FastAPI application,
including request monitoring, logging, and performance tracking.
"""

import time
import logging
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send

from ..core.observability import log_request, increment_counter
from ..core.exceptions import APIError


logger = logging.getLogger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring API requests.
    
    This middleware logs API requests, tracks performance metrics,
    and adds request IDs for distributed tracing.
    """
    
    def __init__(self, app: ASGIApp):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI application
        """
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        """
        Process a request.
        
        Args:
            request: HTTP request
            call_next: Function to call the next middleware
            
        Returns:
            Response: HTTP response
        """
        # Generate a unique request ID if not provided
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Record request start time
        start_time = time.time()
        
        # Set up request context
        context = {
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client": request.client.host if request.client else "unknown"
        }
        
        # Log the request
        logger.info(
            f"Request started: {request.method} {request.url.path}",
            extra=context
        )
        
        # Count the request
        increment_counter(
            "http_requests_total", 
            tags={
                "method": request.method,
                "path": request.url.path
            }
        )
        
        try:
            # Call the next middleware and get the response
            response = await call_next(request)
            
            # Add the request ID to the response headers
            response.headers["X-Request-ID"] = request_id
            
            # Calculate the request duration
            duration = time.time() - start_time
            
            # Log request completion
            log_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration * 1000,
                request_id=request_id
            )
            
            logger.info(
                f"Request completed: {request.method} {request.url.path} {response.status_code} ({duration:.3f}s)",
                extra={**context, "status_code": response.status_code, "duration": duration}
            )
            
            return response
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            logger.error(f"API error: {str(e)}")
            raise APIError(f"API call failed: {str(e)}")
            # Calculate the request duration
            duration = time.time() - start_time
            
            # Log the error
            logger.error(
                f"Request error: {request.method} {request.url.path} ({duration:.3f}s) - {str(e)}",
                exc_info=e,
                extra={**context, "error": str(e), "duration": duration}
            )
            
            # Re-raise the exception
            raise