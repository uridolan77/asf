"""
Middleware for the Medical Research Synthesizer API.

This module provides middleware for the API.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from asf.medical.core.monitoring import log_request, log_error

# Configure logging
logger = logging.getLogger(__name__)

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API requests."""
    
    def __init__(self, app: ASGIApp):
        """
        Initialize the middleware.
        
        Args:
            app: ASGI app
        """
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process a request.
        
        Args:
            request: Request
            call_next: Function to call next middleware
            
        Returns:
            Response
        """
        # Start timer
        start_time = time.time()
        
        # Get request details
        method = request.method
        path = request.url.path
        
        # Get user ID from request if available
        user_id = None
        if hasattr(request.state, "user") and hasattr(request.state.user, "id"):
            user_id = request.state.user.id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log request
            log_request(method, path, response.status_code, duration, user_id)
            
            return response
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            log_error(e, {
                "method": method,
                "path": path,
                "duration": duration,
                "user_id": user_id
            })
            
            # Re-raise exception
            raise
