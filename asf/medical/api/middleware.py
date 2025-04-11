"""
Middleware for the Medical Research Synthesizer API.
This module provides middleware for the API.
"""
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
logger = logging.getLogger(__name__)
class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API requests."""
    def __init__(self, app: ASGIApp):
        """
        Initialize the middleware.
        Args:
            app: ASGI app
        Process a request.
        Args:
            request: Request
            call_next: Function to call next middleware
        Returns:
            Response