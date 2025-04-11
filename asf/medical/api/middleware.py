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