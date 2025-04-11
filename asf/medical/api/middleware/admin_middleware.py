"""
Admin Middleware for the Medical Research Synthesizer.

This module provides a middleware for protecting admin-only routes.
"""

import logging
from typing import Callable, List
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Configure logging
logger = logging.getLogger(__name__)

class AdminRouteMiddleware(BaseHTTPMiddleware):
    """
    Middleware for protecting admin-only routes.
    
    This middleware ensures that only users with admin role can access
    routes with the "admin" tag or routes that match admin path patterns.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        admin_path_patterns: List[str] = None,
        admin_tag: str = "admin"
    ):
        """
        Initialize the admin route middleware.
        
        Args:
            app: ASGI application
            admin_path_patterns: List of path patterns that should be admin-only
            admin_tag: Tag that identifies admin-only routes
        """
        super().__init__(app)
        self.admin_path_patterns = admin_path_patterns or [
            "/v1/admin/",
            "/cache/",
            "/metrics",
            "/v1/model-cache/",
            "/v1/task-management/"
        ]
        self.admin_tag = admin_tag
        logger.info(f"Admin route middleware initialized with patterns: {self.admin_path_patterns}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch the request.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            Response: FastAPI response
        """
        # Check if the path matches any admin path pattern
        path = request.url.path
        is_admin_path = any(path.startswith(pattern) for pattern in self.admin_path_patterns)
        
        # If it's an admin path, check if the user is an admin
        if is_admin_path:
            # Get the user from the request state (set by authentication middleware)
            user = getattr(request.state, "user", None)
            
            # If no user or user is not an admin, return 403 Forbidden
            if not user or getattr(user, "role", None) != "admin":
                logger.warning(f"Unauthorized access attempt to admin route: {path}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin role required"
                )
        
        # Process the request
        return await call_next(request)

def add_admin_middleware(
    app: ASGIApp,
    admin_path_patterns: List[str] = None,
    admin_tag: str = "admin"
):
    """
    Add admin route middleware to an application.
    
    Args:
        app: ASGI application
        admin_path_patterns: List of path patterns that should be admin-only
        admin_tag: Tag that identifies admin-only routes
    """
    app.add_middleware(
        AdminRouteMiddleware,
        admin_path_patterns=admin_path_patterns,
        admin_tag=admin_tag
    )
    logger.info("Added admin route middleware")
