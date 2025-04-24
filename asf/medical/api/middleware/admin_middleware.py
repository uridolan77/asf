"""Admin Middleware for the Medical Research Synthesizer.

This module provides a middleware for protecting admin-only routes.
"""

import logging
from typing import Callable, List
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from asf.medical.storage.models.user import Role

logger = logging.getLogger(__name__)

class AdminRouteMiddleware(BaseHTTPMiddleware):
    """Middleware for protecting admin-only routes.

    This middleware ensures that only users with admin role can access
    routes with the "admin" tag or routes that match admin path patterns.
    """

    def __init__(
        self,
        app: ASGIApp,
        admin_path_patterns: List[str] = None,
        admin_tag: str = "admin"
    ):
        """Initialize the admin route middleware.

        Args:
            app: The ASGI application
            admin_path_patterns: List of URL path patterns that require admin access
            admin_tag: Tag used to identify admin-only routes
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
        """Process a request and enforce admin access for protected routes.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler

        Raises:
            HTTPException: If the user doesn't have admin role for an admin route
        """
        path = request.url.path
        is_admin_path = any(path.startswith(pattern) for pattern in self.admin_path_patterns)

        if is_admin_path:
            user = getattr(request.state, "user", None)

            if not user or getattr(user, "role", None) != Role.ADMIN:
                logger.warning(f"Unauthorized access attempt to admin route: {path}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin role required"
                )

        return await call_next(request)

def add_admin_middleware(
    app: ASGIApp,
    admin_path_patterns: List[str] = None,
    admin_tag: str = "admin"
):
    """Add the admin route middleware to a FastAPI application.

    Args:
        app: The FastAPI application
        admin_path_patterns: List of URL path patterns that require admin access
        admin_tag: Tag used to identify admin-only routes
    """
    app.add_middleware(
        AdminRouteMiddleware,
        admin_path_patterns=admin_path_patterns,
        admin_tag=admin_tag
    )
    logger.info("Added admin route middleware")
