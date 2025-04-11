"""
CSRF Protection Middleware for the Medical Research Synthesizer.

This module provides a middleware for protecting against CSRF attacks.
"""

import secrets
import logging
import time
from typing import Dict, Optional, Any, Callable, List
from fastapi import Request, Response, FastAPI, Cookie, Header
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Configure logging
logger = logging.getLogger(__name__)

class CSRFMiddleware(BaseHTTPMiddleware):
    """
    Middleware for protecting against CSRF attacks.
    
    This middleware implements Double Submit Cookie pattern for CSRF protection.
    It sets a CSRF token in a cookie and requires the same token to be sent in
    the X-CSRF-Token header for all non-GET/HEAD requests.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        cookie_name: str = "csrf_token",
        header_name: str = "X-CSRF-Token",
        cookie_secure: bool = True,
        cookie_httponly: bool = False,  # Must be False to be accessible by JS
        cookie_samesite: str = "lax",
        cookie_max_age: int = 86400,  # 24 hours
        exempt_paths: List[str] = None
    ):
        """
        Initialize the CSRF middleware.
        
        Args:
            app: ASGI application
            cookie_name: Name of the CSRF cookie
            header_name: Name of the CSRF header
            cookie_secure: Whether the cookie should be secure (HTTPS only)
            cookie_httponly: Whether the cookie should be HTTP only
            cookie_samesite: SameSite attribute for the cookie
            cookie_max_age: Max age of the cookie in seconds
            exempt_paths: List of paths exempt from CSRF protection
        """
        super().__init__(app)
        self.cookie_name = cookie_name
        self.header_name = header_name
        self.cookie_secure = cookie_secure
        self.cookie_httponly = cookie_httponly
        self.cookie_samesite = cookie_samesite
        self.cookie_max_age = cookie_max_age
        self.exempt_paths = exempt_paths or ["/docs", "/redoc", "/openapi.json", "/health"]
        logger.info(f"CSRF middleware initialized with cookie_name={cookie_name}, header_name={header_name}")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch the request.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            Response: FastAPI response
        """
        # Skip CSRF protection for exempt paths
        path = request.url.path
        if any(path.startswith(exempt_path) for exempt_path in self.exempt_paths):
            return await call_next(request)
        
        # Skip CSRF protection for safe methods (GET, HEAD, OPTIONS)
        if request.method.upper() in ["GET", "HEAD", "OPTIONS"]:
            response = await call_next(request)
            
            # Set CSRF token cookie if it doesn't exist
            csrf_token = request.cookies.get(self.cookie_name)
            if not csrf_token:
                csrf_token = secrets.token_urlsafe(32)
                response.set_cookie(
                    key=self.cookie_name,
                    value=csrf_token,
                    max_age=self.cookie_max_age,
                    httponly=self.cookie_httponly,
                    secure=self.cookie_secure,
                    samesite=self.cookie_samesite
                )
            
            return response
        
        # For unsafe methods (POST, PUT, DELETE, etc.), verify CSRF token
        csrf_cookie = request.cookies.get(self.cookie_name)
        csrf_header = request.headers.get(self.header_name)
        
        # If no CSRF token in cookie or header, return 403 Forbidden
        if not csrf_cookie or not csrf_header:
            logger.warning(f"CSRF token missing: cookie={bool(csrf_cookie)}, header={bool(csrf_header)}")
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token missing"}
            )
        
        # If CSRF tokens don't match, return 403 Forbidden
        if csrf_cookie != csrf_header:
            logger.warning(f"CSRF token mismatch: cookie={csrf_cookie[:8]}..., header={csrf_header[:8]}...")
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token mismatch"}
            )
        
        # Process the request
        response = await call_next(request)
        
        # Refresh CSRF token cookie
        response.set_cookie(
            key=self.cookie_name,
            value=csrf_cookie,
            max_age=self.cookie_max_age,
            httponly=self.cookie_httponly,
            secure=self.cookie_secure,
            samesite=self.cookie_samesite
        )
        
        return response

def add_csrf_middleware(
    app: FastAPI,
    cookie_name: str = "csrf_token",
    header_name: str = "X-CSRF-Token",
    cookie_secure: bool = True,
    cookie_httponly: bool = False,
    cookie_samesite: str = "lax",
    cookie_max_age: int = 86400,
    exempt_paths: List[str] = None
):
    """
    Add CSRF middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        cookie_name: Name of the CSRF cookie
        header_name: Name of the CSRF header
        cookie_secure: Whether the cookie should be secure (HTTPS only)
        cookie_httponly: Whether the cookie should be HTTP only
        cookie_samesite: SameSite attribute for the cookie
        cookie_max_age: Max age of the cookie in seconds
        exempt_paths: List of paths exempt from CSRF protection
    """
    app.add_middleware(
        CSRFMiddleware,
        cookie_name=cookie_name,
        header_name=header_name,
        cookie_secure=cookie_secure,
        cookie_httponly=cookie_httponly,
        cookie_samesite=cookie_samesite,
        cookie_max_age=cookie_max_age,
        exempt_paths=exempt_paths
    )
    
    logger.info(f"Added CSRF middleware with cookie_name={cookie_name}, header_name={header_name}")
