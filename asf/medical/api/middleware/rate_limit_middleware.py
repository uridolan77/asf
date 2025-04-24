"""Rate Limit Middleware for the Medical Research Synthesizer.

This module provides a middleware for rate limiting API requests.
"""

import logging
from typing import Optional, List, Callable
from fastapi import Request, Response, FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...core.rate_limiter import enhanced_rate_limiter

logger = logging.getLogger(__name__)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting API requests.

    This middleware limits the rate of API requests based on the client's
    IP address or user ID.
    """

    def __init__(
        self,
        app: ASGIApp,
        default_rate: int = 60,  # 60 requests per minute
        default_burst: int = 10,  # 10 requests in a burst
        default_window: int = 60,  # 1 minute window
        exempt_paths: Optional[List[str]] = None,
        get_key: Optional[Callable[[Request], str]] = None
    ):
        """Initialize the rate limit middleware.

        Args:
            app: The ASGI application
            default_rate: Maximum number of requests per minute
            default_burst: Maximum number of requests in a burst
            default_window: Time window in seconds for rate limiting
            exempt_paths: List of URL paths exempt from rate limiting
            get_key: Function to get the rate limit key from the request
        """
        super().__init__(app)
        self.default_rate = default_rate
        self.default_burst = default_burst
        self.default_window = default_window
        self.exempt_paths = exempt_paths or ["/docs", "/redoc", "/openapi.json", "/health"]
        self.get_key = get_key or self._get_client_ip

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process a request and apply rate limiting.

        Args:
            request: The incoming request
            call_next: The next middleware or route handler

        Returns:
            The response from the next middleware or route handler or a rate limit error
        """
        path = request.url.path
        if any(path.startswith(exempt_path) for exempt_path in self.exempt_paths):
            return await call_next(request)

        key = self.get_key(request)

        is_limited, limit_info = await enhanced_rate_limiter.is_rate_limited(
            key=key,
            rate=self.default_rate,
            burst=self.default_burst,
            window=self.default_window
        )

        headers = {
            "X-RateLimit-Limit": str(limit_info["limit"]),
            "X-RateLimit-Remaining": str(limit_info["remaining"]),
            "X-RateLimit-Reset": str(limit_info["reset"])
        }

        if is_limited:
            logger.warning(f"Rate limited: {key} ({request.client.host}) - {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests",
                    "limit": limit_info["limit"],
                    "remaining": limit_info["remaining"],
                    "reset": limit_info["reset"],
                    "window": limit_info["window"]
                },
                headers=headers
            )

        response = await call_next(request)

        for header_name, header_value in headers.items():
            response.headers[header_name] = header_value

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Get the client IP address from the request.

        Args:
            request: FastAPI request

        Returns:
            str: Client IP address
        """
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        return request.client.host

def add_rate_limit_middleware(
    app: FastAPI,
    default_rate: int = 60,
    default_burst: int = 10,
    default_window: int = 60,
    exempt_paths: Optional[List[str]] = None,
    get_key: Optional[Callable[[Request], str]] = None
):
    """Add the rate limit middleware to a FastAPI application.

    Args:
        app: The FastAPI application
        default_rate: Maximum number of requests per minute
        default_burst: Maximum number of requests in a burst
        default_window: Time window in seconds for rate limiting
        exempt_paths: List of URL paths exempt from rate limiting
        get_key: Function to get the rate limit key from the request
    """
    app.add_middleware(
        RateLimitMiddleware,
        default_rate=default_rate,
        default_burst=default_burst,
        default_window=default_window,
        exempt_paths=exempt_paths,
        get_key=get_key
    )

    logger.info(f"Added rate limit middleware with rate={default_rate}, burst={default_burst}, window={default_window}s")


def get_key_from_user_id(request: Request) -> str:
    """
    Get the rate limit key from the user ID.

    Args:
        request: FastAPI request

    Returns:
        str: Rate limit key
    """
    user_id = getattr(request.state, "user_id", None)

    if user_id:
        return f"user:{user_id}"

    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"

    return f"ip:{request.client.host}"

def get_key_from_api_key(request: Request) -> str:
    """
    Get the rate limit key from the API key.

    Args:
        request: FastAPI request

    Returns:
        str: Rate limit key
    """
    api_key = request.headers.get("X-API-Key")

    if api_key:
        return f"api_key:{api_key}"

    return get_key_from_user_id(request)
