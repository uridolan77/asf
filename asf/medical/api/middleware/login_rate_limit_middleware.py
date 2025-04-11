"""
Login Rate Limit Middleware for the Medical Research Synthesizer.

This module provides a middleware for rate limiting login attempts.
"""

import time
import logging
from typing import Dict, Optional, Any, Callable, List
from fastapi import Request, Response, FastAPI
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from asf.medical.core.enhanced_rate_limiter import enhanced_rate_limiter

# Configure logging
logger = logging.getLogger(__name__)

class LoginRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting login attempts.
    
    This middleware limits the rate of login attempts based on the client's
    IP address to prevent brute force attacks.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        login_path: str = "/v1/auth/token",
        rate: int = 5,  # 5 attempts per minute
        burst: int = 3,  # 3 attempts in a burst
        window: int = 60,  # 1 minute window
        block_time: int = 300  # 5 minutes block time after too many attempts
    ):
        """
        Initialize the login rate limit middleware.
        
        Args:
            app: ASGI application
            login_path: Path of the login endpoint
            rate: Rate limit in attempts per window
            burst: Burst limit in attempts
            window: Window size in seconds
            block_time: Block time in seconds after too many attempts
        """
        super().__init__(app)
        self.login_path = login_path
        self.rate = rate
        self.burst = burst
        self.window = window
        self.block_time = block_time
        logger.info(f"Login rate limit middleware initialized with rate={rate}, burst={burst}, window={window}s, block_time={block_time}s")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch the request.
        
        Args:
            request: FastAPI request
            call_next: Function to call the next middleware
            
        Returns:
            Response: FastAPI response
        """
        # Only rate limit login attempts
        if request.url.path != self.login_path or request.method != "POST":
            return await call_next(request)
        
        # Get client IP address
        client_ip = self._get_client_ip(request)
        
        # Check if client is blocked
        is_blocked, block_info = await enhanced_rate_limiter.is_rate_limited(
            key=f"login_block:{client_ip}",
            rate=1,
            burst=1,
            window=self.block_time
        )
        
        if is_blocked:
            logger.warning(f"Blocked login attempt from {client_ip} - too many failed attempts")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many login attempts, please try again later",
                    "reset": block_info["reset"],
                    "retry_after": block_info["reset"] - int(time.time())
                },
                headers={
                    "Retry-After": str(block_info["reset"] - int(time.time()))
                }
            )
        
        # Check rate limit for login attempts
        is_limited, limit_info = await enhanced_rate_limiter.is_rate_limited(
            key=f"login:{client_ip}",
            rate=self.rate,
            burst=self.burst,
            window=self.window
        )
        
        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(limit_info["limit"]),
            "X-RateLimit-Remaining": str(limit_info["remaining"]),
            "X-RateLimit-Reset": str(limit_info["reset"])
        }
        
        # Return 429 Too Many Requests if rate limited
        if is_limited:
            logger.warning(f"Rate limited login attempt from {client_ip}")
            
            # Block client if they hit the rate limit
            await enhanced_rate_limiter.set_rate_limit(
                key=f"login_block:{client_ip}",
                rate=1,
                burst=1,
                window=self.block_time
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many login attempts, please try again later",
                    "limit": limit_info["limit"],
                    "remaining": limit_info["remaining"],
                    "reset": limit_info["reset"],
                    "retry_after": limit_info["reset"] - int(time.time())
                },
                headers={
                    **headers,
                    "Retry-After": str(limit_info["reset"] - int(time.time()))
                }
            )
        
        # Process the request
        response = await call_next(request)
        
        # Check if login failed (status code 401)
        if response.status_code == 401:
            # Increment failed login counter
            await enhanced_rate_limiter.increment_counter(
                key=f"login_failed:{client_ip}",
                window=self.window
            )
            
            # Check if too many failed attempts
            failed_count = await enhanced_rate_limiter.get_counter(
                key=f"login_failed:{client_ip}"
            )
            
            if failed_count >= self.burst:
                # Block client for block_time
                await enhanced_rate_limiter.set_rate_limit(
                    key=f"login_block:{client_ip}",
                    rate=1,
                    burst=1,
                    window=self.block_time
                )
                
                logger.warning(f"Blocked {client_ip} for {self.block_time}s after {failed_count} failed login attempts")
        
        # Add rate limit headers to the response
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
        # Try to get the real IP from X-Forwarded-For header
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the list
            return forwarded_for.split(",")[0].strip()
        
        # Fall back to the client's host
        return request.client.host

def add_login_rate_limit_middleware(
    app: FastAPI,
    login_path: str = "/v1/auth/token",
    rate: int = 5,
    burst: int = 3,
    window: int = 60,
    block_time: int = 300
):
    """
    Add login rate limit middleware to a FastAPI application.
    
    Args:
        app: FastAPI application
        login_path: Path of the login endpoint
        rate: Rate limit in attempts per minute
        burst: Burst limit in attempts
        window: Window size in seconds
        block_time: Block time in seconds after too many attempts
    """
    app.add_middleware(
        LoginRateLimitMiddleware,
        login_path=login_path,
        rate=rate,
        burst=burst,
        window=window,
        block_time=block_time
    )
    
    logger.info(f"Added login rate limit middleware with rate={rate}, burst={burst}, window={window}s, block_time={block_time}s")
