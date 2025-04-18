"""
WebSocket authentication utilities for MCP provider.

This module provides enhanced authentication mechanisms for WebSocket connections,
including token validation, refresh handling, and role-based access control.
"""

import asyncio
import jwt
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Tuple, List

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, status
from starlette.websockets import WebSocketState

from api.auth.dependencies import get_current_user_ws
from models.user import User
from api.services.auth_service import get_auth_service

logger = logging.getLogger(__name__)

# Store active tokens with their expiration times
_token_cache: Dict[str, Tuple[datetime, User]] = {}
_token_cache_lock = asyncio.Lock()

# Token refresh settings
TOKEN_REFRESH_MARGIN_SECONDS = 300  # Refresh token if less than 5 minutes left
TOKEN_CACHE_CLEANUP_INTERVAL = 3600  # Clean up token cache every hour

# Role-based access control
REQUIRED_ROLES_FOR_MCP = ["admin", "developer", "operator"]


async def authenticate_ws_connection(websocket: WebSocket) -> Optional[User]:
    """
    Authenticate a WebSocket connection with enhanced token handling.

    Args:
        websocket: The WebSocket connection

    Returns:
        User object if authentication successful, None otherwise
    """
    try:
        # Get token from query parameters or headers
        token = websocket.query_params.get("token")
        if not token:
            # Try to get token from headers
            auth_header = websocket.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

        if not token:
            logger.warning("No token provided in WebSocket connection")
            await websocket.close(code=1008)  # Policy violation
            return None

        # Check token cache first
        user = await _get_cached_user(token)
        if user:
            # Check if token needs refresh
            if await _should_refresh_token(token):
                # Send refresh message to client
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "type": "token_refresh_required",
                        "message": "Authentication token is about to expire"
                    })
            return user

        # If not in cache, validate with auth service
        user = await get_current_user_ws(websocket)
        if user:
            # Cache the token
            await _cache_token(token, user)
            return user

        return None

    except Exception as e:
        logger.error(f"Error in WebSocket authentication: {str(e)}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011)  # Internal error
        return None


async def check_mcp_access(user: User) -> bool:
    """
    Check if user has access to MCP functionality.

    Args:
        user: User to check

    Returns:
        True if user has access, False otherwise
    """
    if not user:
        return False

    # Get auth service
    auth_service = get_auth_service()

    # Check user role
    user_role = await auth_service.get_user_role(user.id)
    if not user_role:
        return False

    # Check if role has access to MCP
    return user_role.name.lower() in REQUIRED_ROLES_FOR_MCP


async def _get_cached_user(token: str) -> Optional[User]:
    """
    Get user from token cache if token is valid.

    Args:
        token: JWT token

    Returns:
        User object if token is valid and in cache, None otherwise
    """
    async with _token_cache_lock:
        if token in _token_cache:
            expiry, user = _token_cache[token]
            if datetime.utcnow() < expiry:
                return user
            else:
                # Token expired, remove from cache
                del _token_cache[token]
    return None


async def _cache_token(token: str, user: User) -> None:
    """
    Cache a token with its expiration time.

    Args:
        token: JWT token
        user: User object
    """
    try:
        # Decode token to get expiration time
        payload = jwt.decode(token, options={"verify_signature": False})
        if "exp" in payload:
            expiry = datetime.fromtimestamp(payload["exp"])
            async with _token_cache_lock:
                _token_cache[token] = (expiry, user)
    except Exception as e:
        logger.error(f"Error caching token: {str(e)}")


async def _should_refresh_token(token: str) -> bool:
    """
    Check if token should be refreshed.

    Args:
        token: JWT token

    Returns:
        True if token should be refreshed, False otherwise
    """
    async with _token_cache_lock:
        if token in _token_cache:
            expiry, _ = _token_cache[token]
            refresh_threshold = datetime.utcnow() + timedelta(seconds=TOKEN_REFRESH_MARGIN_SECONDS)
            return expiry < refresh_threshold
    return False


async def cleanup_token_cache() -> None:
    """
    Periodically clean up expired tokens from the cache.
    """
    while True:
        try:
            await asyncio.sleep(TOKEN_CACHE_CLEANUP_INTERVAL)

            now = datetime.utcnow()
            expired_tokens = []

            async with _token_cache_lock:
                for token, (expiry, _) in _token_cache.items():
                    if now > expiry:
                        expired_tokens.append(token)

                for token in expired_tokens:
                    del _token_cache[token]

            logger.debug(f"Cleaned up {len(expired_tokens)} expired tokens from cache")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in token cache cleanup: {str(e)}")
            await asyncio.sleep(60)  # Wait a bit before retrying


# Start token cache cleanup task
cleanup_task = None

def start_cleanup_task():
    """Start the token cache cleanup task."""
    global cleanup_task
    if cleanup_task is None or cleanup_task.done():
        cleanup_task = asyncio.create_task(cleanup_token_cache())
        logger.info("Started token cache cleanup task")

def stop_cleanup_task():
    """Stop the token cache cleanup task."""
    global cleanup_task
    if cleanup_task and not cleanup_task.done():
        cleanup_task.cancel()
        logger.info("Stopped token cache cleanup task")
