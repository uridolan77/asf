"""
Authentication dependencies for the BO backend.
"""

import jwt
from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import OAuth2PasswordBearer
import logging
import os

from asf.bollm.backend.models.user import User
from config.config import SessionLocal

logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv('BO_SECRET_KEY', 'your-secret-key')
ALGORITHM = 'HS256'

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

def get_db():
    """
    Get database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(oauth2_scheme), db = Depends(get_db)):
    """
    Get current user from token.

    Args:
        token: JWT token
        db: Database session

    Returns:
        User: Current user

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    # In a real app, this would query the database
    # For now, we'll just return a mock user based on the user_id
    if user_id == "1":
        user = User(
            id=1,
            username="admin",
            email="admin@example.com",
            role_id=1,
            password_hash="hashed_password"  # Add required field
        )
        # Store token as an attribute instead of a constructor parameter
        user.token = token
        return user
    elif user_id == "2":
        user = User(
            id=2,
            username="user",
            email="user@example.com",
            role_id=2,
            password_hash="hashed_password"  # Add required field
        )
        # Store token as an attribute instead of a constructor parameter
        user.token = token
        return user

    raise credentials_exception

async def get_current_user_ws(websocket: WebSocket):
    """
    Get current user from WebSocket connection.

    Args:
        websocket: WebSocket connection

    Returns:
        User: Current user

    Raises:
        WebSocketDisconnect: If token is invalid or user not found
    """
    try:
        # Get token from query parameters
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

        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            logger.warning("Invalid token in WebSocket connection")
            await websocket.close(code=1008)  # Policy violation
            return None

        # In a real app, this would query the database
        # For now, we'll just return a mock user based on the user_id
        if user_id == "1":
            user = User(
                id=1,
                username="admin",
                email="admin@example.com",
                role_id=1,
                password_hash="hashed_password"  # Add required field
            )
            # Store token as an attribute instead of a constructor parameter
            user.token = token
            return user
        elif user_id == "2":
            user = User(
                id=2,
                username="user",
                email="user@example.com",
                role_id=2,
                password_hash="hashed_password"  # Add required field
            )
            # Store token as an attribute instead of a constructor parameter
            user.token = token
            return user

        logger.warning(f"User not found for ID: {user_id}")
        await websocket.close(code=1008)  # Policy violation
        return None

    except jwt.PyJWTError as e:
        logger.warning(f"JWT error in WebSocket connection: {str(e)}")
        await websocket.close(code=1008)  # Policy violation
        return None
    except Exception as e:
        logger.error(f"Error in WebSocket authentication: {str(e)}")
        await websocket.close(code=1011)  # Internal error
        return None
