"""
API dependencies for LLM Gateway.

This module provides dependencies for the FastAPI application, including
database session, current user, and encryption key.
"""

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
import logging
from typing import Dict, Any, Optional

# Import database session
from asf.medical.llm_gateway.db.session import get_db_session

# Set up logging
logger = logging.getLogger(__name__)

# Set up security
security = HTTPBearer()

def get_db() -> Session:
    """
    Get a database session.
    
    This dependency provides a database session for API endpoints.
    
    Returns:
        SQLAlchemy database session
    """
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()

def get_encryption_key() -> Optional[bytes]:
    """
    Get the encryption key for sensitive data.
    
    This dependency provides the encryption key for API endpoints.
    
    Returns:
        Encryption key as bytes or None if not configured
    """
    # Get encryption key from environment variable
    encryption_key = os.environ.get("LLM_GATEWAY_ENCRYPTION_KEY")
    if not encryption_key:
        logger.warning("Encryption key not configured. Sensitive data will not be encrypted.")
        return None
    
    # Convert to bytes
    return encryption_key.encode()

async def get_current_user(
    authorization: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get the current user from the authorization header.
    
    This dependency provides the current user for API endpoints.
    
    Args:
        authorization: HTTP authorization credentials
        db: Database session
        
    Returns:
        Current user information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Get token from authorization header
        token = authorization.credentials
        
        # Validate token
        # This is a placeholder for actual token validation
        # In a real application, you would validate the token against a database or auth service
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from token
        # This is a placeholder for actual user retrieval
        # In a real application, you would get the user from a database or auth service
        user = {
            "id": 1,
            "username": "admin",
            "email": "admin@example.com",
            "roles": ["admin"]
        }
        
        return user
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_active_user(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get the current active user.
    
    This dependency provides the current active user for API endpoints.
    
    Args:
        current_user: Current user information
        
    Returns:
        Current active user information
        
    Raises:
        HTTPException: If the user is inactive
    """
    # Check if user is active
    # This is a placeholder for actual user status check
    # In a real application, you would check the user's status in a database
    if current_user.get("disabled"):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
