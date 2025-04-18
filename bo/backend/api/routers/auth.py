"""
Authentication router for the BO backend.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
import logging

from ..auth.dependencies import get_db, get_current_user, SECRET_KEY, ALGORITHM
from models.user import User

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["auth"])

# Function to check if user is admin
def get_current_admin_user(user: User = Depends(get_current_user)):
    """
    Get current admin user.

    Args:
        user: Current user

    Returns:
        User: Current admin user

    Raises:
        HTTPException: If user is not an admin
    """
    # Check if user is admin (role_id = 1)
    if user.role_id != 1:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions")

    return user

# Access token expiration time
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def create_access_token(data: dict, expires_delta: timedelta = None):
    """
    Create a JWT access token.

    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time

    Returns:
        str: JWT token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """
    Login endpoint.

    Args:
        form_data: OAuth2 password request form
        db: Database session

    Returns:
        dict: Access token and token type
    """
    logger.info(f"Login attempt with username: {form_data.username}")

    # For development, accept any credentials
    # Create a mock user based on the provided username
    user = {
        "id": 1,
        "username": form_data.username.split('@')[0] if '@' in form_data.username else form_data.username,
        "email": form_data.username,
        "role_id": 1
    }

    access_token = create_access_token(data={"sub": str(user["id"])})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me")
async def read_users_me(user: User = Depends(get_current_user)):
    """
    Get current user.

    Args:
        user: Current user

    Returns:
        dict: User information
    """
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role_id": user.role_id
    }
