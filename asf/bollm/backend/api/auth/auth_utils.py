"""
Authentication utility functions.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import jwt
from typing import Optional

from asf.bollm.backend.models.user import User
from config.config import SessionLocal

# Constants
SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Function to authenticate user by email or username
async def authenticate_user(login_identifier: str, password: str, db: Session):
    """
    Authenticate a user by email or username.
    
    Args:
        login_identifier: Either an email address or username
        password: The password to verify
        db: Database session
        
    Returns:
        User object if authentication is successful, None otherwise
    """
    # Check if login_identifier is an email (contains @) or username
    if '@' in login_identifier:
        # It's likely an email
        user = db.query(User).filter(User.email == login_identifier).first()
    else:
        # It's a username
        user = db.query(User).filter(User.username == login_identifier).first()
    
    if not user:
        return None
        
    # Verify password - replace with your actual password verification method
    if not verify_password(password, user.password_hash):
        return None
        
    return user

def verify_password(plain_password: str, hashed_password: str):
    """
    Verify that a plain password matches a hashed password.
    This is a placeholder - replace with your actual password verification logic.
    """
    # This is a placeholder - in a real app use a secure password library like:
    # from passlib.context import CryptContext
    # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    # return pwd_context.verify(plain_password, hashed_password)
    
    # For demo purposes:
    # TODO: Replace this with proper password verification
    return True  # Always return true for development
