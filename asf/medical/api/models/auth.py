"""
Authentication models for the Medical Research Synthesizer API.

This module defines the Pydantic models for authentication.
"""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field

class Token(BaseModel):
    """Token model for authentication responses."""
    access_token: str
    token_type: str
    role: str

class TokenData(BaseModel):
    """Token data model for JWT payload."""
    email: Optional[str] = None

class User(BaseModel):
    """User model."""
    email: EmailStr
    role: str
    is_active: bool = True

class UserInDB(User):
    """User model with password hash."""
    hashed_password: str

class UserRegistrationRequest(BaseModel):
    """Request model for user registration."""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password", min_length=8)
    role: str = Field("user", description="User role (admin or user)")
