"""
Dependencies for the BO backend API.
"""
from typing import Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from config.database import get_db
from models.user import User

async def get_current_user(db: Session = Depends(get_db)) -> Optional[User]:
    """
    Get the current authenticated user.
    
    This is a mock implementation that always returns a test user.
    In a real application, this would validate a token and return the user.
    
    Args:
        db: Database session
        
    Returns:
        User object if authenticated, None otherwise
    """
    # Mock implementation - always return a test user
    # In a real application, this would validate a token and return the user
    test_user = User(
        id=1,
        username="test_user",
        email="test@example.com",
        full_name="Test User",
        role_id=1
    )
    return test_user
