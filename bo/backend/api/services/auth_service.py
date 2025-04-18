"""
Authentication service for the BO backend.

This module provides a service for authentication and authorization.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status

from models.user import User, Role
from config.config import SessionLocal

logger = logging.getLogger(__name__)

class AuthService:
    """
    Service for authentication and authorization.
    """
    
    def __init__(self):
        """
        Initialize the authentication service.
        """
        self._db = SessionLocal()
    
    async def get_user_role(self, user_id: int) -> Optional[Role]:
        """
        Get the role of a user.
        
        Args:
            user_id: User ID
            
        Returns:
            User role or None if not found
        """
        try:
            # In a real implementation, this would query the database
            # For now, we'll just return a mock role based on the user ID
            if user_id == 1:
                return Role(id=1, name="admin", description="Administrator")
            elif user_id == 2:
                return Role(id=2, name="user", description="Regular user")
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting user role: {str(e)}")
            return None
    
    async def check_user_access(self, user_id: int, resource: str) -> bool:
        """
        Check if a user has access to a resource.
        
        Args:
            user_id: User ID
            resource: Resource name
            
        Returns:
            True if user has access, False otherwise
        """
        try:
            # Get user role
            role = await self.get_user_role(user_id)
            if not role:
                return False
            
            # Check access based on role and resource
            if role.name == "admin":
                # Admins have access to everything
                return True
            elif role.name == "user":
                # Users have access to some resources
                allowed_resources = ["search", "kb", "contradiction", "terminology"]
                return resource in allowed_resources
            else:
                return False
        except Exception as e:
            logger.error(f"Error checking user access: {str(e)}")
            return False

# Singleton instance
_auth_service = None

def get_auth_service():
    """
    Get the authentication service instance.
    
    Returns:
        Authentication service instance
    """
    global _auth_service
    
    if _auth_service is None:
        _auth_service = AuthService()
    
    return _auth_service
