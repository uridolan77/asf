from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
import sys
import os

# Add the backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Use absolute imports
from config.database import get_db
from services.user_provider_service import UserProviderService
from utils.crypto import generate_key
from api.auth import get_current_user
from models.user import User
from pydantic import BaseModel

router = APIRouter(
    prefix="/api/user-providers",
    tags=["user-providers"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Get encryption key (in production, this should be loaded from a secure source)
ENCRYPTION_KEY = generate_key()

# Pydantic models for request/response
class UserProviderAssignment(BaseModel):
    user_id: int
    provider_id: str
    role: str = "user"

class UserProviderResponse(BaseModel):
    user_id: int
    provider_id: str
    role: str
    username: Optional[str] = None
    email: Optional[str] = None
    display_name: Optional[str] = None
    provider_type: Optional[str] = None
    assigned_at: Optional[str] = None

# Endpoints for managing users-providers relationship

@router.post("/assign", response_model=UserProviderResponse, status_code=status.HTTP_201_CREATED)
async def assign_user_to_provider(
    assignment: UserProviderAssignment,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Assign a user to a provider with a specific role."""
    # Check if current user has admin role
    if current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can assign users to providers"
        )

    service = UserProviderService(db, ENCRYPTION_KEY, current_user.id)
    result = service.assign_user_to_provider(
        assignment.provider_id,
        assignment.user_id,
        assignment.role
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assign user to provider"
        )

    # Get the user's role for the provider
    role = service.get_user_role_for_provider(assignment.provider_id, assignment.user_id)

    return UserProviderResponse(
        user_id=assignment.user_id,
        provider_id=assignment.provider_id,
        role=role
    )

@router.delete("/{provider_id}/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_user_from_provider(
    provider_id: str,
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Remove a user from a provider."""
    # Check if current user has admin role
    if current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators can remove users from providers"
        )

    service = UserProviderService(db, ENCRYPTION_KEY, current_user.id)
    result = service.remove_user_from_provider(provider_id, user_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove user from provider"
        )

    return None

@router.get("/providers/{provider_id}/users", response_model=List[UserProviderResponse])
async def get_users_for_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all users assigned to a provider."""
    service = UserProviderService(db, ENCRYPTION_KEY, current_user.id)

    # Check if current user has access to the provider
    if current_user.role.name != "admin" and not service.check_user_has_access(provider_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this provider"
        )

    users = service.get_users_for_provider(provider_id)

    return [
        UserProviderResponse(
            user_id=user["user_id"],
            provider_id=provider_id,
            role=user["role"],
            username=user["username"],
            email=user["email"],
            assigned_at=user["assigned_at"].isoformat() if user.get("assigned_at") else None
        )
        for user in users
    ]

@router.get("/users/{user_id}/providers", response_model=List[UserProviderResponse])
async def get_providers_for_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all providers assigned to a user."""
    # Check if current user is admin or the requested user
    if current_user.role.name != "admin" and current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only view your own providers unless you're an administrator"
        )

    service = UserProviderService(db, ENCRYPTION_KEY, current_user.id)
    providers = service.get_providers_for_user(user_id)

    return [
        UserProviderResponse(
            user_id=user_id,
            provider_id=provider["provider_id"],
            role=provider["role"],
            display_name=provider["display_name"],
            provider_type=provider["provider_type"],
            assigned_at=provider["assigned_at"].isoformat() if provider.get("assigned_at") else None
        )
        for provider in providers
    ]

@router.get("/check-access/{provider_id}", response_model=Dict[str, Any])
async def check_user_access(
    provider_id: str,
    required_role: str = "user",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Check if the current user has access to a provider with a specific role."""
    service = UserProviderService(db, ENCRYPTION_KEY, current_user.id)

    # Admins always have access
    if current_user.role.name == "admin":
        return {"has_access": True, "role": "admin"}

    # Check if user has the required role
    has_access = service.check_user_has_access(provider_id, current_user.id, required_role)
    role = service.get_user_role_for_provider(provider_id, current_user.id)

    return {"has_access": has_access, "role": role}
