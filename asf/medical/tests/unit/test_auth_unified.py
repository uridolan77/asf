"""
Unit tests for the unified authentication system.

This module provides unit tests for the unified authentication system.
"""

import pytest
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock
from fastapi import HTTPException
from jose import jwt

from asf.medical.api.auth_unified import (
    AuthService, get_current_user, get_current_active_user, get_admin_user,
    has_role, has_any_role, Token, TokenData, User, UserCreate, UserUpdate
)
from asf.medical.core.config import settings
from asf.medical.core.security import get_password_hash, verify_password, create_access_token
from asf.medical.storage.models import User as DBUser

# Configure logging
logger = logging.getLogger(__name__)

# Test data
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_USER_ROLE = "user"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"
TEST_ADMIN_ROLE = "admin"

@pytest.fixture
def mock_user_repository():
    """Mock user repository for testing."""
    repository = MagicMock()
    
    # Create mock users
    test_user = MagicMock(spec=DBUser)
    test_user.id = 1
    test_user.email = TEST_USER_EMAIL
    test_user.hashed_password = get_password_hash(TEST_USER_PASSWORD)
    test_user.role = TEST_USER_ROLE
    test_user.is_active = True
    
    test_admin = MagicMock(spec=DBUser)
    test_admin.id = 2
    test_admin.email = TEST_ADMIN_EMAIL
    test_admin.hashed_password = get_password_hash(TEST_ADMIN_PASSWORD)
    test_admin.role = TEST_ADMIN_ROLE
    test_admin.is_active = True
    
    inactive_user = MagicMock(spec=DBUser)
    inactive_user.id = 3
    inactive_user.email = "inactive@example.com"
    inactive_user.hashed_password = get_password_hash("inactivepassword")
    inactive_user.role = "user"
    inactive_user.is_active = False
    
    # Set up repository methods
    repository.get_by_email_async = AsyncMock()
    repository.get_by_email_async.side_effect = lambda db, email: {
        TEST_USER_EMAIL: test_user,
        TEST_ADMIN_EMAIL: test_admin,
        "inactive@example.com": inactive_user
    }.get(email)
    
    repository.get_by_id_async = AsyncMock()
    repository.get_by_id_async.side_effect = lambda db, id: {
        1: test_user,
        2: test_admin,
        3: inactive_user
    }.get(id)
    
    repository.create_user_async = AsyncMock()
    repository.create_user_async.return_value = test_user
    
    repository.update_user_async = AsyncMock()
    repository.update_user_async.return_value = test_user
    
    repository.delete_user_async = AsyncMock()
    repository.delete_user_async.return_value = True
    
    repository.get_users_async = AsyncMock()
    repository.get_users_async.return_value = [test_user, test_admin, inactive_user]
    
    return repository

@pytest.fixture
def auth_service(mock_user_repository):
    """Auth service for testing."""
    return AuthService(mock_user_repository)

@pytest.mark.unit
@pytest.mark.service
@pytest.mark.async_test
class TestAuthService:
    """Test cases for AuthService."""
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, auth_service):
        """Test successful user authentication."""
        # Authenticate user
        user = await auth_service.authenticate_user(None, TEST_USER_EMAIL, TEST_USER_PASSWORD)
        
        # Assertions
        assert user is not None
        assert user.email == TEST_USER_EMAIL
        assert user.role == TEST_USER_ROLE
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, auth_service):
        """Test user authentication with wrong password."""
        # Authenticate user with wrong password
        user = await auth_service.authenticate_user(None, TEST_USER_EMAIL, "wrongpassword")
        
        # Assertions
        assert user is None
    
    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, auth_service):
        """Test user authentication with non-existent user."""
        # Authenticate non-existent user
        user = await auth_service.authenticate_user(None, "nonexistent@example.com", TEST_USER_PASSWORD)
        
        # Assertions
        assert user is None
    
    @pytest.mark.asyncio
    async def test_authenticate_user_inactive(self, auth_service):
        """Test user authentication with inactive user."""
        # Authenticate inactive user
        user = await auth_service.authenticate_user(None, "inactive@example.com", "inactivepassword")
        
        # Assertions
        assert user is None
    
    @pytest.mark.asyncio
    async def test_get_current_user_success(self, auth_service):
        """Test getting current user from token."""
        # Create token
        access_token = create_access_token(subject=TEST_USER_EMAIL)
        
        # Get current user
        user = await auth_service.get_current_user(None, access_token)
        
        # Assertions
        assert user is not None
        assert user.email == TEST_USER_EMAIL
        assert user.role == TEST_USER_ROLE
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, auth_service):
        """Test getting current user with invalid token."""
        # Create invalid token
        access_token = "invalid.token.here"
        
        # Get current user with invalid token
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, access_token)
        
        # Assertions
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Could not validate credentials"
    
    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(self, auth_service):
        """Test getting current user with expired token."""
        # Create expired token
        expires_delta = timedelta(minutes=-1)  # Expired 1 minute ago
        access_token = create_access_token(subject=TEST_USER_EMAIL, expires_delta=expires_delta)
        
        # Get current user with expired token
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, access_token)
        
        # Assertions
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Could not validate credentials"
    
    @pytest.mark.asyncio
    async def test_get_current_user_user_not_found(self, auth_service):
        """Test getting current user with token for non-existent user."""
        # Create token for non-existent user
        access_token = create_access_token(subject="nonexistent@example.com")
        
        # Get current user with token for non-existent user
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, access_token)
        
        # Assertions
        assert excinfo.value.status_code == 401
        assert excinfo.value.detail == "Could not validate credentials"
    
    @pytest.mark.asyncio
    async def test_register_user_success(self, auth_service):
        """Test successful user registration."""
        # Register user
        user = await auth_service.register_user(None, "newuser@example.com", "newpassword", "user")
        
        # Assertions
        assert user is not None
        assert auth_service.user_repository.create_user_async.called
    
    @pytest.mark.asyncio
    async def test_register_user_existing_email(self, auth_service):
        """Test user registration with existing email."""
        # Register user with existing email
        user = await auth_service.register_user(None, TEST_USER_EMAIL, "newpassword", "user")
        
        # Assertions
        assert user is None
    
    @pytest.mark.asyncio
    async def test_update_user_success(self, auth_service):
        """Test successful user update."""
        # Update user
        user = await auth_service.update_user(None, 1, {"email": "updated@example.com"})
        
        # Assertions
        assert user is not None
        assert auth_service.user_repository.update_user_async.called
    
    @pytest.mark.asyncio
    async def test_update_user_not_found(self, auth_service):
        """Test user update with non-existent user."""
        # Set up repository to return None for non-existent user
        auth_service.user_repository.get_by_id_async.side_effect = lambda db, id: None if id == 999 else MagicMock()
        
        # Update non-existent user
        user = await auth_service.update_user(None, 999, {"email": "updated@example.com"})
        
        # Assertions
        assert user is None
    
    @pytest.mark.asyncio
    async def test_update_user_with_password(self, auth_service):
        """Test user update with password."""
        # Update user with password
        user = await auth_service.update_user(None, 1, {"password": "newpassword"})
        
        # Assertions
        assert user is not None
        assert auth_service.user_repository.update_user_async.called
        
        # Check that password was hashed
        call_args = auth_service.user_repository.update_user_async.call_args
        assert "hashed_password" in call_args[0][2]
        assert "password" not in call_args[0][2]
    
    @pytest.mark.asyncio
    async def test_delete_user_success(self, auth_service):
        """Test successful user deletion."""
        # Delete user
        result = await auth_service.delete_user(None, 1)
        
        # Assertions
        assert result is True
        assert auth_service.user_repository.delete_user_async.called
    
    @pytest.mark.asyncio
    async def test_get_users(self, auth_service):
        """Test getting all users."""
        # Get users
        users = await auth_service.get_users(None)
        
        # Assertions
        assert users is not None
        assert len(users) == 3
        assert auth_service.user_repository.get_users_async.called

@pytest.mark.unit
@pytest.mark.dependency
@pytest.mark.async_test
class TestAuthDependencies:
    """Test cases for authentication dependencies."""
    
    @pytest.mark.asyncio
    async def test_get_current_active_user_success(self):
        """Test getting current active user."""
        # Create mock user
        user = MagicMock(spec=DBUser)
        user.is_active = True
        
        # Get current active user
        result = await get_current_active_user(user)
        
        # Assertions
        assert result is user
    
    @pytest.mark.asyncio
    async def test_get_current_active_user_inactive(self):
        """Test getting current active user with inactive user."""
        # Create mock inactive user
        user = MagicMock(spec=DBUser)
        user.is_active = False
        
        # Get current active user with inactive user
        with pytest.raises(HTTPException) as excinfo:
            await get_current_active_user(user)
        
        # Assertions
        assert excinfo.value.status_code == 400
        assert excinfo.value.detail == "Inactive user"
    
    @pytest.mark.asyncio
    async def test_get_admin_user_success(self):
        """Test getting admin user."""
        # Create mock admin user
        user = MagicMock(spec=DBUser)
        user.role = "admin"
        
        # Get admin user
        result = await get_admin_user(user)
        
        # Assertions
        assert result is user
    
    @pytest.mark.asyncio
    async def test_get_admin_user_not_admin(self):
        """Test getting admin user with non-admin user."""
        # Create mock non-admin user
        user = MagicMock(spec=DBUser)
        user.role = "user"
        
        # Get admin user with non-admin user
        with pytest.raises(HTTPException) as excinfo:
            await get_admin_user(user)
        
        # Assertions
        assert excinfo.value.status_code == 403
        assert excinfo.value.detail == "Not enough permissions"
    
    @pytest.mark.asyncio
    async def test_has_role_success(self):
        """Test has_role dependency with matching role."""
        # Create mock user with matching role
        user = MagicMock(spec=DBUser)
        user.role = "researcher"
        
        # Create has_role dependency
        has_researcher_role = has_role("researcher")
        
        # Check role
        result = await has_researcher_role(user)
        
        # Assertions
        assert result is user
    
    @pytest.mark.asyncio
    async def test_has_role_failure(self):
        """Test has_role dependency with non-matching role."""
        # Create mock user with non-matching role
        user = MagicMock(spec=DBUser)
        user.role = "user"
        
        # Create has_role dependency
        has_researcher_role = has_role("researcher")
        
        # Check role
        with pytest.raises(HTTPException) as excinfo:
            await has_researcher_role(user)
        
        # Assertions
        assert excinfo.value.status_code == 403
        assert excinfo.value.detail == "Role 'researcher' required"
    
    @pytest.mark.asyncio
    async def test_has_any_role_success(self):
        """Test has_any_role dependency with matching role."""
        # Create mock user with matching role
        user = MagicMock(spec=DBUser)
        user.role = "researcher"
        
        # Create has_any_role dependency
        has_researcher_or_admin_role = has_any_role(["researcher", "admin"])
        
        # Check role
        result = await has_researcher_or_admin_role(user)
        
        # Assertions
        assert result is user
    
    @pytest.mark.asyncio
    async def test_has_any_role_failure(self):
        """Test has_any_role dependency with non-matching role."""
        # Create mock user with non-matching role
        user = MagicMock(spec=DBUser)
        user.role = "user"
        
        # Create has_any_role dependency
        has_researcher_or_admin_role = has_any_role(["researcher", "admin"])
        
        # Check role
        with pytest.raises(HTTPException) as excinfo:
            await has_researcher_or_admin_role(user)
        
        # Assertions
        assert excinfo.value.status_code == 403
        assert excinfo.value.detail == "One of roles ['researcher', 'admin'] required"
