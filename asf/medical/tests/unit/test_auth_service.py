"""
Unit tests for the AuthService.

This module provides unit tests for the AuthService.
"""

import pytest
import logging
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta

from fastapi import HTTPException
from jose import jwt, JWTError

from asf.medical.api.auth import AuthService, TokenData, get_current_user, get_current_active_user, get_admin_user
from asf.medical.core.config import settings
from asf.medical.core.security import get_password_hash, create_access_token, create_refresh_token

# Configure logging
logger = logging.getLogger(__name__)

# Test data
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"

@pytest.fixture
def mock_user_repository():
    """Mock UserRepository for testing."""
    mock = AsyncMock()
    
    # Create test users
    test_user = MagicMock(
        id=1,
        email=TEST_USER_EMAIL,
        hashed_password=get_password_hash(TEST_USER_PASSWORD),
        is_active=True,
        role="user"
    )
    
    admin_user = MagicMock(
        id=2,
        email=TEST_ADMIN_EMAIL,
        hashed_password=get_password_hash(TEST_ADMIN_PASSWORD),
        is_active=True,
        role="admin"
    )
    
    inactive_user = MagicMock(
        id=3,
        email="inactive@example.com",
        hashed_password=get_password_hash("inactivepassword"),
        is_active=False,
        role="user"
    )
    
    # Mock get_by_email_async method
    async def get_by_email_async(db, email):
        if email == TEST_USER_EMAIL:
            return test_user
        elif email == TEST_ADMIN_EMAIL:
            return admin_user
        elif email == "inactive@example.com":
            return inactive_user
        else:
            return None
    
    mock.get_by_email_async = get_by_email_async
    
    # Mock get_by_id_async method
    async def get_by_id_async(db, id):
        if id == 1:
            return test_user
        elif id == 2:
            return admin_user
        elif id == 3:
            return inactive_user
        else:
            return None
    
    mock.get_by_id_async = get_by_id_async
    
    # Mock create_async method
    mock.create_async.return_value = test_user
    
    # Mock update_async method
    mock.update_async.return_value = test_user
    
    # Mock delete_async method
    mock.delete_async.return_value = True
    
    return mock

@pytest.fixture
def auth_service(mock_user_repository):
    """AuthService instance for testing."""
    return AuthService(user_repository=mock_user_repository)

@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.service
class TestAuthService:
    """Test cases for AuthService."""
    
    async def test_authenticate_user_success(self, auth_service, mock_user_repository):
        """Test authenticate_user with valid credentials."""
        # Call the authenticate_user method with valid credentials
        user = await auth_service.authenticate_user(None, TEST_USER_EMAIL, TEST_USER_PASSWORD)
        
        # Check that the user repository's get_by_email_async method was called
        # This is handled by the fixture
        
        # Check the result
        assert user is not None
        assert user.email == TEST_USER_EMAIL
        assert user.is_active is True
        assert user.role == "user"
    
    async def test_authenticate_user_invalid_email(self, auth_service, mock_user_repository):
        """Test authenticate_user with invalid email."""
        # Call the authenticate_user method with invalid email
        user = await auth_service.authenticate_user(None, "invalid@example.com", TEST_USER_PASSWORD)
        
        # Check the result
        assert user is None
    
    async def test_authenticate_user_invalid_password(self, auth_service, mock_user_repository):
        """Test authenticate_user with invalid password."""
        # Call the authenticate_user method with invalid password
        user = await auth_service.authenticate_user(None, TEST_USER_EMAIL, "invalidpassword")
        
        # Check the result
        assert user is None
    
    async def test_authenticate_user_inactive(self, auth_service, mock_user_repository):
        """Test authenticate_user with inactive user."""
        # Call the authenticate_user method with inactive user
        user = await auth_service.authenticate_user(None, "inactive@example.com", "inactivepassword")
        
        # Check the result
        assert user is None
    
    async def test_get_current_user_valid_token(self, auth_service, mock_user_repository):
        """Test get_current_user with valid token."""
        # Create a valid token
        access_token = create_access_token(subject=TEST_USER_EMAIL)
        
        # Call the get_current_user method
        user = await auth_service.get_current_user(None, access_token)
        
        # Check the result
        assert user is not None
        assert user.email == TEST_USER_EMAIL
        assert user.is_active is True
        assert user.role == "user"
    
    async def test_get_current_user_invalid_token(self, auth_service, mock_user_repository):
        """Test get_current_user with invalid token."""
        # Create an invalid token
        invalid_token = "invalid.token.here"
        
        # Call the get_current_user method and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, invalid_token)
        
        # Check the exception
        assert excinfo.value.status_code == 401
        assert "Could not validate credentials" in excinfo.value.detail
    
    async def test_get_current_user_expired_token(self, auth_service, mock_user_repository):
        """Test get_current_user with expired token."""
        # Create an expired token
        expired_token = create_access_token(
            subject=TEST_USER_EMAIL,
            expires_delta=timedelta(minutes=-10)  # Token expired 10 minutes ago
        )
        
        # Call the get_current_user method and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, expired_token)
        
        # Check the exception
        assert excinfo.value.status_code == 401
        assert "Could not validate credentials" in excinfo.value.detail
    
    async def test_get_current_user_user_not_found(self, auth_service, mock_user_repository):
        """Test get_current_user with user not found."""
        # Create a token for a non-existent user
        token = create_access_token(subject="nonexistent@example.com")
        
        # Call the get_current_user method and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, token)
        
        # Check the exception
        assert excinfo.value.status_code == 401
        assert "User not found" in excinfo.value.detail
    
    async def test_get_current_user_refresh_token(self, auth_service, mock_user_repository):
        """Test get_current_user with refresh token."""
        # Create a refresh token
        refresh_token = create_refresh_token(subject=TEST_USER_EMAIL)
        
        # Call the get_current_user method and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await auth_service.get_current_user(None, refresh_token)
        
        # Check the exception
        assert excinfo.value.status_code == 401
        assert "Cannot use refresh token for authentication" in excinfo.value.detail
    
    async def test_register_user(self, auth_service, mock_user_repository):
        """Test register_user."""
        # Call the register_user method
        user = await auth_service.register_user(
            db=None,
            email="newuser@example.com",
            password="newpassword",
            role="user"
        )
        
        # Check that the user repository's create_async method was called
        mock_user_repository.create_async.assert_called_once()
        
        # Check the result
        assert user is not None
        assert user.email == TEST_USER_EMAIL  # This is from the mock
    
    async def test_register_user_email_exists(self, auth_service, mock_user_repository):
        """Test register_user with existing email."""
        # Mock the user repository to return a user for the email check
        mock_user_repository.get_by_email_async.return_value = MagicMock()
        
        # Call the register_user method
        user = await auth_service.register_user(
            db=None,
            email=TEST_USER_EMAIL,
            password="newpassword",
            role="user"
        )
        
        # Check the result
        assert user is None
    
    async def test_get_user_by_email(self, auth_service, mock_user_repository):
        """Test get_user_by_email."""
        # Call the get_user_by_email method
        user = await auth_service.get_user_by_email(None, TEST_USER_EMAIL)
        
        # Check the result
        assert user is not None
        assert user.email == TEST_USER_EMAIL
    
    async def test_get_user_by_id(self, auth_service, mock_user_repository):
        """Test get_user_by_id."""
        # Call the get_user_by_id method
        user = await auth_service.get_user_by_id(None, 1)
        
        # Check the result
        assert user is not None
        assert user.id == 1
    
    async def test_update_user(self, auth_service, mock_user_repository):
        """Test update_user."""
        # Call the update_user method
        user = await auth_service.update_user(
            db=None,
            user_id=1,
            user_data={"email": "updated@example.com"}
        )
        
        # Check that the user repository's update_async method was called
        mock_user_repository.update_async.assert_called_once()
        
        # Check the result
        assert user is not None
    
    async def test_delete_user(self, auth_service, mock_user_repository):
        """Test delete_user."""
        # Call the delete_user method
        result = await auth_service.delete_user(None, 1)
        
        # Check that the user repository's delete_async method was called
        mock_user_repository.delete_async.assert_called_once()
        
        # Check the result
        assert result is True

@pytest.mark.asyncio
@pytest.mark.unit
class TestAuthDependencies:
    """Test cases for authentication dependencies."""
    
    async def test_get_current_user_dependency(self, auth_service):
        """Test get_current_user dependency."""
        # Create a valid token
        access_token = create_access_token(subject=TEST_USER_EMAIL)
        
        # Mock the auth_service dependency
        mock_auth_service = AsyncMock()
        mock_auth_service.get_current_user.return_value = MagicMock(
            id=1,
            email=TEST_USER_EMAIL,
            is_active=True,
            role="user"
        )
        
        # Call the get_current_user dependency
        user = await get_current_user(access_token, None, mock_auth_service)
        
        # Check that the auth service's get_current_user method was called
        mock_auth_service.get_current_user.assert_called_once_with(None, access_token)
        
        # Check the result
        assert user is not None
        assert user.email == TEST_USER_EMAIL
    
    async def test_get_current_active_user_dependency(self):
        """Test get_current_active_user dependency."""
        # Create an active user
        active_user = MagicMock(
            id=1,
            email=TEST_USER_EMAIL,
            is_active=True,
            role="user"
        )
        
        # Call the get_current_active_user dependency
        user = await get_current_active_user(active_user)
        
        # Check the result
        assert user is not None
        assert user.email == TEST_USER_EMAIL
    
    async def test_get_current_active_user_dependency_inactive(self):
        """Test get_current_active_user dependency with inactive user."""
        # Create an inactive user
        inactive_user = MagicMock(
            id=3,
            email="inactive@example.com",
            is_active=False,
            role="user"
        )
        
        # Call the get_current_active_user dependency and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await get_current_active_user(inactive_user)
        
        # Check the exception
        assert excinfo.value.status_code == 400
        assert "Inactive user" in excinfo.value.detail
    
    async def test_get_admin_user_dependency(self):
        """Test get_admin_user dependency."""
        # Create an admin user
        admin_user = MagicMock(
            id=2,
            email=TEST_ADMIN_EMAIL,
            is_active=True,
            role="admin"
        )
        
        # Call the get_admin_user dependency
        user = await get_admin_user(admin_user)
        
        # Check the result
        assert user is not None
        assert user.email == TEST_ADMIN_EMAIL
        assert user.role == "admin"
    
    async def test_get_admin_user_dependency_non_admin(self):
        """Test get_admin_user dependency with non-admin user."""
        # Create a non-admin user
        non_admin_user = MagicMock(
            id=1,
            email=TEST_USER_EMAIL,
            is_active=True,
            role="user"
        )
        
        # Call the get_admin_user dependency and expect an HTTPException
        with pytest.raises(HTTPException) as excinfo:
            await get_admin_user(non_admin_user)
        
        # Check the exception
        assert excinfo.value.status_code == 403
        assert "Not enough permissions" in excinfo.value.detail
