"""
Security tests for the authentication system.

This module provides tests for the security aspects of the authentication system.
"""

import pytest
import jwt
from datetime import datetime, timedelta
from fastapi import HTTPException
from jose import JWTError

from asf.medical.core.config import settings
from asf.medical.core.security import create_access_token, create_refresh_token
from asf.medical.api.auth import get_current_user, TokenData
from asf.medical.storage.models import User as DBUser

# Mock user for testing
mock_user = DBUser(
    id=1,
    email="test@example.com",
    hashed_password="hashed_password",
    is_active=True,
    role="user"
)

# Mock admin user for testing
mock_admin = DBUser(
    id=2,
    email="admin@example.com",
    hashed_password="hashed_password",
    is_active=True,
    role="admin"
)

@pytest.mark.asyncio
async def test_expired_token_rejected(mocker):
    """Test that expired tokens are rejected."""
    # Create an expired token
    expired_token = create_access_token(
        subject="test@example.com",
        expires_delta=timedelta(minutes=-10)  # Token expired 10 minutes ago
    )
    
    # Mock the database session and user repository
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    
    # Mock the get_current_user function to raise an exception
    mock_auth_service.get_current_user.side_effect = JWTError("Token expired")
    
    # Test that the token is rejected
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(expired_token, mock_db, mock_auth_service)
    
    # Check that the exception has the correct status code
    assert excinfo.value.status_code == 401
    assert "Could not validate credentials" in excinfo.value.detail

@pytest.mark.asyncio
async def test_refresh_token_rejected_for_authentication(mocker):
    """Test that refresh tokens are rejected for authentication."""
    # Create a refresh token
    refresh_token = create_refresh_token(
        subject="test@example.com"
    )
    
    # Mock the database session and user repository
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    
    # Mock the jwt.decode function to return a payload with type=refresh
    mocker.patch(
        "jose.jwt.decode",
        return_value={"sub": "test@example.com", "type": "refresh"}
    )
    
    # Test that the token is rejected
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(refresh_token, mock_db, mock_auth_service)
    
    # Check that the exception has the correct status code
    assert excinfo.value.status_code == 401
    assert "Cannot use refresh token for authentication" in excinfo.value.detail

@pytest.mark.asyncio
async def test_tampered_token_rejected(mocker):
    """Test that tampered tokens are rejected."""
    # Create a valid token
    valid_token = create_access_token(
        subject="test@example.com"
    )
    
    # Tamper with the token by changing a character
    tampered_token = valid_token[:-1] + ("1" if valid_token[-1] != "1" else "2")
    
    # Mock the database session and user repository
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    
    # Mock the get_current_user function to raise an exception
    mock_auth_service.get_current_user.side_effect = JWTError("Invalid signature")
    
    # Test that the token is rejected
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(tampered_token, mock_db, mock_auth_service)
    
    # Check that the exception has the correct status code
    assert excinfo.value.status_code == 401
    assert "Could not validate credentials" in excinfo.value.detail

@pytest.mark.asyncio
async def test_inactive_user_rejected(mocker):
    """Test that inactive users are rejected."""
    # Create a valid token
    valid_token = create_access_token(
        subject="inactive@example.com"
    )
    
    # Mock the database session and user repository
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    
    # Create an inactive user
    inactive_user = DBUser(
        id=3,
        email="inactive@example.com",
        hashed_password="hashed_password",
        is_active=False,
        role="user"
    )
    
    # Mock the get_current_user function to return the inactive user
    mock_auth_service.get_current_user.return_value = inactive_user
    
    # Test that the user is rejected
    with pytest.raises(HTTPException) as excinfo:
        from asf.medical.api.auth import get_current_active_user
        await get_current_active_user(inactive_user)
    
    # Check that the exception has the correct status code
    assert excinfo.value.status_code == 400
    assert "Inactive user" in excinfo.value.detail

@pytest.mark.asyncio
async def test_non_admin_user_rejected_for_admin_endpoint(mocker):
    """Test that non-admin users are rejected for admin endpoints."""
    # Create a valid token for a non-admin user
    valid_token = create_access_token(
        subject="test@example.com"
    )
    
    # Mock the database session and user repository
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    
    # Mock the get_current_user function to return a non-admin user
    mock_auth_service.get_current_user.return_value = mock_user
    
    # Test that the user is rejected for admin endpoints
    with pytest.raises(HTTPException) as excinfo:
        from asf.medical.api.auth import get_admin_user
        await get_admin_user(mock_user)
    
    # Check that the exception has the correct status code
    assert excinfo.value.status_code == 403
    assert "Not enough permissions" in excinfo.value.detail
