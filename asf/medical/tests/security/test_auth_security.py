"""
Security tests for the authentication system.
This module provides tests for the security aspects of the authentication system.
"""
import pytest
import jwt
from fastapi import HTTPException
from asf.medical.core.security import create_access_token, create_refresh_token
from asf.medical.api.auth import get_current_user
from asf.medical.storage.models import MedicalUser as DBUser

mock_user = DBUser(
    id=1,
    email="test@example.com",
    hashed_password="hashed_password",
    is_active=True,
    role="user"
)

mock_admin = DBUser(
    id=2,
    email="admin@example.com",
    hashed_password="hashed_password",
    is_active=True,
    role="admin"
)

@pytest.mark.asyncio
async def test_expired_token_rejected(mocker):
    refresh_token = create_refresh_token(
        subject="test@example.com"
    )
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    mocker.patch(
        "jose.jwt.decode",
        return_value={"sub": "test@example.com", "type": "refresh"}
    )
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(refresh_token, mock_db, mock_auth_service)
    assert excinfo.value.status_code == 401
    assert "Cannot use refresh token for authentication" in excinfo.value.detail

@pytest.mark.asyncio
async def test_tampered_token_rejected(mocker):
    valid_token = create_access_token(
        subject="inactive@example.com"
    )
    mock_db = mocker.AsyncMock()
    mock_auth_service = mocker.AsyncMock()
    inactive_user = DBUser(
        id=3,
        email="inactive@example.com",
        hashed_password="hashed_password",
        is_active=False,
        role="user"
    )
    mock_auth_service.get_current_user.return_value = inactive_user
    with pytest.raises(HTTPException) as excinfo:
        from asf.medical.api.auth import get_current_active_user
        await get_current_active_user(inactive_user)
    assert excinfo.value.status_code == 400
    assert "Inactive user" in excinfo.value.detail

@pytest.mark.asyncio
async def test_non_admin_user_rejected_for_admin_endpoint(mocker):