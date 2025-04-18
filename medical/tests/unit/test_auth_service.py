"""
Unit tests for the AuthService.
This module provides unit tests for the AuthService.
"""
import pytest
import logging
from fastapi import HTTPException
from unittest.mock import AsyncMock, MagicMock
from ...core.security import get_password_hash, create_access_token
from ...services.auth_service import AuthService
from ...api.auth import get_current_user, get_current_active_user, get_admin_user
logger = logging.getLogger(__name__)
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"
@pytest.fixture
def mock_user_repository():
    """Mock UserRepository for testing.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    mock = AsyncMock()
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
    mock.create_async.return_value = test_user
    mock.update_async.return_value = test_user
    mock.delete_async.return_value = True
    return mock
@pytest.fixture
def auth_service(mock_user_repository):
    """AuthService instance for testing.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
    return AuthService(user_repository=mock_user_repository)
@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.service
class TestAuthService:
    """Test cases for AuthService."""
    async def test_authenticate_user_success(self, auth_service, mock_user_repository):
        """
        Test authenticate_user with valid credentials.
        Args:
            # TODO: Add parameter descriptions
        Returns:
            # TODO: Add return description
        """
    async def test_get_current_user_dependency(self, auth_service):
        access_token = create_access_token(subject=TEST_USER_EMAIL)
        mock_auth_service = AsyncMock()
        mock_auth_service.get_current_user.return_value = MagicMock(
            id=1,
            email=TEST_USER_EMAIL,
            is_active=True,
            role="user"
        )
        user = await get_current_user(access_token, None, mock_auth_service)
        mock_auth_service.get_current_user.assert_called_once_with(None, access_token)
        assert user is not None
        assert user.email == TEST_USER_EMAIL
    async def test_get_current_active_user_dependency(self):
        inactive_user = MagicMock(
            id=3,
            email="inactive@example.com",
            is_active=False,
            role="user"
        )
        with pytest.raises(HTTPException) as excinfo:
            await get_current_active_user(inactive_user)
        assert excinfo.value.status_code == 400
        assert "Inactive user" in excinfo.value.detail
    async def test_get_admin_user_dependency(self):
        non_admin_user = MagicMock(
            id=1,
            email=TEST_USER_EMAIL,
            is_active=True,
            role="user"
        )
        with pytest.raises(HTTPException) as excinfo:
            await get_admin_user(non_admin_user)
        assert excinfo.value.status_code == 403
        assert "Not enough permissions" in excinfo.value.detail