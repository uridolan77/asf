"""
Integration tests for the authentication API.

This module provides integration tests for the authentication API endpoints.
"""

import pytest
import logging
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from asf.medical.api.main import app
from asf.medical.api.auth import AuthService

# Configure logging
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

# Test data
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"

@pytest.fixture
def auth_token():
    """Get authentication token for testing."""
    # Login to get token
    response = client.post(
        "/v1/auth/token",
        data={
            "username": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
    )
    
    # Check if login was successful
    if response.status_code != 200:
        pytest.skip("Failed to get authentication token")
    
    # Return the token
    return response.json()["access_token"]

@pytest.fixture
def refresh_token():
    """Get refresh token for testing."""
    # Login to get token
    response = client.post(
        "/v1/auth/token",
        data={
            "username": TEST_USER_EMAIL,
            "password": TEST_USER_PASSWORD
        }
    )
    
    # Check if login was successful
    if response.status_code != 200:
        pytest.skip("Failed to get refresh token")
    
    # Return the refresh token
    return response.json()["refresh_token"]

@pytest.fixture
def admin_token():
    """Get admin authentication token for testing."""
    # Login to get token
    response = client.post(
        "/v1/auth/token",
        data={
            "username": TEST_ADMIN_EMAIL,
            "password": TEST_ADMIN_PASSWORD
        }
    )
    
    # Check if login was successful
    if response.status_code != 200:
        pytest.skip("Failed to get admin authentication token")
    
    # Return the token
    return response.json()["access_token"]

@pytest.mark.integration
@pytest.mark.api
class TestAuthAPI:
    """Test cases for authentication API."""
    
    def test_login_success(self):
        """Test login with valid credentials."""
        # Call the login endpoint
        response = client.post(
            "/v1/auth/token",
            data={
                "username": TEST_USER_EMAIL,
                "password": TEST_USER_PASSWORD
            }
        )
        
        # Check the response
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert "refresh_token" in response.json()
        assert response.json()["token_type"] == "bearer"
        assert response.json()["role"] == "user"
        assert "expires_in" in response.json()
    
    def test_login_invalid_credentials(self):
        """Test login with invalid credentials."""
        # Call the login endpoint with invalid credentials
        response = client.post(
            "/v1/auth/token",
            data={
                "username": TEST_USER_EMAIL,
                "password": "wrongpassword"
            }
        )
        
        # Check the response
        assert response.status_code == 401
        assert "Incorrect email or password" in response.json()["detail"]
    
    def test_refresh_token_endpoint(self, refresh_token):
        """Test refresh token endpoint."""
        # Call the refresh token endpoint
        response = client.post(
            "/v1/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        
        # Check the response
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert "refresh_token" in response.json()
        assert response.json()["token_type"] == "bearer"
        assert "role" in response.json()
        assert "expires_in" in response.json()
    
    def test_refresh_token_endpoint_invalid_token(self):
        """Test refresh token endpoint with invalid token."""
        # Call the refresh token endpoint with invalid token
        response = client.post(
            "/v1/auth/refresh",
            json={"refresh_token": "invalid.token.here"}
        )
        
        # Check the response
        assert response.status_code == 401
        assert "Invalid refresh token" in response.json()["detail"]
    
    def test_get_current_user_endpoint(self, auth_token):
        """Test get_current_user endpoint."""
        # Call the get_current_user endpoint
        response = client.get(
            "/v1/auth/me",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["email"] == TEST_USER_EMAIL
        assert response.json()["is_active"] is True
        assert response.json()["role"] == "user"
    
    def test_get_current_user_endpoint_invalid_token(self):
        """Test get_current_user endpoint with invalid token."""
        # Call the get_current_user endpoint with invalid token
        response = client.get(
            "/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"}
        )
        
        # Check the response
        assert response.status_code == 401
        assert "Could not validate credentials" in response.json()["detail"]
    
    @patch.object(AuthService, "register_user")
    def test_register_user_endpoint(self, mock_register_user, admin_token):
        """Test register_user endpoint."""
        # Mock the register_user method
        mock_register_user.return_value = MagicMock(
            id=3,
            email="newuser@example.com",
            is_active=True,
            role="user"
        )
        
        # Call the register_user endpoint
        response = client.post(
            "/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "newpassword",
                "role": "user"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 201
        assert response.json()["email"] == "newuser@example.com"
        assert response.json()["is_active"] is True
        assert response.json()["role"] == "user"
    
    @patch.object(AuthService, "register_user")
    def test_register_user_endpoint_email_exists(self, mock_register_user, admin_token):
        """Test register_user endpoint with existing email."""
        # Mock the register_user method to return None (email exists)
        mock_register_user.return_value = None
        
        # Call the register_user endpoint
        response = client.post(
            "/v1/auth/register",
            json={
                "email": TEST_USER_EMAIL,
                "password": "newpassword",
                "role": "user"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 400
        assert "Email already registered" in response.json()["detail"]
    
    def test_register_user_endpoint_unauthorized(self):
        """Test register_user endpoint without admin authentication."""
        # Call the register_user endpoint without admin authentication
        response = client.post(
            "/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "newpassword",
                "role": "user"
            }
        )
        
        # Check the response
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
    
    def test_register_user_endpoint_non_admin(self, auth_token):
        """Test register_user endpoint with non-admin authentication."""
        # Call the register_user endpoint with non-admin authentication
        response = client.post(
            "/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "newpassword",
                "role": "user"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 403
        assert "Not enough permissions" in response.json()["detail"]
    
    @patch.object(AuthService, "update_user")
    def test_update_current_user_endpoint(self, mock_update_user, auth_token):
        """Test update_current_user endpoint."""
        # Mock the update_user method
        mock_update_user.return_value = MagicMock(
            id=1,
            email="updated@example.com",
            is_active=True,
            role="user"
        )
        
        # Call the update_current_user endpoint
        response = client.put(
            "/v1/auth/me",
            json={"email": "updated@example.com"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["email"] == "updated@example.com"
    
    @patch.object(AuthService, "get_users")
    def test_get_users_endpoint(self, mock_get_users, admin_token):
        """Test get_users endpoint."""
        # Mock the get_users method
        mock_get_users.return_value = [
            MagicMock(
                id=1,
                email=TEST_USER_EMAIL,
                is_active=True,
                role="user"
            ),
            MagicMock(
                id=2,
                email=TEST_ADMIN_EMAIL,
                is_active=True,
                role="admin"
            )
        ]
        
        # Call the get_users endpoint
        response = client.get(
            "/v1/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert len(response.json()) == 2
        assert response.json()[0]["email"] == TEST_USER_EMAIL
        assert response.json()[1]["email"] == TEST_ADMIN_EMAIL
    
    def test_get_users_endpoint_non_admin(self, auth_token):
        """Test get_users endpoint with non-admin authentication."""
        # Call the get_users endpoint with non-admin authentication
        response = client.get(
            "/v1/auth/users",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 403
        assert "Not enough permissions" in response.json()["detail"]
    
    @patch.object(AuthService, "get_user_by_id")
    def test_get_user_by_id_endpoint(self, mock_get_user_by_id, admin_token):
        """Test get_user_by_id endpoint."""
        # Mock the get_user_by_id method
        mock_get_user_by_id.return_value = MagicMock(
            id=1,
            email=TEST_USER_EMAIL,
            is_active=True,
            role="user"
        )
        
        # Call the get_user_by_id endpoint
        response = client.get(
            "/v1/auth/users/1",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["email"] == TEST_USER_EMAIL
    
    @patch.object(AuthService, "get_user_by_id")
    def test_get_user_by_id_endpoint_not_found(self, mock_get_user_by_id, admin_token):
        """Test get_user_by_id endpoint with user not found."""
        # Mock the get_user_by_id method to return None
        mock_get_user_by_id.return_value = None
        
        # Call the get_user_by_id endpoint
        response = client.get(
            "/v1/auth/users/999",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]
    
    @patch.object(AuthService, "update_user")
    def test_update_user_endpoint(self, mock_update_user, admin_token):
        """Test update_user endpoint."""
        # Mock the update_user method
        mock_update_user.return_value = MagicMock(
            id=1,
            email="updated@example.com",
            is_active=True,
            role="user"
        )
        
        # Call the update_user endpoint
        response = client.put(
            "/v1/auth/users/1",
            json={"email": "updated@example.com"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["email"] == "updated@example.com"
    
    @patch.object(AuthService, "delete_user")
    def test_delete_user_endpoint(self, mock_delete_user, admin_token):
        """Test delete_user endpoint."""
        # Mock the delete_user method
        mock_delete_user.return_value = True
        
        # Call the delete_user endpoint
        response = client.delete(
            "/v1/auth/users/1",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["status"] == "success"
    
    @patch.object(AuthService, "delete_user")
    def test_delete_user_endpoint_not_found(self, mock_delete_user, admin_token):
        """Test delete_user endpoint with user not found."""
        # Mock the delete_user method to return False
        mock_delete_user.return_value = False
        
        # Call the delete_user endpoint
        response = client.delete(
            "/v1/auth/users/999",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Check the response
        assert response.status_code == 404
        assert "User not found" in response.json()["detail"]
