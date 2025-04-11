"""
Security tests for CSRF protection.

This module provides tests for the CSRF protection middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from asf.medical.api.middleware.csrf_middleware import CSRFMiddleware, add_csrf_middleware

def test_csrf_middleware_get_request():
    """Test that GET requests are allowed without CSRF token."""
    # Create a test app
    app = FastAPI()
    
    # Add CSRF middleware
    add_csrf_middleware(app)
    
    # Add a test endpoint
    @app.get("/test")
    def test_endpoint():
        return {"message": "success"}
    
    # Create a test client
    client = TestClient(app)
    
    # Test that GET requests are allowed without CSRF token
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "success"}
    
    # Check that the CSRF token cookie is set
    assert "csrf_token" in response.cookies

def test_csrf_middleware_post_request_without_token():
    """Test that POST requests are rejected without CSRF token."""
    # Create a test app
    app = FastAPI()
    
    # Add CSRF middleware
    add_csrf_middleware(app)
    
    # Add a test endpoint
    @app.post("/test")
    def test_endpoint():
        return {"message": "success"}
    
    # Create a test client
    client = TestClient(app)
    
    # Test that POST requests are rejected without CSRF token
    response = client.post("/test")
    assert response.status_code == 403
    assert "CSRF token missing" in response.json()["detail"]

def test_csrf_middleware_post_request_with_token():
    """Test that POST requests are allowed with valid CSRF token."""
    # Create a test app
    app = FastAPI()
    
    # Add CSRF middleware
    add_csrf_middleware(app)
    
    # Add a test endpoint
    @app.post("/test")
    def test_endpoint():
        return {"message": "success"}
    
    # Create a test client
    client = TestClient(app)
    
    # First, make a GET request to get the CSRF token
    response = client.get("/test")
    csrf_token = response.cookies["csrf_token"]
    
    # Test that POST requests are allowed with valid CSRF token
    response = client.post(
        "/test",
        headers={"X-CSRF-Token": csrf_token}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "success"}

def test_csrf_middleware_post_request_with_invalid_token():
    """Test that POST requests are rejected with invalid CSRF token."""
    # Create a test app
    app = FastAPI()
    
    # Add CSRF middleware
    add_csrf_middleware(app)
    
    # Add a test endpoint
    @app.post("/test")
    def test_endpoint():
        return {"message": "success"}
    
    # Create a test client
    client = TestClient(app)
    
    # First, make a GET request to get the CSRF token
    response = client.get("/test")
    
    # Test that POST requests are rejected with invalid CSRF token
    response = client.post(
        "/test",
        headers={"X-CSRF-Token": "invalid_token"}
    )
    assert response.status_code == 403
    assert "CSRF token mismatch" in response.json()["detail"]

def test_csrf_middleware_exempt_paths():
    """Test that exempt paths are allowed without CSRF token."""
    # Create a test app
    app = FastAPI()
    
    # Add CSRF middleware with exempt paths
    add_csrf_middleware(
        app,
        exempt_paths=["/exempt"]
    )
    
    # Add test endpoints
    @app.post("/test")
    def test_endpoint():
        return {"message": "success"}
    
    @app.post("/exempt")
    def exempt_endpoint():
        return {"message": "exempt"}
    
    # Create a test client
    client = TestClient(app)
    
    # Test that POST requests to exempt paths are allowed without CSRF token
    response = client.post("/exempt")
    assert response.status_code == 200
    assert response.json() == {"message": "exempt"}
    
    # Test that POST requests to non-exempt paths are rejected without CSRF token
    response = client.post("/test")
    assert response.status_code == 403
    assert "CSRF token missing" in response.json()["detail"]
