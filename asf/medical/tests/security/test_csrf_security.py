Security tests for CSRF protection.

This module provides tests for the CSRF protection middleware.

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_csrf_middleware_get_request():
    """Test that GET requests are allowed without CSRF token.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    app = FastAPI()
    
    add_csrf_middleware(app)
    
    @app.get("/test")
    def test_endpoint():
        """
        test_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "success"}
    
    client = TestClient(app)
    
    response = client.get("/test")
    assert response.status_code == 200
    assert response.json() == {"message": "success"}
    
    assert "csrf_token" in response.cookies

def test_csrf_middleware_post_request_without_token():
    """Test that POST requests are rejected without CSRF token.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    app = FastAPI()
    
    add_csrf_middleware(app)
    
    @app.post("/test")
    def test_endpoint():
        """
        test_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "success"}
    
    client = TestClient(app)
    
    response = client.post("/test")
    assert response.status_code == 403
    assert "CSRF token missing" in response.json()["detail"]

def test_csrf_middleware_post_request_with_token():
    """Test that POST requests are allowed with valid CSRF token.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    app = FastAPI()
    
    add_csrf_middleware(app)
    
    @app.post("/test")
    def test_endpoint():
        """
        test_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "success"}
    
    client = TestClient(app)
    
    response = client.get("/test")
    csrf_token = response.cookies["csrf_token"]
    
    response = client.post(
        "/test",
        headers={"X-CSRF-Token": csrf_token}
    )
    assert response.status_code == 200
    assert response.json() == {"message": "success"}

def test_csrf_middleware_post_request_with_invalid_token():
    """Test that POST requests are rejected with invalid CSRF token.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    app = FastAPI()
    
    add_csrf_middleware(app)
    
    @app.post("/test")
    def test_endpoint():
        """
        test_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "success"}
    
    client = TestClient(app)
    
    response = client.get("/test")
    
    response = client.post(
        "/test",
        headers={"X-CSRF-Token": "invalid_token"}
    )
    assert response.status_code == 403
    assert "CSRF token mismatch" in response.json()["detail"]

def test_csrf_middleware_exempt_paths():
    """Test that exempt paths are allowed without CSRF token.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    app = FastAPI()
    
    add_csrf_middleware(
        app,
        exempt_paths=["/exempt"]
    )
    
    @app.post("/test")
    def test_endpoint():
        """
        test_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "success"}
    
    @app.post("/exempt")
    def exempt_endpoint():
        """
        exempt_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "exempt"}
    
    client = TestClient(app)
    
    response = client.post("/exempt")
    assert response.status_code == 200
    assert response.json() == {"message": "exempt"}
    
    response = client.post("/test")
    assert response.status_code == 403
    assert "CSRF token missing" in response.json()["detail"]
