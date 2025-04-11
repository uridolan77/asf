Security tests for rate limiting.

This module provides tests for the rate limiting middleware.

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from asf.medical.api.middleware.login_rate_limit_middleware import add_login_rate_limit_middleware
from asf.medical.core.enhanced_rate_limiter import enhanced_rate_limiter

@pytest.fixture
def rate_limited_app():
    """Create a test app with rate limiting middleware.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    app = FastAPI()
    
    add_login_rate_limit_middleware(
        app,
        login_path="/login",
        rate=3,  # 3 attempts per minute
        burst=2,  # 2 attempts in a burst
        window=60,  # 1 minute window
        block_time=300  # 5 minutes block time
    )
    
    @app.post("/login")
    def login_endpoint(username: str, password: str):
        """
        login_endpoint function.
        
        This function provides functionality for...
        Args:
            username: Description of username
            password: Description of password
        """
        if username == "test" and password == "password":
            return {"message": "success"}
        return {"message": "failure"}, 401
    
    @app.post("/other")
    def other_endpoint():
        """
        other_endpoint function.
        
        This function provides functionality for..."""
        return {"message": "success"}
    
    return app

@pytest.fixture
def reset_rate_limiter():
    """Reset the rate limiter before and after each test.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    enhanced_rate_limiter.local_limits = {}
    enhanced_rate_limiter.local_tokens = {}
    enhanced_rate_limiter.local_last_refill = {}
    enhanced_rate_limiter.local_counters = {}
    
    yield
    
    enhanced_rate_limiter.local_limits = {}
    enhanced_rate_limiter.local_tokens = {}
    enhanced_rate_limiter.local_last_refill = {}
    enhanced_rate_limiter.local_counters = {}

def test_rate_limiting_login_endpoint(rate_limited_app, reset_rate_limiter):
    """Test that login endpoint is rate limited.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    client = TestClient(rate_limited_app)
    
    for i in range(3):
        response = client.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} was rate limited"
    
    response = client.post("/login", json={"username": "test", "password": "wrong"})
    assert response.status_code == 429
    assert "Too many login attempts" in response.json()["detail"]
    
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers
    assert "Retry-After" in response.headers

def test_non_login_endpoint_not_rate_limited(rate_limited_app, reset_rate_limiter):
    """Test that non-login endpoints are not rate limited.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    client = TestClient(rate_limited_app)
    
    for i in range(10):
        response = client.post("/other")
        assert response.status_code != 429, f"Request {i+1} was rate limited"

def test_different_ips_separate_rate_limits(rate_limited_app, reset_rate_limiter, monkeypatch):
    """Test that different IPs have separate rate limits.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    client1 = TestClient(rate_limited_app)
    client2 = TestClient(rate_limited_app)
    
    def get_client_ip_1(request):
        """
        get_client_ip_1 function.
        
        This function provides functionality for...
        Args:
            request: Description of request
        
        Returns:
            Description of return value
        """
        return "192.168.1.1"
    
    def get_client_ip_2(request):
        """
        get_client_ip_2 function.
        
        This function provides functionality for...
        Args:
            request: Description of request
        
        Returns:
            Description of return value
        """
        return "192.168.1.2"
    
    from asf.medical.api.middleware.login_rate_limit_middleware import LoginRateLimitMiddleware
    original_get_client_ip = LoginRateLimitMiddleware._get_client_ip
    
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", get_client_ip_1)
    
    for i in range(3):
        response = client1.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} from client1 was rate limited"
    
    response = client1.post("/login", json={"username": "test", "password": "wrong"})
    assert response.status_code == 429
    
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", get_client_ip_2)
    
    for i in range(3):
        response = client2.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} from client2 was rate limited"
    
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", original_get_client_ip)

def test_block_after_too_many_failed_attempts(rate_limited_app, reset_rate_limiter, monkeypatch):
    """Test that client is blocked after too many failed login attempts.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    client = TestClient(rate_limited_app)
    
    def get_client_ip(request):
        """
        get_client_ip function.
        
        This function provides functionality for...
        Args:
            request: Description of request
        
        Returns:
            Description of return value
        """
        return "192.168.1.3"
    
    from asf.medical.api.middleware.login_rate_limit_middleware import LoginRateLimitMiddleware
    original_get_client_ip = LoginRateLimitMiddleware._get_client_ip
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", get_client_ip)
    
    for i in range(3):
        response = client.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} was rate limited"
    
    response = client.post("/login", json={"username": "test", "password": "wrong"})
    assert response.status_code == 429
    assert "Too many login attempts" in response.json()["detail"]
    
    response = client.post("/login", json={"username": "test", "password": "password"})
    assert response.status_code == 429
    
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", original_get_client_ip)
