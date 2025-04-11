"""
Security tests for rate limiting.

This module provides tests for the rate limiting middleware.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from asf.medical.api.middleware.login_rate_limit_middleware import add_login_rate_limit_middleware
from asf.medical.core.enhanced_rate_limiter import enhanced_rate_limiter

@pytest.fixture
def rate_limited_app():
    """Create a test app with rate limiting middleware."""
    # Create a test app
    app = FastAPI()
    
    # Add rate limiting middleware
    add_login_rate_limit_middleware(
        app,
        login_path="/login",
        rate=3,  # 3 attempts per minute
        burst=2,  # 2 attempts in a burst
        window=60,  # 1 minute window
        block_time=300  # 5 minutes block time
    )
    
    # Add a test login endpoint
    @app.post("/login")
    def login_endpoint(username: str, password: str):
        if username == "test" and password == "password":
            return {"message": "success"}
        return {"message": "failure"}, 401
    
    # Add a test non-login endpoint
    @app.post("/other")
    def other_endpoint():
        return {"message": "success"}
    
    return app

@pytest.fixture
def reset_rate_limiter():
    """Reset the rate limiter before and after each test."""
    # Reset before test
    enhanced_rate_limiter.local_limits = {}
    enhanced_rate_limiter.local_tokens = {}
    enhanced_rate_limiter.local_last_refill = {}
    enhanced_rate_limiter.local_counters = {}
    
    yield
    
    # Reset after test
    enhanced_rate_limiter.local_limits = {}
    enhanced_rate_limiter.local_tokens = {}
    enhanced_rate_limiter.local_last_refill = {}
    enhanced_rate_limiter.local_counters = {}

def test_rate_limiting_login_endpoint(rate_limited_app, reset_rate_limiter):
    """Test that login endpoint is rate limited."""
    client = TestClient(rate_limited_app)
    
    # First 3 requests should succeed
    for i in range(3):
        response = client.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} was rate limited"
    
    # 4th request should be rate limited
    response = client.post("/login", json={"username": "test", "password": "wrong"})
    assert response.status_code == 429
    assert "Too many login attempts" in response.json()["detail"]
    
    # Headers should include rate limit information
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers
    assert "Retry-After" in response.headers

def test_non_login_endpoint_not_rate_limited(rate_limited_app, reset_rate_limiter):
    """Test that non-login endpoints are not rate limited."""
    client = TestClient(rate_limited_app)
    
    # Make many requests to non-login endpoint
    for i in range(10):
        response = client.post("/other")
        assert response.status_code != 429, f"Request {i+1} was rate limited"

def test_different_ips_separate_rate_limits(rate_limited_app, reset_rate_limiter, monkeypatch):
    """Test that different IPs have separate rate limits."""
    client1 = TestClient(rate_limited_app)
    client2 = TestClient(rate_limited_app)
    
    # Mock client IP addresses
    def get_client_ip_1(request):
        return "192.168.1.1"
    
    def get_client_ip_2(request):
        return "192.168.1.2"
    
    # Patch the _get_client_ip method for client1
    from asf.medical.api.middleware.login_rate_limit_middleware import LoginRateLimitMiddleware
    original_get_client_ip = LoginRateLimitMiddleware._get_client_ip
    
    # Use monkeypatch to patch the method
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", get_client_ip_1)
    
    # First 3 requests from client1 should succeed
    for i in range(3):
        response = client1.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} from client1 was rate limited"
    
    # 4th request from client1 should be rate limited
    response = client1.post("/login", json={"username": "test", "password": "wrong"})
    assert response.status_code == 429
    
    # Change the mock to client2's IP
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", get_client_ip_2)
    
    # Requests from client2 should still succeed
    for i in range(3):
        response = client2.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} from client2 was rate limited"
    
    # Restore the original method
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", original_get_client_ip)

def test_block_after_too_many_failed_attempts(rate_limited_app, reset_rate_limiter, monkeypatch):
    """Test that client is blocked after too many failed login attempts."""
    client = TestClient(rate_limited_app)
    
    # Mock client IP address
    def get_client_ip(request):
        return "192.168.1.3"
    
    # Patch the _get_client_ip method
    from asf.medical.api.middleware.login_rate_limit_middleware import LoginRateLimitMiddleware
    original_get_client_ip = LoginRateLimitMiddleware._get_client_ip
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", get_client_ip)
    
    # Make 3 failed login attempts
    for i in range(3):
        response = client.post("/login", json={"username": "test", "password": "wrong"})
        assert response.status_code != 429, f"Request {i+1} was rate limited"
    
    # Client should be blocked
    response = client.post("/login", json={"username": "test", "password": "wrong"})
    assert response.status_code == 429
    assert "Too many login attempts" in response.json()["detail"]
    
    # Even with correct credentials, client should still be blocked
    response = client.post("/login", json={"username": "test", "password": "password"})
    assert response.status_code == 429
    
    # Restore the original method
    monkeypatch.setattr(LoginRateLimitMiddleware, "_get_client_ip", original_get_client_ip)
