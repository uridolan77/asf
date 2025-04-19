import pytest
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from agentor.llm_gateway.api.auth import validate_api_key, get_settings, Settings


def test_validate_api_key():
    """Test the validate_api_key function."""
    # Create a FastAPI app
    app = FastAPI()
    
    # Create a test endpoint
    @app.get("/test")
    async def test_endpoint(api_key: str = Depends(validate_api_key)):
        return {"api_key": api_key}
    
    # Create a test client
    client = TestClient(app)
    
    # Test with a valid API key
    with patch('agentor.llm_gateway.api.auth.get_settings', return_value=Settings(api_key="test-api-key")):
        response = client.get("/test", headers={"X-API-KEY": "test-api-key"})
        assert response.status_code == 200
        assert response.json() == {"api_key": "test-api-key"}
    
    # Test with an invalid API key
    with patch('agentor.llm_gateway.api.auth.get_settings', return_value=Settings(api_key="test-api-key")):
        response = client.get("/test", headers={"X-API-KEY": "invalid-api-key"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid API Key"}
    
    # Test with no API key
    response = client.get("/test")
    assert response.status_code == 422  # Validation error


def test_rbac_middleware():
    """Test the RBACMiddleware class."""
    # This is a placeholder for testing the RBAC middleware
    # In a real implementation, we would test that the middleware
    # correctly checks permissions based on the JWT
    pass
