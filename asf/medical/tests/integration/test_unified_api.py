"""
Integration tests for the unified API.

This module provides integration tests for the unified API.
"""

import pytest
import logging
import asyncio
from typing import Dict, Any
from httpx import AsyncClient

from asf.medical.api.main_unified import app
from asf.medical.ml.services.prisma_screening_service import ScreeningStage
from asf.medical.ml.services.bias_assessment_service import BiasDomain, BiasRisk
from asf.medical.ml.services.enhanced_contradiction_service import ContradictionType, ContradictionConfidence

# Configure logging
logger = logging.getLogger(__name__)

# Test data
TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"

@pytest.fixture
async def client():
    """Get an async client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def user_token(client):
    """Get a user token for testing."""
    response = await client.post(
        "/v1/auth/token",
        data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    )
    return response.json()["access_token"]

@pytest.fixture
async def admin_token(client):
    """Get an admin token for testing."""
    response = await client.post(
        "/v1/auth/token",
        data={"username": TEST_ADMIN_EMAIL, "password": TEST_ADMIN_PASSWORD}
    )
    return response.json()["access_token"]

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestAuthAPI:
    """Test cases for authentication API."""
    
    async def test_login_success(self, client):
        """Test successful login."""
        # Login
        response = await client.post(
            "/v1/auth/token",
            data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert "token_type" in response.json()
        assert "role" in response.json()
        assert "expires_in" in response.json()
        assert response.json()["token_type"] == "bearer"
        assert response.json()["role"] == "user"
    
    async def test_login_failure(self, client):
        """Test login with wrong credentials."""
        # Login with wrong password
        response = await client.post(
            "/v1/auth/token",
            data={"username": TEST_USER_EMAIL, "password": "wrongpassword"}
        )
        
        # Assertions
        assert response.status_code == 401
        assert "detail" in response.json()
        assert response.json()["detail"] == "Incorrect email or password"
    
    async def test_get_current_user(self, client, user_token):
        """Test getting current user information."""
        # Get current user
        response = await client.get(
            "/v1/auth/me",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "email" in response.json()
        assert "role" in response.json()
        assert "is_active" in response.json()
        assert response.json()["email"] == TEST_USER_EMAIL
        assert response.json()["role"] == "user"
        assert response.json()["is_active"] is True
    
    async def test_get_users_admin(self, client, admin_token):
        """Test getting all users as admin."""
        # Get users
        response = await client.get(
            "/v1/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) > 0
    
    async def test_get_users_not_admin(self, client, user_token):
        """Test getting all users as non-admin."""
        # Get users
        response = await client.get(
            "/v1/auth/users",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 403
        assert "detail" in response.json()
        assert response.json()["detail"] == "Not enough permissions"

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestSearchAPI:
    """Test cases for search API."""
    
    async def test_search_endpoint(self, client, user_token):
        """Test search endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5
        }
        
        # Make request
        response = await client.post(
            "/v1/search",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "query" in response.json()["meta"]
        assert response.json()["meta"]["query"] == data["query"]
        assert "max_results" in response.json()["meta"]
        assert response.json()["meta"]["max_results"] == data["max_results"]
    
    async def test_pico_search_endpoint(self, client, user_token):
        """Test PICO search endpoint."""
        # Test data
        data = {
            "condition": "hypertension",
            "interventions": ["lisinopril", "amlodipine"],
            "outcomes": ["blood pressure reduction", "cardiovascular events"],
            "max_results": 5
        }
        
        # Make request
        response = await client.post(
            "/v1/search/pico",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "condition" in response.json()["meta"]
        assert response.json()["meta"]["condition"] == data["condition"]
        assert "interventions" in response.json()["meta"]
        assert response.json()["meta"]["interventions"] == data["interventions"]

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestContradictionAPI:
    """Test cases for contradiction API."""
    
    async def test_analyze_contradictions_endpoint(self, client, user_token):
        """Test analyze contradictions endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "threshold": 0.7,
            "use_all_methods": True
        }
        
        # Make request
        response = await client.post(
            "/v1/contradiction/analyze",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "query" in response.json()["data"]
        assert response.json()["data"]["query"] == data["query"]
        assert "total_articles" in response.json()["data"]
        assert "contradictions_found" in response.json()["data"]
        assert "contradiction_types" in response.json()["data"]
        assert "contradictions" in response.json()["data"]
        assert "analysis_id" in response.json()["data"]
    
    async def test_detect_contradiction_endpoint(self, client, user_token):
        """Test detect contradiction endpoint."""
        # Test data
        data = {
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "use_all_methods": True
        }
        
        # Make request
        response = await client.post(
            "/v1/contradiction/detect",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "is_contradiction" in response.json()["data"]
        assert "contradiction_score" in response.json()["data"]
        assert "contradiction_type" in response.json()["data"]
        assert "confidence" in response.json()["data"]
        assert "explanation" in response.json()["data"]

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestScreeningAPI:
    """Test cases for screening API."""
    
    async def test_prisma_screening_endpoint(self, client, user_token):
        """Test PRISMA screening endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "stage": ScreeningStage.SCREENING,
            "criteria": {
                "include": ["randomized controlled trial", "cardiovascular outcomes"],
                "exclude": ["animal study", "in vitro"]
            }
        }
        
        # Make request
        response = await client.post(
            "/v1/screening/prisma",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "query" in response.json()["data"]
        assert response.json()["data"]["query"] == data["query"]
        assert "stage" in response.json()["data"]
        assert response.json()["data"]["stage"] == data["stage"]
        assert "total_articles" in response.json()["data"]
        assert "included" in response.json()["data"]
        assert "excluded" in response.json()["data"]
        assert "uncertain" in response.json()["data"]
        assert "results" in response.json()["data"]
        assert "flow_data" in response.json()["data"]
    
    async def test_bias_assessment_endpoint(self, client, user_token):
        """Test bias assessment endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "domains": [
                BiasDomain.RANDOMIZATION,
                BiasDomain.BLINDING,
                BiasDomain.ALLOCATION_CONCEALMENT,
                BiasDomain.SAMPLE_SIZE,
                BiasDomain.ATTRITION
            ]
        }
        
        # Make request
        response = await client.post(
            "/v1/screening/bias-assessment",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "query" in response.json()["data"]
        assert response.json()["data"]["query"] == data["query"]
        assert "total_articles" in response.json()["data"]
        assert "low_risk" in response.json()["data"]
        assert "moderate_risk" in response.json()["data"]
        assert "high_risk" in response.json()["data"]
        assert "unclear_risk" in response.json()["data"]
        assert "results" in response.json()["data"]

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestAnalysisAPI:
    """Test cases for analysis API."""
    
    async def test_analyze_contradictions_endpoint(self, client, user_token):
        """Test analyze contradictions endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "threshold": 0.7,
            "use_biomedlm": True,
            "use_tsmixer": False,
            "use_lorentz": False
        }
        
        # Make request
        response = await client.post(
            "/v1/analysis/contradictions",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "query" in response.json()["data"]
        assert response.json()["data"]["query"] == data["query"]
        assert "total_articles" in response.json()["data"]
        assert "contradictions" in response.json()["data"]
        assert "analysis_id" in response.json()["data"]
    
    async def test_analyze_cap_endpoint(self, client, user_token):
        """Test analyze CAP endpoint."""
        # Make request
        response = await client.get(
            "/v1/analysis/cap",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "total_articles" in response.json()["data"]
        assert "analysis" in response.json()["data"]
        assert "analysis_id" in response.json()["data"]

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestKnowledgeBaseAPI:
    """Test cases for knowledge base API."""
    
    async def test_create_knowledge_base_endpoint(self, client, user_token):
        """Test create knowledge base endpoint."""
        # Test data
        data = {
            "name": f"test_kb_{asyncio.get_event_loop().time()}",
            "query": "statin therapy cardiovascular",
            "update_schedule": "weekly"
        }
        
        # Make request
        response = await client.post(
            "/v1/knowledge-base",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "kb_id" in response.json()["data"]
        assert "name" in response.json()["data"]
        assert response.json()["data"]["name"] == data["name"]
        assert "query" in response.json()["data"]
        assert response.json()["data"]["query"] == data["query"]
        assert "update_schedule" in response.json()["data"]
        assert response.json()["data"]["update_schedule"] == data["update_schedule"]
        
        # Store KB ID for later tests
        kb_id = response.json()["data"]["kb_id"]
        
        # Test list knowledge bases endpoint
        list_response = await client.get(
            "/v1/knowledge-base",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert list_response.status_code == 200
        assert "success" in list_response.json()
        assert "message" in list_response.json()
        assert "data" in list_response.json()
        assert "meta" in list_response.json()
        assert list_response.json()["success"] is True
        assert isinstance(list_response.json()["data"], list)
        assert len(list_response.json()["data"]) > 0
        
        # Test get knowledge base endpoint
        get_response = await client.get(
            f"/v1/knowledge-base/{kb_id}",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert get_response.status_code == 200
        assert "success" in get_response.json()
        assert "message" in get_response.json()
        assert "data" in get_response.json()
        assert "meta" in get_response.json()
        assert get_response.json()["success"] is True
        assert "kb_id" in get_response.json()["data"]
        assert get_response.json()["data"]["kb_id"] == kb_id
        assert "name" in get_response.json()["data"]
        assert get_response.json()["data"]["name"] == data["name"]
        
        # Test update knowledge base endpoint
        update_response = await client.post(
            f"/v1/knowledge-base/{kb_id}/update",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert update_response.status_code == 200
        assert "success" in update_response.json()
        assert "message" in update_response.json()
        assert "data" in update_response.json()
        assert "meta" in update_response.json()
        assert update_response.json()["success"] is True
        assert "kb_id" in update_response.json()["data"]
        assert update_response.json()["data"]["kb_id"] == kb_id
        assert "status" in update_response.json()["data"]
        assert update_response.json()["data"]["status"] == "updating"
        
        # Test delete knowledge base endpoint
        delete_response = await client.delete(
            f"/v1/knowledge-base/{kb_id}",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert delete_response.status_code == 200
        assert "success" in delete_response.json()
        assert "message" in delete_response.json()
        assert "data" in delete_response.json()
        assert "meta" in delete_response.json()
        assert delete_response.json()["success"] is True
        assert "kb_id" in delete_response.json()["data"]
        assert delete_response.json()["data"]["kb_id"] == kb_id
        assert "status" in delete_response.json()["data"]
        assert delete_response.json()["data"]["status"] == "deleted"

@pytest.mark.integration
@pytest.mark.api
@pytest.mark.asyncio
class TestExportAPI:
    """Test cases for export API."""
    
    async def test_export_with_query_endpoint(self, client, user_token):
        """Test export with query endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5
        }
        
        # Make request
        response = await client.post(
            "/v1/export/json",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "file_url" in response.json()["data"]
