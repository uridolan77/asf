"""
Integration tests for the unified API.

This module provides integration tests for the unified API.
"""

import pytest
import logging
from typing import Dict, Any
from fastapi.testclient import TestClient

from asf.medical.api.main_unified import app
from asf.medical.ml.services.prisma_screening_service import ScreeningStage
from asf.medical.ml.services.bias_assessment_service import BiasDomain, BiasRisk
from asf.medical.ml.services.enhanced_contradiction_service import ContradictionType, ContradictionConfidence

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
def user_token():
    """Get a user token for testing."""
    response = client.post(
        "/v1/auth/token",
        data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    )
    return response.json()["access_token"]

@pytest.fixture
def admin_token():
    """Get an admin token for testing."""
    response = client.post(
        "/v1/auth/token",
        data={"username": TEST_ADMIN_EMAIL, "password": TEST_ADMIN_PASSWORD}
    )
    return response.json()["access_token"]

@pytest.mark.integration
@pytest.mark.api
class TestAuthAPI:
    """Test cases for authentication API."""
    
    def test_login_success(self):
        """Test successful login."""
        # Login
        response = client.post(
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
    
    def test_login_failure(self):
        """Test login with wrong credentials."""
        # Login with wrong password
        response = client.post(
            "/v1/auth/token",
            data={"username": TEST_USER_EMAIL, "password": "wrongpassword"}
        )
        
        # Assertions
        assert response.status_code == 401
        assert "detail" in response.json()
        assert response.json()["detail"] == "Incorrect email or password"
    
    def test_get_current_user(self, user_token):
        """Test getting current user information."""
        # Get current user
        response = client.get(
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
    
    def test_get_users_admin(self, admin_token):
        """Test getting all users as admin."""
        # Get users
        response = client.get(
            "/v1/auth/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) > 0
    
    def test_get_users_not_admin(self, user_token):
        """Test getting all users as non-admin."""
        # Get users
        response = client.get(
            "/v1/auth/users",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 403
        assert "detail" in response.json()
        assert response.json()["detail"] == "Not enough permissions"

@pytest.mark.integration
@pytest.mark.api
class TestSearchAPI:
    """Test cases for search API."""
    
    def test_search_endpoint(self, user_token):
        """Test search endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5
        }
        
        # Make request
        response = client.post(
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
    
    def test_pico_search_endpoint(self, user_token):
        """Test PICO search endpoint."""
        # Test data
        data = {
            "condition": "hypertension",
            "interventions": ["lisinopril", "amlodipine"],
            "outcomes": ["blood pressure reduction", "cardiovascular events"],
            "max_results": 5
        }
        
        # Make request
        response = client.post(
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
class TestContradictionAPI:
    """Test cases for contradiction API."""
    
    def test_analyze_contradictions_endpoint(self, user_token):
        """Test analyze contradictions endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "threshold": 0.7,
            "use_all_methods": True
        }
        
        # Make request
        response = client.post(
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
    
    def test_detect_contradiction_endpoint(self, user_token):
        """Test detect contradiction endpoint."""
        # Test data
        data = {
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "use_all_methods": True
        }
        
        # Make request
        response = client.post(
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
        assert "claim1" in response.json()["data"]
        assert response.json()["data"]["claim1"] == data["claim1"]
        assert "claim2" in response.json()["data"]
        assert response.json()["data"]["claim2"] == data["claim2"]
        assert "is_contradiction" in response.json()["data"]
        assert "contradiction_score" in response.json()["data"]
        assert "contradiction_type" in response.json()["data"]
        assert "confidence" in response.json()["data"]
        assert "explanation" in response.json()["data"]

@pytest.mark.integration
@pytest.mark.api
class TestScreeningAPI:
    """Test cases for screening API."""
    
    def test_prisma_screening_endpoint(self, user_token):
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
        response = client.post(
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
    
    def test_bias_assessment_endpoint(self, user_token):
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
        response = client.post(
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
    
    def test_flow_diagram_endpoint(self, user_token):
        """Test flow diagram endpoint."""
        # Make request
        response = client.get(
            "/v1/screening/flow-diagram",
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        # Assertions
        assert response.status_code == 200
        assert "success" in response.json()
        assert "message" in response.json()
        assert "data" in response.json()
        assert "meta" in response.json()
        assert response.json()["success"] is True
        assert "identification" in response.json()["data"]
        assert "screening" in response.json()["data"]
        assert "eligibility" in response.json()["data"]
        assert "included" in response.json()["data"]

@pytest.mark.integration
@pytest.mark.api
class TestExportAPI:
    """Test cases for export API."""
    
    def test_export_json_endpoint(self, user_token):
        """Test export JSON endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5
        }
        
        # First, perform a search to get a result_id
        search_response = client.post(
            "/v1/search",
            json=data,
            headers={"Authorization": f"Bearer {user_token}"}
        )
        
        result_id = search_response.json()["data"].get("result_id")
        
        if not result_id:
            pytest.skip("No result_id available for export test")
        
        # Test data for export
        export_data = {
            "result_id": result_id
        }
        
        # Make request
        response = client.post(
            "/v1/export/json",
            json=export_data,
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
    
    def test_export_with_query_endpoint(self, user_token):
        """Test export with query endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5
        }
        
        # Make request
        response = client.post(
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
