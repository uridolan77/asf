"""
Integration tests for the analysis API.

This module provides integration tests for the analysis API endpoints.
"""

import pytest
import logging
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from asf.medical.api.main import app
from asf.medical.services.analysis_service import AnalysisService

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
class TestAnalysisAPI:
    """Test cases for analysis API."""
    
    @patch.object(AnalysisService, "analyze_contradictions")
    def test_analyze_contradictions_endpoint(self, mock_analyze_contradictions, auth_token):
        """Test analyze_contradictions endpoint."""
        # Mock the analyze_contradictions method
        mock_analyze_contradictions.return_value = {
            "query": "test query",
            "total_articles": 2,
            "contradictions": [
                {
                    "article1": {
                        "pmid": "12345",
                        "title": "Test Article 1",
                        "abstract": "This is a test abstract for article 1."
                    },
                    "article2": {
                        "pmid": "67890",
                        "title": "Test Article 2",
                        "abstract": "This is a test abstract for article 2."
                    },
                    "contradiction_score": 0.85,
                    "contradiction_type": "negation",
                    "explanation": "The articles contradict each other on the effectiveness of the treatment."
                }
            ],
            "analysis_id": "test-analysis-id",
            "detection_method": "biomedlm"
        }
        
        # Call the analyze_contradictions endpoint
        response = client.post(
            "/v1/analysis/contradictions",
            json={
                "query": "test query",
                "max_results": 10,
                "threshold": 0.7,
                "use_biomedlm": True
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["query"] == "test query"
        assert response.json()["total_articles"] == 2
        assert len(response.json()["contradictions"]) == 1
        assert response.json()["contradictions"][0]["contradiction_score"] == 0.85
        assert response.json()["contradictions"][0]["contradiction_type"] == "negation"
        assert response.json()["analysis_id"] == "test-analysis-id"
        assert response.json()["detection_method"] == "biomedlm"
    
    @patch.object(AnalysisService, "analyze_contradictions")
    def test_analyze_contradictions_endpoint_with_multiple_methods(self, mock_analyze_contradictions, auth_token):
        """Test analyze_contradictions endpoint with multiple detection methods."""
        # Mock the analyze_contradictions method
        mock_analyze_contradictions.return_value = {
            "query": "test query",
            "total_articles": 2,
            "contradictions": [
                {
                    "article1": {
                        "pmid": "12345",
                        "title": "Test Article 1"
                    },
                    "article2": {
                        "pmid": "67890",
                        "title": "Test Article 2"
                    },
                    "contradiction_score": 0.85,
                    "contradiction_type": "negation"
                }
            ],
            "analysis_id": "test-analysis-id",
            "detection_method": "biomedlm,tsmixer,lorentz"
        }
        
        # Call the analyze_contradictions endpoint with multiple detection methods
        response = client.post(
            "/v1/analysis/contradictions",
            json={
                "query": "test query",
                "max_results": 10,
                "threshold": 0.7,
                "use_biomedlm": True,
                "use_tsmixer": True,
                "use_lorentz": True
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert "biomedlm" in response.json()["detection_method"]
        assert "tsmixer" in response.json()["detection_method"]
        assert "lorentz" in response.json()["detection_method"]
        
        # Check that the parameters were passed correctly
        mock_analyze_contradictions.assert_called_with(
            query="test query",
            max_results=10,
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=True,
            use_lorentz=True,
            user_id=None  # This would be set in a real request
        )
    
    @patch.object(AnalysisService, "detect_contradiction")
    def test_detect_contradiction_endpoint(self, mock_detect_contradiction, auth_token):
        """Test detect_contradiction endpoint."""
        # Mock the detect_contradiction method
        mock_detect_contradiction.return_value = {
            "is_contradiction": True,
            "score": 0.85,
            "type": "negation",
            "explanation": "The claims contradict each other on the effectiveness of the treatment."
        }
        
        # Call the detect_contradiction endpoint
        response = client.post(
            "/v1/analysis/detect-contradiction",
            json={
                "claim1": "Treatment X is effective for condition Y.",
                "claim2": "Treatment X is not effective for condition Y.",
                "threshold": 0.7
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["is_contradiction"] is True
        assert response.json()["score"] == 0.85
        assert response.json()["type"] == "negation"
        assert "explanation" in response.json()
    
    @patch.object(AnalysisService, "detect_contradiction")
    def test_detect_contradiction_endpoint_with_multiple_methods(self, mock_detect_contradiction, auth_token):
        """Test detect_contradiction endpoint with multiple detection methods."""
        # Mock the detect_contradiction method
        mock_detect_contradiction.return_value = {
            "is_contradiction": True,
            "score": 0.85,
            "type": "negation",
            "explanation": "The claims contradict each other on the effectiveness of the treatment."
        }
        
        # Call the detect_contradiction endpoint with multiple detection methods
        response = client.post(
            "/v1/analysis/detect-contradiction",
            json={
                "claim1": "Treatment X is effective for condition Y.",
                "claim2": "Treatment X is not effective for condition Y.",
                "threshold": 0.7,
                "use_biomedlm": True,
                "use_tsmixer": True,
                "use_lorentz": True,
                "use_temporal": True
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        
        # Check that the parameters were passed correctly
        mock_detect_contradiction.assert_called_with(
            claim1="Treatment X is effective for condition Y.",
            claim2="Treatment X is not effective for condition Y.",
            threshold=0.7,
            use_biomedlm=True,
            use_tsmixer=True,
            use_lorentz=True,
            use_temporal=True,
            skip_cache=False
        )
    
    @patch.object(AnalysisService, "get_analysis_by_id")
    def test_get_analysis_by_id_endpoint(self, mock_get_analysis_by_id, auth_token):
        """Test get_analysis_by_id endpoint."""
        # Mock the get_analysis_by_id method
        mock_get_analysis_by_id.return_value = {
            "query": "test query",
            "contradictions": [
                {
                    "article1": {
                        "pmid": "12345",
                        "title": "Test Article 1"
                    },
                    "article2": {
                        "pmid": "67890",
                        "title": "Test Article 2"
                    },
                    "contradiction_score": 0.85,
                    "contradiction_type": "negation"
                }
            ],
            "timestamp": "2023-01-01T12:00:00",
            "user_id": 1
        }
        
        # Call the get_analysis_by_id endpoint
        response = client.get(
            "/v1/analysis/result/test-analysis-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["query"] == "test query"
        assert len(response.json()["contradictions"]) == 1
        assert response.json()["contradictions"][0]["contradiction_score"] == 0.85
        assert response.json()["timestamp"] == "2023-01-01T12:00:00"
    
    @patch.object(AnalysisService, "get_analysis_by_id")
    def test_get_analysis_by_id_endpoint_not_found(self, mock_get_analysis_by_id, auth_token):
        """Test get_analysis_by_id endpoint with analysis not found."""
        # Mock the get_analysis_by_id method to return None
        mock_get_analysis_by_id.return_value = None
        
        # Call the get_analysis_by_id endpoint
        response = client.get(
            "/v1/analysis/result/test-analysis-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_analyze_contradictions_endpoint_unauthorized(self):
        """Test analyze_contradictions endpoint without authentication."""
        # Call the analyze_contradictions endpoint without authentication
        response = client.post(
            "/v1/analysis/contradictions",
            json={
                "query": "test query",
                "max_results": 10,
                "threshold": 0.7
            }
        )
        
        # Check the response
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
