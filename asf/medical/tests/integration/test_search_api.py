"""
Integration tests for the search API.

This module provides integration tests for the search API endpoints.
"""

import pytest
import logging
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock, AsyncMock

from fastapi.testclient import TestClient

from asf.medical.api.main import app
from asf.medical.services.search_service import SearchService, SearchMethod

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
class TestSearchAPI:
    """Test cases for search API."""
    
    @patch.object(SearchService, "search")
    def test_search_endpoint(self, mock_search, auth_token):
        """Test search endpoint."""
        # Mock the search method
        mock_search.return_value = {
            "query": "test query",
            "results": [
                {
                    "pmid": "12345",
                    "title": "Test Article 1",
                    "abstract": "This is a test abstract for article 1."
                },
                {
                    "pmid": "67890",
                    "title": "Test Article 2",
                    "abstract": "This is a test abstract for article 2."
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_pages": 1,
                "total_results": 2
            }
        }
        
        # Call the search endpoint
        response = client.get(
            "/v1/search",
            params={"query": "test query", "max_results": 10},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["query"] == "test query"
        assert len(response.json()["results"]) == 2
        assert response.json()["results"][0]["pmid"] == "12345"
        assert response.json()["results"][1]["pmid"] == "67890"
        assert response.json()["pagination"]["page"] == 1
        assert response.json()["pagination"]["total_results"] == 2
    
    @patch.object(SearchService, "search")
    def test_search_endpoint_with_pagination(self, mock_search, auth_token):
        """Test search endpoint with pagination."""
        # Mock the search method
        mock_search.return_value = {
            "query": "test query",
            "results": [
                {
                    "pmid": "67890",
                    "title": "Test Article 2",
                    "abstract": "This is a test abstract for article 2."
                }
            ],
            "pagination": {
                "page": 2,
                "page_size": 1,
                "total_pages": 2,
                "total_results": 2
            }
        }
        
        # Call the search endpoint with pagination
        response = client.get(
            "/v1/search",
            params={"query": "test query", "max_results": 10, "page": 2, "page_size": 1},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["pagination"]["page"] == 2
        assert response.json()["pagination"]["page_size"] == 1
        assert len(response.json()["results"]) == 1
        assert response.json()["results"][0]["pmid"] == "67890"
    
    @patch.object(SearchService, "search")
    def test_search_endpoint_with_search_method(self, mock_search, auth_token):
        """Test search endpoint with search method."""
        # Mock the search method
        mock_search.return_value = {
            "query": "test query",
            "results": [
                {
                    "nct_id": "NCT12345",
                    "title": "Test Clinical Trial 1",
                    "summary": "This is a test summary for clinical trial 1."
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_pages": 1,
                "total_results": 1
            }
        }
        
        # Call the search endpoint with clinical_trials search method
        response = client.get(
            "/v1/search",
            params={"query": "test query", "max_results": 10, "search_method": "clinical_trials"},
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["results"][0]["nct_id"] == "NCT12345"
        
        # Check that the search method was passed correctly
        mock_search.assert_called_with(
            query="test query",
            max_results=10,
            page=1,
            page_size=20,
            user_id=None,  # This would be set in a real request
            search_method=SearchMethod.CLINICAL_TRIALS,
            use_graph_rag=False,
            use_vector_search=True,
            use_graph_search=True
        )
    
    @patch.object(SearchService, "search")
    def test_search_endpoint_with_graph_rag(self, mock_search, auth_token):
        """Test search endpoint with GraphRAG."""
        # Mock the search method
        mock_search.return_value = {
            "query": "test query",
            "results": [
                {
                    "pmid": "12345",
                    "title": "Test Article 1",
                    "abstract": "This is a test abstract for article 1.",
                    "relevance_score": 0.95
                }
            ],
            "pagination": {
                "page": 1,
                "page_size": 20,
                "total_pages": 1,
                "total_results": 1
            },
            "search_method": "graph_rag"
        }
        
        # Call the search endpoint with GraphRAG
        response = client.get(
            "/v1/search",
            params={
                "query": "test query",
                "max_results": 10,
                "use_graph_rag": "true",
                "use_vector_search": "true",
                "use_graph_search": "true"
            },
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["results"][0]["pmid"] == "12345"
        assert response.json()["results"][0]["relevance_score"] == 0.95
        
        # Check that the GraphRAG parameters were passed correctly
        mock_search.assert_called_with(
            query="test query",
            max_results=10,
            page=1,
            page_size=20,
            user_id=None,  # This would be set in a real request
            search_method=SearchMethod.GRAPH_RAG,
            use_graph_rag=True,
            use_vector_search=True,
            use_graph_search=True
        )
    
    @patch.object(SearchService, "get_result_by_id")
    def test_get_search_result_by_id(self, mock_get_result_by_id, auth_token):
        """Test get_search_result_by_id endpoint."""
        # Mock the get_result_by_id method
        mock_get_result_by_id.return_value = {
            "query": "test query",
            "results": [
                {
                    "pmid": "12345",
                    "title": "Test Article 1",
                    "abstract": "This is a test abstract for article 1."
                }
            ],
            "timestamp": "2023-01-01T12:00:00",
            "user_id": 1
        }
        
        # Call the get_search_result_by_id endpoint
        response = client.get(
            "/v1/search/result/test-result-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 200
        assert response.json()["query"] == "test query"
        assert len(response.json()["results"]) == 1
        assert response.json()["results"][0]["pmid"] == "12345"
        assert response.json()["timestamp"] == "2023-01-01T12:00:00"
    
    @patch.object(SearchService, "get_result_by_id")
    def test_get_search_result_by_id_not_found(self, mock_get_result_by_id, auth_token):
        """Test get_search_result_by_id endpoint with result not found."""
        # Mock the get_result_by_id method to return None
        mock_get_result_by_id.return_value = None
        
        # Call the get_search_result_by_id endpoint
        response = client.get(
            "/v1/search/result/test-result-id",
            headers={"Authorization": f"Bearer {auth_token}"}
        )
        
        # Check the response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]
    
    def test_search_endpoint_unauthorized(self):
        """Test search endpoint without authentication."""
        # Call the search endpoint without authentication
        response = client.get(
            "/v1/search",
            params={"query": "test query", "max_results": 10}
        )
        
        # Check the response
        assert response.status_code == 401
        assert "Not authenticated" in response.json()["detail"]
