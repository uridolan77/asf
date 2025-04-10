"""
Integration tests for the contradiction API.

This module provides integration tests for the contradiction API endpoints.
"""

import pytest
import logging
from typing import Dict, Any
from fastapi.testclient import TestClient

from asf.medical.api.main_enhanced import app
from asf.medical.ml.services.enhanced_contradiction_service import ContradictionType, ContradictionConfidence

# Configure logging
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

@pytest.mark.integration
@pytest.mark.api
class TestContradictionAPI:
    """Test cases for contradiction API."""

    def test_analyze_contradictions_endpoint(self):
        """Test analyze contradictions endpoint."""
        # Test data
        data = {
            "query": "statin therapy cardiovascular",
            "max_results": 5,
            "threshold": 0.7,
            "use_all_methods": True
        }
        
        # Make request
        response = client.post("/api/v1/contradiction/analyze", json=data)
        
        # Assertions
        assert response.status_code == 200
        assert "query" in response.json()
        assert "total_articles" in response.json()
        assert "contradictions_found" in response.json()
        assert "contradiction_types" in response.json()
        assert "contradictions" in response.json()
        assert "analysis_id" in response.json()
        assert response.json()["query"] == data["query"]
        assert isinstance(response.json()["total_articles"], int)
        assert isinstance(response.json()["contradictions_found"], int)
        assert isinstance(response.json()["contradiction_types"], dict)
        assert isinstance(response.json()["contradictions"], list)
        assert isinstance(response.json()["analysis_id"], str)
    
    def test_detect_contradiction_endpoint(self):
        """Test detect contradiction endpoint."""
        # Test data
        data = {
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "metadata1": {
                "publication_date": "2020-01-01",
                "study_design": "randomized controlled trial",
                "sample_size": 1000,
                "p_value": 0.001
            },
            "metadata2": {
                "publication_date": "2021-06-15",
                "study_design": "randomized controlled trial",
                "sample_size": 2000,
                "p_value": 0.45
            },
            "use_all_methods": True
        }
        
        # Make request
        response = client.post("/api/v1/contradiction/detect", json=data)
        
        # Assertions
        assert response.status_code == 200
        assert "claim1" in response.json()
        assert "claim2" in response.json()
        assert "is_contradiction" in response.json()
        assert "contradiction_score" in response.json()
        assert "contradiction_type" in response.json()
        assert "confidence" in response.json()
        assert "explanation" in response.json()
        assert "methods_used" in response.json()
        assert "details" in response.json()
        assert response.json()["claim1"] == data["claim1"]
        assert response.json()["claim2"] == data["claim2"]
        assert isinstance(response.json()["is_contradiction"], bool)
        assert isinstance(response.json()["contradiction_score"], float)
        assert response.json()["contradiction_type"] in [ct.value for ct in ContradictionType]
        assert response.json()["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(response.json()["explanation"], str)
        assert isinstance(response.json()["methods_used"], list)
        assert isinstance(response.json()["details"], dict)
