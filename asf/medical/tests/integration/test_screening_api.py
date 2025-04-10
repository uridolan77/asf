"""
Integration tests for the screening API.

This module provides integration tests for the screening API endpoints.
"""

import pytest
import logging
from typing import Dict, Any
from fastapi.testclient import TestClient

from asf.medical.api.main_enhanced import app
from asf.medical.ml.services.prisma_screening_service import ScreeningStage
from asf.medical.ml.services.bias_assessment_service import BiasDomain, BiasRisk

# Configure logging
logger = logging.getLogger(__name__)

# Create test client
client = TestClient(app)

@pytest.mark.integration
@pytest.mark.api
class TestScreeningAPI:
    """Test cases for screening API."""

    def test_prisma_screening_endpoint(self):
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
        response = client.post("/api/v1/screening/prisma", json=data)
        
        # Assertions
        assert response.status_code == 200
        assert "query" in response.json()
        assert "stage" in response.json()
        assert "total_articles" in response.json()
        assert "included" in response.json()
        assert "excluded" in response.json()
        assert "uncertain" in response.json()
        assert "results" in response.json()
        assert "flow_data" in response.json()
        assert response.json()["query"] == data["query"]
        assert response.json()["stage"] == data["stage"]
        assert isinstance(response.json()["total_articles"], int)
        assert isinstance(response.json()["included"], int)
        assert isinstance(response.json()["excluded"], int)
        assert isinstance(response.json()["uncertain"], int)
        assert isinstance(response.json()["results"], list)
        assert isinstance(response.json()["flow_data"], dict)
    
    def test_bias_assessment_endpoint(self):
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
        response = client.post("/api/v1/screening/bias-assessment", json=data)
        
        # Assertions
        assert response.status_code == 200
        assert "query" in response.json()
        assert "total_articles" in response.json()
        assert "low_risk" in response.json()
        assert "moderate_risk" in response.json()
        assert "high_risk" in response.json()
        assert "unclear_risk" in response.json()
        assert "results" in response.json()
        assert response.json()["query"] == data["query"]
        assert isinstance(response.json()["total_articles"], int)
        assert isinstance(response.json()["low_risk"], int)
        assert isinstance(response.json()["moderate_risk"], int)
        assert isinstance(response.json()["high_risk"], int)
        assert isinstance(response.json()["unclear_risk"], int)
        assert isinstance(response.json()["results"], list)
    
    def test_flow_diagram_endpoint(self):
        """Test flow diagram endpoint."""
        # Make request
        response = client.get("/api/v1/screening/flow-diagram")
        
        # Assertions
        assert response.status_code == 200
        assert "identification" in response.json()
        assert "screening" in response.json()
        assert "eligibility" in response.json()
        assert "included" in response.json()
        assert "records_identified" in response.json()["identification"]
        assert "records_removed" in response.json()["identification"]
        assert "records_remaining" in response.json()["identification"]
        assert "records_screened" in response.json()["screening"]
        assert "records_excluded" in response.json()["screening"]
        assert "records_remaining" in response.json()["screening"]
        assert "full_text_assessed" in response.json()["eligibility"]
        assert "full_text_excluded" in response.json()["eligibility"]
        assert "exclusion_reasons" in response.json()["eligibility"]
        assert "studies_included" in response.json()["included"]
