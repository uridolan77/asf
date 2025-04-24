Integration tests for the API.

This module provides integration tests for the API.

import pytest
import logging
from fastapi.testclient import TestClient

from asf.medical.api.main import app

logger = logging.getLogger(__name__)

client = TestClient(app)

TEST_USER_EMAIL = "test@example.com"
TEST_USER_PASSWORD = "testpassword"
TEST_ADMIN_EMAIL = "admin@example.com"
TEST_ADMIN_PASSWORD = "adminpassword"

@pytest.fixture
def user_token():
    """Get a user token for testing.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/auth/token",
        data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    )
    return response.json()["access_token"]

@pytest.fixture
def admin_token():
    """Get an admin token for testing.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/auth/token",
        data={"username": TEST_ADMIN_EMAIL, "password": TEST_ADMIN_PASSWORD}
    )
    return response.json()["access_token"]

def test_root():
    """Test the root endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    """Test the health endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"

def test_metrics():
    """Test the metrics endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "metrics" in response.json()

def test_auth_token():
    """Test the auth token endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/auth/token",
        data={"username": TEST_USER_EMAIL, "password": TEST_USER_PASSWORD}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert "token_type" in response.json()
    assert "role" in response.json()
    assert "expires_in" in response.json()

def test_auth_token_invalid():
    """Test the auth token endpoint with invalid credentials.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/auth/token",
        data={"username": TEST_USER_EMAIL, "password": "wrongpassword"}
    )
    assert response.status_code == 401

def test_search(user_token):
    """Test the search endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/search",
        json={"query": "statin therapy", "max_results": 5},
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "results" in response.json()
    assert "total" in response.json()
    assert "query" in response.json()

def test_search_pico(user_token):
    """Test the PICO search endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/search/pico",
        json={
            "population": "adults with high cholesterol",
            "intervention": "statin therapy",
            "comparison": "placebo",
            "outcome": "cardiovascular events",
            "max_results": 5
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "results" in response.json()
    assert "total" in response.json()
    assert "query" in response.json()

def test_enhanced_contradiction_detect(user_token):
    """Test the enhanced contradiction detection endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/enhanced-contradiction/detect",
        json={
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "use_biomedlm": True,
            "use_tsmixer": True,
            "use_lorentz": True,
            "threshold": 0.5
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "is_contradiction" in response.json()
    assert "contradiction_score" in response.json()
    assert "contradiction_type" in response.json()
    assert "confidence" in response.json()
    assert "classification" in response.json()

def test_enhanced_contradiction_analyze(user_token):
    """Test the enhanced contradiction analysis endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/enhanced-contradiction/analyze",
        json={
            "claims": [
                {
                    "text": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
                    "metadata": {
                        "publication_year": 2020,
                        "study_design": "randomized controlled trial",
                        "sample_size": 5000
                    }
                },
                {
                    "text": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
                    "metadata": {
                        "publication_year": 2015,
                        "study_design": "observational study",
                        "sample_size": 1000
                    }
                }
            ],
            "threshold": 0.5,
            "use_biomedlm": True,
            "use_tsmixer": True,
            "use_lorentz": True
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "total_claims" in response.json()
    assert "total_contradictions" in response.json()
    assert "contradictions" in response.json()

def test_contradiction_resolution_resolve(user_token):
    """Test the contradiction resolution endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/contradiction-resolution/resolve",
        json={
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "metadata1": {
                "publication_year": 2020,
                "study_design": "randomized controlled trial",
                "sample_size": 5000
            },
            "metadata2": {
                "publication_year": 2015,
                "study_design": "observational study",
                "sample_size": 1000
            },
            "strategy": "evidence_hierarchy",
            "use_combined_evidence": False
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "recommendation" in response.json()
    assert "confidence" in response.json()
    assert "confidence_score" in response.json()
    assert "recommended_claim" in response.json()
    assert "strategy" in response.json()
    assert "explanation" in response.json()
    assert "timestamp" in response.json()

def test_contradiction_resolution_combined(user_token):
    """Test the contradiction resolution with combined evidence endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/contradiction-resolution/resolve",
        json={
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "metadata1": {
                "publication_year": 2020,
                "study_design": "randomized controlled trial",
                "sample_size": 5000
            },
            "metadata2": {
                "publication_year": 2015,
                "study_design": "observational study",
                "sample_size": 1000
            },
            "use_combined_evidence": True
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "recommendation" in response.json()
    assert "confidence" in response.json()
    assert "confidence_score" in response.json()
    assert "recommended_claim" in response.json()
    assert "strategy" in response.json()
    assert "explanation" in response.json()
    assert "timestamp" in response.json()

def test_contradiction_resolution_history(user_token):
    """Test the contradiction resolution history endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.get(
        "/v1/contradiction-resolution/history",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "total_entries" in response.json()
    assert "history" in response.json()

def test_screening(user_token):
    """Test the screening endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/screening",
        json={
            "title": "Effects of statin therapy on cardiovascular events",
            "abstract": "Background: Statin therapy is widely used for primary and secondary prevention of cardiovascular disease. However, the effects of statin therapy on cardiovascular events in different populations remain controversial. Methods: We conducted a systematic review and meta-analysis of randomized controlled trials to evaluate the effects of statin therapy on cardiovascular events. Results: Statin therapy significantly reduced the risk of cardiovascular events in patients with high cholesterol. Conclusion: Statin therapy is effective for reducing cardiovascular events in patients with high cholesterol.",
            "stage": "title_abstract"
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "decision" in response.json()
    assert "confidence" in response.json()
    assert "explanation" in response.json()

def test_bias_assessment(user_token):
    """Test the bias assessment endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/screening/bias-assessment",
        json={
            "title": "Effects of statin therapy on cardiovascular events",
            "abstract": "Background: Statin therapy is widely used for primary and secondary prevention of cardiovascular disease. However, the effects of statin therapy on cardiovascular events in different populations remain controversial. Methods: We conducted a randomized, double-blind, placebo-controlled trial to evaluate the effects of statin therapy on cardiovascular events. Results: Statin therapy significantly reduced the risk of cardiovascular events in patients with high cholesterol. Conclusion: Statin therapy is effective for reducing cardiovascular events in patients with high cholesterol.",
            "full_text": "...",
            "domains": ["randomization", "blinding", "allocation_concealment", "sample_size", "attrition"]
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "overall_risk" in response.json()
    assert "domain_assessments" in response.json()
    assert "explanation" in response.json()

def test_export(user_token):
    """Test the export endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/export",
        json={
            "query_id": "123",
            "format": "json"
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code in [200, 404]  # 404 if query doesn't exist

def test_analysis(user_token):
    """Test the analysis endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.post(
        "/v1/analysis/contradictions",
        json={
            "query": "statin therapy cardiovascular",
            "max_results": 10,
            "threshold": 0.5,
            "use_biomedlm": True,
            "use_tsmixer": True,
            "use_lorentz": True
        },
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "query" in response.json()
    assert "total_articles" in response.json()
    assert "total_contradictions" in response.json()
    assert "contradictions" in response.json()

def test_knowledge_base(user_token):
    """Test the knowledge base endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.get(
        "/v1/knowledge-base",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    assert "knowledge_bases" in response.json()

def test_admin_endpoint(admin_token):
    """Test an admin endpoint.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    response = client.get(
        "/v1/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code in [200, 403, 404]  # 403 if not admin, 404 if endpoint doesn't exist
