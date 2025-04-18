"""
Tests for the Medical Clinical Data Service integration.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.services.medical_clinical_data_service import MedicalClinicalDataService
from api.endpoints import app

client = TestClient(app)

# Mock authentication for tests
@pytest.fixture
def mock_auth():
    with patch("api.dependencies.get_current_user") as mock:
        mock.return_value = MagicMock(id=1, username="testuser")
        yield mock

# Mock the clinical data service
@pytest.fixture
def mock_clinical_data_service():
    with patch("api.routers.medical_clinical_data.get_medical_clinical_data_service") as mock:
        service_mock = MagicMock(spec=MedicalClinicalDataService)
        mock.return_value = service_mock
        yield service_mock

def test_search_concept_and_trials(mock_auth, mock_clinical_data_service):
    """Test the search_concept_and_trials endpoint."""
    # Setup mock response
    mock_clinical_data_service.search_concept_and_trials.return_value = {
        "success": True,
        "message": "Found 5 trials for term: diabetes",
        "data": {
            "term": "diabetes",
            "concepts": [
                {"conceptId": "73211009", "preferredTerm": "Diabetes mellitus"}
            ],
            "trials": [
                {"NCTId": "NCT01234567", "BriefTitle": "Diabetes Treatment Study"}
            ]
        }
    }
    
    # Make request
    response = client.get("/api/medical/clinical-data/search?term=diabetes&max_trials=5")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "Found 5 trials" in response.json()["message"]
    assert response.json()["data"]["term"] == "diabetes"
    
    # Verify service was called correctly
    mock_clinical_data_service.search_concept_and_trials.assert_called_once_with("diabetes", max_trials=5)

def test_get_trials_by_concept(mock_auth, mock_clinical_data_service):
    """Test the get_trials_by_concept endpoint."""
    # Setup mock response
    mock_clinical_data_service.search_by_concept_id.return_value = {
        "success": True,
        "message": "Found 3 trials for concept ID: 73211009",
        "data": {
            "concept": {"conceptId": "73211009", "preferredTerm": "Diabetes mellitus"},
            "trials": [
                {"NCTId": "NCT01234567", "BriefTitle": "Diabetes Treatment Study"}
            ]
        }
    }
    
    # Make request
    response = client.get("/api/medical/clinical-data/concept/73211009/trials?terminology=SNOMEDCT&max_trials=3")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "Found 3 trials" in response.json()["message"]
    assert response.json()["data"]["concept"]["conceptId"] == "73211009"
    
    # Verify service was called correctly
    mock_clinical_data_service.search_by_concept_id.assert_called_once_with(
        "73211009", terminology="SNOMEDCT", max_trials=3
    )

def test_map_trial_conditions(mock_auth, mock_clinical_data_service):
    """Test the map_trial_conditions endpoint."""
    # Setup mock response
    mock_clinical_data_service.map_trial_conditions.return_value = {
        "success": True,
        "message": "Mapped conditions for trial NCT01234567 to SNOMED CT concepts",
        "data": {
            "nct_id": "NCT01234567",
            "conditions": {
                "Diabetes": [
                    {"conceptId": "73211009", "preferredTerm": "Diabetes mellitus"}
                ]
            }
        }
    }
    
    # Make request
    response = client.get("/api/medical/clinical-data/trial/NCT01234567/mapping")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "Mapped conditions" in response.json()["message"]
    assert response.json()["data"]["nct_id"] == "NCT01234567"
    
    # Verify service was called correctly
    mock_clinical_data_service.map_trial_conditions.assert_called_once_with("NCT01234567")

def test_find_trials_with_semantic_expansion(mock_auth, mock_clinical_data_service):
    """Test the find_trials_with_semantic_expansion endpoint."""
    # Setup mock response
    mock_clinical_data_service.find_trials_with_semantic_expansion.return_value = {
        "success": True,
        "message": "Found 7 trials with semantic expansion for term: heart attack",
        "data": {
            "normalized_term": "Myocardial infarction",
            "search_terms_used": ["Myocardial infarction", "Cardiac infarction", "Heart attack"],
            "trials": [
                {"NCTId": "NCT01234567", "BriefTitle": "Heart Attack Treatment Study"}
            ]
        }
    }
    
    # Make request
    response = client.get("/api/medical/clinical-data/semantic-search?term=heart%20attack&include_similar=true&max_trials=10")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "Found 7 trials" in response.json()["message"]
    assert response.json()["data"]["normalized_term"] == "Myocardial infarction"
    
    # Verify service was called correctly
    mock_clinical_data_service.find_trials_with_semantic_expansion.assert_called_once_with(
        "heart attack", include_similar=True, max_trials=10
    )

def test_get_trial_semantic_context(mock_auth, mock_clinical_data_service):
    """Test the get_trial_semantic_context endpoint."""
    # Setup mock response
    mock_clinical_data_service.get_trial_semantic_context.return_value = {
        "success": True,
        "message": "Retrieved semantic context for trial NCT01234567",
        "data": {
            "nct_id": "NCT01234567",
            "study_title": "Heart Attack Treatment Study",
            "condition_mappings": {
                "Heart Attack": [
                    {"conceptId": "22298006", "preferredTerm": "Myocardial infarction"}
                ]
            },
            "intervention_mappings": {
                "Aspirin": [
                    {"conceptId": "387458008", "preferredTerm": "Aspirin"}
                ]
            }
        }
    }
    
    # Make request
    response = client.get("/api/medical/clinical-data/trial/NCT01234567/semantic-context")
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "Retrieved semantic context" in response.json()["message"]
    assert response.json()["data"]["nct_id"] == "NCT01234567"
    
    # Verify service was called correctly
    mock_clinical_data_service.get_trial_semantic_context.assert_called_once_with("NCT01234567")

def test_analyze_trial_phases_by_concept(mock_auth, mock_clinical_data_service):
    """Test the analyze_trial_phases_by_concept endpoint."""
    # Setup mock response
    mock_clinical_data_service.analyze_trial_phases_by_concept.return_value = {
        "success": True,
        "message": "Analyzed trial phases for concept ID: 73211009",
        "data": {
            "concept": {"conceptId": "73211009", "preferredTerm": "Diabetes mellitus"},
            "phase_distribution": {
                "Phase 1": 15,
                "Phase 2": 28,
                "Phase 3": 42,
                "Phase 4": 10,
                "Not Applicable": 5
            },
            "total_trials": 100
        }
    }
    
    # Make request
    response = client.get(
        "/api/medical/clinical-data/concept/73211009/phase-analysis?terminology=SNOMEDCT&include_descendants=true&max_results=500"
    )
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    assert "Analyzed trial phases" in response.json()["message"]
    assert response.json()["data"]["concept"]["conceptId"] == "73211009"
    
    # Verify service was called correctly
    mock_clinical_data_service.analyze_trial_phases_by_concept.assert_called_once_with(
        "73211009", terminology="SNOMEDCT", include_descendants=True, max_results=500
    )
