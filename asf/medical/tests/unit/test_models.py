"""
Unit tests for Pydantic models.

This module provides unit tests for Pydantic models to ensure data validation rules are effective.
"""

import pytest
from pydantic import ValidationError
from typing import Dict, Any, List, Optional

from asf.medical.api.models.user import UserCreate, UserUpdate, UserInDB
from asf.medical.api.models.token import Token, TokenPayload
from asf.medical.api.models.study import StudyCreate, StudyUpdate, StudyInDB
from asf.medical.api.models.contradiction import ContradictionRequest, ContradictionResponse
from asf.medical.api.models.screening import ScreeningRequest, ScreeningResponse
from asf.medical.api.models.bias import BiasAssessmentRequest, BiasAssessmentResponse

@pytest.mark.unit
class TestUserModels:
    """Test cases for user models."""
    
    def test_user_create_valid(self):
        """Test valid UserCreate model."""
        user = UserCreate(
            email="test@example.com",
            password="password123",
            full_name="Test User",
        )
        assert user.email == "test@example.com"
        assert user.password == "password123"
        assert user.full_name == "Test User"
    
    def test_user_create_invalid_email(self):
        """Test invalid email in UserCreate model."""
        with pytest.raises(ValidationError):
            UserCreate(
                email="invalid-email",
                password="password123",
                full_name="Test User",
            )
    
    def test_user_create_short_password(self):
        """Test short password in UserCreate model."""
        with pytest.raises(ValidationError):
            UserCreate(
                email="test@example.com",
                password="short",
                full_name="Test User",
            )
    
    def test_user_update_valid(self):
        """Test valid UserUpdate model."""
        user = UserUpdate(
            full_name="Updated User",
        )
        assert user.full_name == "Updated User"
        assert user.password is None
    
    def test_user_in_db_valid(self):
        """Test valid UserInDB model."""
        user = UserInDB(
            id=1,
            email="test@example.com",
            hashed_password="hashed_password",
            full_name="Test User",
            is_active=True,
            is_superuser=False,
        )
        assert user.id == 1
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed_password"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.is_superuser is False


@pytest.mark.unit
class TestTokenModels:
    """Test cases for token models."""
    
    def test_token_valid(self):
        """Test valid Token model."""
        token = Token(
            access_token="access_token",
            token_type="bearer",
        )
        assert token.access_token == "access_token"
        assert token.token_type == "bearer"
    
    def test_token_payload_valid(self):
        """Test valid TokenPayload model."""
        payload = TokenPayload(
            sub="test@example.com",
            exp=1234567890,
        )
        assert payload.sub == "test@example.com"
        assert payload.exp == 1234567890


@pytest.mark.unit
class TestStudyModels:
    """Test cases for study models."""
    
    def test_study_create_valid(self):
        """Test valid StudyCreate model."""
        study = StudyCreate(
            title="Test Study",
            abstract="This is a test study.",
            authors=["Author 1", "Author 2"],
            publication_date="2023-01-01",
            journal="Test Journal",
            doi="10.1234/test",
            pmid="12345678",
            url="https://example.com/study",
        )
        assert study.title == "Test Study"
        assert study.abstract == "This is a test study."
        assert study.authors == ["Author 1", "Author 2"]
        assert study.publication_date == "2023-01-01"
        assert study.journal == "Test Journal"
        assert study.doi == "10.1234/test"
        assert study.pmid == "12345678"
        assert study.url == "https://example.com/study"
    
    def test_study_create_invalid_url(self):
        """Test invalid URL in StudyCreate model."""
        with pytest.raises(ValidationError):
            StudyCreate(
                title="Test Study",
                abstract="This is a test study.",
                authors=["Author 1", "Author 2"],
                publication_date="2023-01-01",
                journal="Test Journal",
                doi="10.1234/test",
                pmid="12345678",
                url="invalid-url",
            )
    
    def test_study_update_valid(self):
        """Test valid StudyUpdate model."""
        study = StudyUpdate(
            title="Updated Study",
        )
        assert study.title == "Updated Study"
        assert study.abstract is None
    
    def test_study_in_db_valid(self):
        """Test valid StudyInDB model."""
        study = StudyInDB(
            id=1,
            title="Test Study",
            abstract="This is a test study.",
            authors=["Author 1", "Author 2"],
            publication_date="2023-01-01",
            journal="Test Journal",
            doi="10.1234/test",
            pmid="12345678",
            url="https://example.com/study",
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T00:00:00",
            owner_id=1,
        )
        assert study.id == 1
        assert study.title == "Test Study"
        assert study.abstract == "This is a test study."
        assert study.authors == ["Author 1", "Author 2"]
        assert study.publication_date == "2023-01-01"
        assert study.journal == "Test Journal"
        assert study.doi == "10.1234/test"
        assert study.pmid == "12345678"
        assert study.url == "https://example.com/study"
        assert study.created_at == "2023-01-01T00:00:00"
        assert study.updated_at == "2023-01-01T00:00:00"
        assert study.owner_id == 1


@pytest.mark.unit
class TestContradictionModels:
    """Test cases for contradiction models."""
    
    def test_contradiction_request_valid(self):
        """Test valid ContradictionRequest model."""
        request = ContradictionRequest(
            claim1="Claim 1",
            claim2="Claim 2",
            context="Context",
        )
        assert request.claim1 == "Claim 1"
        assert request.claim2 == "Claim 2"
        assert request.context == "Context"
    
    def test_contradiction_request_empty_claims(self):
        """Test empty claims in ContradictionRequest model."""
        with pytest.raises(ValidationError):
            ContradictionRequest(
                claim1="",
                claim2="Claim 2",
                context="Context",
            )
        
        with pytest.raises(ValidationError):
            ContradictionRequest(
                claim1="Claim 1",
                claim2="",
                context="Context",
            )
    
    def test_contradiction_response_valid(self):
        """Test valid ContradictionResponse model."""
        response = ContradictionResponse(
            claim1="Claim 1",
            claim2="Claim 2",
            contradiction_score=0.8,
            contradiction_type="semantic",
            confidence="high",
            explanation="Explanation",
            evidence=[
                {"text": "Evidence 1", "score": 0.9},
                {"text": "Evidence 2", "score": 0.8},
            ],
        )
        assert response.claim1 == "Claim 1"
        assert response.claim2 == "Claim 2"
        assert response.contradiction_score == 0.8
        assert response.contradiction_type == "semantic"
        assert response.confidence == "high"
        assert response.explanation == "Explanation"
        assert len(response.evidence) == 2
        assert response.evidence[0]["text"] == "Evidence 1"
        assert response.evidence[0]["score"] == 0.9


@pytest.mark.unit
class TestScreeningModels:
    """Test cases for screening models."""
    
    def test_screening_request_valid(self):
        """Test valid ScreeningRequest model."""
        request = ScreeningRequest(
            title="Test Study",
            abstract="This is a test study.",
            full_text="This is the full text of the test study.",
            stage="title_abstract",
        )
        assert request.title == "Test Study"
        assert request.abstract == "This is a test study."
        assert request.full_text == "This is the full text of the test study."
        assert request.stage == "title_abstract"
    
    def test_screening_request_invalid_stage(self):
        """Test invalid stage in ScreeningRequest model."""
        with pytest.raises(ValidationError):
            ScreeningRequest(
                title="Test Study",
                abstract="This is a test study.",
                full_text="This is the full text of the test study.",
                stage="invalid_stage",
            )
    
    def test_screening_response_valid(self):
        """Test valid ScreeningResponse model."""
        response = ScreeningResponse(
            decision="include",
            confidence=0.8,
            reasons=["Reason 1", "Reason 2"],
            stage="title_abstract",
        )
        assert response.decision == "include"
        assert response.confidence == 0.8
        assert response.reasons == ["Reason 1", "Reason 2"]
        assert response.stage == "title_abstract"


@pytest.mark.unit
class TestBiasModels:
    """Test cases for bias assessment models."""
    
    def test_bias_assessment_request_valid(self):
        """Test valid BiasAssessmentRequest model."""
        request = BiasAssessmentRequest(
            study_text="This is a randomized controlled trial...",
            domains=["randomization", "blinding", "allocation_concealment"],
        )
        assert request.study_text == "This is a randomized controlled trial..."
        assert request.domains == ["randomization", "blinding", "allocation_concealment"]
    
    def test_bias_assessment_request_invalid_domain(self):
        """Test invalid domain in BiasAssessmentRequest model."""
        with pytest.raises(ValidationError):
            BiasAssessmentRequest(
                study_text="This is a randomized controlled trial...",
                domains=["randomization", "invalid_domain"],
            )
    
    def test_bias_assessment_response_valid(self):
        """Test valid BiasAssessmentResponse model."""
        response = BiasAssessmentResponse(
            domains={
                "randomization": {
                    "risk": "low",
                    "evidence": [
                        {"context": "randomized controlled trial", "score": 0.9},
                    ],
                    "explanation": "The study used proper randomization.",
                },
                "blinding": {
                    "risk": "high",
                    "evidence": [
                        {"context": "no blinding was used", "score": 0.8},
                    ],
                    "explanation": "The study did not use blinding.",
                },
            },
            overall={
                "risk": "moderate",
                "explanation": "The study has low risk of bias in randomization but high risk in blinding.",
            },
        )
        assert response.domains["randomization"]["risk"] == "low"
        assert response.domains["blinding"]["risk"] == "high"
        assert response.overall["risk"] == "moderate"
