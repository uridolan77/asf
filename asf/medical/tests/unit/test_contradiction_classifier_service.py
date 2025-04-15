"""Unit tests for the contradiction classifier service."""

import pytest

from ...ml.services.contradiction_service import (
    ContradictionClassifierService,
    ContradictionType,
    ContradictionConfidence,
    ClinicalSignificance,
    EvidenceQuality
)


@pytest.fixture
def classifier_service():
    """Create a contradiction classifier service for testing."""
    return ContradictionClassifierService()


@pytest.fixture
def sample_contradiction():
    """Create a sample contradiction for testing."""
    return {
        "claim1": "Aspirin reduces the risk of heart attack.",
        "claim2": "Aspirin does not significantly affect heart attack risk.",
        "metadata1": {
            "publication_date": "2020-01-01T00:00:00Z",
            "study_type": "randomized controlled trial",
            "sample_size": 1000,
            "population": "adults with cardiovascular risk factors",
            "age_range": "45-75",
            "gender": "mixed",
            "ethnicity": "diverse",
            "location": "United States",
            "intervention": "daily aspirin 81mg",
            "outcome": "heart attack incidence",
            "statistical_method": "Cox proportional hazards",
            "journal_impact_factor": 12.5
        },
        "metadata2": {
            "publication_date": "2010-01-01T00:00:00Z",
            "study_type": "observational study",
            "sample_size": 500,
            "population": "healthy adults",
            "age_range": "30-60",
            "gender": "male",
            "ethnicity": "caucasian",
            "location": "Europe",
            "intervention": "daily aspirin 325mg",
            "outcome": "cardiovascular events",
            "statistical_method": "logistic regression",
            "journal_impact_factor": 5.2
        }
    }


async def test_assess_clinical_significance(classifier_service, sample_contradiction):
    """Test assessing clinical significance."""
    result = await classifier_service._assess_clinical_significance(
        claim1=sample_contradiction["claim1"],
        claim2=sample_contradiction["claim2"],
        metadata1=sample_contradiction["metadata1"],
        metadata2=sample_contradiction["metadata2"]
    )
    
    assert isinstance(result, dict)
    assert "significance" in result
    assert "score" in result
    assert "terms" in result
    assert isinstance(result["terms"], list)


def test_assess_evidence_quality(classifier_service, sample_contradiction):
    """Test assessing evidence quality."""
    result = classifier_service._assess_evidence_quality(sample_contradiction["metadata1"])
    
    assert isinstance(result, dict)
    assert "quality" in result
    assert "score" in result
    assert "factors" in result
    assert isinstance(result["factors"], list)


def test_assess_temporal_factor(classifier_service, sample_contradiction):
    """Test assessing temporal factor."""
    result = classifier_service._assess_temporal_factor(
        metadata1=sample_contradiction["metadata1"],
        metadata2=sample_contradiction["metadata2"]
    )
    
    assert isinstance(result, dict)
    assert "detected" in result
    assert "score" in result
    assert "factors" in result
    assert isinstance(result["factors"], list)


def test_assess_population_difference(classifier_service, sample_contradiction):
    """Test assessing population difference."""
    result = classifier_service._assess_population_difference(
        claim1=sample_contradiction["claim1"],
        claim2=sample_contradiction["claim2"],
        metadata1=sample_contradiction["metadata1"],
        metadata2=sample_contradiction["metadata2"]
    )
    
    assert isinstance(result, dict)
    assert "detected" in result
    assert "score" in result
    assert "differences" in result
    assert isinstance(result["differences"], list)


def test_assess_methodological_difference(classifier_service, sample_contradiction):
    """Test assessing methodological difference."""
    result = classifier_service._assess_methodological_difference(
        claim1=sample_contradiction["claim1"],
        claim2=sample_contradiction["claim2"],
        metadata1=sample_contradiction["metadata1"],
        metadata2=sample_contradiction["metadata2"]
    )
    
    assert isinstance(result, dict)
    assert "detected" in result
    assert "score" in result
    assert "differences" in result
    assert isinstance(result["differences"], list)


async def test_classify_contradiction(classifier_service, sample_contradiction):
    """Test classifying a contradiction."""
    result = await classifier_service.classify_contradiction(
        claim1=sample_contradiction["claim1"],
        claim2=sample_contradiction["claim2"],
        metadata1=sample_contradiction["metadata1"],
        metadata2=sample_contradiction["metadata2"]
    )
    
    assert isinstance(result, dict)
    assert "is_contradiction" in result
    assert "contradiction_type" in result
    assert "confidence" in result
    assert "score" in result
    assert "explanation" in result
    assert "evidence" in result
    assert "analysis" in result
