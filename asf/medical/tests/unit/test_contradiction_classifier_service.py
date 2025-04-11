Unit tests for the contradiction classifier service.

import pytest

from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionClassifierService,
    ContradictionType,
    ContradictionConfidence,
    ClinicalSignificance,
    EvidenceQuality
)

@pytest.fixture
def classifier_service():
    Create a contradiction classifier service for testing.
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
    assert isinstance(result["significance"], ClinicalSignificance)
    assert isinstance(result["score"], float)
    assert isinstance(result["terms"], list)

def test_assess_evidence_quality(classifier_service, sample_contradiction):
    Test assessing evidence quality.
    
    Args:
        classifier_service: Description of classifier_service
        sample_contradiction: Description of sample_contradiction
    
    result = classifier_service._assess_evidence_quality(sample_contradiction["metadata1"])
    
    assert isinstance(result, dict)
    assert "quality" in result
    assert "score" in result
    assert "factors" in result
    assert isinstance(result["quality"], EvidenceQuality)
    assert isinstance(result["score"], float)
    assert isinstance(result["factors"], list)

def test_assess_temporal_factor(classifier_service, sample_contradiction):
    Test assessing temporal factor.
    
    Args:
        classifier_service: Description of classifier_service
        sample_contradiction: Description of sample_contradiction
    
    result = classifier_service._assess_temporal_factor(
        metadata1=sample_contradiction["metadata1"],
        metadata2=sample_contradiction["metadata2"]
    )
    
    assert isinstance(result, dict)
    assert "detected" in result
    assert "score" in result
    assert "publication_date_difference" in result
    assert "factors" in result
    assert isinstance(result["detected"], bool)
    assert isinstance(result["score"], float)
    assert isinstance(result["factors"], list)

def test_assess_population_difference(classifier_service, sample_contradiction):
    Test assessing population difference.
    
    Args:
        classifier_service: Description of classifier_service
        sample_contradiction: Description of sample_contradiction
    
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
    assert isinstance(result["detected"], bool)
    assert isinstance(result["score"], float)
    assert isinstance(result["differences"], list)

def test_assess_methodological_difference(classifier_service, sample_contradiction):
    Test assessing methodological difference.
    
    Args:
        classifier_service: Description of classifier_service
        sample_contradiction: Description of sample_contradiction
    
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
    assert isinstance(result["detected"], bool)
    assert isinstance(result["score"], float)
    assert isinstance(result["differences"], list)
