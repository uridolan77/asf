"""Unit tests for the enhanced contradiction classifier."""

import pytest

from ...ml.services.contradiction_service import (
    ContradictionClassifierService,
    ContradictionType,
    ContradictionConfidence,
    ClinicalSignificance,
    EvidenceQuality,
    StudyDesignHierarchy
)

@pytest.fixture
def enhanced_classifier():
    """Create an enhanced contradiction classifier for testing.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    return ContradictionClassifierService()

@pytest.fixture
def sample_contradiction():
    """Sample contradiction for testing.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    return {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
        "is_contradiction": True,
        "contradiction_score": 0.85,
        "contradiction_type": ContradictionType.NEGATION,
        "confidence": ContradictionConfidence.HIGH,
        "methods_used": ["biomedlm", "negation"],
        "explanation": "Claim 2 is a negation of Claim 1 with similarity 0.85.",
        "metadata1": {
            "publication_year": 2020,
            "study_design": "randomized controlled trial",
            "sample_size": 5000,
            "population": "adults with high cholesterol",
            "p_value": 0.01,
            "journal": "New England Journal of Medicine",
            "impact_factor": 70.6
        },
        "metadata2": {
            "publication_year": 2015,
            "study_design": "observational study",
            "sample_size": 1000,
            "population": "elderly patients with high cholesterol",
            "p_value": 0.08,
            "journal": "Journal of Clinical Investigation",
            "impact_factor": 14.8
        }
    }

def test_enhanced_classifier_initialization(enhanced_classifier):
    """Test that the enhanced contradiction classifier initializes correctly.

    Args:
        enhanced_classifier: The classifier instance to test
    """

    assert enhanced_classifier is not None
    assert enhanced_classifier.thresholds is not None
    assert enhanced_classifier.thresholds[ContradictionType.DIRECT] > 0

@pytest.mark.asyncio
async def test_classify_contradiction(enhanced_classifier, sample_contradiction):
    """Test the contradiction classification functionality.

    Args:
        enhanced_classifier: The classifier instance to test
        sample_contradiction: Sample contradiction data for testing
    """
    high_significance = await enhanced_classifier._assess_clinical_significance(
        "Statin therapy reduces mortality in patients with cardiovascular disease.",
        "Statin therapy increases mortality in patients with cardiovascular disease."
    )
    assert high_significance["significance"] == ClinicalSignificance.HIGH

    moderate_significance = await enhanced_classifier._assess_clinical_significance(
        "Aspirin reduces pain in patients with headache.",
        "Acetaminophen is more effective than aspirin for pain relief in patients with headache."
    )
    assert moderate_significance["significance"] == ClinicalSignificance.MODERATE

    low_significance = await enhanced_classifier._assess_clinical_significance(
        "Vitamin C supplements may cause mild gastrointestinal discomfort.",
        "Vitamin C supplements are well-tolerated with minimal side effects."
    )
    assert low_significance["significance"] == ClinicalSignificance.LOW

@pytest.mark.asyncio
async def test_assess_evidence_quality(enhanced_classifier):
    significant_temporal = enhanced_classifier._assess_temporal_factor(
        {"publication_year": 2023},
        {"publication_year": 2010}
    )
    assert significant_temporal["detected"] == True
    assert significant_temporal["publication_date_difference"] == 13

    no_temporal = enhanced_classifier._assess_temporal_factor(
        {"publication_year": 2022},
        {"publication_year": 2021}
    )
    assert no_temporal["detected"] == False

@pytest.mark.asyncio
async def test_assess_population_difference(enhanced_classifier):
    method_diff = enhanced_classifier._assess_methodological_difference(
        "The treatment was effective.",
        "The treatment was not effective.",
        {
            "study_design": "randomized controlled trial",
            "sample_size": 5000
        },
        {
            "study_design": "observational study",
            "sample_size": 500
        }
    )
    assert method_diff["detected"] == True

    no_diff = enhanced_classifier._assess_methodological_difference(
        "The treatment was effective.",
        "The treatment was not effective.",
        {
            "study_design": "randomized controlled trial",
            "sample_size": 1000
        },
        {
            "study_design": "randomized controlled trial",
            "sample_size": 1200
        }
    )
    assert no_diff["detected"] == False
