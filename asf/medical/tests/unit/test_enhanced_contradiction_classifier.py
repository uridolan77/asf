"""
Unit tests for the enhanced contradiction classifier.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

from asf.medical.ml.services.enhanced_contradiction_classifier import (
    EnhancedContradictionClassifier,
    ContradictionType,
    ContradictionConfidence,
    ClinicalSignificance,
    EvidenceQuality,
    StudyDesignHierarchy
)

@pytest.fixture
def enhanced_classifier():
    """Create an enhanced contradiction classifier for testing."""
    return EnhancedContradictionClassifier()

@pytest.fixture
def sample_contradiction():
    """Sample contradiction for testing."""
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

@pytest.mark.asyncio
async def test_classify_contradiction(enhanced_classifier, sample_contradiction):
    """Test classifying a contradiction."""
    # Classify the contradiction
    classified = await enhanced_classifier.classify_contradiction(sample_contradiction)
    
    # Check that classification was added
    assert "classification" in classified
    
    # Check clinical significance
    assert "clinical_significance" in classified["classification"]
    assert classified["classification"]["clinical_significance"] == ClinicalSignificance.HIGH
    
    # Check evidence quality
    assert "evidence_quality" in classified["classification"]
    assert classified["classification"]["evidence_quality"]["claim1"] == EvidenceQuality.HIGH
    assert classified["classification"]["evidence_quality"]["claim2"] in [EvidenceQuality.MODERATE, EvidenceQuality.LOW]
    
    # Check temporal factor
    assert "temporal_factor" in classified["classification"]
    assert classified["classification"]["temporal_factor"]["detected"] == True
    assert classified["classification"]["temporal_factor"]["publication_date_difference"] == 5
    
    # Check population difference
    assert "population_difference" in classified["classification"]
    assert classified["classification"]["population_difference"]["detected"] == True
    
    # Check methodological difference
    assert "methodological_difference" in classified["classification"]
    assert classified["classification"]["methodological_difference"]["detected"] == True

@pytest.mark.asyncio
async def test_assess_clinical_significance(enhanced_classifier):
    """Test assessing clinical significance."""
    # Test high clinical significance
    high_significance = await enhanced_classifier._assess_clinical_significance(
        "Statin therapy reduces mortality in patients with cardiovascular disease.",
        "Statin therapy increases mortality in patients with cardiovascular disease."
    )
    assert high_significance["significance"] == ClinicalSignificance.HIGH
    
    # Test moderate clinical significance
    moderate_significance = await enhanced_classifier._assess_clinical_significance(
        "Aspirin reduces pain in patients with headache.",
        "Acetaminophen is more effective than aspirin for pain relief in patients with headache."
    )
    assert moderate_significance["significance"] == ClinicalSignificance.MODERATE
    
    # Test low clinical significance
    low_significance = await enhanced_classifier._assess_clinical_significance(
        "Vitamin C supplements may cause mild gastrointestinal discomfort.",
        "Vitamin C supplements are well-tolerated with minimal side effects."
    )
    assert low_significance["significance"] == ClinicalSignificance.LOW

@pytest.mark.asyncio
async def test_assess_evidence_quality(enhanced_classifier):
    """Test assessing evidence quality."""
    # Test high evidence quality
    high_quality = enhanced_classifier._assess_evidence_quality({
        "study_design": "systematic review and meta-analysis",
        "sample_size": 10000,
        "publication_year": 2023,
        "journal_impact_factor": 25.0,
        "bias_risk": "low"
    })
    assert high_quality["quality"] == EvidenceQuality.HIGH
    
    # Test moderate evidence quality
    moderate_quality = enhanced_classifier._assess_evidence_quality({
        "study_design": "cohort study",
        "sample_size": 1000,
        "publication_year": 2020,
        "journal_impact_factor": 5.0,
        "bias_risk": "moderate"
    })
    assert moderate_quality["quality"] == EvidenceQuality.MODERATE
    
    # Test low evidence quality
    low_quality = enhanced_classifier._assess_evidence_quality({
        "study_design": "case report",
        "sample_size": 10,
        "publication_year": 2010,
        "journal_impact_factor": 1.0,
        "bias_risk": "high"
    })
    assert low_quality["quality"] in [EvidenceQuality.LOW, EvidenceQuality.VERY_LOW]

@pytest.mark.asyncio
async def test_assess_temporal_factor(enhanced_classifier):
    """Test assessing temporal factor."""
    # Test significant temporal difference
    significant_temporal = enhanced_classifier._assess_temporal_factor(
        {"publication_year": 2023},
        {"publication_year": 2010}
    )
    assert significant_temporal["detected"] == True
    assert significant_temporal["publication_date_difference"] == 13
    
    # Test no temporal difference
    no_temporal = enhanced_classifier._assess_temporal_factor(
        {"publication_year": 2022},
        {"publication_year": 2021}
    )
    assert no_temporal["detected"] == False

@pytest.mark.asyncio
async def test_assess_population_difference(enhanced_classifier):
    """Test assessing population difference."""
    # Test population difference
    population_diff = enhanced_classifier._assess_population_difference(
        "The treatment was effective in children.",
        "The treatment was effective in adults.",
        {"population": "pediatric patients"},
        {"population": "adult patients"}
    )
    assert population_diff["detected"] == True
    
    # Test no population difference
    no_diff = enhanced_classifier._assess_population_difference(
        "The treatment was effective in adults.",
        "The treatment was not effective in adults.",
        {"population": "adult patients"},
        {"population": "adult patients"}
    )
    assert no_diff["detected"] == False

@pytest.mark.asyncio
async def test_assess_methodological_difference(enhanced_classifier):
    """Test assessing methodological difference."""
    # Test methodological difference
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
    
    # Test no methodological difference
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
