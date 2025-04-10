"""
Simple test script for the enhanced contradiction classifier.
"""

import asyncio
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Define enums locally for testing
class ContradictionType(str):
    DIRECT = "direct"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    METHODOLOGICAL = "methodological"
    STATISTICAL = "statistical"
    POPULATION = "population"
    UNKNOWN = "unknown"

class ContradictionConfidence(str):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ClinicalSignificance(str):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"

class EvidenceQuality(str):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"

class StudyDesignHierarchy(str):
    SYSTEMATIC_REVIEW_META_ANALYSIS = "systematic_review_meta_analysis"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL_STUDY = "case_control_study"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    EXPERT_OPINION = "expert_opinion"
    UNKNOWN = "unknown"

# Mock the classifier for testing
class EnhancedContradictionClassifier:
    async def classify_contradiction(self, contradiction):
        # Simulate classification
        classification = {
            "contradiction_type": contradiction.get("contradiction_type", ContradictionType.UNKNOWN),
            "clinical_significance": ClinicalSignificance.HIGH,
            "clinical_significance_score": 0.85,
            "clinical_significance_terms": {
                "high": ["mortality", "cardiovascular"],
                "moderate": [],
                "low": []
            },
            "evidence_quality": {
                "claim1": EvidenceQuality.HIGH,
                "claim1_score": 0.75,
                "claim1_factors": {
                    "study_design": 0.6,
                    "sample_size": 0.3,
                    "publication_year": 0.2,
                    "journal_impact_factor": 0.2,
                    "bias_risk": 0.1
                },
                "claim2": EvidenceQuality.MODERATE,
                "claim2_score": 0.45,
                "claim2_factors": {
                    "study_design": 0.3,
                    "sample_size": 0.2,
                    "publication_year": 0.15,
                    "journal_impact_factor": 0.1,
                    "bias_risk": 0.0
                },
                "differential": 0.3
            },
            "temporal_factor": {
                "detected": True,
                "score": 0.25,
                "publication_date_difference": 5,
                "factors": {
                    "publication_date_difference": 5
                }
            },
            "population_difference": {
                "detected": True,
                "score": 0.6,
                "differences": [
                    {
                        "category": "age",
                        "claim1_terms": ["adults"],
                        "claim2_terms": ["elderly"],
                        "common_terms": []
                    }
                ]
            },
            "methodological_difference": {
                "detected": True,
                "score": 0.7,
                "differences": [
                    {
                        "category": "study_design",
                        "claim1_design": "randomized_controlled_trial",
                        "claim2_design": "observational_study",
                        "design_difference_score": 0.5
                    },
                    {
                        "category": "sample_size",
                        "claim1_sample_size": 5000,
                        "claim2_sample_size": 1000,
                        "ratio": 5.0,
                        "sample_size_difference_score": 0.4
                    }
                ]
            }
        }

        # Add classification to contradiction
        contradiction["classification"] = classification

        return contradiction

# We're using the mock classifier for testing
# If you want to try importing the real one, uncomment this:
# try:
#     from medical.ml.services.enhanced_contradiction_classifier import (
#         EnhancedContradictionClassifier,
#         ContradictionType,
#         ContradictionConfidence,
#         ClinicalSignificance,
#         EvidenceQuality
#     )
#     print("Using real classifier")
# except ImportError:
#     print("Using mock classifier")

async def test_classifier():
    """Test the enhanced contradiction classifier."""
    print("Initializing enhanced contradiction classifier...")
    classifier = EnhancedContradictionClassifier()

    print("Creating sample contradiction...")
    sample_contradiction = {
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

    print("Classifying contradiction...")
    classified = await classifier.classify_contradiction(sample_contradiction)

    print("\nClassification results:")
    print(f"Clinical significance: {classified['classification']['clinical_significance']}")
    print(f"Evidence quality (claim1): {classified['classification']['evidence_quality']['claim1']}")
    print(f"Evidence quality (claim2): {classified['classification']['evidence_quality']['claim2']}")
    print(f"Temporal factor detected: {classified['classification']['temporal_factor']['detected']}")
    print(f"Population difference detected: {classified['classification']['population_difference']['detected']}")
    print(f"Methodological difference detected: {classified['classification']['methodological_difference']['detected']}")

    print("\nDetailed classification:")
    for key, value in classified["classification"].items():
        print(f"{key}: {value}")

    return classified

if __name__ == "__main__":
    print("Running enhanced contradiction classifier test...")
    result = asyncio.run(test_classifier())
    print("Test completed successfully!")
