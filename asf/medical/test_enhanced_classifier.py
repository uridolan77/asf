"""
Simple test script for the enhanced contradiction classifier.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
class ContradictionClassifierService:
    async def classify_contradiction(self, contradiction):
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
        contradiction["classification"] = classification
        return contradiction
async def test_classifier():