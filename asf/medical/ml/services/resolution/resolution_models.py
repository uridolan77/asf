"""Models for medical contradiction resolution.

This module defines the data models and enums used for resolving
contradictions in medical literature.
"""

from enum import Enum

class ResolutionStrategy(str, Enum):
    """Resolution strategies for medical contradictions.
    
    This enum defines the various strategies that can be used to resolve
    contradictions in medical literature.
    """
    EVIDENCE_HIERARCHY = "evidence_hierarchy"
    SAMPLE_SIZE_WEIGHTING = "sample_size_weighting"
    RECENCY_PREFERENCE = "recency_preference"
    POPULATION_SPECIFICITY = "population_specificity"
    METHODOLOGICAL_QUALITY = "methodological_quality"
    STATISTICAL_SIGNIFICANCE = "statistical_significance"
    COMBINED_EVIDENCE = "combined_evidence"
    UNKNOWN = "unknown"

class ResolutionConfidence(str, Enum):
    """Confidence levels for contradiction resolution.
    
    This enum defines the confidence levels for the resolution
    of medical contradictions.
    """
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"

class RecommendationType(str, Enum):
    """Recommendation types for contradiction resolution.
    
    This enum defines the different types of recommendations that can be made
    when resolving medical contradictions.
    """
    FAVOR_CLAIM1 = "favor_claim1"
    FAVOR_CLAIM2 = "favor_claim2"
    INCONCLUSIVE = "inconclusive"
    CONDITIONAL = "conditional"
    FURTHER_RESEARCH = "further_research"

# Study design hierarchy for evidence quality assessment
STUDY_DESIGN_HIERARCHY = {
    "systematic_review_meta_analysis": 7,
    "randomized_controlled_trial": 6,
    "cohort_study": 5,
    "case_control_study": 4,
    "case_series": 3,
    "case_report": 2,
    "expert_opinion": 1,
    "unknown": 0
}

# Study design keywords for detection
STUDY_DESIGN_KEYWORDS = {
    "systematic_review_meta_analysis": [
        "systematic review", "meta-analysis", "meta analysis", "metaanalysis"
    ],
    "randomized_controlled_trial": [
        "randomized controlled trial", "rct", "randomised controlled trial",
        "randomized clinical trial", "randomised clinical trial"
    ],
    "cohort_study": [
        "cohort study", "cohort analysis", "longitudinal study",
        "prospective study", "retrospective study", "follow-up study"
    ],
    "case_control_study": [
        "case-control study", "case control study", "case-control analysis"
    ],
    "case_series": [
        "case series", "case study series", "clinical series"
    ],
    "case_report": [
        "case report", "case study", "patient report"
    ],
    "expert_opinion": [
        "expert opinion", "expert consensus", "clinical opinion",
        "narrative review", "commentary", "editorial"
    ]
}

# Clinical significance terms
CLINICAL_SIGNIFICANCE_TERMS = {
    "high": [
        "mortality", "death", "survival", "fatal", "life-threatening",
        "cardiovascular", "stroke", "heart attack", "myocardial infarction",
        "cancer", "malignancy", "tumor", "severe", "critical", "emergency",
        "hospitalization", "intensive care", "icu", "organ failure",
        "permanent disability", "irreversible"
    ],
    "moderate": [
        "morbidity", "complication", "adverse event", "side effect",
        "quality of life", "functional status", "disability", "impairment",
        "chronic", "long-term", "persistent", "recurrent", "relapse",
        "readmission", "hospitalization", "surgery", "intervention"
    ],
    "low": [
        "mild", "minor", "transient", "temporary", "self-limiting",
        "benign", "cosmetic", "symptomatic", "discomfort", "inconvenience",
        "laboratory", "biomarker", "surrogate", "non-significant"
    ]
}

# Resolution result template
RESOLUTION_RESULT_TEMPLATE = {
    "recommendation": RecommendationType.INCONCLUSIVE,
    "confidence": ResolutionConfidence.LOW,
    "confidence_score": 0.3,
    "recommended_claim": None,
    "recommendation_note": "",
    "strategy": ResolutionStrategy.UNKNOWN,
    "timestamp": None,
    "explanation": {
        "summary": "",
        "detailed_reasoning": "",
        "clinical_implications": "",
        "limitations": "",
        "references": []
    }
}
