"""Resolution strategies for medical contradictions.

This module provides strategies for resolving contradictions in medical literature
based on evidence-based medicine principles.
"""
import logging
from datetime import datetime
from typing import Dict, Any
from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionType,
    ClinicalSignificance,
    EvidenceQuality
)
from asf.medical.ml.services.resolution.resolution_models import StudyDesignHierarchy
from asf.medical.ml.services.resolution.resolution_models import (
    ResolutionStrategy,
    ResolutionConfidence,
    RecommendationType,
    STUDY_DESIGN_HIERARCHY,
    STUDY_DESIGN_KEYWORDS
)
from asf.medical.ml.services.resolution.resolution_utils import (
    calculate_confidence_from_hierarchy_diff,
    assess_population_relevance
)
logger = logging.getLogger(__name__)
async def resolve_by_evidence_hierarchy(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on evidence hierarchy.

    This strategy resolves contradictions by comparing the study designs and evidence quality
    of the contradicting claims, following the evidence-based medicine hierarchy.

    Args:
        contradiction: Dictionary containing contradiction details

    Returns:
        Dictionary with resolution recommendation and confidence
    """
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    metadata1 = contradiction.get("metadata1", {})
    metadata2 = contradiction.get("metadata2", {})
    classification = contradiction.get("classification", {})
    evidence_quality = classification.get("evidence_quality", {})
    quality1 = evidence_quality.get("claim1", EvidenceQuality.UNKNOWN)
    quality2 = evidence_quality.get("claim2", EvidenceQuality.UNKNOWN)
    quality_score1 = evidence_quality.get("claim1_score", 0)
    quality_score2 = evidence_quality.get("claim2_score", 0)
    study_design1 = metadata1.get("study_design", "unknown").lower()
    study_design2 = metadata2.get("study_design", "unknown").lower()
    design_type1 = StudyDesignHierarchy.UNKNOWN
    design_type2 = StudyDesignHierarchy.UNKNOWN
    for design, keywords in STUDY_DESIGN_KEYWORDS.items():
        if any(keyword in study_design1 for keyword in keywords):
            design_type1 = design
        if any(keyword in study_design2 for keyword in keywords):
            design_type2 = design
    hierarchy_pos1 = STUDY_DESIGN_HIERARCHY.get(design_type1, 0)
    hierarchy_pos2 = STUDY_DESIGN_HIERARCHY.get(design_type2, 0)
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "evidence_comparison": {
            "claim1_design": str(design_type1),
            "claim2_design": str(design_type2),
            "claim1_hierarchy_position": hierarchy_pos1,
            "claim2_hierarchy_position": hierarchy_pos2,
            "claim1_quality": quality1,
            "claim2_quality": quality2,
            "claim1_quality_score": quality_score1,
            "claim2_quality_score": quality_score2,
            "hierarchy_differential": hierarchy_pos1 - hierarchy_pos2,
            "quality_differential": quality_score1 - quality_score2
        }
    }
    if hierarchy_pos1 > hierarchy_pos2:
        result["recommendation"] = RecommendationType.FAVOR_CLAIM1
        result["recommended_claim"] = claim1
        result["confidence"] = calculate_confidence_from_hierarchy_diff(hierarchy_pos1, hierarchy_pos2)
        result["confidence_score"] = min(0.9, 0.5 + (hierarchy_pos1 - hierarchy_pos2) * 0.1)
    elif hierarchy_pos2 > hierarchy_pos1:
        result["recommendation"] = RecommendationType.FAVOR_CLAIM2
        result["recommended_claim"] = claim2
        result["confidence"] = calculate_confidence_from_hierarchy_diff(hierarchy_pos2, hierarchy_pos1)
        result["confidence_score"] = min(0.9, 0.5 + (hierarchy_pos2 - hierarchy_pos1) * 0.1)
    else:
        if quality_score1 > quality_score2 + 0.2:  # Significant difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
        elif quality_score2 > quality_score1 + 0.2:  # Significant difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
        else:
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "Evidence is insufficient to determine which claim is more reliable."
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.3
    return result
async def resolve_by_sample_size(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on sample size.

    This strategy resolves contradictions by comparing the sample sizes of the studies
    supporting each claim, with larger sample sizes generally providing more reliable evidence.

    Args:
        contradiction: Dictionary containing contradiction details

    Returns:
        Dictionary with resolution recommendation and confidence
    """
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    metadata1 = contradiction.get("metadata1", {})
    metadata2 = contradiction.get("metadata2", {})
    sample_size1 = metadata1.get("sample_size", 0)
    sample_size2 = metadata2.get("sample_size", 0)
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "sample_size_comparison": {
            "claim1_sample_size": sample_size1,
            "claim2_sample_size": sample_size2,
            "ratio": 1.0,
            "absolute_difference": abs(sample_size1 - sample_size2)
        }
    }
    if sample_size1 <= 0 or sample_size2 <= 0:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Sample size information is not available for one or both claims."
        return result
    ratio = max(sample_size1, sample_size2) / max(1, min(sample_size1, sample_size2))
    result["sample_size_comparison"]["ratio"] = ratio
    if sample_size1 > sample_size2:
        if ratio >= 10:  # 10x larger
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.HIGH
            result["confidence_score"] = 0.8
        elif ratio >= 5:  # 5x larger
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.7
        elif ratio >= 2:  # 2x larger
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.6
        else:  # Less than 2x larger
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "The difference in sample size is not significant enough to favor one claim over the other."
            result["confidence"] = ResolutionConfidence.VERY_LOW
            result["confidence_score"] = 0.3
    elif sample_size2 > sample_size1:
        if ratio >= 10:  # 10x larger
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.HIGH
            result["confidence_score"] = 0.8
        elif ratio >= 5:  # 5x larger
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.7
        elif ratio >= 2:  # 2x larger
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.6
        else:  # Less than 2x larger
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "The difference in sample size is not significant enough to favor one claim over the other."
            result["confidence"] = ResolutionConfidence.VERY_LOW
            result["confidence_score"] = 0.3
    else:  # Equal sample sizes
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Both claims have the same sample size."
        result["confidence"] = ResolutionConfidence.VERY_LOW
        result["confidence_score"] = 0.3
    power1 = metadata1.get("statistical_power")
    power2 = metadata2.get("statistical_power")
    if power1 is not None and power2 is not None:
        result["sample_size_comparison"]["claim1_power"] = power1
        result["sample_size_comparison"]["claim2_power"] = power2
        if result["recommendation"] == RecommendationType.FAVOR_CLAIM1 and power1 > 0.8 and power1 > power2:
            result["confidence_score"] += 0.1
        elif result["recommendation"] == RecommendationType.FAVOR_CLAIM2 and power2 > 0.8 and power2 > power1:
            result["confidence_score"] += 0.1
    return result
async def resolve_by_recency(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on publication recency.

    This strategy resolves contradictions by comparing the publication dates of the studies
    supporting each claim, with more recent studies generally providing more up-to-date evidence.

    Args:
        contradiction: Dictionary containing contradiction details

    Returns:
        Dictionary with resolution recommendation and confidence
    """
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    metadata1 = contradiction.get("metadata1", {})
    metadata2 = contradiction.get("metadata2", {})
    pub_year1 = metadata1.get("publication_year", 0)
    pub_year2 = metadata2.get("publication_year", 0)
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "recency_comparison": {
            "claim1_year": pub_year1,
            "claim2_year": pub_year2,
            "year_difference": abs(pub_year1 - pub_year2),
            "current_year": datetime.now().year
        }
    }
    if pub_year1 <= 0 or pub_year2 <= 0:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Publication year information is not available for one or both claims."
        return result
    year_diff = abs(pub_year1 - pub_year2)
    result["recency_comparison"]["year_difference"] = year_diff
    if pub_year1 > pub_year2:
        if year_diff >= 10:  # 10+ years difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.HIGH
            result["confidence_score"] = 0.8
        elif year_diff >= 5:  # 5-9 years difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.7
        elif year_diff >= 2:  # 2-4 years difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.6
        else:  # Less than 2 years difference
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "The difference in publication year is not significant enough to favor one claim over the other."
            result["confidence"] = ResolutionConfidence.VERY_LOW
            result["confidence_score"] = 0.3
    elif pub_year2 > pub_year1:
        if year_diff >= 10:  # 10+ years difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.HIGH
            result["confidence_score"] = 0.8
        elif year_diff >= 5:  # 5-9 years difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.7
        elif year_diff >= 2:  # 2-4 years difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.6
        else:  # Less than 2 years difference
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "The difference in publication year is not significant enough to favor one claim over the other."
            result["confidence"] = ResolutionConfidence.VERY_LOW
            result["confidence_score"] = 0.3
    else:  # Same publication year
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Both claims were published in the same year."
        result["confidence"] = ResolutionConfidence.VERY_LOW
        result["confidence_score"] = 0.3
    current_year = datetime.now().year
    claim1_age = current_year - pub_year1
    claim2_age = current_year - pub_year2
    result["recency_comparison"]["claim1_age"] = claim1_age
    result["recency_comparison"]["claim2_age"] = claim2_age
    if claim1_age > 15 and claim2_age > 15:
        result["confidence_score"] = max(0.2, result["confidence_score"] - 0.2)
        result["confidence"] = ResolutionConfidence.LOW
        result["recommendation_note"] = "Both studies are more than 15 years old, which reduces confidence in the resolution."
    if result["recommendation"] == RecommendationType.FAVOR_CLAIM1 and claim1_age > 15 and claim2_age <= 15:
        result["recommendation"] = RecommendationType.CONDITIONAL
        result["recommended_claim"] = f"While {claim1} is from a more recent study, it is still over 15 years old. Consider the possibility that more recent evidence may exist."
        result["confidence_score"] = max(0.3, result["confidence_score"] - 0.3)
        result["confidence"] = ResolutionConfidence.LOW
    elif result["recommendation"] == RecommendationType.FAVOR_CLAIM2 and claim2_age > 15 and claim1_age <= 15:
        result["recommendation"] = RecommendationType.CONDITIONAL
        result["recommended_claim"] = f"While {claim2} is from a more recent study, it is still over 15 years old. Consider the possibility that more recent evidence may exist."
        result["confidence_score"] = max(0.3, result["confidence_score"] - 0.3)
        result["confidence"] = ResolutionConfidence.LOW
    return result
async def resolve_by_population_specificity(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on population specificity.

    This strategy resolves contradictions by comparing the populations studied in each claim,
    with more specific or relevant populations providing more applicable evidence for particular cases.

    Args:
        contradiction: Dictionary containing contradiction details

    Returns:
        Dictionary with resolution recommendation and confidence
    """
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    metadata1 = contradiction.get("metadata1", {})
    metadata2 = contradiction.get("metadata2", {})
    classification = contradiction.get("classification", {})
    population1 = metadata1.get("population", "")
    population2 = metadata2.get("population", "")
    population_difference = classification.get("population_difference", {})
    differences = population_difference.get("differences", [])
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "population_comparison": {
            "claim1_population": population1,
            "claim2_population": population2,
            "differences": differences
        }
    }
    if not population1 and not population2 and not differences:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Population information is not available for one or both claims."
        return result
    population_relevance = assess_population_relevance(population1, population2, differences)
    result["population_comparison"]["relevance_assessment"] = population_relevance
    if population_relevance["more_relevant_population"] == "claim1":
        result["recommendation"] = RecommendationType.FAVOR_CLAIM1
        result["recommended_claim"] = claim1
        result["confidence"] = ResolutionConfidence.MODERATE
        result["confidence_score"] = 0.7
        result["recommendation_note"] = f"Favoring claim 1 because it studied a more clinically relevant population: {population1}"
    elif population_relevance["more_relevant_population"] == "claim2":
        result["recommendation"] = RecommendationType.FAVOR_CLAIM2
        result["recommended_claim"] = claim2
        result["confidence"] = ResolutionConfidence.MODERATE
        result["confidence_score"] = 0.7
        result["recommendation_note"] = f"Favoring claim 2 because it studied a more clinically relevant population: {population2}"
    else:
        if differences:
            result["recommendation"] = RecommendationType.CONDITIONAL
            result["recommended_claim"] = "The contradiction may be explained by population differences. Consider which population is more relevant to your specific clinical context."
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
        else:
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "There is insufficient information about population differences to resolve the contradiction."
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.3
    return result