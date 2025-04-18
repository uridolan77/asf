Additional resolution strategies for medical contradictions.

This module provides additional strategies for resolving contradictions in medical literature
based on evidence-based medicine principles.
import logging
from typing import Dict, List, Any
from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionType,
    ClinicalSignificance,
    EvidenceQuality
)
from asf.medical.ml.services.resolution.resolution_models import (
    ResolutionStrategy,
    ResolutionConfidence,
    RecommendationType,
    STUDY_DESIGN_HIERARCHY,
    STUDY_DESIGN_KEYWORDS
)
logger = logging.getLogger(__name__)
async def resolve_by_methodological_quality(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on methodological quality.

    This strategy resolves contradictions by comparing the methodological quality of the studies
    supporting each claim, including study design, blinding, and randomization.

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
    methodological_difference = classification.get("methodological_difference", {})
    differences = methodological_difference.get("differences", [])
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "methodological_comparison": {
            "differences": differences,
            "quality_assessment": {}
        }
    }
    if not differences:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Methodological quality information is not available for comparison."
        return result
    study_design_diff = None
    for diff in differences:
        category = diff.get("category", "")
        if category == "study_design":
            study_design_diff = diff
    if study_design_diff:
        claim1_design = study_design_diff.get("claim1_design", "")
        claim2_design = study_design_diff.get("claim2_design", "")
        hierarchy_pos1 = 0
        hierarchy_pos2 = 0
        for design, keywords in STUDY_DESIGN_KEYWORDS.items():
            if any(keyword in claim1_design for keyword in keywords):
                hierarchy_pos1 = STUDY_DESIGN_HIERARCHY.get(design, 0)
            if any(keyword in claim2_design for keyword in keywords):
                hierarchy_pos2 = STUDY_DESIGN_HIERARCHY.get(design, 0)
        result["methodological_comparison"]["study_design"] = {
            "claim1_design": claim1_design,
            "claim2_design": claim2_design,
            "claim1_hierarchy_position": hierarchy_pos1,
            "claim2_hierarchy_position": hierarchy_pos2,
            "hierarchy_differential": hierarchy_pos1 - hierarchy_pos2
        }
        if hierarchy_pos1 - hierarchy_pos2 >= 2:
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.7
            result["recommendation_note"] = f"Favoring claim 1 because it used a stronger study design ({claim1_design} vs. {claim2_design})."
        elif hierarchy_pos2 - hierarchy_pos1 >= 2:
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.7
            result["recommendation_note"] = f"Favoring claim 2 because it used a stronger study design ({claim2_design} vs. {claim1_design})."
    blinding1 = metadata1.get("blinding", "none").lower()
    blinding2 = metadata2.get("blinding", "none").lower()
    randomization1 = metadata1.get("randomization", "none").lower()
    randomization2 = metadata2.get("randomization", "none").lower()
    blinding_score1 = 0
    if "double" in blinding1 or "triple" in blinding1:
        blinding_score1 = 2
    elif "single" in blinding1:
        blinding_score1 = 1
    blinding_score2 = 0
    if "double" in blinding2 or "triple" in blinding2:
        blinding_score2 = 2
    elif "single" in blinding2:
        blinding_score2 = 1
    randomization_score1 = 0
    if randomization1 != "none":
        randomization_score1 = 1
        if "stratified" in randomization1 or "block" in randomization1:
            randomization_score1 = 2
    randomization_score2 = 0
    if randomization2 != "none":
        randomization_score2 = 1
        if "stratified" in randomization2 or "block" in randomization2:
            randomization_score2 = 2
    result["methodological_comparison"]["blinding"] = {
        "claim1_blinding": blinding1,
        "claim2_blinding": blinding2,
        "claim1_score": blinding_score1,
        "claim2_score": blinding_score2,
        "differential": blinding_score1 - blinding_score2
    }
    result["methodological_comparison"]["randomization"] = {
        "claim1_randomization": randomization1,
        "claim2_randomization": randomization2,
        "claim1_score": randomization_score1,
        "claim2_score": randomization_score2,
        "differential": randomization_score1 - randomization_score2
    }
    quality_score1 = hierarchy_pos1 / 7.0 * 0.5  # 50% weight to study design
    quality_score1 += blinding_score1 / 2.0 * 0.25  # 25% weight to blinding
    quality_score1 += randomization_score1 / 2.0 * 0.25  # 25% weight to randomization
    quality_score2 = hierarchy_pos2 / 7.0 * 0.5  # 50% weight to study design
    quality_score2 += blinding_score2 / 2.0 * 0.25  # 25% weight to blinding
    quality_score2 += randomization_score2 / 2.0 * 0.25  # 25% weight to randomization
    result["methodological_comparison"]["quality_assessment"] = {
        "claim1_quality_score": quality_score1,
        "claim2_quality_score": quality_score2,
        "differential": quality_score1 - quality_score2
    }
    if result["recommendation"] == RecommendationType.INCONCLUSIVE:
        if quality_score1 - quality_score2 >= 0.3:  # Significant difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
            result["recommendation_note"] = "Favoring claim 1 because it has better overall methodological quality."
        elif quality_score2 - quality_score1 >= 0.3:  # Significant difference
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
            result["recommendation_note"] = "Favoring claim 2 because it has better overall methodological quality."
        else:
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "The methodological quality of both claims is similar."
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.4
    return result
async def resolve_by_statistical_significance(contradiction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on statistical significance.

    This strategy resolves contradictions by comparing the statistical significance of the studies
    supporting each claim, including p-values, confidence intervals, and effect sizes.

    Args:
        contradiction: Dictionary containing contradiction details

    Returns:
        Dictionary with resolution recommendation and confidence
    """
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    metadata1 = contradiction.get("metadata1", {})
    metadata2 = contradiction.get("metadata2", {})
    p_value1 = metadata1.get("p_value")
    p_value2 = metadata2.get("p_value")
    confidence_interval1 = metadata1.get("confidence_interval")
    confidence_interval2 = metadata2.get("confidence_interval")
    effect_size1 = metadata1.get("effect_size")
    effect_size2 = metadata2.get("effect_size")
    sample_size1 = metadata1.get("sample_size", 0)
    sample_size2 = metadata2.get("sample_size", 0)
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "statistical_comparison": {
            "p_values": {
                "claim1": p_value1,
                "claim2": p_value2
            },
            "confidence_intervals": {
                "claim1": confidence_interval1,
                "claim2": confidence_interval2
            },
            "effect_sizes": {
                "claim1": effect_size1,
                "claim2": effect_size2
            },
            "sample_sizes": {
                "claim1": sample_size1,
                "claim2": sample_size2
            }
        }
    }
    if p_value1 is None or p_value2 is None:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Statistical significance information (p-values) is not available for one or both claims."
        return result
    if (p_value1 <= 0.05 and p_value2 > 0.05):
        result["recommendation"] = RecommendationType.FAVOR_CLAIM1
        result["recommended_claim"] = claim1
        result["confidence"] = ResolutionConfidence.MODERATE
        result["confidence_score"] = 0.7
        result["recommendation_note"] = f"Favoring claim 1 because it shows statistical significance (p={p_value1}) while claim 2 does not (p={p_value2})."
    elif (p_value2 <= 0.05 and p_value1 > 0.05):
        result["recommendation"] = RecommendationType.FAVOR_CLAIM2
        result["recommended_claim"] = claim2
        result["confidence"] = ResolutionConfidence.MODERATE
        result["confidence_score"] = 0.7
        result["recommendation_note"] = f"Favoring claim 2 because it shows statistical significance (p={p_value2}) while claim 1 does not (p={p_value1})."
    elif p_value1 <= 0.05 and p_value2 <= 0.05:
        p_ratio = min(p_value1, p_value2) / max(p_value1, p_value2)
        result["statistical_comparison"]["p_value_ratio"] = p_ratio
        if p_ratio <= 0.1 and p_value1 < p_value2:
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
            result["recommendation_note"] = f"Favoring claim 1 because it shows stronger statistical significance (p={p_value1} vs p={p_value2})."
        elif p_ratio <= 0.1 and p_value2 < p_value1:
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.MODERATE
            result["confidence_score"] = 0.6
            result["recommendation_note"] = f"Favoring claim 2 because it shows stronger statistical significance (p={p_value2} vs p={p_value1})."
        else:
            result["recommendation"] = RecommendationType.INCONCLUSIVE
            result["recommended_claim"] = "Both claims show statistical significance with similar p-values."
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.4
    else:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "Neither claim shows statistical significance."
        result["confidence"] = ResolutionConfidence.LOW
        result["confidence_score"] = 0.3
    if effect_size1 is not None and effect_size2 is not None:
        result["statistical_comparison"]["effect_size_ratio"] = abs(effect_size1) / max(0.001, abs(effect_size2))
        if result["recommendation"] == RecommendationType.INCONCLUSIVE and abs(effect_size1) >= 2 * abs(effect_size2):
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.4
            result["recommendation_note"] = f"Favoring claim 1 because it shows a larger effect size ({effect_size1} vs {effect_size2})."
        elif result["recommendation"] == RecommendationType.INCONCLUSIVE and abs(effect_size2) >= 2 * abs(effect_size1):
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.4
            result["recommendation_note"] = f"Favoring claim 2 because it shows a larger effect size ({effect_size2} vs {effect_size1})."
    if isinstance(confidence_interval1, list) and len(confidence_interval1) == 2 and \
       isinstance(confidence_interval2, list) and len(confidence_interval2) == 2:
        width1 = confidence_interval1[1] - confidence_interval1[0]
        width2 = confidence_interval2[1] - confidence_interval2[0]
        result["statistical_comparison"]["ci_widths"] = {
            "claim1": width1,
            "claim2": width2,
            "ratio": width1 / max(0.001, width2)
        }
        if result["recommendation"] == RecommendationType.INCONCLUSIVE and width1 <= 0.5 * width2:
            result["recommendation"] = RecommendationType.FAVOR_CLAIM1
            result["recommended_claim"] = claim1
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.4
            result["recommendation_note"] = f"Favoring claim 1 because it has a narrower confidence interval ({width1} vs {width2})."
        elif result["recommendation"] == RecommendationType.INCONCLUSIVE and width2 <= 0.5 * width1:
            result["recommendation"] = RecommendationType.FAVOR_CLAIM2
            result["recommended_claim"] = claim2
            result["confidence"] = ResolutionConfidence.LOW
            result["confidence_score"] = 0.4
            result["recommendation_note"] = f"Favoring claim 2 because it has a narrower confidence interval ({width2} vs {width1})."
    return result
async def resolve_by_combined_evidence(contradiction: Dict[str, Any], resolution_strategies: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve contradictions based on combined evidence from multiple strategies.

    This strategy resolves contradictions by combining the results of multiple resolution strategies
    to provide a more comprehensive assessment of the evidence.

    Args:
        contradiction: Dictionary containing contradiction details
        resolution_strategies: Dictionary mapping strategy names to resolution functions

    Returns:
        Dictionary with resolution recommendation and confidence
    """
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    result = {
        "recommendation": RecommendationType.INCONCLUSIVE,
        "confidence": ResolutionConfidence.LOW,
        "confidence_score": 0.3,
        "recommended_claim": None,
        "combined_evidence": {
            "strategies_applied": [],
            "strategy_results": {}
        }
    }
    strategies = [
        ResolutionStrategy.EVIDENCE_HIERARCHY,
        ResolutionStrategy.SAMPLE_SIZE_WEIGHTING,
        ResolutionStrategy.RECENCY_PREFERENCE,
        ResolutionStrategy.POPULATION_SPECIFICITY,
        ResolutionStrategy.METHODOLOGICAL_QUALITY,
        ResolutionStrategy.STATISTICAL_SIGNIFICANCE
    ]
    strategy_results = {}
    for strategy in strategies:
        resolution_func = resolution_strategies.get(strategy)
        if resolution_func:
            strategy_result = await resolution_func(contradiction)
            strategy_results[strategy] = strategy_result
            result["combined_evidence"]["strategies_applied"].append(strategy)
            result["combined_evidence"]["strategy_results"][strategy] = {
                "recommendation": strategy_result.get("recommendation"),
                "confidence": strategy_result.get("confidence"),
                "confidence_score": strategy_result.get("confidence_score"),
                "recommendation_note": strategy_result.get("recommendation_note")
            }
    claim1_count = 0
    claim2_count = 0
    inconclusive_count = 0
    conditional_count = 0
    further_research_count = 0
    claim1_score = 0.0
    claim2_score = 0.0
    total_weight = 0.0
    for strategy, strategy_result in strategy_results.items():
        recommendation = strategy_result.get("recommendation")
        confidence_score = strategy_result.get("confidence_score", 0.3)
        if recommendation == RecommendationType.FAVOR_CLAIM1:
            claim1_count += 1
            claim1_score += confidence_score
        elif recommendation == RecommendationType.FAVOR_CLAIM2:
            claim2_count += 1
            claim2_score += confidence_score
        elif recommendation == RecommendationType.INCONCLUSIVE:
            inconclusive_count += 1
        elif recommendation == RecommendationType.CONDITIONAL:
            conditional_count += 1
        elif recommendation == RecommendationType.FURTHER_RESEARCH:
            further_research_count += 1
        total_weight += confidence_score
    if total_weight > 0:
        claim1_score /= total_weight
        claim2_score /= total_weight
    result["combined_evidence"]["recommendation_counts"] = {
        "favor_claim1": claim1_count,
        "favor_claim2": claim2_count,
        "inconclusive": inconclusive_count,
        "conditional": conditional_count,
        "further_research": further_research_count
    }
    result["combined_evidence"]["weighted_scores"] = {
        "claim1_score": claim1_score,
        "claim2_score": claim2_score,
        "differential": claim1_score - claim2_score
    }
    total_strategies = len(strategy_results)
    if total_strategies == 0:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "No resolution strategies could be applied due to insufficient data."
        result["confidence"] = ResolutionConfidence.VERY_LOW
        result["confidence_score"] = 0.1
    elif claim1_count > claim2_count and claim1_count > inconclusive_count:
        result["recommendation"] = RecommendationType.FAVOR_CLAIM1
        result["recommended_claim"] = claim1
        result["confidence_score"] = min(0.9, 0.5 + (claim1_score - claim2_score))
        if result["confidence_score"] >= 0.7:
            result["confidence"] = ResolutionConfidence.HIGH
        elif result["confidence_score"] >= 0.5:
            result["confidence"] = ResolutionConfidence.MODERATE
        else:
            result["confidence"] = ResolutionConfidence.LOW
        result["recommendation_note"] = f"Favoring claim 1 based on {claim1_count} out of {total_strategies} resolution strategies."
    elif claim2_count > claim1_count and claim2_count > inconclusive_count:
        result["recommendation"] = RecommendationType.FAVOR_CLAIM2
        result["recommended_claim"] = claim2
        result["confidence_score"] = min(0.9, 0.5 + (claim2_score - claim1_score))
        if result["confidence_score"] >= 0.7:
            result["confidence"] = ResolutionConfidence.HIGH
        elif result["confidence_score"] >= 0.5:
            result["confidence"] = ResolutionConfidence.MODERATE
        else:
            result["confidence"] = ResolutionConfidence.LOW
        result["recommendation_note"] = f"Favoring claim 2 based on {claim2_count} out of {total_strategies} resolution strategies."
    elif conditional_count > 0 and conditional_count >= claim1_count and conditional_count >= claim2_count:
        result["recommendation"] = RecommendationType.CONDITIONAL
        result["recommended_claim"] = "The contradiction may be resolved differently depending on specific clinical context."
        result["confidence"] = ResolutionConfidence.MODERATE
        result["confidence_score"] = 0.5
        result["recommendation_note"] = "Multiple resolution strategies suggest that the contradiction resolution depends on specific clinical context."
    elif further_research_count > 0 and further_research_count >= claim1_count and further_research_count >= claim2_count:
        result["recommendation"] = RecommendationType.FURTHER_RESEARCH
        result["recommended_claim"] = "Further research is needed to resolve this contradiction."
        result["confidence"] = ResolutionConfidence.LOW
        result["confidence_score"] = 0.3
        result["recommendation_note"] = "Multiple resolution strategies suggest that further research is needed to resolve this contradiction."
    else:
        result["recommendation"] = RecommendationType.INCONCLUSIVE
        result["recommended_claim"] = "The evidence is insufficient to resolve this contradiction conclusively."
        result["confidence"] = ResolutionConfidence.LOW
        result["confidence_score"] = 0.3
        result["recommendation_note"] = f"No clear resolution emerged from the {total_strategies} strategies applied."
    return result