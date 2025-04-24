"""Explanation generator for medical contradiction resolution.

This module provides functions for generating explanations for
contradiction resolution results.
"""

import logging
from typing import Dict, List, Any

from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionType,
    ClinicalSignificance
)
from asf.medical.ml.services.resolution.resolution_models import (
    ResolutionStrategy,
    ResolutionConfidence,
    RecommendationType,
    CLINICAL_SIGNIFICANCE_TERMS
)

logger = logging.getLogger(__name__)


def generate_explanation(
    contradiction: Dict[str, Any],
    resolution_result: Dict[str, Any],
    include_details: bool = False
) -> str:
    """Generate explanation for contradiction resolution.
    
    This function generates a human-readable explanation for the resolution
    of a contradiction between two medical claims.
    
    Args:
        contradiction: The contradiction information
        resolution_result: The resolution result
        include_details: Whether to include detailed information
        
    Returns:
        A human-readable explanation
    """
    # Extract basic information
    claim1 = contradiction.get("claim1", "")
    claim2 = contradiction.get("claim2", "")
    classification = contradiction.get("classification", {})
    
    # Extract resolution information
    recommendation = resolution_result.get("recommendation", RecommendationType.INCONCLUSIVE)
    confidence = resolution_result.get("confidence", ResolutionConfidence.LOW)
    strategy = resolution_result.get("strategy", ResolutionStrategy.COMBINED_EVIDENCE)
    recommended_claim = resolution_result.get("recommended_claim", "")
    
    # Generate basic explanation
    explanation = f"The contradiction between the claims has been analyzed using the {strategy.replace('_', ' ')} approach. "
    
    if recommendation == RecommendationType.FAVOR_CLAIM1:
        explanation += f"The evidence favors the first claim: '{claim1}'. "
    elif recommendation == RecommendationType.FAVOR_CLAIM2:
        explanation += f"The evidence favors the second claim: '{claim2}'. "
    elif recommendation == RecommendationType.CONDITIONAL:
        explanation += "The resolution is conditional on specific factors. "
        explanation += f"{recommended_claim} "
    elif recommendation == RecommendationType.FURTHER_RESEARCH:
        explanation += "Further research is needed to resolve this contradiction. "
        explanation += f"{recommended_claim} "
    else:  # INCONCLUSIVE
        explanation += "The contradiction could not be conclusively resolved. "
        explanation += f"{recommended_claim} "
    
    explanation += f"Confidence: {confidence}."
    
    # Add detailed reasoning if requested
    if include_details:
        explanation += "\n\n"
        
        # Add strategy-specific explanation
        if strategy == ResolutionStrategy.EVIDENCE_HIERARCHY:
            explanation += generate_evidence_hierarchy_explanation(resolution_result)
        elif strategy == ResolutionStrategy.SAMPLE_SIZE_WEIGHTING:
            explanation += generate_sample_size_explanation(resolution_result)
        elif strategy == ResolutionStrategy.RECENCY_PREFERENCE:
            explanation += generate_recency_explanation(resolution_result)
        elif strategy == ResolutionStrategy.POPULATION_SPECIFICITY:
            explanation += generate_population_explanation(resolution_result)
        elif strategy == ResolutionStrategy.METHODOLOGICAL_QUALITY:
            explanation += generate_methodological_explanation(resolution_result)
        elif strategy == ResolutionStrategy.STATISTICAL_SIGNIFICANCE:
            explanation += generate_statistical_explanation(resolution_result)
        elif strategy == ResolutionStrategy.COMBINED_EVIDENCE:
            explanation += generate_combined_evidence_explanation(resolution_result)
        
        # Add clinical implications
        clinical_significance = classification.get("clinical_significance", {}).get("significance", ClinicalSignificance.UNKNOWN)
        explanation += "\n\n" + generate_clinical_implications(contradiction, resolution_result, clinical_significance)
        
        # Add limitations
        explanation += "\n\n" + generate_limitations(contradiction, resolution_result, strategy)
        
        # Add references
        references = generate_references(contradiction, resolution_result, strategy)
        explanation += "\n\nReferences:\n" + "\n".join([f"- {ref}" for ref in references])
    
    return explanation


def generate_evidence_hierarchy_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for evidence hierarchy resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    evidence_comparison = resolution.get("evidence_comparison", {})
    claim1_design = evidence_comparison.get("claim1_design", "unknown")
    claim2_design = evidence_comparison.get("claim2_design", "unknown")
    hierarchy_pos1 = evidence_comparison.get("claim1_hierarchy_position", 0)
    hierarchy_pos2 = evidence_comparison.get("claim2_hierarchy_position", 0)
    quality1 = evidence_comparison.get("claim1_quality", "unknown")
    quality2 = evidence_comparison.get("claim2_quality", "unknown")
    
    explanation = "The contradiction was resolved based on the evidence hierarchy principle, "
    explanation += "which prioritizes evidence from higher-quality study designs. "
    
    if hierarchy_pos1 > hierarchy_pos2:
        explanation += f"The first claim is supported by a {claim1_design} (evidence level: {hierarchy_pos1}), "
        explanation += f"which is higher in the evidence hierarchy than the {claim2_design} (evidence level: {hierarchy_pos2}) "
        explanation += "supporting the second claim. "
        explanation += f"The quality of evidence was assessed as {quality1} for the first claim and {quality2} for the second claim."
    elif hierarchy_pos2 > hierarchy_pos1:
        explanation += f"The second claim is supported by a {claim2_design} (evidence level: {hierarchy_pos2}), "
        explanation += f"which is higher in the evidence hierarchy than the {claim1_design} (evidence level: {hierarchy_pos1}) "
        explanation += "supporting the first claim. "
        explanation += f"The quality of evidence was assessed as {quality1} for the first claim and {quality2} for the second claim."
    else:
        explanation += f"Both claims are supported by similar study designs ({claim1_design} and {claim2_design}). "
        explanation += f"The quality of evidence was assessed as {quality1} for the first claim and {quality2} for the second claim."
    
    return explanation


def generate_sample_size_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for sample size resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    sample_comparison = resolution.get("sample_size_comparison", {})
    sample_size1 = sample_comparison.get("claim1_sample_size", 0)
    sample_size2 = sample_comparison.get("claim2_sample_size", 0)
    ratio = sample_comparison.get("ratio", 1.0)
    
    explanation = "The contradiction was resolved based on sample size considerations, "
    explanation += "which prioritizes evidence from studies with larger sample sizes. "
    
    if sample_size1 > sample_size2:
        explanation += f"The first claim is supported by a study with {sample_size1} participants, "
        explanation += f"which is {ratio:.1f} times larger than the study supporting the second claim ({sample_size2} participants). "
        explanation += "Larger sample sizes generally provide more reliable estimates and greater statistical power."
    elif sample_size2 > sample_size1:
        explanation += f"The second claim is supported by a study with {sample_size2} participants, "
        explanation += f"which is {ratio:.1f} times larger than the study supporting the first claim ({sample_size1} participants). "
        explanation += "Larger sample sizes generally provide more reliable estimates and greater statistical power."
    else:
        explanation += f"Both claims are supported by studies with similar sample sizes ({sample_size1} and {sample_size2} participants)."
    
    power1 = sample_comparison.get("claim1_power")
    power2 = sample_comparison.get("claim2_power")
    
    if power1 is not None and power2 is not None:
        explanation += f" The statistical power was estimated at {power1:.2f} for the first study and {power2:.2f} for the second study."
    
    return explanation


def generate_recency_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for recency resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    recency_comparison = resolution.get("recency_comparison", {})
    year1 = recency_comparison.get("claim1_year", 0)
    year2 = recency_comparison.get("claim2_year", 0)
    year_diff = recency_comparison.get("year_difference", 0)
    
    explanation = "The contradiction was resolved based on publication recency, "
    explanation += "which prioritizes more recent evidence as it may incorporate newer methodologies, "
    explanation += "larger datasets, or reflect evolving understanding of the subject matter. "
    
    if year1 > year2:
        explanation += f"The first claim is supported by a more recent study (published in {year1}), "
        explanation += f"which is {year_diff} years more recent than the study supporting the second claim (published in {year2}). "
    elif year2 > year1:
        explanation += f"The second claim is supported by a more recent study (published in {year2}), "
        explanation += f"which is {year_diff} years more recent than the study supporting the first claim (published in {year1}). "
    else:
        explanation += f"Both claims are supported by studies published in the same year ({year1})."
    
    age1 = recency_comparison.get("claim1_age")
    age2 = recency_comparison.get("claim2_age")
    
    if age1 is not None and age2 is not None:
        if age1 > 15 and age2 > 15:
            explanation += f" It's worth noting that both studies are relatively old ({age1} and {age2} years old), "
            explanation += "which may limit the applicability of this resolution approach."
        elif age1 > 15:
            explanation += f" It's worth noting that the first study is relatively old ({age1} years old), "
            explanation += "which may limit the applicability of this resolution approach."
        elif age2 > 15:
            explanation += f" It's worth noting that the second study is relatively old ({age2} years old), "
            explanation += "which may limit the applicability of this resolution approach."
    
    return explanation


def generate_population_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for population specificity resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    population_comparison = resolution.get("population_comparison", {})
    population1 = population_comparison.get("claim1_population", "")
    population2 = population_comparison.get("claim2_population", "")
    differences = population_comparison.get("differences", [])
    relevance_assessment = population_comparison.get("relevance_assessment", {})
    
    explanation = "The contradiction was resolved based on population specificity considerations, "
    explanation += "which evaluates whether differences in study populations might explain the contradictory findings. "
    
    if differences:
        explanation += "The following population differences were identified: "
        for diff in differences[:3]:  # Limit to first 3 differences for brevity
            category = diff.get("category", "")
            claim1_terms = diff.get("claim1_terms", [])
            claim2_terms = diff.get("claim2_terms", [])
            explanation += f"{category.capitalize()}: {', '.join(claim1_terms)} vs. {', '.join(claim2_terms)}; "
    
    more_relevant = relevance_assessment.get("more_relevant_population")
    relevance_factors = relevance_assessment.get("relevance_factors", [])
    
    if more_relevant == "claim1":
        explanation += f"The population in the first study ({population1}) was assessed as more clinically relevant "
        if relevance_factors:
            explanation += f"due to: {', '.join(relevance_factors)}. "
        explanation += "This suggests that the first claim may be more applicable to typical clinical scenarios."
    elif more_relevant == "claim2":
        explanation += f"The population in the second study ({population2}) was assessed as more clinically relevant "
        if relevance_factors:
            explanation += f"due to: {', '.join(relevance_factors)}. "
        explanation += "This suggests that the second claim may be more applicable to typical clinical scenarios."
    else:
        explanation += f"Both populations ({population1} and {population2}) were assessed as having similar clinical relevance. "
        explanation += "The contradiction may be explained by population differences, but neither population is clearly more relevant for general clinical practice."
    
    return explanation


def generate_methodological_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for methodological quality resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    methodological_comparison = resolution.get("methodological_comparison", {})
    quality_assessment = methodological_comparison.get("quality_assessment", {})
    study_design = methodological_comparison.get("study_design", {})
    blinding = methodological_comparison.get("blinding", {})
    randomization = methodological_comparison.get("randomization", {})
    
    explanation = "The contradiction was resolved based on methodological quality considerations, "
    explanation += "which evaluates the rigor and reliability of the research methods used in each study. "
    
    if study_design:
        claim1_design = study_design.get("claim1_design", "unknown")
        claim2_design = study_design.get("claim2_design", "unknown")
        hierarchy_diff = study_design.get("hierarchy_differential", 0)
        explanation += f"The first study used a {claim1_design} design, while the second used a {claim2_design} design. "
        
        if hierarchy_diff > 0:
            explanation += "The first study's design is higher in the evidence hierarchy. "
        elif hierarchy_diff < 0:
            explanation += "The second study's design is higher in the evidence hierarchy. "
        else:
            explanation += "Both study designs are at similar levels in the evidence hierarchy. "
    
    if blinding:
        claim1_blinding = blinding.get("claim1_blinding", "unknown")
        claim2_blinding = blinding.get("claim2_blinding", "unknown")
        blinding_diff = blinding.get("differential", 0)
        explanation += f"Regarding blinding, the first study used {claim1_blinding} blinding, "
        explanation += f"while the second used {claim2_blinding} blinding. "
        
        if blinding_diff > 0:
            explanation += "The first study had more robust blinding procedures. "
        elif blinding_diff < 0:
            explanation += "The second study had more robust blinding procedures. "
    
    if randomization:
        claim1_randomization = randomization.get("claim1_randomization", "unknown")
        claim2_randomization = randomization.get("claim2_randomization", "unknown")
        randomization_diff = randomization.get("differential", 0)
        explanation += f"For randomization, the first study used {claim1_randomization} randomization, "
        explanation += f"while the second used {claim2_randomization} randomization. "
        
        if randomization_diff > 0:
            explanation += "The first study had more robust randomization procedures. "
        elif randomization_diff < 0:
            explanation += "The second study had more robust randomization procedures. "
    
    if quality_assessment:
        claim1_score = quality_assessment.get("claim1_quality_score", 0)
        claim2_score = quality_assessment.get("claim2_quality_score", 0)
        quality_diff = quality_assessment.get("differential", 0)
        explanation += f"Overall methodological quality was scored at {claim1_score:.2f} for the first study "
        explanation += f"and {claim2_score:.2f} for the second study. "
        
        if quality_diff > 0.3:
            explanation += "The first study demonstrated substantially better methodological quality."
        elif quality_diff < -0.3:
            explanation += "The second study demonstrated substantially better methodological quality."
        else:
            explanation += "Both studies demonstrated similar overall methodological quality."
    
    return explanation


def generate_statistical_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for statistical significance resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    statistical_comparison = resolution.get("statistical_comparison", {})
    p_values = statistical_comparison.get("p_values", {})
    confidence_intervals = statistical_comparison.get("confidence_intervals", {})
    effect_sizes = statistical_comparison.get("effect_sizes", {})
    
    p_value1 = p_values.get("claim1")
    p_value2 = p_values.get("claim2")
    effect_size1 = effect_sizes.get("claim1")
    effect_size2 = effect_sizes.get("claim2")
    ci1 = confidence_intervals.get("claim1")
    ci2 = confidence_intervals.get("claim2")
    
    explanation = "The contradiction was resolved based on statistical significance considerations, "
    explanation += "which evaluates the strength and reliability of the statistical evidence supporting each claim. "
    
    if p_value1 is not None and p_value2 is not None:
        explanation += f"The first study reported a p-value of {p_value1}, "
        explanation += f"while the second reported a p-value of {p_value2}. "
        
        if p_value1 <= 0.05 and p_value2 > 0.05:
            explanation += "The first study's results are statistically significant, while the second's are not. "
        elif p_value2 <= 0.05 and p_value1 > 0.05:
            explanation += "The second study's results are statistically significant, while the first's are not. "
        elif p_value1 <= 0.05 and p_value2 <= 0.05:
            explanation += "Both studies report statistically significant results. "
            if p_value1 < p_value2:
                explanation += f"The first study shows stronger statistical significance (smaller p-value). "
            elif p_value2 < p_value1:
                explanation += f"The second study shows stronger statistical significance (smaller p-value). "
        else:
            explanation += "Neither study reports statistically significant results. "
    
    if effect_size1 is not None and effect_size2 is not None:
        explanation += f"The effect size was {effect_size1} in the first study and {effect_size2} in the second. "
        
        if abs(effect_size1) > abs(effect_size2):
            explanation += "The first study demonstrated a larger effect size. "
        elif abs(effect_size2) > abs(effect_size1):
            explanation += "The second study demonstrated a larger effect size. "
        else:
            explanation += "Both studies demonstrated similar effect sizes. "
    
    if isinstance(ci1, list) and len(ci1) == 2 and isinstance(ci2, list) and len(ci2) == 2:
        width1 = ci1[1] - ci1[0]
        width2 = ci2[1] - ci2[0]
        explanation += f"The 95% confidence interval was [{ci1[0]}, {ci1[1]}] in the first study "
        explanation += f"and [{ci2[0]}, {ci2[1]}] in the second. "
        
        if width1 < width2:
            explanation += "The first study had a narrower confidence interval, suggesting more precise results. "
        elif width2 < width1:
            explanation += "The second study had a narrower confidence interval, suggesting more precise results. "
        else:
            explanation += "Both studies had similar confidence interval widths. "
    
    return explanation


def generate_combined_evidence_explanation(resolution: Dict[str, Any]) -> str:
    """Generate explanation for combined evidence resolution.
    
    Args:
        resolution: Resolution result
        
    Returns:
        Explanation text
    """
    combined_evidence = resolution.get("combined_evidence", {})
    strategies_applied = combined_evidence.get("strategies_applied", [])
    recommendation_counts = combined_evidence.get("recommendation_counts", {})
    weighted_scores = combined_evidence.get("weighted_scores", {})
    
    claim1_count = recommendation_counts.get("favor_claim1", 0)
    claim2_count = recommendation_counts.get("favor_claim2", 0)
    inconclusive_count = recommendation_counts.get("inconclusive", 0)
    conditional_count = recommendation_counts.get("conditional", 0)
    further_research_count = recommendation_counts.get("further_research", 0)
    
    claim1_score = weighted_scores.get("claim1_score", 0)
    claim2_score = weighted_scores.get("claim2_score", 0)
    
    explanation = "The contradiction was resolved using a combined evidence approach, "
    explanation += "which integrates results from multiple resolution strategies to reach a more robust conclusion. "
    explanation += f"A total of {len(strategies_applied)} resolution strategies were applied: "
    explanation += ", ".join([s.replace("_", " ").title() for s in strategies_applied]) + ". "
    
    explanation += f"Of these strategies, {claim1_count} favored the first claim, "
    explanation += f"{claim2_count} favored the second claim, {inconclusive_count} were inconclusive, "
    explanation += f"{conditional_count} suggested a conditional resolution, and "
    explanation += f"{further_research_count} suggested that further research is needed. "
    
    explanation += "When weighted by confidence, "
    explanation += f"the first claim received a score of {claim1_score:.2f} and "
    explanation += f"the second claim received a score of {claim2_score:.2f}. "
    
    if claim1_count > claim2_count and claim1_count > inconclusive_count:
        explanation += "The majority of strategies favored the first claim, "
        explanation += "leading to the conclusion that it is more likely to be correct."
    elif claim2_count > claim1_count and claim2_count > inconclusive_count:
        explanation += "The majority of strategies favored the second claim, "
        explanation += "leading to the conclusion that it is more likely to be correct."
    elif conditional_count > 0 and conditional_count >= claim1_count and conditional_count >= claim2_count:
        explanation += "Multiple strategies suggested that the resolution depends on specific clinical context, "
        explanation += "indicating that both claims may be valid under different circumstances."
    elif further_research_count > 0 and further_research_count >= claim1_count and further_research_count >= claim2_count:
        explanation += "Multiple strategies suggested that further research is needed, "
        explanation += "indicating that the current evidence is insufficient to resolve the contradiction."
    else:
        explanation += "No clear consensus emerged from the different strategies, "
        explanation += "suggesting that the evidence is insufficient to conclusively resolve the contradiction."
    
    return explanation


def generate_clinical_implications(
    contradiction: Dict[str, Any],
    resolution: Dict[str, Any],
    clinical_significance: str
) -> str:
    """Generate clinical implications for contradiction resolution.
    
    Args:
        contradiction: Classified contradiction
        resolution: Resolution result
        clinical_significance: Clinical significance of the contradiction
        
    Returns:
        Clinical implications text
    """
    recommendation = resolution.get("recommendation", RecommendationType.INCONCLUSIVE)
    recommended_claim = resolution.get("recommended_claim", "")
    confidence = resolution.get("confidence", ResolutionConfidence.LOW)
    
    implications = "Clinical Implications: "
    
    if clinical_significance == ClinicalSignificance.HIGH:
        implications += "This contradiction has high clinical significance, "
        implications += "meaning it could substantially impact patient outcomes, treatment decisions, or diagnostic accuracy. "
    elif clinical_significance == ClinicalSignificance.MODERATE:
        implications += "This contradiction has moderate clinical significance, "
        implications += "meaning it may impact treatment decisions or patient management but is unlikely to cause severe harm. "
    elif clinical_significance == ClinicalSignificance.LOW:
        implications += "This contradiction has low clinical significance, "
        implications += "meaning it is unlikely to substantially impact patient outcomes or treatment decisions. "
    else:
        implications += "The clinical significance of this contradiction is unknown. "
    
    if recommendation == RecommendationType.FAVOR_CLAIM1 or recommendation == RecommendationType.FAVOR_CLAIM2:
        if confidence == ResolutionConfidence.HIGH:
            implications += f"Clinicians should strongly consider the recommended claim ({recommended_claim}) "
            implications += "when making clinical decisions. "
        elif confidence == ResolutionConfidence.MODERATE:
            implications += f"Clinicians should consider the recommended claim ({recommended_claim}) "
            implications += "when making clinical decisions, but should remain aware of the contradictory evidence. "
        else:  # LOW or VERY_LOW
            implications += f"While the evidence slightly favors one claim ({recommended_claim}), "
            implications += "clinicians should exercise caution and consider both perspectives when making clinical decisions. "
    elif recommendation == RecommendationType.CONDITIONAL:
        implications += "Clinicians should consider the specific clinical context and patient characteristics "
        implications += "when deciding which claim to apply. Different patients or scenarios may warrant different approaches. "
    elif recommendation == RecommendationType.FURTHER_RESEARCH:
        implications += "Given the current state of evidence, clinicians should exercise caution "
        implications += "and rely on established clinical guidelines until further research clarifies this contradiction. "
    else:  # INCONCLUSIVE
        implications += "Given the inconclusive resolution, clinicians should exercise caution, "
        implications += "consider both perspectives, and rely on established clinical guidelines when making decisions. "
    
    return implications


def generate_limitations(
    contradiction: Dict[str, Any],
    resolution: Dict[str, Any],
    strategy: ResolutionStrategy
) -> str:
    """Generate limitations for contradiction resolution.
    
    Args:
        contradiction: Classified contradiction
        resolution: Resolution result
        strategy: Resolution strategy used
        
    Returns:
        Limitations text
    """
    limitations = "Limitations: "
    
    if strategy == ResolutionStrategy.EVIDENCE_HIERARCHY:
        limitations += "The evidence hierarchy approach has limitations. "
        limitations += "Higher-level study designs don't always guarantee better quality research, "
        limitations += "and lower-level designs may sometimes be more appropriate for certain research questions. "
        limitations += "This approach also doesn't account for the quality of implementation within each study design."
    elif strategy == ResolutionStrategy.SAMPLE_SIZE_WEIGHTING:
        limitations += "The sample size approach has limitations. "
        limitations += "Larger sample sizes don't always guarantee better quality research, "
        limitations += "especially if there are methodological flaws or biases in the study design. "
        limitations += "Smaller, well-designed studies may sometimes provide more reliable evidence than larger, flawed studies."
    elif strategy == ResolutionStrategy.RECENCY_PREFERENCE:
        limitations += "The recency approach has limitations. "
        limitations += "More recent studies aren't always better than older ones, "
        limitations += "especially if the older studies used more rigorous methods or had larger sample sizes. "
        limitations += "This approach also doesn't account for the quality of the studies."
    elif strategy == ResolutionStrategy.POPULATION_SPECIFICITY:
        limitations += "The population specificity approach has limitations. "
        limitations += "Determining which population is more clinically relevant is subjective "
        limitations += "and depends on the specific clinical context. "
        limitations += "This approach also doesn't account for other methodological differences between studies."
    elif strategy == ResolutionStrategy.METHODOLOGICAL_QUALITY:
        limitations += "The methodological quality approach has limitations. "
        limitations += "Quality assessment is inherently subjective and may not capture all relevant aspects of study design. "
        limitations += "Different quality assessment tools may yield different results, "
        limitations += "and the relative importance of different methodological features is not always clear."
    elif strategy == ResolutionStrategy.STATISTICAL_SIGNIFICANCE:
        limitations += "The statistical significance approach has limitations. "
        limitations += "P-values and statistical significance don't always reflect clinical significance or practical importance. "
        limitations += "This approach also doesn't account for other methodological differences between studies, "
        limitations += "and may be affected by publication bias or selective reporting."
    elif strategy == ResolutionStrategy.COMBINED_EVIDENCE:
        limitations += "The combined evidence approach has limitations. "
        limitations += "It assumes that all resolution strategies are equally valid and applicable to the contradiction at hand. "
        limitations += "The approach may also be affected by limitations in the underlying data and methods used in each strategy."
    
    limitations += " Additionally, this resolution is based on the available information, "
    limitations += "which may be incomplete. The assessment doesn't account for unpublished studies or data, "
    limitations += "and may be affected by publication bias. The resolution should be updated as new evidence emerges."
    
    return limitations


def generate_references(
    contradiction: Dict[str, Any],
    resolution: Dict[str, Any],
    strategy: ResolutionStrategy
) -> List[str]:
    """Generate references for contradiction resolution.
    
    Args:
        contradiction: Classified contradiction
        resolution: Resolution result
        strategy: Resolution strategy used
        
    Returns:
        List of references
    """
    references = [
        "Guyatt GH, et al. GRADE: an emerging consensus on rating quality of evidence and strength of recommendations. BMJ. 2008;336(7650):924-926.",
        "Higgins JPT, et al. Cochrane Handbook for Systematic Reviews of Interventions. Version 6.3. Cochrane, 2022.",
        "Ioannidis JPA. Contradicted and initially stronger effects in highly cited clinical research. JAMA. 2005;294(2):218-228."
    ]
    
    if strategy == ResolutionStrategy.EVIDENCE_HIERARCHY:
        references.append("Burns PB, et al. The levels of evidence and their role in evidence-based medicine. Plast Reconstr Surg. 2011;128(1):305-310.")
    elif strategy == ResolutionStrategy.SAMPLE_SIZE_WEIGHTING:
        references.append("Button KS, et al. Power failure: why small sample size undermines the reliability of neuroscience. Nat Rev Neurosci. 2013;14(5):365-376.")
    elif strategy == ResolutionStrategy.RECENCY_PREFERENCE:
        references.append("Shojania KG, et al. How quickly do systematic reviews go out of date? A survival analysis. Ann Intern Med. 2007;147(4):224-233.")
    elif strategy == ResolutionStrategy.POPULATION_SPECIFICITY:
        references.append("Rothwell PM. External validity of randomised controlled trials: 'to whom do the results of this trial apply?' Lancet. 2005;365(9453):82-93.")
    elif strategy == ResolutionStrategy.METHODOLOGICAL_QUALITY:
        references.append("Higgins JPT, et al. The Cochrane Collaboration's tool for assessing risk of bias in randomised trials. BMJ. 2011;343:d5928.")
    elif strategy == ResolutionStrategy.STATISTICAL_SIGNIFICANCE:
        references.append("Wasserstein RL, Lazar NA. The ASA Statement on p-Values: Context, Process, and Purpose. Am Stat. 2016;70(2):129-133.")
    
    return references
