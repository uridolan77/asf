"""
Utilities for medical contradiction resolution.

This module provides utility functions for resolving contradictions
in medical literature.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from asf.medical.ml.services.resolution.resolution_models import (
    ResolutionStrategy,
    ResolutionConfidence,
    RecommendationType,
    STUDY_DESIGN_HIERARCHY,
    STUDY_DESIGN_KEYWORDS
)

# Set up logging
logger = logging.getLogger(__name__)

def extract_metadata(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract metadata from an article.
    
    Args:
        article: Article data
        
    Returns:
        Metadata dictionary
    """
    metadata = {}
    
    # Extract basic metadata
    if "pmid" in article:
        metadata["pmid"] = article["pmid"]
    if "doi" in article:
        metadata["doi"] = article["doi"]
    if "title" in article:
        metadata["title"] = article["title"]
    if "journal" in article:
        metadata["journal"] = article["journal"]
    if "publication_year" in article:
        metadata["publication_year"] = article["publication_year"]
    if "authors" in article:
        metadata["authors"] = article["authors"]
    
    # Extract study design
    if "study_design" in article:
        metadata["study_design"] = article["study_design"]
    elif "abstract" in article:
        metadata["study_design"] = detect_study_design(article["abstract"])
    
    # Extract sample size
    if "sample_size" in article:
        metadata["sample_size"] = article["sample_size"]
    elif "abstract" in article:
        metadata["sample_size"] = extract_sample_size(article["abstract"])
    
    # Extract population
    if "population" in article:
        metadata["population"] = article["population"]
    elif "abstract" in article:
        metadata["population"] = extract_population(article["abstract"])
    
    # Extract statistical information
    if "p_value" in article:
        metadata["p_value"] = article["p_value"]
    if "confidence_interval" in article:
        metadata["confidence_interval"] = article["confidence_interval"]
    if "effect_size" in article:
        metadata["effect_size"] = article["effect_size"]
    
    # Extract methodological information
    if "blinding" in article:
        metadata["blinding"] = article["blinding"]
    if "randomization" in article:
        metadata["randomization"] = article["randomization"]
    if "allocation_concealment" in article:
        metadata["allocation_concealment"] = article["allocation_concealment"]
    if "intention_to_treat" in article:
        metadata["intention_to_treat"] = article["intention_to_treat"]
    
    return metadata

def detect_study_design(text: str) -> str:
    """
    Detect study design from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Detected study design
    """
    text = text.lower()
    
    for design, keywords in STUDY_DESIGN_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text:
                return design
    
    return "unknown"

def extract_sample_size(text: str) -> int:
    """
    Extract sample size from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Extracted sample size
    """
    # This is a placeholder for a more sophisticated extraction method
    # In a real implementation, this would use regex or NLP to extract sample size
    return 0

def extract_population(text: str) -> str:
    """
    Extract population from text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Extracted population
    """
    # This is a placeholder for a more sophisticated extraction method
    # In a real implementation, this would use NLP to extract population
    return ""

def calculate_confidence_from_hierarchy_diff(higher_pos: int, lower_pos: int) -> ResolutionConfidence:
    """
    Calculate confidence based on difference in hierarchy positions.
    
    Args:
        higher_pos: Higher position in hierarchy
        lower_pos: Lower position in hierarchy
        
    Returns:
        Confidence level
    """
    diff = higher_pos - lower_pos
    
    if diff >= 3:
        return ResolutionConfidence.HIGH
    elif diff >= 2:
        return ResolutionConfidence.MODERATE
    elif diff >= 1:
        return ResolutionConfidence.LOW
    else:
        return ResolutionConfidence.VERY_LOW

def assess_population_relevance(population1: str, population2: str, differences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Assess which population is more clinically relevant.
    
    Args:
        population1: Population description for claim 1
        population2: Population description for claim 2
        differences: List of detected population differences
        
    Returns:
        Population relevance assessment
    """
    # Initialize result
    result = {
        "more_relevant_population": None,
        "relevance_score": 0.0,
        "relevance_factors": []
    }
    
    # Check for specific population characteristics that might indicate
    # higher clinical relevance
    
    # 1. Check for specific disease populations vs. general population
    pop1_specific_disease = any(term in population1.lower() for term in [
        "patient", "disease", "disorder", "syndrome", "condition", "diagnosed"
    ])
    pop2_specific_disease = any(term in population2.lower() for term in [
        "patient", "disease", "disorder", "syndrome", "condition", "diagnosed"
    ])
    
    if pop1_specific_disease and not pop2_specific_disease:
        result["more_relevant_population"] = "claim1"
        result["relevance_score"] += 0.3
        result["relevance_factors"].append("specific_disease_population")
    elif pop2_specific_disease and not pop1_specific_disease:
        result["more_relevant_population"] = "claim2"
        result["relevance_score"] += 0.3
        result["relevance_factors"].append("specific_disease_population")
    
    # 2. Check for comorbidities
    pop1_comorbidities = any(term in population1.lower() for term in [
        "comorbid", "comorbidity", "multimorbidity", "multiple conditions"
    ])
    pop2_comorbidities = any(term in population2.lower() for term in [
        "comorbid", "comorbidity", "multimorbidity", "multiple conditions"
    ])
    
    if pop1_comorbidities and not pop2_comorbidities:
        # If current recommendation is not claim1, update it
        if result["more_relevant_population"] != "claim1":
            result["more_relevant_population"] = "claim1"
        result["relevance_score"] += 0.2
        result["relevance_factors"].append("comorbidities")
    elif pop2_comorbidities and not pop1_comorbidities:
        # If current recommendation is not claim2, update it
        if result["more_relevant_population"] != "claim2":
            result["more_relevant_population"] = "claim2"
        result["relevance_score"] += 0.2
        result["relevance_factors"].append("comorbidities")
    
    # 3. Check for real-world vs. highly selected population
    pop1_real_world = any(term in population1.lower() for term in [
        "real-world", "real world", "pragmatic", "community", "primary care"
    ])
    pop2_real_world = any(term in population2.lower() for term in [
        "real-world", "real world", "pragmatic", "community", "primary care"
    ])
    
    if pop1_real_world and not pop2_real_world:
        # If current recommendation is not claim1, update it only if score is low
        if result["more_relevant_population"] != "claim1" or result["relevance_score"] < 0.3:
            result["more_relevant_population"] = "claim1"
        result["relevance_score"] += 0.2
        result["relevance_factors"].append("real_world_population")
    elif pop2_real_world and not pop1_real_world:
        # If current recommendation is not claim2, update it only if score is low
        if result["more_relevant_population"] != "claim2" or result["relevance_score"] < 0.3:
            result["more_relevant_population"] = "claim2"
        result["relevance_score"] += 0.2
        result["relevance_factors"].append("real_world_population")
    
    # If relevance score is too low, don't make a recommendation
    if result["relevance_score"] < 0.2:
        result["more_relevant_population"] = None
    
    return result
