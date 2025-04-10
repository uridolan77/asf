#!/usr/bin/env python
"""
Simple test script for the contradiction detection feature.

This script implements a simplified version of the contradiction detection logic
and tests it with sample data.
"""

import re
import json
import logging
from enum import Enum
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContradictionType(str, Enum):
    """Contradiction type enum."""
    NONE = "none"
    DIRECT = "direct"
    NEGATION = "negation"
    STATISTICAL = "statistical"
    UNKNOWN = "unknown"

class ContradictionConfidence(str, Enum):
    """Contradiction confidence enum."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

# Test data
TEST_CLAIMS = [
    {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
        "metadata1": {
            "publication_date": "2020-01-01",
            "study_design": "randomized controlled trial",
            "sample_size": 1000,
            "p_value": 0.001,
            "effect_size": 0.3
        },
        "metadata2": {
            "publication_date": "2021-06-15",
            "study_design": "randomized controlled trial",
            "sample_size": 2000,
            "p_value": 0.45,
            "effect_size": -0.05
        }
    },
    {
        "claim1": "Antibiotics are effective for treating bacterial pneumonia.",
        "claim2": "Antibiotics are ineffective for treating bacterial pneumonia.",
        "metadata1": None,
        "metadata2": None
    },
    {
        "claim1": "Regular exercise improves cardiovascular health.",
        "claim2": "Physical activity has positive effects on heart health.",
        "metadata1": None,
        "metadata2": None
    },
    {
        "claim1": "Vitamin D supplementation prevents respiratory infections.",
        "claim2": "Vitamin D supplementation has no effect on respiratory infection risk.",
        "metadata1": {
            "publication_date": "2019-05-10",
            "study_design": "meta-analysis",
            "sample_size": 5000,
            "p_value": 0.02,
            "effect_size": 0.15
        },
        "metadata2": {
            "publication_date": "2022-03-20",
            "study_design": "randomized controlled trial",
            "sample_size": 3000,
            "p_value": 0.3,
            "effect_size": 0.05
        }
    }
]

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using a simple Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    # Tokenize texts
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def detect_negation_contradiction(claim1: str, claim2: str) -> Dict[str, Any]:
    """
    Detect negation contradiction between two claims.
    
    Args:
        claim1: First claim
        claim2: Second claim
        
    Returns:
        Negation contradiction detection result
    """
    # Initialize result
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None
    }
    
    # Negation patterns
    negation_patterns = [
        ("not ", ""),
        ("no ", ""),
        ("never ", ""),
        ("doesn't ", "does "),
        ("does not ", "does "),
        ("don't ", "do "),
        ("do not ", "do "),
        ("isn't ", "is "),
        ("is not ", "is "),
        ("aren't ", "are "),
        ("are not ", "are "),
        ("ineffective ", "effective "),
        ("inefficacy ", "efficacy ")
    ]
    
    # Convert claims to lowercase for case-insensitive matching
    claim1_lower = claim1.lower()
    claim2_lower = claim2.lower()
    
    # Check if one claim is the negation of the other
    for pattern, replacement in negation_patterns:
        # Check if claim1 contains negation and claim2 doesn't
        if pattern in claim1_lower and pattern not in claim2_lower:
            # Replace negation in claim1
            modified_claim1 = claim1_lower.replace(pattern, replacement)
            
            # Calculate similarity between modified claim1 and claim2
            similarity = calculate_text_similarity(modified_claim1, claim2_lower)
            
            if similarity > 0.8:
                result["is_contradiction"] = True
                result["score"] = similarity
                result["confidence"] = ContradictionConfidence.HIGH
                result["explanation"] = f"Claim 1 is a negation of Claim 2 with similarity {similarity:.2f}."
                return result
        
        # Check if claim2 contains negation and claim1 doesn't
        if pattern in claim2_lower and pattern not in claim1_lower:
            # Replace negation in claim2
            modified_claim2 = claim2_lower.replace(pattern, replacement)
            
            # Calculate similarity between claim1 and modified claim2
            similarity = calculate_text_similarity(claim1_lower, modified_claim2)
            
            if similarity > 0.8:
                result["is_contradiction"] = True
                result["score"] = similarity
                result["confidence"] = ContradictionConfidence.HIGH
                result["explanation"] = f"Claim 2 is a negation of Claim 1 with similarity {similarity:.2f}."
                return result
    
    return result

def detect_statistical_contradiction(
    claim1: str,
    claim2: str,
    metadata1: Optional[Dict[str, Any]],
    metadata2: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Detect statistical contradiction between two claims.
    
    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim
        
    Returns:
        Statistical contradiction detection result
    """
    # Initialize result
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None
    }
    
    # Check if metadata is available
    if not metadata1 or not metadata2:
        return result
    
    # Check if p-values are available
    p_value1 = metadata1.get("p_value")
    p_value2 = metadata2.get("p_value")
    
    if p_value1 is not None and p_value2 is not None:
        # Check if one p-value is significant and the other is not
        is_significant1 = p_value1 < 0.05
        is_significant2 = p_value2 < 0.05
        
        if is_significant1 != is_significant2:
            # Calculate score based on p-value difference
            p_value_diff = abs(p_value1 - p_value2)
            score = min(1.0, p_value_diff)
            
            # Set result
            result["is_contradiction"] = True
            result["score"] = score
            
            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW
            
            # Generate explanation
            result["explanation"] = f"Statistical contradiction: p-value1={p_value1:.3f} (significant={is_significant1}), p-value2={p_value2:.3f} (significant={is_significant2})."
    
    # Check if effect sizes are available
    effect_size1 = metadata1.get("effect_size")
    effect_size2 = metadata2.get("effect_size")
    
    if effect_size1 is not None and effect_size2 is not None:
        # Check if effect sizes have opposite signs
        if effect_size1 * effect_size2 < 0:
            # Calculate score based on effect size difference
            effect_size_diff = abs(effect_size1 - effect_size2)
            score = min(1.0, effect_size_diff)
            
            # Set result if score is higher than current score
            if score > result["score"]:
                result["is_contradiction"] = True
                result["score"] = score
                
                # Set confidence based on score
                if score > 0.9:
                    result["confidence"] = ContradictionConfidence.HIGH
                elif score > 0.8:
                    result["confidence"] = ContradictionConfidence.MEDIUM
                else:
                    result["confidence"] = ContradictionConfidence.LOW
                
                # Generate explanation
                result["explanation"] = f"Statistical contradiction: effect_size1={effect_size1:.3f}, effect_size2={effect_size2:.3f} (opposite directions)."
    
    return result

def detect_contradiction(
    claim1: str,
    claim2: str,
    metadata1: Optional[Dict[str, Any]] = None,
    metadata2: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Detect contradiction between two claims.
    
    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim
        
    Returns:
        Contradiction detection result
    """
    logger.info(f"Detecting contradiction between claims: '{claim1}' and '{claim2}'")
    
    # Initialize result
    result = {
        "is_contradiction": False,
        "contradiction_score": 0.0,
        "contradiction_type": ContradictionType.NONE,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None,
        "methods_used": [],
        "details": {}
    }
    
    # Detect negation contradiction
    negation_result = detect_negation_contradiction(claim1, claim2)
    result["methods_used"].append("negation")
    result["details"]["negation"] = negation_result
    
    # Update result if negation contradiction is detected
    if negation_result["is_contradiction"]:
        result["is_contradiction"] = True
        result["contradiction_score"] = negation_result["score"]
        result["contradiction_type"] = ContradictionType.NEGATION
        result["confidence"] = negation_result["confidence"]
        result["explanation"] = negation_result["explanation"]
    
    # Detect statistical contradiction if metadata is available
    if metadata1 and metadata2:
        statistical_result = detect_statistical_contradiction(claim1, claim2, metadata1, metadata2)
        result["methods_used"].append("statistical")
        result["details"]["statistical"] = statistical_result
        
        # Update result if statistical contradiction is detected and has higher score
        if statistical_result["is_contradiction"] and statistical_result["score"] > result["contradiction_score"]:
            result["is_contradiction"] = True
            result["contradiction_score"] = statistical_result["score"]
            result["contradiction_type"] = ContradictionType.STATISTICAL
            result["confidence"] = statistical_result["confidence"]
            result["explanation"] = statistical_result["explanation"]
    
    logger.info(f"Contradiction detection result: {result['is_contradiction']} (score: {result['contradiction_score']}, type: {result['contradiction_type']})")
    return result

def main():
    """Main function."""
    logger.info("Starting contradiction detection tests...")
    
    # Test each claim pair
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        
        # Detect contradiction
        result = detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"]
        )
        
        # Print result
        logger.info(f"Result: {'Contradiction' if result['is_contradiction'] else 'No contradiction'}")
        logger.info(f"Score: {result['contradiction_score']:.2f}")
        logger.info(f"Type: {result['contradiction_type']}")
        logger.info(f"Confidence: {result['confidence']}")
        if result["explanation"]:
            logger.info(f"Explanation: {result['explanation']}")
        logger.info("---")
    
    logger.info("All tests completed successfully")

if __name__ == "__main__":
    main()
