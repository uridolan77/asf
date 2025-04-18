"""
Module description.

This module provides functionality for...
"""
import re
import logging
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class ContradictionType(str, Enum):
    Contradiction type enum.
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))
    
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
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None
    }
    
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
    
    claim1_lower = claim1.lower()
    claim2_lower = claim2.lower()
    
    for pattern, replacement in negation_patterns:
        if pattern in claim1_lower and pattern not in claim2_lower:
            modified_claim1 = claim1_lower.replace(pattern, replacement)
            
            similarity = calculate_text_similarity(modified_claim1, claim2_lower)
            
            if similarity > 0.8:
                result["is_contradiction"] = True
                result["score"] = similarity
                result["confidence"] = ContradictionConfidence.HIGH
                result["explanation"] = f"Claim 1 is a negation of Claim 2 with similarity {similarity:.2f}."
                return result
        
        if pattern in claim2_lower and pattern not in claim1_lower:
            modified_claim2 = claim2_lower.replace(pattern, replacement)
            
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
        """
        detect_statistical_contradiction function.
        
        This function provides functionality for...
        Args:
            claim1: Description of claim1
            claim2: Description of claim2
            metadata1: Description of metadata1
            metadata2: Description of metadata2
        
        Returns:
            Description of return value
        """
    claim2: str,
    metadata1: Optional[Dict[str, Any]],
    metadata2: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None
    }
    
    if not metadata1 or not metadata2:
        return result
    
    p_value1 = metadata1.get("p_value")
    p_value2 = metadata2.get("p_value")
    
    if p_value1 is not None and p_value2 is not None:
        is_significant1 = p_value1 < 0.05
        is_significant2 = p_value2 < 0.05
        
        if is_significant1 != is_significant2:
            p_value_diff = abs(p_value1 - p_value2)
            score = min(1.0, p_value_diff)
            
            result["is_contradiction"] = True
            result["score"] = score
            
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW
            
            result["explanation"] = f"Statistical contradiction: p-value1={p_value1:.3f} (significant={is_significant1}), p-value2={p_value2:.3f} (significant={is_significant2})."
    
    effect_size1 = metadata1.get("effect_size")
    effect_size2 = metadata2.get("effect_size")
    
    if effect_size1 is not None and effect_size2 is not None:
        if effect_size1 * effect_size2 < 0:
            effect_size_diff = abs(effect_size1 - effect_size2)
            score = min(1.0, effect_size_diff)
            
            if score > result["score"]:
                result["is_contradiction"] = True
                result["score"] = score
                
                if score > 0.9:
                    result["confidence"] = ContradictionConfidence.HIGH
                elif score > 0.8:
                    result["confidence"] = ContradictionConfidence.MEDIUM
                else:
                    result["confidence"] = ContradictionConfidence.LOW
                
                result["explanation"] = f"Statistical contradiction: effect_size1={effect_size1:.3f}, effect_size2={effect_size2:.3f} (opposite directions)."
    
    return result

def detect_contradiction(
    claim1: str,
        """
        detect_contradiction function.
        
        This function provides functionality for...
        Args:
            claim1: Description of claim1
            claim2: Description of claim2
            metadata1: Description of metadata1
            metadata2: Description of metadata2
        
        Returns:
            Description of return value
        """
    claim2: str,
    metadata1: Optional[Dict[str, Any]] = None,
    metadata2: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    logger.info(f"Detecting contradiction between claims: '{claim1}' and '{claim2}'")
    
    result = {
        "is_contradiction": False,
        "contradiction_score": 0.0,
        "contradiction_type": ContradictionType.NONE,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None,
        "methods_used": [],
        "details": {}
    }
    
    negation_result = detect_negation_contradiction(claim1, claim2)
    result["methods_used"].append("negation")
    result["details"]["negation"] = negation_result
    
    if negation_result["is_contradiction"]:
        result["is_contradiction"] = True
        result["contradiction_score"] = negation_result["score"]
        result["contradiction_type"] = ContradictionType.NEGATION
        result["confidence"] = negation_result["confidence"]
        result["explanation"] = negation_result["explanation"]
    
    if metadata1 and metadata2:
        statistical_result = detect_statistical_contradiction(claim1, claim2, metadata1, metadata2)
        result["methods_used"].append("statistical")
        result["details"]["statistical"] = statistical_result
        
        if statistical_result["is_contradiction"] and statistical_result["score"] > result["contradiction_score"]:
            result["is_contradiction"] = True
            result["contradiction_score"] = statistical_result["score"]
            result["contradiction_type"] = ContradictionType.STATISTICAL
            result["confidence"] = statistical_result["confidence"]
            result["explanation"] = statistical_result["explanation"]
    
    logger.info(f"Contradiction detection result: {result['is_contradiction']} (score: {result['contradiction_score']}, type: {result['contradiction_type']})")
    return result

def main():
    """Main function.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    logger.info("Starting contradiction detection tests...")
    
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        
        result = detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"]
        )
        
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
