#!/usr/bin/env python
"""
Simple test script for BioMedLM integration with contradiction detection.

This script implements a simplified version of the BioMedLM integration with
the contradiction detection logic and tests it with sample data.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

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

class MockBioMedLMService:
    """
    Mock implementation of the BioMedLM service for testing.
    
    This class provides a simplified implementation of the BioMedLM service
    that can be used for testing without requiring the actual model.
    """
    
    def __init__(self):
        """Initialize the mock BioMedLM service."""
        logger.info("Mock BioMedLM service initialized")
        
        # Negation patterns for detecting contradictions
        self.negation_patterns = [
            ("not ", ""),
            ("no ", ""),
            ("never ", ""),
            ("doesn't ", "does "),
            ("does not ", "does "),
            ("don't ", "do "),
            ("do not ", "do "),
            ("ineffective ", "effective "),
            ("inefficacy ", "efficacy ")
        ]
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text using a simple bag-of-words approach.
        
        Args:
            text: Text to encode
            
        Returns:
            Text embedding
        """
        # Tokenize text
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Create a simple embedding (random but deterministic)
        np.random.seed(sum(ord(c) for c in text))
        embedding = np.random.rand(768)
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Check for exact match
        if text1.lower() == text2.lower():
            return 1.0
        
        # Check for negation
        for pattern, replacement in self.negation_patterns:
            # Check if text1 contains negation and text2 doesn't
            if pattern in text1.lower() and pattern not in text2.lower():
                # Replace negation in text1
                modified_text1 = text1.lower().replace(pattern, replacement)
                
                # Check if modified text1 is similar to text2
                if self._calculate_text_similarity(modified_text1, text2.lower()) > 0.8:
                    return 0.1  # Low similarity for contradictions
            
            # Check if text2 contains negation and text1 doesn't
            if pattern in text2.lower() and pattern not in text1.lower():
                # Replace negation in text2
                modified_text2 = text2.lower().replace(pattern, replacement)
                
                # Check if text1 is similar to modified text2
                if self._calculate_text_similarity(text1.lower(), modified_text2) > 0.8:
                    return 0.1  # Low similarity for contradictions
        
        # Get embeddings
        embedding1 = self.encode(text1)
        embedding2 = self.encode(text2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        
        return float(similarity)
    
    def detect_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """
        Detect contradiction between two claims.
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            Tuple of (is_contradiction, confidence)
        """
        # Calculate similarity
        similarity = self.calculate_similarity(claim1, claim2)
        
        # Check for negation
        for pattern, replacement in self.negation_patterns:
            # Check if claim1 contains negation and claim2 doesn't
            if pattern in claim1.lower() and pattern not in claim2.lower():
                # Replace negation in claim1
                modified_claim1 = claim1.lower().replace(pattern, replacement)
                
                # Check if modified claim1 is similar to claim2
                if self._calculate_text_similarity(modified_claim1, claim2.lower()) > 0.8:
                    return True, 0.9  # High confidence for negation contradictions
            
            # Check if claim2 contains negation and claim1 doesn't
            if pattern in claim2.lower() and pattern not in claim1.lower():
                # Replace negation in claim2
                modified_claim2 = claim2.lower().replace(pattern, replacement)
                
                # Check if claim1 is similar to modified claim2
                if self._calculate_text_similarity(claim1.lower(), modified_claim2) > 0.8:
                    return True, 0.9  # High confidence for negation contradictions
        
        # Invert similarity to get contradiction score
        contradiction_score = 1.0 - similarity
        
        # Determine if it's a contradiction
        is_contradiction = contradiction_score > 0.5
        
        return is_contradiction, contradiction_score
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
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

class ContradictionService:
    """
    Service for detecting contradictions between medical claims.
    
    This service provides methods for detecting contradictions between medical claims
    using rule-based approaches, text similarity, and BioMedLM for semantic analysis.
    """
    
    def __init__(self, biomedlm_service=None):
        """
        Initialize the contradiction service.
        
        Args:
            biomedlm_service: BioMedLM service for semantic analysis
        """
        # Initialize BioMedLM service
        self.biomedlm_service = biomedlm_service
        
        # Negation patterns for rule-based contradiction detection
        self.negation_patterns = [
            ("not ", ""),
            ("no ", ""),
            ("never ", ""),
            ("doesn't ", "does "),
            ("does not ", "does "),
            ("don't ", "do "),
            ("do not ", "do "),
            ("ineffective ", "effective "),
            ("inefficacy ", "efficacy ")
        ]
        
        # Contradiction keywords for rule-based contradiction detection
        self.contradiction_keywords = [
            "contrary",
            "opposite",
            "disagree",
            "conflict",
            "contradict",
            "inconsistent",
            "refute",
            "disprove",
            "rebut",
            "counter",
            "oppose",
            "challenge",
            "dispute",
            "reject",
            "deny",
            "negate"
        ]
        
        # Thresholds for contradiction detection
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.NEGATION: 0.8,
            ContradictionType.STATISTICAL: 0.7
        }
        
        logger.info("Contradiction service initialized")
    
    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        threshold: float = 0.7,
        use_biomedlm: bool = True
    ) -> Dict[str, Any]:
        """
        Detect contradiction between two claims.
        
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim
            threshold: Contradiction detection threshold
            use_biomedlm: Whether to use BioMedLM for semantic analysis
            
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
        
        # Detect semantic contradiction using BioMedLM if available and requested
        if use_biomedlm and self.biomedlm_service:
            semantic_result = await self._detect_semantic_contradiction(claim1, claim2)
            result["methods_used"].append("biomedlm")
            result["details"]["semantic"] = semantic_result
            
            # Update result if semantic contradiction is detected
            if semantic_result["is_contradiction"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = semantic_result["score"]
                result["contradiction_type"] = ContradictionType.DIRECT
                result["confidence"] = semantic_result["confidence"]
                result["explanation"] = semantic_result["explanation"]
        
        # Detect negation contradiction
        negation_result = self._detect_negation_contradiction(claim1, claim2)
        result["methods_used"].append("negation")
        result["details"]["negation"] = negation_result
        
        # Update result if negation contradiction is detected and has higher score
        if negation_result["is_contradiction"] and negation_result["score"] > result["contradiction_score"]:
            result["is_contradiction"] = True
            result["contradiction_score"] = negation_result["score"]
            result["contradiction_type"] = ContradictionType.NEGATION
            result["confidence"] = negation_result["confidence"]
            result["explanation"] = negation_result["explanation"]
        
        # Detect keyword-based contradiction
        keyword_result = self._detect_keyword_contradiction(claim1, claim2)
        result["methods_used"].append("keyword")
        result["details"]["keyword"] = keyword_result
        
        # Update result if keyword contradiction is detected and has higher score
        if keyword_result["is_contradiction"] and keyword_result["score"] > result["contradiction_score"]:
            result["is_contradiction"] = True
            result["contradiction_score"] = keyword_result["score"]
            result["contradiction_type"] = ContradictionType.DIRECT
            result["confidence"] = keyword_result["confidence"]
            result["explanation"] = keyword_result["explanation"]
        
        # Detect statistical contradiction if metadata is available
        if metadata1 and metadata2:
            statistical_result = self._detect_statistical_contradiction(metadata1, metadata2)
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
    
    async def _detect_semantic_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """
        Detect semantic contradiction between two claims using BioMedLM.
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            Semantic contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }
        
        try:
            # Use BioMedLM to detect contradiction
            is_contradiction, score = self.biomedlm_service.detect_contradiction(claim1, claim2)
            
            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)
            
            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW
            
            # Generate explanation
            if result["is_contradiction"]:
                result["explanation"] = f"The claims semantically contradict each other with a score of {score:.2f}."
            
            return result
        except Exception as e:
            logger.error(f"Error detecting semantic contradiction: {str(e)}")
            return result
    
    def _detect_negation_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
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
        
        # Convert claims to lowercase for case-insensitive matching
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        # Check if one claim is the negation of the other
        for pattern, replacement in self.negation_patterns:
            # Check if claim1 contains negation and claim2 doesn't
            if pattern in claim1_lower and pattern not in claim2_lower:
                # Replace negation in claim1
                modified_claim1 = claim1_lower.replace(pattern, replacement)
                
                # Calculate similarity between modified claim1 and claim2
                if self.biomedlm_service:
                    # Use BioMedLM for semantic similarity if available
                    try:
                        similarity = self.biomedlm_service.calculate_similarity(modified_claim1, claim2_lower)
                    except Exception as e:
                        logger.error(f"Error calculating BioMedLM similarity: {str(e)}")
                        similarity = self._calculate_text_similarity(modified_claim1, claim2_lower)
                else:
                    # Fall back to Jaccard similarity
                    similarity = self._calculate_text_similarity(modified_claim1, claim2_lower)
                
                if similarity > self.thresholds[ContradictionType.NEGATION]:
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
                if self.biomedlm_service:
                    # Use BioMedLM for semantic similarity if available
                    try:
                        similarity = self.biomedlm_service.calculate_similarity(claim1_lower, modified_claim2)
                    except Exception as e:
                        logger.error(f"Error calculating BioMedLM similarity: {str(e)}")
                        similarity = self._calculate_text_similarity(claim1_lower, modified_claim2)
                else:
                    # Fall back to Jaccard similarity
                    similarity = self._calculate_text_similarity(claim1_lower, modified_claim2)
                
                if similarity > self.thresholds[ContradictionType.NEGATION]:
                    result["is_contradiction"] = True
                    result["score"] = similarity
                    result["confidence"] = ContradictionConfidence.HIGH
                    result["explanation"] = f"Claim 2 is a negation of Claim 1 with similarity {similarity:.2f}."
                    return result
        
        return result
    
    def _detect_keyword_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect keyword-based contradiction between two claims.
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            Keyword contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }
        
        # Convert claims to lowercase for case-insensitive matching
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()
        
        # Combine claims for keyword search
        combined_text = f"{claim1_lower} {claim2_lower}"
        
        # Count contradiction keywords
        keyword_count = 0
        found_keywords = []
        
        for keyword in self.contradiction_keywords:
            if keyword in combined_text:
                keyword_count += 1
                found_keywords.append(keyword)
        
        # Calculate score based on keyword count
        if keyword_count > 0:
            # Normalize score between 0 and 1, with diminishing returns after 5 keywords
            score = min(1.0, keyword_count / 5.0)
            
            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = score
            
            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW
            
            # Generate explanation
            if result["is_contradiction"]:
                result["explanation"] = f"Found contradiction keywords: {', '.join(found_keywords[:5])}."
                if len(found_keywords) > 5:
                    result["explanation"] += f" and {len(found_keywords) - 5} more."
        
        return result
    
    def _detect_statistical_contradiction(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect statistical contradiction between two claims.
        
        Args:
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
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
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

async def test_biomedlm_service():
    """Test the BioMedLM service."""
    logger.info("Testing BioMedLM service...")
    
    # Initialize service
    biomedlm_service = MockBioMedLMService()
    
    # Test similarity calculation
    claim1 = "Statin therapy reduces the risk of cardiovascular events."
    claim2 = "Statin therapy lowers the chance of heart problems."
    
    logger.info(f"Calculating similarity between: '{claim1}' and '{claim2}'")
    similarity = biomedlm_service.calculate_similarity(claim1, claim2)
    logger.info(f"Similarity: {similarity:.4f}")
    
    # Test contradiction detection
    claim1 = "Statin therapy reduces the risk of cardiovascular events."
    claim2 = "Statin therapy does not reduce the risk of cardiovascular events."
    
    logger.info(f"Detecting contradiction between: '{claim1}' and '{claim2}'")
    is_contradiction, score = biomedlm_service.detect_contradiction(claim1, claim2)
    logger.info(f"Contradiction: {is_contradiction}, Score: {score:.4f}")
    
    logger.info("BioMedLM service tests completed")

async def test_contradiction_service_with_biomedlm():
    """Test the contradiction service with BioMedLM integration."""
    logger.info("Testing contradiction service with BioMedLM integration...")
    
    # Initialize services
    biomedlm_service = MockBioMedLMService()
    contradiction_service = ContradictionService(biomedlm_service=biomedlm_service)
    
    # Test each claim pair
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        
        # Detect contradiction with BioMedLM
        result_with_biomedlm = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True
        )
        
        # Detect contradiction without BioMedLM
        result_without_biomedlm = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=False
        )
        
        # Print results
        logger.info("With BioMedLM:")
        logger.info(f"  Result: {'Contradiction' if result_with_biomedlm['is_contradiction'] else 'No contradiction'}")
        logger.info(f"  Score: {result_with_biomedlm['contradiction_score']:.4f}")
        logger.info(f"  Type: {result_with_biomedlm['contradiction_type']}")
        logger.info(f"  Confidence: {result_with_biomedlm['confidence']}")
        logger.info(f"  Methods: {', '.join(result_with_biomedlm['methods_used'])}")
        if result_with_biomedlm["explanation"]:
            logger.info(f"  Explanation: {result_with_biomedlm['explanation']}")
        
        logger.info("Without BioMedLM:")
        logger.info(f"  Result: {'Contradiction' if result_without_biomedlm['is_contradiction'] else 'No contradiction'}")
        logger.info(f"  Score: {result_without_biomedlm['contradiction_score']:.4f}")
        logger.info(f"  Type: {result_without_biomedlm['contradiction_type']}")
        logger.info(f"  Confidence: {result_without_biomedlm['confidence']}")
        logger.info(f"  Methods: {', '.join(result_without_biomedlm['methods_used'])}")
        if result_without_biomedlm["explanation"]:
            logger.info(f"  Explanation: {result_without_biomedlm['explanation']}")
        
        logger.info("---")
    
    logger.info("Contradiction service tests completed")

async def main():
    """Main function."""
    logger.info("Starting BioMedLM contradiction detection tests...")
    
    try:
        # Test BioMedLM service
        await test_biomedlm_service()
        
        # Test contradiction service with BioMedLM integration
        await test_contradiction_service_with_biomedlm()
        
        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
