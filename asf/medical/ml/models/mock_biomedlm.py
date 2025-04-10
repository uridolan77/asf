"""
Mock BioMedLM service for testing.

This module provides a mock implementation of the BioMedLM service for testing purposes.
"""

import logging
import re
import numpy as np
from typing import Tuple

# Set up logging
logger = logging.getLogger(__name__)

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
            ("didn't ", "did "),
            ("did not ", "did "),
            ("isn't ", "is "),
            ("is not ", "is "),
            ("aren't ", "are "),
            ("are not ", "are "),
            ("wasn't ", "was "),
            ("was not ", "was "),
            ("weren't ", "were "),
            ("were not ", "were "),
            ("hasn't ", "has "),
            ("has not ", "has "),
            ("haven't ", "have "),
            ("have not ", "have "),
            ("hadn't ", "had "),
            ("had not ", "had "),
            ("cannot ", "can "),
            ("can't ", "can "),
            ("couldn't ", "could "),
            ("could not ", "could "),
            ("shouldn't ", "should "),
            ("should not ", "should "),
            ("wouldn't ", "would "),
            ("would not ", "would "),
            ("won't ", "will "),
            ("will not ", "will "),
            ("without ", "with "),
            ("absence of ", "presence of "),
            ("lack of ", "presence of "),
            ("failed to ", "succeeded to "),
            ("failure to ", "success to "),
            ("ineffective ", "effective "),
            ("inefficacy ", "efficacy "),
            ("insufficient ", "sufficient "),
            ("inadequate ", "adequate "),
            ("unable to ", "able to "),
            ("inability to ", "ability to "),
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
