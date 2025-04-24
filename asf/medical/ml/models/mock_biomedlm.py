"""Mock BioMedLM service for testing.

This module provides a mock implementation of the BioMedLM service for testing purposes.
"""
import logging
import numpy as np
from typing import Tuple
logger = logging.getLogger(__name__)
class MockBioMedLMService:
    """Mock implementation of the BioMedLM service for testing.
    
    This class provides a simplified implementation of the BioMedLM service
    that can be used for testing without requiring the actual model.
    """
    def __init__(self):
        """Initialize the mock BioMedLM service.
        
        This method sets up the mock service with predefined negation patterns
        for simple contradiction detection.
        """
        logger.info("Mock BioMedLM service initialized")
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
            Text embedding as a numpy array
        """
        # Simple bag-of-words encoding
        words = set(text.lower().split())
        # Create a random embedding based on the hash of the words
        np.random.seed(hash(frozenset(words)) % 2**32)
        return np.random.rand(768)  # Same dimension as BioMedLM embeddings

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        return self._jaccard_similarity(text1, text2)

    def detect_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Tuple of (is_contradiction, confidence)
        """
        # Check if one claim is a negation of the other
        negated_claim1 = self._apply_negation(claim1)
        similarity = self._jaccard_similarity(negated_claim1, claim2)

        # If the negated claim1 is similar to claim2, they are contradictory
        is_contradiction = similarity > 0.6
        confidence = similarity if is_contradiction else 1.0 - similarity

        return (is_contradiction, confidence)

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using a simple Jaccard similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union

    def _apply_negation(self, text: str) -> str:
        """
        Apply negation patterns to a text.

        Args:
            text: Text to negate

        Returns:
            Negated text
        """
        result = text.lower()
        for pattern, replacement in self.negation_patterns:
            result = result.replace(pattern, replacement)
        return result