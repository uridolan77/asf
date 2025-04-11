"""
Mock BioMedLM service for testing.

This module provides a mock implementation of the BioMedLM service for testing purposes.
"""

import logging
import re
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class MockBioMedLMService:
    """
    Mock implementation of the BioMedLM service for testing.
    
    This class provides a simplified implementation of the BioMedLM service
    that can be used for testing without requiring the actual model.
    """
    
    def __init__(self):
        """Initialize the mock BioMedLM service.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
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
            Text embedding
        Calculate the similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        Detect contradiction between two claims.
        
        Args:
            claim1: First claim
            claim2: Second claim
            
        Returns:
            Tuple of (is_contradiction, confidence)
        Calculate text similarity using a simple Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1