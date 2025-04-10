"""
Temporal Confidence Scorer for BioMedLM

This module integrates the temporal knowledge confidence model with BioMedLM
for time-aware contradiction detection in medical claims.
"""

import logging
import datetime
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from scipy.stats import beta

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("temporal-confidence-scorer")

class TemporalConfidenceScorer:
    """
    Temporal confidence scorer for medical claims.
    
    This class uses Beta distributions to model confidence in medical claims over time,
    with domain-specific decay rates for different medical specialties.
    """
    
    def __init__(self, domain_decay_rates: Optional[Dict[str, float]] = None):
        """
        Initialize the temporal confidence scorer.
        
        Args:
            domain_decay_rates: Dictionary mapping medical domains to decay rates
        """
        # Default decay rates for medical domains (per day)
        self.domain_decay_rates = domain_decay_rates or {
            "general_medicine": 0.00005,  # Very slow decay
            "oncology": 0.0001,
            "cardiology": 0.00008,
            "neurology": 0.00007,
            "infectious_disease": 0.0002,  # Faster decay due to rapid developments
            "pharmacology": 0.00015,
            "genetics": 0.00012,
            "nutrition": 0.0001,
            "surgery": 0.00006,
            "pediatrics": 0.00007,
            "geriatrics": 0.00006,
            "psychiatry": 0.00005,
            "emergency_medicine": 0.0001,
            "radiology": 0.00004,
            "default": 0.0001  # Default decay rate
        }
        
        self.decay_type = "exponential"  # Can be "linear" or "exponential"
    
    def calculate_confidence(
        self, 
        initial_confidence: float, 
        domain: str, 
        creation_time: datetime.datetime,
        current_time: Optional[datetime.datetime] = None
    ) -> float:
        """
        Calculate the current confidence of a medical claim based on temporal decay.
        
        Args:
            initial_confidence: Initial confidence score (0-1)
            domain: Medical domain of the claim
            creation_time: When the claim was created
            current_time: Current time (defaults to now)
            
        Returns:
            Current confidence score (0-1)
        """
        if current_time is None:
            current_time = datetime.datetime.now()
        
        # Get decay rate for the domain
        decay_rate = self.domain_decay_rates.get(domain, self.domain_decay_rates["default"])
        
        # Calculate time difference in days
        time_diff = (current_time - creation_time).total_seconds() / (24 * 3600)
        
        # Apply decay based on decay type
        if self.decay_type == "linear":
            # Linear decay: confidence = initial_confidence - (decay_rate * time_diff)
            confidence = initial_confidence - (decay_rate * time_diff)
        else:
            # Exponential decay: confidence = initial_confidence * exp(-decay_rate * time_diff)
            confidence = initial_confidence * np.exp(-decay_rate * time_diff)
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def get_beta_parameters(self, confidence: float, certainty: float = 10.0) -> Tuple[float, float]:
        """
        Convert confidence score to Beta distribution parameters.
        
        Args:
            confidence: Confidence score (0-1)
            certainty: Certainty level (higher means more certain)
            
        Returns:
            Tuple of (alpha, beta) parameters
        """
        alpha = confidence * certainty
        beta_param = (1 - confidence) * certainty
        return alpha, beta_param
    
    def combine_confidences(self, confidences: List[float], certainties: Optional[List[float]] = None) -> float:
        """
        Combine multiple confidence scores using Beta distributions.
        
        Args:
            confidences: List of confidence scores
            certainties: List of certainty levels (optional)
            
        Returns:
            Combined confidence score
        """
        if not confidences:
            return 0.0
        
        if certainties is None:
            certainties = [10.0] * len(confidences)
        
        # Convert confidences to Beta parameters
        alpha_sum = 0.0
        beta_sum = 0.0
        
        for confidence, certainty in zip(confidences, certainties):
            alpha, beta_param = self.get_beta_parameters(confidence, certainty)
            alpha_sum += alpha
            beta_sum += beta_param
        
        # Calculate combined confidence
        combined_confidence = alpha_sum / (alpha_sum + beta_sum)
        
        return combined_confidence
    
    def weight_contradiction_score(
        self, 
        contradiction_score: float, 
        claim1_confidence: float, 
        claim2_confidence: float
    ) -> float:
        """
        Weight contradiction score based on the confidence of the claims.
        
        Args:
            contradiction_score: Original contradiction score
            claim1_confidence: Confidence in first claim
            claim2_confidence: Confidence in second claim
            
        Returns:
            Weighted contradiction score
        """
        # Calculate average confidence
        avg_confidence = (claim1_confidence + claim2_confidence) / 2
        
        # Weight contradiction score based on confidence
        # Higher confidence means we trust the contradiction score more
        weighted_score = contradiction_score * avg_confidence
        
        return weighted_score
    
    def get_claim_metadata(self, claim: str) -> Dict[str, Any]:
        """
        Extract metadata from a medical claim for confidence calculation.
        
        Args:
            claim: Medical claim text
            
        Returns:
            Dictionary with metadata
        """
        # This is a placeholder implementation
        # In a real system, this would use NLP to extract domain, publication date, etc.
        
        # Default metadata
        metadata = {
            "domain": "general_medicine",
            "creation_time": datetime.datetime.now() - datetime.timedelta(days=30),  # Assume 30 days old
            "initial_confidence": 0.8
        }
        
        # Simple keyword-based domain detection
        domain_keywords = {
            "cancer": "oncology",
            "tumor": "oncology",
            "heart": "cardiology",
            "cardiac": "cardiology",
            "brain": "neurology",
            "neural": "neurology",
            "infection": "infectious_disease",
            "virus": "infectious_disease",
            "bacteria": "infectious_disease",
            "drug": "pharmacology",
            "medication": "pharmacology",
            "gene": "genetics",
            "genetic": "genetics",
            "diet": "nutrition",
            "food": "nutrition",
            "surgery": "surgery",
            "surgical": "surgery",
            "child": "pediatrics",
            "children": "pediatrics",
            "elderly": "geriatrics",
            "aging": "geriatrics",
            "mental": "psychiatry",
            "psychological": "psychiatry",
            "emergency": "emergency_medicine",
            "trauma": "emergency_medicine",
            "imaging": "radiology",
            "scan": "radiology"
        }
        
        # Check for domain keywords
        claim_lower = claim.lower()
        for keyword, domain in domain_keywords.items():
            if keyword in claim_lower:
                metadata["domain"] = domain
                break
        
        # Look for date indicators
        date_indicators = [
            "published in", "reported in", "study from", "research from",
            "trial in", "conducted in", "findings from"
        ]
        
        for indicator in date_indicators:
            if indicator in claim_lower:
                # Look for year after the indicator
                pos = claim_lower.find(indicator) + len(indicator)
                year_match = re.search(r'\b(19|20)\d{2}\b', claim_lower[pos:pos+20])
                if year_match:
                    year = int(year_match.group(0))
                    # Create a date from the year (assume middle of the year)
                    metadata["creation_time"] = datetime.datetime(year, 6, 15)
        
        return metadata
