"""
Temporal service for the Medical Research Synthesizer.

This module provides a service for temporal analysis of medical literature.
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import datetime

from asf.medical.ml.models import TSMixerService, BioMedLMService
from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class TemporalService:
    """
    Service for temporal analysis of medical literature.
    
    This service provides methods for analyzing how medical knowledge evolves over time.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the temporal service.
        
        Returns:
            TemporalService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(TemporalService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the temporal service."""
        self.tsmixer_service = None
        self.biomedlm_service = None
        
        # Domain-specific decay rates (half-life in days)
        self.decay_rates = {
            "oncology": 365 * 2,  # 2 years
            "infectious_disease": 365 * 1,  # 1 year
            "cardiology": 365 * 3,  # 3 years
            "neurology": 365 * 4,  # 4 years
            "psychiatry": 365 * 5,  # 5 years
            "default": 365 * 2.5  # 2.5 years
        }
        
        logger.info("Temporal service initialized")
    
    def _get_tsmixer_service(self) -> TSMixerService:
        """
        Get the TSMixer service.
        
        Returns:
            TSMixerService: The TSMixer service
        """
        if self.tsmixer_service is None:
            logger.info("Initializing TSMixer service")
            self.tsmixer_service = TSMixerService()
        return self.tsmixer_service
    
    def _get_biomedlm_service(self) -> BioMedLMService:
        """
        Get the BioMedLM service.
        
        Returns:
            BioMedLMService: The BioMedLM service
        """
        if self.biomedlm_service is None:
            logger.info("Initializing BioMedLM service")
            self.biomedlm_service = BioMedLMService()
        return self.biomedlm_service
    
    def calculate_temporal_confidence(
        self,
        publication_date: str,
        domain: str = "default",
        reference_date: Optional[str] = None
    ) -> float:
        """
        Calculate temporal confidence for a publication.
        
        Args:
            publication_date: Publication date (YYYY-MM-DD)
            domain: Medical domain
            reference_date: Reference date (YYYY-MM-DD, default: today)
            
        Returns:
            Temporal confidence (0-1)
        """
        # Parse dates
        try:
            pub_date = datetime.datetime.strptime(publication_date, "%Y-%m-%d")
        except ValueError:
            try:
                # Try with just year and month
                pub_date = datetime.datetime.strptime(publication_date, "%Y-%m")
            except ValueError:
                try:
                    # Try with just year
                    pub_date = datetime.datetime.strptime(publication_date, "%Y")
                except ValueError:
                    logger.warning(f"Invalid publication date: {publication_date}")
                    return 0.5  # Default confidence
        
        if reference_date:
            try:
                ref_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid reference date: {reference_date}")
                ref_date = datetime.datetime.now()
        else:
            ref_date = datetime.datetime.now()
        
        # Get decay rate for domain
        half_life = self.decay_rates.get(domain.lower(), self.decay_rates["default"])
        
        # Calculate time difference in days
        time_diff = (ref_date - pub_date).days
        
        # Calculate confidence using exponential decay
        if time_diff < 0:
            # Future publication (shouldn't happen)
            return 0.5
        
        confidence = np.exp(-np.log(2) * time_diff / half_life)
        
        return float(confidence)
    
    def calculate_beta_distribution_parameters(
        self,
        confidence: float,
        certainty: float = 0.8
    ) -> Tuple[float, float]:
        """
        Calculate Beta distribution parameters for a confidence value.
        
        Args:
            confidence: Confidence value (0-1)
            certainty: Certainty about the confidence (0-1)
            
        Returns:
            Tuple of (alpha, beta) parameters
        """
        # Scale certainty to control the concentration
        concentration = 10 * certainty
        
        # Calculate alpha and beta
        alpha = concentration * confidence
        beta = concentration * (1 - confidence)
        
        return alpha, beta
    
    def analyze_temporal_sequence(
        self,
        claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze a temporal sequence of claims.
        
        Args:
            claims: List of claims with timestamps and text
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing temporal sequence of {len(claims)} claims")
        
        # Sort claims by timestamp
        sorted_claims = sorted(claims, key=lambda x: x["timestamp"])
        
        # Get TSMixer service
        tsmixer_service = self._get_tsmixer_service()
        
        # Get BioMedLM service for embeddings
        biomedlm_service = self._get_biomedlm_service()
        
        # Create sequence for TSMixer
        sequence = []
        for claim in sorted_claims:
            sequence.append({
                "claim": claim["text"],
                "timestamp": claim["timestamp"]
            })
        
        # Analyze sequence
        analysis = tsmixer_service.analyze_temporal_sequence(
            sequence,
            embedding_fn=biomedlm_service.encode
        )
        
        # Calculate temporal confidence for each claim
        confidences = []
        for claim in sorted_claims:
            if "publication_date" in claim and claim["publication_date"]:
                domain = claim.get("domain", "default")
                confidence = self.calculate_temporal_confidence(
                    claim["publication_date"],
                    domain
                )
                
                # Calculate Beta distribution parameters
                alpha, beta = self.calculate_beta_distribution_parameters(confidence)
                
                confidences.append({
                    "claim_id": claim.get("id", ""),
                    "confidence": confidence,
                    "alpha": alpha,
                    "beta": beta
                })
        
        # Add confidences to analysis
        analysis["temporal_confidences"] = confidences
        
        # Calculate overall trend
        if len(sorted_claims) > 1:
            # Get contradiction scores
            contradiction_scores = analysis["contradiction_scores"]
            
            # Calculate trend
            trend = np.polyfit(range(len(contradiction_scores)), contradiction_scores, 1)[0]
            
            # Add trend to analysis
            analysis["trend"] = {
                "slope": float(trend),
                "direction": "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable",
                "magnitude": "strong" if abs(trend) > 0.1 else "moderate" if abs(trend) > 0.05 else "weak"
            }
        
        logger.info("Temporal sequence analysis completed")
        
        return analysis
    
    def analyze_claim_evolution(
        self,
        claim: str,
        related_claims: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze how a claim has evolved over time.
        
        Args:
            claim: The main claim
            related_claims: List of related claims with timestamps
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing evolution of claim: {claim}")
        
        # Get BioMedLM service
        biomedlm_service = self._get_biomedlm_service()
        
        # Calculate similarity between main claim and each related claim
        similarities = []
        for related_claim in related_claims:
            similarity = biomedlm_service.calculate_similarity(
                claim,
                related_claim["text"]
            )
            
            similarities.append({
                "claim_id": related_claim.get("id", ""),
                "text": related_claim["text"],
                "timestamp": related_claim["timestamp"],
                "similarity": similarity
            })
        
        # Sort by timestamp
        similarities = sorted(similarities, key=lambda x: x["timestamp"])
        
        # Calculate evolution metrics
        if len(similarities) > 1:
            # Get similarity values
            similarity_values = [s["similarity"] for s in similarities]
            
            # Calculate trend
            trend = np.polyfit(range(len(similarity_values)), similarity_values, 1)[0]
            
            # Calculate volatility
            volatility = np.std(similarity_values)
            
            # Calculate divergence (1 - final similarity)
            divergence = 1.0 - similarity_values[-1]
            
            evolution = {
                "trend": {
                    "slope": float(trend),
                    "direction": "converging" if trend > 0 else "diverging" if trend < 0 else "stable",
                    "magnitude": "strong" if abs(trend) > 0.1 else "moderate" if abs(trend) > 0.05 else "weak"
                },
                "volatility": float(volatility),
                "divergence": float(divergence)
            }
        else:
            evolution = {
                "trend": {
                    "slope": 0.0,
                    "direction": "stable",
                    "magnitude": "weak"
                },
                "volatility": 0.0,
                "divergence": 0.0
            }
        
        # Create result
        result = {
            "claim": claim,
            "similarities": similarities,
            "evolution": evolution
        }
        
        logger.info("Claim evolution analysis completed")
        
        return result
    
    def unload_models(self):
        """Unload all models from memory."""
        if self.tsmixer_service is not None:
            self.tsmixer_service.unload_model()
        
        if self.biomedlm_service is not None:
            self.biomedlm_service.unload_model()
        
        logger.info("All models unloaded")
