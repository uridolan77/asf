"""Temporal Service for Medical Research Synthesizer.

This module provides a service for temporal analysis of medical literature.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime

from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.core.exceptions import OperationError


logger = logging.getLogger(__name__)

class TemporalService:
    """Temporal analysis service for medical literature.
    
    This service provides functionality for analyzing temporal aspects of medical claims,
    including temporal confidence calculation, temporal contradiction detection,
    and temporal sequence analysis.
    """

    def __init__(self, tsmixer_service: Optional[TSMixerService] = None, biomedlm_service: Optional[BioMedLMService] = None):
        """
        Initialize the temporal service.

        Args:
            tsmixer_service: TSMixer service for temporal sequence analysis
            biomedlm_service: BioMedLM service for semantic analysis
        """
        self.tsmixer_service = tsmixer_service or TSMixerService()
        self.biomedlm_service = biomedlm_service

        # Domain-specific characteristics for temporal confidence calculation
        self.domain_characteristics = {
            "cardiology": {
                "half_life": 365 * 3,  # 3 years
                "evolution_rate": "moderate",
                "evidence_stability": "moderate",
                "technology_dependence": "high",
                "description": "Cardiology knowledge evolves at a moderate rate with high technology dependence"
            },
            "oncology": {
                "half_life": 365 * 2,  # 2 years
                "evolution_rate": "rapid",
                "evidence_stability": "low",
                "technology_dependence": "very high",
                "description": "Oncology knowledge evolves rapidly with very high technology dependence"
            },
            "infectious_disease": {
                "half_life": 365 * 1.5,  # 1.5 years
                "evolution_rate": "very rapid",
                "evidence_stability": "very low",
                "technology_dependence": "high",
                "description": "Infectious disease knowledge evolves very rapidly, especially during outbreaks"
            },
            "anatomy": {
                "half_life": 365 * 10,  # 10 years
                "evolution_rate": "very slow",
                "evidence_stability": "very high",
                "technology_dependence": "low",
                "description": "Anatomical knowledge is very stable with slow evolution"
            },
            "genetics": {
                "half_life": 365 * 2,  # 2 years
                "evolution_rate": "rapid",
                "evidence_stability": "moderate",
                "technology_dependence": "very high",
                "description": "Genetics knowledge evolves rapidly with very high technology dependence"
            },
            "psychiatry": {
                "half_life": 365 * 5,  # 5 years
                "evolution_rate": "slow",
                "evidence_stability": "moderate",
                "technology_dependence": "low",
                "description": "Psychiatric knowledge evolves relatively slowly"
            },
            "pharmacology": {
                "half_life": 365 * 3,  # 3 years
                "evolution_rate": "moderate",
                "evidence_stability": "moderate",
                "technology_dependence": "high",
                "description": "Pharmacological knowledge evolves at a moderate rate"
            },
            "general": {
                "half_life": 365 * 2.5,  # 2.5 years
                "evolution_rate": "moderate",
                "evidence_stability": "moderate",
                "technology_dependence": "moderate",
                "description": "General medical knowledge with moderate evolution rate"
            }
        }
        self.decay_rates = {domain: info["half_life"] for domain, info in self.domain_characteristics.items()}
        logger.info("Temporal service initialized")

    def _get_tsmixer_service(self) -> TSMixerService:
        """
        Get the TSMixer service.

        Returns:
            TSMixerService: The TSMixer service
        """
        return self.tsmixer_service

    async def calculate_temporal_confidence(
        self,
        publication_date: str,
        domain: str = "general",
        reference_date: Optional[str] = None,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate temporal confidence for a publication with domain-specific characteristics.

        Args:
            publication_date: Publication date (YYYY-MM-DD)
            domain: Medical domain
            reference_date: Reference date (YYYY-MM-DD, default: today)
            include_details: Whether to include detailed information in the result

        Returns:
            Temporal confidence (0-1) or dict with confidence and details
        """
        # Default to general domain if specified domain not found
        if domain not in self.domain_characteristics:
            domain = "general"
            logger.warning(f"Domain '{domain}' not found, using 'general' instead")

        # Get domain characteristics
        domain_info = self.domain_characteristics[domain]
        half_life = domain_info["half_life"]

        # Parse dates
        try:
            pub_date = datetime.fromisoformat(publication_date.replace("Z", "+00:00"))

            if reference_date:
                ref_date = datetime.fromisoformat(reference_date.replace("Z", "+00:00"))
            else:
                ref_date = datetime.now()

            # Calculate time difference in days
            time_diff = (ref_date - pub_date).days

            # Convert to years for easier interpretation
            time_diff_years = time_diff / 365.0

            # Calculate confidence using exponential decay formula
            # confidence = e^(-ln(2) * time_diff / half_life)
            confidence = np.exp(-0.693 * time_diff / half_life)

            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))

            # Generate explanation
            explanation = self._generate_temporal_confidence_explanation(
                domain_info, confidence, time_diff_years, domain
            )

            if include_details:
                return {
                    "confidence": float(confidence),
                    "explanation": explanation,
                    "domain": domain,
                    "domain_info": domain_info,
                    "time_diff_years": time_diff_years,
                    "half_life_years": half_life / 365.0
                }
            else:
                return {"confidence": float(confidence), "explanation": explanation}

        except Exception as e:
            logger.error(f"Error calculating temporal confidence: {e}")
            raise OperationError(f"Operation failed: {str(e)}")

    def _generate_temporal_confidence_explanation(
        self,
        domain_info: Dict[str, Any],
        confidence: float,
        time_diff_years: float,
        domain: str
    ) -> str:
        """
        Generate an explanation for the temporal confidence calculation.

        Args:
            domain_info: Domain characteristics
            confidence: Calculated confidence
            time_diff_years: Time difference in years
            domain: Medical domain

        Returns:
            Explanation string
        """
        if confidence > 0.9:
            confidence_level = "very high"
        elif confidence > 0.7:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "moderate"
        elif confidence > 0.3:
            confidence_level = "low"
        else:
            confidence_level = "very low"

        evolution_rate = domain_info.get("evolution_rate", "moderate")

        explanation = (
            f"The temporal confidence is {confidence_level} ({confidence:.2f}) based on "
            f"the publication being {time_diff_years:.1f} years old. "
            f"In {domain}, knowledge typically evolves at a {evolution_rate} rate "
            f"with a half-life of approximately {domain_info['half_life']/365:.1f} years."
        )

        return explanation

    def _calculate_beta_parameters(self, confidence: float, certainty: float = 0.8) -> Tuple[float, float]:
        """
        Calculate Beta distribution parameters for a confidence value.

        Args:
            confidence: Confidence value (0-1)
            certainty: Certainty about the confidence (0-1)

        Returns:
            Tuple of (alpha, beta) parameters
        """
        # Ensure inputs are valid
        confidence = max(0.01, min(0.99, confidence))
        certainty = max(0.01, min(0.99, certainty))

        # Calculate concentration parameter
        concentration = certainty * 100

        # Calculate alpha and beta
        alpha = confidence * concentration
        beta = (1 - confidence) * concentration

        return (alpha, beta)

    async def analyze_temporal_sequence(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a temporal sequence of claims.

        Args:
            claims: List of claims with timestamps and text

        Returns:
            Analysis results
        """
        if not claims or len(claims) < 2:
            return {"error": "Need at least 2 claims for temporal sequence analysis"}

        try:
            # Sort claims by timestamp
            sorted_claims = sorted(claims, key=lambda x: x.get("timestamp", ""))

            # Extract text and timestamps
            texts = [claim.get("text", "") for claim in sorted_claims]
            timestamps = [claim.get("timestamp", "") for claim in sorted_claims]

            # Use TSMixer for sequence analysis
            if self.tsmixer_service:
                sequence_analysis = await self.tsmixer_service.analyze_sequence(texts, timestamps)
                return sequence_analysis
            else:
                return {"error": "TSMixer service not available"}

        except Exception as e:
            logger.error(f"Error analyzing temporal sequence: {e}")
            raise OperationError(f"Operation failed: {str(e)}")

    async def analyze_claim_evolution(self, claim: str, related_claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze how a claim has evolved over time.

        Args:
            claim: The main claim
            related_claims: List of related claims with timestamps

        Returns:
            Analysis results
        """
        if not claim or not related_claims:
            return {"error": "Need main claim and related claims for evolution analysis"}

        try:
            # Sort related claims by timestamp
            sorted_claims = sorted(related_claims, key=lambda x: x.get("timestamp", ""))

            # Extract text and timestamps
            texts = [claim] + [rc.get("text", "") for rc in sorted_claims]
            timestamps = [""] + [rc.get("timestamp", "") for rc in sorted_claims]

            # Use TSMixer for evolution analysis
            if self.tsmixer_service:
                evolution_analysis = await self.tsmixer_service.analyze_evolution(texts, timestamps)
                return evolution_analysis
            else:
                return {"error": "TSMixer service not available"}

        except Exception as e:
            logger.error(f"Error analyzing claim evolution: {e}")
            raise OperationError(f"Operation failed: {str(e)}")

    async def analyze_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        date1: str,
        date2: str
    ) -> Tuple[bool, float, str]:
        """
        Analyze temporal contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            date1: Date of first claim (YYYY-MM-DD)
            date2: Date of second claim (YYYY-MM-DD)

        Returns:
            Tuple of (is_contradiction, score, explanation)
        """
        try:
            # Check if dates are valid
            if not date1 or not date2:
                return False, 0.0, "Missing dates for temporal contradiction analysis"

            # Parse dates
            date1_obj = datetime.fromisoformat(date1.replace("Z", "+00:00"))
            date2_obj = datetime.fromisoformat(date2.replace("Z", "+00:00"))

            # Calculate time difference in years
            time_diff_years = abs((date1_obj - date2_obj).days / 365.0)

            # Use TSMixer for temporal contradiction detection
            if self.tsmixer_service:
                is_contradiction, score, explanation = await self.tsmixer_service.detect_temporal_contradiction(
                    claim1, claim2, date1, date2
                )
                return is_contradiction, score, explanation
            else:
                # Fallback to simple heuristic
                if time_diff_years > 5:
                    score = min(0.7, time_diff_years / 10)
                    return True, score, f"Claims are separated by {time_diff_years:.1f} years, suggesting potential temporal contradiction"
                else:
                    return False, 0.0, f"Claims are only {time_diff_years:.1f} years apart, insufficient for temporal contradiction"

        except Exception as e:
            logger.error(f"Error analyzing temporal contradiction: {e}")
            raise OperationError(f"Operation failed: {str(e)}")

    def unload_models(self):
        """
        Unload all models to free up memory.

        Returns:
            None
        """
        if self.tsmixer_service is not None:
            self.tsmixer_service.unload_model()
        if self.biomedlm_service is not None:
            self.biomedlm_service.unload_model()
        logger.info("All models unloaded")
