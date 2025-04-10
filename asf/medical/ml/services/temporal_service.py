"""
Temporal service for the Medical Research Synthesizer.

This module provides a service for temporal analysis of medical literature.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import datetime

from asf.medical.ml.models import TSMixerService, BioMedLMService

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

        # Domain-specific decay rates (half-life in days) and characteristics
        self.domain_characteristics = {
            # Rapidly evolving fields with frequent new treatments and research
            "oncology": {
                "half_life": 365 * 2,  # 2 years
                "evolution_rate": "rapid",
                "evidence_stability": "moderate",
                "technology_dependence": "high",
                "description": "Cancer research evolves rapidly with new treatments and targeted therapies"
            },
            "infectious_disease": {
                "half_life": 365 * 1,  # 1 year
                "evolution_rate": "very_rapid",
                "evidence_stability": "low",
                "technology_dependence": "high",
                "description": "Infectious disease knowledge changes quickly with emerging pathogens and resistance patterns"
            },
            "immunology": {
                "half_life": 365 * 1.5,  # 1.5 years
                "evolution_rate": "rapid",
                "evidence_stability": "moderate",
                "technology_dependence": "high",
                "description": "Immunology research advances quickly with new understanding of immune mechanisms"
            },

            # Moderately evolving fields
            "cardiology": {
                "half_life": 365 * 3,  # 3 years
                "evolution_rate": "moderate",
                "evidence_stability": "high",
                "technology_dependence": "moderate",
                "description": "Cardiovascular medicine has established foundations with incremental advances"
            },
            "endocrinology": {
                "half_life": 365 * 3,  # 3 years
                "evolution_rate": "moderate",
                "evidence_stability": "high",
                "technology_dependence": "moderate",
                "description": "Endocrinology knowledge evolves steadily with well-established principles"
            },
            "gastroenterology": {
                "half_life": 365 * 2.5,  # 2.5 years
                "evolution_rate": "moderate",
                "evidence_stability": "moderate",
                "technology_dependence": "moderate",
                "description": "Gastroenterology combines stable knowledge with evolving treatment approaches"
            },

            # Slowly evolving fields with established principles
            "neurology": {
                "half_life": 365 * 4,  # 4 years
                "evolution_rate": "slow",
                "evidence_stability": "high",
                "technology_dependence": "moderate",
                "description": "Neurological principles change slowly despite technological advances in imaging"
            },
            "psychiatry": {
                "half_life": 365 * 5,  # 5 years
                "evolution_rate": "slow",
                "evidence_stability": "moderate",
                "technology_dependence": "low",
                "description": "Psychiatric knowledge evolves gradually with long-term studies and observations"
            },
            "anatomy": {
                "half_life": 365 * 10,  # 10 years
                "evolution_rate": "very_slow",
                "evidence_stability": "very_high",
                "technology_dependence": "low",
                "description": "Human anatomy knowledge is highly stable with rare significant changes"
            },

            # Specialized fields
            "genomics": {
                "half_life": 365 * 1.5,  # 1.5 years
                "evolution_rate": "very_rapid",
                "evidence_stability": "moderate",
                "technology_dependence": "very_high",
                "description": "Genomic medicine advances rapidly with new sequencing and analysis technologies"
            },
            "pharmacology": {
                "half_life": 365 * 2,  # 2 years
                "evolution_rate": "rapid",
                "evidence_stability": "moderate",
                "technology_dependence": "high",
                "description": "Pharmacological knowledge evolves with new drug development and understanding of mechanisms"
            },
            "public_health": {
                "half_life": 365 * 3,  # 3 years
                "evolution_rate": "moderate",
                "evidence_stability": "moderate",
                "technology_dependence": "low",
                "description": "Public health knowledge combines stable principles with evolving population data"
            },

            # Default for unknown domains
            "default": {
                "half_life": 365 * 2.5,  # 2.5 years
                "evolution_rate": "moderate",
                "evidence_stability": "moderate",
                "technology_dependence": "moderate",
                "description": "General medical knowledge with moderate evolution rate"
            }
        }

        # Extract decay rates for backward compatibility
        self.decay_rates = {domain: info["half_life"] for domain, info in self.domain_characteristics.items()}

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
        reference_date: Optional[str] = None,
        include_details: bool = False
    ) -> Union[float, Dict[str, Any]]:
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
                    if include_details:
                        return {
                            "confidence": 0.5,
                            "reason": "Invalid publication date format",
                            "domain": domain,
                            "domain_characteristics": self.domain_characteristics.get(domain.lower(), self.domain_characteristics["default"])
                        }
                    return 0.5  # Default confidence

        if reference_date:
            try:
                ref_date = datetime.datetime.strptime(reference_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid reference date: {reference_date}")
                ref_date = datetime.datetime.now()
        else:
            ref_date = datetime.datetime.now()

        # Get domain characteristics
        domain_key = domain.lower()
        domain_info = self.domain_characteristics.get(domain_key, self.domain_characteristics["default"])
        half_life = domain_info["half_life"]
        evolution_rate = domain_info["evolution_rate"]
        evidence_stability = domain_info["evidence_stability"]

        # Calculate time difference in days
        time_diff = (ref_date - pub_date).days
        time_diff_years = time_diff / 365.0

        # Calculate confidence using exponential decay
        if time_diff < 0:
            # Future publication (shouldn't happen)
            if include_details:
                return {
                    "confidence": 0.5,
                    "reason": "Publication date is in the future",
                    "domain": domain,
                    "domain_characteristics": domain_info
                }
            return 0.5

        # Base confidence using exponential decay
        base_confidence = np.exp(-np.log(2) * time_diff / half_life)

        # Apply domain-specific adjustments
        adjusted_confidence = base_confidence

        # Adjust based on evidence stability
        if evidence_stability == "very_high":
            # Very stable evidence decays more slowly
            stability_factor = 1.2
        elif evidence_stability == "high":
            stability_factor = 1.1
        elif evidence_stability == "moderate":
            stability_factor = 1.0
        elif evidence_stability == "low":
            stability_factor = 0.9
        else:  # very_low
            stability_factor = 0.8

        adjusted_confidence *= stability_factor

        # Ensure confidence is between 0 and 1
        final_confidence = max(0.0, min(1.0, adjusted_confidence))

        if include_details:
            return {
                "confidence": float(final_confidence),
                "base_confidence": float(base_confidence),
                "stability_factor": stability_factor,
                "time_diff_days": time_diff,
                "time_diff_years": time_diff_years,
                "half_life_days": half_life,
                "half_life_years": half_life / 365.0,
                "domain": domain,
                "domain_characteristics": domain_info,
                "explanation": self._generate_confidence_explanation(
                    domain_info, final_confidence, time_diff_years, domain
                )
            }

        return float(final_confidence)

    def _generate_confidence_explanation(
        self,
        domain_info: Dict[str, Any],
        confidence: float,
        time_diff_years: float,
        domain: str = "medical"
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
        domain_description = domain_info["description"]
        evolution_rate = domain_info["evolution_rate"]
        half_life_years = domain_info["half_life"] / 365.0

        # Confidence level description
        if confidence > 0.9:
            confidence_desc = "very high"
        elif confidence > 0.7:
            confidence_desc = "high"
        elif confidence > 0.5:
            confidence_desc = "moderate"
        elif confidence > 0.3:
            confidence_desc = "low"
        else:
            confidence_desc = "very low"

        # Evolution rate description
        if evolution_rate == "very_rapid":
            evolution_desc = "very rapidly evolving"
            rate_explanation = "knowledge can change significantly even within a few years"
        elif evolution_rate == "rapid":
            evolution_desc = "rapidly evolving"
            rate_explanation = "significant advances occur frequently"
        elif evolution_rate == "moderate":
            evolution_desc = "moderately evolving"
            rate_explanation = "knowledge evolves steadily over time"
        elif evolution_rate == "slow":
            evolution_desc = "slowly evolving"
            rate_explanation = "fundamental principles remain stable for many years"
        else:  # very_slow
            evolution_desc = "very slowly evolving"
            rate_explanation = "core knowledge remains stable for decades"

        # Format domain name for better readability
        formatted_domain = domain.replace("_", " ").title()

        # Generate explanation
        explanation = f"This evidence has {confidence_desc} temporal confidence ({confidence:.2f}) "
        explanation += f"based on its age ({time_diff_years:.1f} years) and domain characteristics. "
        explanation += f"In {evolution_desc} {formatted_domain} medicine, {rate_explanation}. "
        explanation += f"Evidence typically has a half-life of {half_life_years:.1f} years. "
        explanation += f"{domain_description}."

        return explanation

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
