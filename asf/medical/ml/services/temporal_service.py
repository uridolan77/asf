"""
Temporal service for the Medical Research Synthesizer.

This module provides a service for temporal analysis of medical literature.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import datetime

from asf.medical.ml.models import TSMixerService, BioMedLMService

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
        """Initialize the temporal service.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        self.tsmixer_service = None
        self.biomedlm_service = None

        self.domain_characteristics = {
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

            "default": {
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
        Get the BioMedLM service.

        Returns:
            BioMedLMService: The BioMedLM service
        Calculate temporal confidence for a publication with domain-specific characteristics.

        Args:
            publication_date: Publication date (YYYY-MM-DD)
            domain: Medical domain
            reference_date: Reference date (YYYY-MM-DD, default: today)
            include_details: Whether to include detailed information in the result

        Returns:
            Temporal confidence (0-1) or dict with confidence and details
        Generate an explanation for the temporal confidence calculation.

        Args:
            domain_info: Domain characteristics
            confidence: Calculated confidence
            time_diff_years: Time difference in years
            domain: Medical domain

        Returns:
            Explanation string
        Calculate Beta distribution parameters for a confidence value.

        Args:
            confidence: Confidence value (0-1)
            certainty: Certainty about the confidence (0-1)

        Returns:
            Tuple of (alpha, beta) parameters
        Analyze a temporal sequence of claims.

        Args:
            claims: List of claims with timestamps and text

        Returns:
            Analysis results
        Analyze how a claim has evolved over time.

        Args:
            claim: The main claim
            related_claims: List of related claims with timestamps

        Returns:
            Analysis results
        if self.tsmixer_service is not None:
            self.tsmixer_service.unload_model()

        if self.biomedlm_service is not None:
            self.biomedlm_service.unload_model()

        logger.info("All models unloaded")
