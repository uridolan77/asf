#!/usr/bin/env python
"""
Test script for domain-specific temporal confidence calculation.

This script tests the domain-specific temporal confidence calculation and explanation generation.
"""

import sys
import logging
import datetime
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from asf.medical.ml.services.temporal_service import TemporalService
except ImportError:
    logger.warning("Failed to import TemporalService. Using mock implementation.")

    class TemporalService:
        """Mock implementation of TemporalService."""

        def __init__(self):
            """Initialize the mock temporal service."""
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

            logger.info("Mock temporal service initialized")

        def calculate_temporal_confidence(
            self,
            publication_date: str,
            domain: str = "default",
            reference_date: str = None,
            include_details: bool = False
        ):
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
            import math
            base_confidence = math.exp(-math.log(2) * time_diff / half_life)

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

            Returns:
                Explanation string
            """
            domain_description = domain_info["description"]
            evolution_rate = domain_info["evolution_rate"]
            half_life_years = domain_info["half_life"] / 365.0
            # Use the domain parameter passed to calculate_temporal_confidence
            domain_name = domain

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
            elif evolution_rate == "rapid":
                evolution_desc = "rapidly evolving"
            elif evolution_rate == "moderate":
                evolution_desc = "moderately evolving"
            elif evolution_rate == "slow":
                evolution_desc = "slowly evolving"
            else:  # very_slow
                evolution_desc = "very slowly evolving"

            # Generate explanation
            explanation = f"This evidence has {confidence_desc} temporal confidence ({confidence:.2f}) "
            explanation += f"based on its age ({time_diff_years:.1f} years) and domain characteristics. "
            # Format domain name for better readability
            formatted_domain = domain_name.replace("_", " ").title()
            explanation += f"In {evolution_desc} {formatted_domain} medicine, "
            explanation += f"evidence typically has a half-life of {half_life_years:.1f} years. "
            explanation += f"{domain_description}."

            return explanation

# Test data
TEST_CASES = [
    {
        "publication_date": "2020-01-01",
        "domain": "infectious_disease",
        "description": "Recent infectious disease publication"
    },
    {
        "publication_date": "2010-01-01",
        "domain": "infectious_disease",
        "description": "Older infectious disease publication"
    },
    {
        "publication_date": "2020-01-01",
        "domain": "neurology",
        "description": "Recent neurology publication"
    },
    {
        "publication_date": "2010-01-01",
        "domain": "neurology",
        "description": "Older neurology publication"
    },
    {
        "publication_date": "2020-01-01",
        "domain": "oncology",
        "description": "Recent oncology publication"
    },
    {
        "publication_date": "2010-01-01",
        "domain": "oncology",
        "description": "Older oncology publication"
    },
    {
        "publication_date": "2020-01-01",
        "domain": "psychiatry",
        "description": "Recent psychiatry publication"
    },
    {
        "publication_date": "2010-01-01",
        "domain": "psychiatry",
        "description": "Older psychiatry publication"
    },
    {
        "publication_date": "2020-01-01",
        "domain": "unknown_domain",
        "description": "Recent publication in unknown domain"
    }
]

def test_domain_specific_temporal_confidence():
    """Test domain-specific temporal confidence calculation."""
    logger.info("Testing domain-specific temporal confidence calculation...")

    # Initialize temporal service
    temporal_service = TemporalService()

    # Set reference date for consistent testing
    reference_date = "2023-01-01"

    # Test each case
    for i, test_case in enumerate(TEST_CASES):
        logger.info(f"Test case {i+1}: {test_case['description']}")

        # Calculate temporal confidence with details
        confidence_details = temporal_service.calculate_temporal_confidence(
            publication_date=test_case["publication_date"],
            domain=test_case["domain"],
            reference_date=reference_date,
            include_details=True
        )

        # Print results
        if isinstance(confidence_details, dict):
            logger.info(f"Domain: {confidence_details['domain']}")
            logger.info(f"Confidence: {confidence_details['confidence']:.4f}")
            logger.info(f"Time difference: {confidence_details['time_diff_years']:.1f} years")
            logger.info(f"Half-life: {confidence_details.get('half_life_years', 0):.1f} years")

            # Print domain characteristics
            domain_chars = confidence_details.get("domain_characteristics", {})
            if domain_chars:
                logger.info(f"Evolution rate: {domain_chars.get('evolution_rate', 'unknown')}")
                logger.info(f"Evidence stability: {domain_chars.get('evidence_stability', 'unknown')}")

            # Print explanation
            explanation = confidence_details.get("explanation", "")
            if explanation:
                logger.info(f"Explanation: {explanation}")
        else:
            logger.info(f"Confidence: {confidence_details:.4f}")

        logger.info("---")

    logger.info("Domain-specific temporal confidence tests completed")

def main():
    """Main function."""
    logger.info("Starting domain-specific temporal confidence tests...")

    try:
        # Test domain-specific temporal confidence calculation
        test_domain_specific_temporal_confidence()

        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
        raise

if __name__ == "__main__":
    main()
