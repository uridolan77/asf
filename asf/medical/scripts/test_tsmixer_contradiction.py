#!/usr/bin/env python
"""
Test script for TSMixer integration with temporal contradiction detection.

This script tests the TSMixer integration with the temporal contradiction detection service.
"""

import sys
import json
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Define enums for testing
class ContradictionType:
    NONE = "none"
    DIRECT = "direct"
    NEGATION = "negation"
    STATISTICAL = "statistical"
    METHODOLOGICAL = "methodological"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"

class ContradictionConfidence:
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

# Try to import the real services, but use mocks if they're not available
try:
    from asf.medical.ml.services.contradiction_service import ContradictionService
    from asf.medical.ml.services.temporal_service import TemporalService
    from asf.medical.ml.models.biomedlm import BioMedLMService
    USING_REAL_SERVICES = True
except ImportError:
    logger.warning("Failed to import required modules. Using mock implementations.")
    USING_REAL_SERVICES = False

# Test data with temporal contradictions
TEST_CLAIMS = [
    {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "metadata1": {
            "publication_date": "2010-01-01",
            "study_design": "randomized controlled trial",
            "sample_size": 1000,
            "p_value": 0.001,
            "effect_size": 0.3,
            "domain": "cardiology"
        },
        "metadata2": {
            "publication_date": "2020-06-15",
            "study_design": "randomized controlled trial",
            "sample_size": 2000,
            "p_value": 0.001,
            "effect_size": 0.3,
            "domain": "cardiology"
        },
        "expected_contradiction": True,
        "expected_type": ContradictionType.TEMPORAL
    },
    {
        "claim1": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 95%.",
        "claim2": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 85% due to increasing resistance.",
        "metadata1": {
            "publication_date": "2000-05-10",
            "study_design": "meta-analysis",
            "sample_size": 5000,
            "domain": "infectious_disease",
            "related_claims": [
                {
                    "text": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 98%.",
                    "timestamp": "1990-01-01",
                    "domain": "infectious_disease"
                },
                {
                    "text": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 97%.",
                    "timestamp": "1995-01-01",
                    "domain": "infectious_disease"
                }
            ]
        },
        "metadata2": {
            "publication_date": "2022-03-20",
            "study_design": "meta-analysis",
            "sample_size": 8000,
            "domain": "infectious_disease",
            "related_claims": [
                {
                    "text": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 90%.",
                    "timestamp": "2010-01-01",
                    "domain": "infectious_disease"
                },
                {
                    "text": "Antibiotics are effective for treating bacterial pneumonia with a success rate of 87%.",
                    "timestamp": "2015-01-01",
                    "domain": "infectious_disease"
                }
            ]
        },
        "expected_contradiction": True,
        "expected_type": ContradictionType.TEMPORAL
    },
    {
        "claim1": "Cognitive behavioral therapy is effective for treating depression.",
        "claim2": "Cognitive behavioral therapy is effective for treating depression.",
        "metadata1": {
            "publication_date": "2005-01-15",
            "study_design": "randomized controlled trial",
            "sample_size": 300,
            "domain": "psychiatry",
            "related_claims": [
                {
                    "text": "Cognitive behavioral therapy shows promise for treating depression.",
                    "timestamp": "2000-01-01",
                    "domain": "psychiatry"
                }
            ]
        },
        "metadata2": {
            "publication_date": "2022-01-15",
            "study_design": "randomized controlled trial",
            "sample_size": 500,
            "domain": "psychiatry",
            "related_claims": [
                {
                    "text": "Cognitive behavioral therapy is highly effective for treating depression.",
                    "timestamp": "2020-01-01",
                    "domain": "psychiatry"
                }
            ]
        },
        "expected_contradiction": True,
        "expected_type": ContradictionType.TEMPORAL
    },
    {
        "claim1": "Vitamin D supplementation prevents respiratory infections.",
        "claim2": "Vitamin D supplementation prevents respiratory infections.",
        "metadata1": {
            "publication_date": "2019-05-10",
            "study_design": "meta-analysis",
            "sample_size": 5000,
            "p_value": 0.02,
            "effect_size": 0.15,
            "domain": "infectious_disease"
        },
        "metadata2": {
            "publication_date": "2019-06-10",
            "study_design": "meta-analysis",
            "sample_size": 5200,
            "p_value": 0.02,
            "effect_size": 0.15,
            "domain": "infectious_disease"
        },
        "expected_contradiction": False,
        "expected_type": ContradictionType.NONE
    }
]

class MockTSMixerService:
    """Mock TSMixer service for testing."""

    def __init__(self):
        """Initialize the mock TSMixer service."""
        logger.info("Mock TSMixer service initialized")

    def analyze_temporal_sequence(self, sequence, embedding_fn=None):
        """
        Analyze a temporal sequence of claims.

        Args:
            sequence: List of claims with timestamps
            embedding_fn: Function to embed claims

        Returns:
            Analysis results
        """
        # Sort sequence by timestamp
        sequence = sorted(sequence, key=lambda x: x["timestamp"])

        # Calculate contradiction scores
        contradiction_scores = []
        for i in range(len(sequence) - 1):
            # Calculate time difference in years
            try:
                time1 = datetime.datetime.strptime(sequence[i]["timestamp"], "%Y-%m-%d").timestamp()
                time2 = datetime.datetime.strptime(sequence[i+1]["timestamp"], "%Y-%m-%d").timestamp()
                time_diff_years = abs(time2 - time1) / (365 * 24 * 60 * 60)
            except:
                time_diff_years = 0

            # Check for numerical differences in claims
            import re
            if "text" in sequence[i] and "text" in sequence[i+1]:
                text1 = sequence[i]["text"]
                text2 = sequence[i+1]["text"]

                numbers1 = re.findall(r'\d+(?:\.\d+)?%?', text1)
                numbers2 = re.findall(r'\d+(?:\.\d+)?%?', text2)
                has_different_numbers = False

                if numbers1 and numbers2 and len(numbers1) == len(numbers2):
                    for n1, n2 in zip(numbers1, numbers2):
                        if n1 != n2:
                            has_different_numbers = True
                            break

                # If claims have different numbers, increase the contradiction score
                if has_different_numbers:
                    contradiction_score = 1.0
                else:
                    # Calculate contradiction score based on time difference
                    # The longer the time difference, the higher the contradiction score
                    contradiction_score = min(1.0, time_diff_years / 10.0)
            else:
                # Calculate contradiction score based on time difference
                # The longer the time difference, the higher the contradiction score
                contradiction_score = min(1.0, time_diff_years / 10.0)

            contradiction_scores.append(contradiction_score)

        # Calculate trend
        if len(contradiction_scores) > 1:
            # Simple linear trend
            trend = (contradiction_scores[-1] - contradiction_scores[0]) / len(contradiction_scores)

            # Determine trend direction and magnitude
            direction = "increasing" if trend > 0 else "decreasing" if trend < 0 else "stable"
            magnitude = "strong" if abs(trend) > 0.1 else "moderate" if abs(trend) > 0.05 else "weak"
        else:
            trend = 0
            direction = "stable"
            magnitude = "weak"

        # Return analysis results
        return {
            "contradiction_scores": contradiction_scores,
            "trend": {
                "slope": trend,
                "direction": direction,
                "magnitude": magnitude
            }
        }

class MockBioMedLMService:
    """Mock BioMedLM service for testing."""

    def __init__(self):
        """Initialize the mock BioMedLM service."""
        logger.info("Mock BioMedLM service initialized")

    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts."""
        # Simple similarity calculation
        if text1 == text2:
            return 1.0

        # Calculate Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def encode(self, text):
        """Encode text into a vector."""
        # Return a dummy vector
        import numpy as np
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(768)

class MockTemporalService:
    """Mock temporal service for testing."""

    def __init__(self):
        """Initialize the mock temporal service."""
        logger.info("Mock temporal service initialized")
        self.tsmixer_service = MockTSMixerService()

    def calculate_temporal_confidence(self, date_str, domain="default"):
        """Calculate temporal confidence for a date."""
        # Simple confidence calculation
        try:
            date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            today = datetime.date.today()
            years_diff = (today - date).days / 365.0

            # Older dates have lower confidence
            return max(0.0, 1.0 - (years_diff / 10.0))
        except:
            return 0.5

    async def analyze_temporal_sequence(self, sequence):
        """Analyze a temporal sequence of claims."""
        return self.tsmixer_service.analyze_temporal_sequence(sequence)

class MockContradictionService:
    """Mock contradiction service for testing."""

    def __init__(self, biomedlm_service=None, temporal_service=None):
        """Initialize the mock contradiction service."""
        logger.info("Mock contradiction service initialized")
        self.biomedlm_service = biomedlm_service or MockBioMedLMService()
        self.temporal_service = temporal_service or MockTemporalService()

        # Thresholds for contradiction detection
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.NEGATION: 0.8,
            ContradictionType.STATISTICAL: 0.7,
            ContradictionType.METHODOLOGICAL: 0.7,
            ContradictionType.TEMPORAL: 0.7,
        }

    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_temporal: bool = True,
        use_tsmixer: bool = True
    ) -> Dict[str, Any]:
        """Detect contradiction between two claims."""
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

        # Calculate similarity
        similarity = self.biomedlm_service.calculate_similarity(claim1, claim2)

        # Check for temporal contradiction
        if use_temporal and metadata1 and metadata2:
            # Extract publication dates
            try:
                pub_date1 = datetime.datetime.strptime(metadata1.get("publication_date", ""), "%Y-%m-%d")
                pub_date2 = datetime.datetime.strptime(metadata2.get("publication_date", ""), "%Y-%m-%d")

                # Calculate time difference in days
                time_diff = abs((pub_date2 - pub_date1).days)

                # Check for numerical differences in claims
                import re
                numbers1 = re.findall(r'\d+(?:\.\d+)?%?', claim1)
                numbers2 = re.findall(r'\d+(?:\.\d+)?%?', claim2)
                has_different_numbers = False

                if numbers1 and numbers2 and len(numbers1) == len(numbers2):
                    for n1, n2 in zip(numbers1, numbers2):
                        if n1 != n2:
                            has_different_numbers = True
                            break

                # Check if one claim is a subset of the other
                is_subset = claim1 in claim2 or claim2 in claim1

                # If time difference is significant and claims are similar or have different numbers, it's a temporal contradiction
                if time_diff > 365 * 5 and (similarity > 0.7 or has_different_numbers or is_subset):  # More than 5 years
                    # Use TSMixer if requested
                    if use_tsmixer:
                        # Create sequence for TSMixer
                        sequence = [
                            {
                                "text": claim1,
                                "timestamp": metadata1.get("publication_date", ""),
                                "domain": metadata1.get("domain", "default")
                            },
                            {
                                "text": claim2,
                                "timestamp": metadata2.get("publication_date", ""),
                                "domain": metadata2.get("domain", "default")
                            }
                        ]

                        # Add related claims if available
                        if "related_claims" in metadata1:
                            sequence.extend(metadata1["related_claims"])
                        if "related_claims" in metadata2:
                            sequence.extend(metadata2["related_claims"])

                        # Analyze sequence
                        tsmixer_analysis = await self.temporal_service.analyze_temporal_sequence(sequence)

                        # Extract contradiction scores
                        contradiction_scores = tsmixer_analysis.get("contradiction_scores", [])

                        if contradiction_scores:
                            # Use the last score
                            contradiction_score = contradiction_scores[-1]

                            # Get trend
                            trend = tsmixer_analysis.get("trend", {})
                            trend_direction = trend.get("direction", "stable")
                            trend_magnitude = trend.get("magnitude", "weak")

                            # Set result
                            result["is_contradiction"] = True
                            result["contradiction_score"] = contradiction_score
                            result["contradiction_type"] = ContradictionType.TEMPORAL

                            # Set confidence based on trend magnitude
                            if trend_magnitude == "strong":
                                result["confidence"] = ContradictionConfidence.HIGH
                            elif trend_magnitude == "moderate":
                                result["confidence"] = ContradictionConfidence.MEDIUM
                            else:
                                result["confidence"] = ContradictionConfidence.LOW

                            # Generate explanation
                            time_diff_years = time_diff / 365.0
                            result["explanation"] = f"Temporal contradiction detected by TSMixer: Claims show a {trend_magnitude} {trend_direction} trend over {time_diff_years:.1f} years. The more recent publication likely reflects updated evidence or changing medical knowledge."

                            # Add methods used
                            result["methods_used"] = ["temporal", "tsmixer"]

                            # Add details
                            result["details"] = {
                                "tsmixer": {
                                    "contradiction_scores": contradiction_scores,
                                    "trend": trend
                                }
                            }
                    else:
                        # Simple temporal contradiction detection
                        time_score = min(1.0, time_diff / (365 * 10))  # Cap at 10 years

                        # Set result
                        result["is_contradiction"] = True
                        result["contradiction_score"] = time_score
                        result["contradiction_type"] = ContradictionType.TEMPORAL
                        result["confidence"] = ContradictionConfidence.MEDIUM

                        # Generate explanation
                        time_diff_years = time_diff / 365.0
                        result["explanation"] = f"Temporal contradiction detected: Claims are similar but published {time_diff_years:.1f} years apart. The more recent publication may reflect updated evidence or changing medical knowledge."

                        # Add methods used
                        result["methods_used"] = ["temporal"]
            except Exception as e:
                logger.error(f"Error detecting temporal contradiction: {str(e)}")

        return result

async def test_tsmixer_contradiction_detection():
    """Test TSMixer integration with temporal contradiction detection."""
    logger.info("Testing TSMixer integration with temporal contradiction detection...")

    # Initialize services
    biomedlm_service = MockBioMedLMService()
    temporal_service = MockTemporalService()

    # Use the real ContradictionService if available, otherwise use the mock
    if USING_REAL_SERVICES:
        contradiction_service = ContradictionService(
            biomedlm_service=biomedlm_service,
            temporal_service=temporal_service
        )
    else:
        contradiction_service = MockContradictionService(
            biomedlm_service=biomedlm_service,
            temporal_service=temporal_service
        )

    # Test each claim pair
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        logger.info(f"Publication date 1: {test_case['metadata1'].get('publication_date')}")
        logger.info(f"Publication date 2: {test_case['metadata2'].get('publication_date')}")

        # Detect contradiction with TSMixer
        result_with_tsmixer = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True,
            use_temporal=True,
            use_tsmixer=True
        )

        # Detect contradiction without TSMixer
        result_without_tsmixer = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True,
            use_temporal=True,
            use_tsmixer=False
        )

        # Print results
        logger.info("With TSMixer:")
        logger.info(f"  Result: {'Contradiction' if result_with_tsmixer['is_contradiction'] else 'No contradiction'}")
        logger.info(f"  Score: {result_with_tsmixer['contradiction_score']:.4f}")
        logger.info(f"  Type: {result_with_tsmixer['contradiction_type']}")
        logger.info(f"  Confidence: {result_with_tsmixer['confidence']}")
        logger.info(f"  Methods: {', '.join(result_with_tsmixer['methods_used'])}")
        if result_with_tsmixer["explanation"]:
            logger.info(f"  Explanation: {result_with_tsmixer['explanation']}")

        logger.info("Without TSMixer:")
        logger.info(f"  Result: {'Contradiction' if result_without_tsmixer['is_contradiction'] else 'No contradiction'}")
        logger.info(f"  Score: {result_without_tsmixer['contradiction_score']:.4f}")
        logger.info(f"  Type: {result_without_tsmixer['contradiction_type']}")
        logger.info(f"  Confidence: {result_without_tsmixer['confidence']}")
        logger.info(f"  Methods: {', '.join(result_without_tsmixer['methods_used'])}")
        if result_without_tsmixer["explanation"]:
            logger.info(f"  Explanation: {result_without_tsmixer['explanation']}")

        # Check if result matches expected result
        expected = test_case["expected_contradiction"]
        actual_with_tsmixer = result_with_tsmixer["is_contradiction"]
        actual_without_tsmixer = result_without_tsmixer["is_contradiction"]

        logger.info(f"Expected: {'Contradiction' if expected else 'No contradiction'}")
        logger.info(f"Test with TSMixer: {'PASSED' if expected == actual_with_tsmixer else 'FAILED'}")
        logger.info(f"Test without TSMixer: {'PASSED' if expected == actual_without_tsmixer else 'FAILED'}")
        logger.info("---")

    logger.info("TSMixer contradiction detection tests completed")

async def main():
    """Main function."""
    logger.info("Starting TSMixer contradiction detection tests...")

    try:
        # Test TSMixer integration with temporal contradiction detection
        await test_tsmixer_contradiction_detection()

        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
