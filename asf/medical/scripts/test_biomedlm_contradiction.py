#!/usr/bin/env python
"""
Test script for BioMedLM integration with contradiction detection.

This script tests the BioMedLM integration with the contradiction detection service.
"""

import sys
import json
import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from asf.medical.ml.models.biomedlm import BioMedLMService
except ImportError:
    logger.warning("BioMedLM service not available, using mock implementation")
    from asf.medical.ml.models.mock_biomedlm import MockBioMedLMService as BioMedLMService
from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService

# Initialize logger
logger = logging.getLogger(__name__)

# Test data
TEST_CLAIMS = [
    {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
        "metadata1": {
            "publication_date": "2020-01-01",
            "study_design": "randomized controlled trial",
            "sample_size": 1000,
            "p_value": 0.001,
            "effect_size": 0.3
        },
        "metadata2": {
            "publication_date": "2021-06-15",
            "study_design": "randomized controlled trial",
            "sample_size": 2000,
            "p_value": 0.45,
            "effect_size": -0.05
        }
    },
    {
        "claim1": "Antibiotics are effective for treating bacterial pneumonia.",
        "claim2": "Antibiotics are ineffective for treating bacterial pneumonia.",
        "metadata1": None,
        "metadata2": None
    },
    {
        "claim1": "Regular exercise improves cardiovascular health.",
        "claim2": "Physical activity has positive effects on heart health.",
        "metadata1": None,
        "metadata2": None
    },
    {
        "claim1": "Vitamin D supplementation prevents respiratory infections.",
        "claim2": "Vitamin D supplementation has no effect on respiratory infection risk.",
        "metadata1": {
            "publication_date": "2019-05-10",
            "study_design": "meta-analysis",
            "sample_size": 5000,
            "p_value": 0.02,
            "effect_size": 0.15
        },
        "metadata2": {
            "publication_date": "2022-03-20",
            "study_design": "randomized controlled trial",
            "sample_size": 3000,
            "p_value": 0.3,
            "effect_size": 0.05
        }
    }
]

async def test_biomedlm_service():
    """Test the BioMedLM service."""
    logger.info("Testing BioMedLM service...")

    # Initialize service
    biomedlm_service = BioMedLMService()

    # Test similarity calculation
    claim1 = "Statin therapy reduces the risk of cardiovascular events."
    claim2 = "Statin therapy lowers the chance of heart problems."

    logger.info(f"Calculating similarity between: '{claim1}' and '{claim2}'")
    similarity = biomedlm_service.calculate_similarity(claim1, claim2)
    logger.info(f"Similarity: {similarity:.4f}")

    # Test contradiction detection
    claim1 = "Statin therapy reduces the risk of cardiovascular events."
    claim2 = "Statin therapy does not reduce the risk of cardiovascular events."

    logger.info(f"Detecting contradiction between: '{claim1}' and '{claim2}'")
    is_contradiction, score = biomedlm_service.detect_contradiction(claim1, claim2)
    logger.info(f"Contradiction: {is_contradiction}, Score: {score:.4f}")

    logger.info("BioMedLM service tests completed")

async def test_contradiction_service_with_biomedlm():
    """Test the contradiction service with BioMedLM integration."""
    logger.info("Testing contradiction service with BioMedLM integration...")

    # Initialize services
    biomedlm_service = BioMedLMService()
    contradiction_service = EnhancedContradictionService(biomedlm_service=biomedlm_service)

    # Test each claim pair
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")

        # Detect contradiction with BioMedLM
        result_with_biomedlm = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True
        )

        # Detect contradiction without BioMedLM
        result_without_biomedlm = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=False
        )

        # Print results
        logger.info("With BioMedLM:")
        logger.info(f"  Result: {'Contradiction' if result_with_biomedlm['is_contradiction'] else 'No contradiction'}")
        logger.info(f"  Score: {result_with_biomedlm['contradiction_score']:.4f}")
        logger.info(f"  Type: {result_with_biomedlm['contradiction_type']}")
        logger.info(f"  Confidence: {result_with_biomedlm['confidence']}")
        logger.info(f"  Methods: {', '.join(result_with_biomedlm['methods_used'])}")
        if result_with_biomedlm["explanation"]:
            logger.info(f"  Explanation: {result_with_biomedlm['explanation']}")

        logger.info("Without BioMedLM:")
        logger.info(f"  Result: {'Contradiction' if result_without_biomedlm['is_contradiction'] else 'No contradiction'}")
        logger.info(f"  Score: {result_without_biomedlm['contradiction_score']:.4f}")
        logger.info(f"  Type: {result_without_biomedlm['contradiction_type']}")
        logger.info(f"  Confidence: {result_without_biomedlm['confidence']}")
        logger.info(f"  Methods: {', '.join(result_without_biomedlm['methods_used'])}")
        if result_without_biomedlm["explanation"]:
            logger.info(f"  Explanation: {result_without_biomedlm['explanation']}")

        logger.info("---")

    logger.info("Contradiction service tests completed")

async def main():
    """Main function."""
    logger.info("Starting BioMedLM contradiction detection tests...")

    try:
        # Test BioMedLM service
        await test_biomedlm_service()

        # Test contradiction service with BioMedLM integration
        await test_contradiction_service_with_biomedlm()

        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
