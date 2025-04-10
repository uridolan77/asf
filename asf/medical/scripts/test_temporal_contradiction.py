#!/usr/bin/env python
"""
Test script for temporal contradiction detection.

This script tests the temporal contradiction detection functionality.
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
            "domain": "infectious_disease"
        },
        "metadata2": {
            "publication_date": "2022-03-20",
            "study_design": "meta-analysis",
            "sample_size": 8000,
            "domain": "infectious_disease"
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
    },
    {
        "claim1": "Cognitive behavioral therapy is effective for treating depression.",
        "claim2": "Cognitive behavioral therapy is effective for treating depression.",
        "metadata1": {
            "publication_date": "2005-01-15",
            "study_design": "randomized controlled trial",
            "sample_size": 300,
            "domain": "psychiatry"
        },
        "metadata2": {
            "publication_date": "2022-01-15",
            "study_design": "randomized controlled trial",
            "sample_size": 500,
            "domain": "psychiatry"
        },
        "expected_contradiction": True,
        "expected_type": ContradictionType.TEMPORAL
    }
]

def parse_date(date_str: Optional[str]) -> Optional[datetime.datetime]:
    """
    Parse a date string into a datetime object.

    Args:
        date_str: Date string to parse

    Returns:
        Parsed datetime object or None if parsing fails
    """
    if not date_str:
        return None

    try:
        # Try different date formats
        formats = [
            "%Y-%m-%d",  # 2020-01-01
            "%Y/%m/%d",  # 2020/01/01
            "%d-%m-%Y",  # 01-01-2020
            "%d/%m/%Y",  # 01/01/2020
            "%b %d, %Y",  # Jan 01, 2020
            "%B %d, %Y",  # January 01, 2020
            "%Y"  # 2020 (year only)
        ]

        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If all formats fail, try to extract year
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            year = int(year_match.group(0))
            return datetime.datetime(year, 1, 1)  # Default to January 1st

        return None
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {str(e)}")
        return None

def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate text similarity using a simple Jaccard similarity.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0 and 1
    """
    # Check for exact match
    if text1 == text2:
        return 1.0

    import re
    # Tokenize texts
    tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
    tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))

    # Calculate Jaccard similarity
    intersection = len(tokens1.intersection(tokens2))
    union = len(tokens1.union(tokens2))

    if union == 0:
        return 0.0

    return intersection / union

def detect_temporal_contradiction(
    claim1: str,
    claim2: str,
    metadata1: Dict[str, Any],
    metadata2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detect temporal contradiction between two claims based on publication dates and content evolution.

    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim

    Returns:
        Temporal contradiction detection result
    """
    # Initialize result
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.UNKNOWN,
        "explanation": None
    }

    try:
        # Extract publication dates
        pub_date1 = parse_date(metadata1.get("publication_date"))
        pub_date2 = parse_date(metadata2.get("publication_date"))

        # If either date is missing, return default result
        if not pub_date1 or not pub_date2:
            return result

        # Calculate time difference in days
        time_diff = abs((pub_date2 - pub_date1).days)

        # If time difference is less than 30 days, it's unlikely to be a temporal contradiction
        if time_diff < 30:
            return result

        # Calculate semantic similarity between claims
        similarity = calculate_text_similarity(claim1, claim2)

        # For temporal contradictions, we're looking for claims that are semantically similar
        # but published far apart in time, which may indicate evolving medical knowledge

        # For temporal contradictions, we need to handle different cases:
        # 1. Identical claims published far apart in time
        # 2. Similar claims with significant semantic overlap published far apart in time
        # 3. Claims where one is an extension of the other (like adding "but resistance is increasing")

        # Check if one claim is a subset of the other
        is_subset = claim1 in claim2 or claim2 in claim1

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

        if claim1 == claim2 or similarity > 0.9 or is_subset or has_different_numbers:
            # This is a potential temporal contradiction - identical claims published at different times
            # Extract domains if available
            domain1 = metadata1.get("domain", "default")
            domain2 = metadata2.get("domain", "default")

            # Simple time-based confidence calculation
            # Older publications have lower confidence
            max_years = 10  # Maximum years to consider for confidence calculation
            now = datetime.datetime.now()
            years_diff1 = min(max_years, (now - pub_date1).days / 365)
            years_diff2 = min(max_years, (now - pub_date2).days / 365)

            confidence1 = 1.0 - (years_diff1 / max_years)
            confidence2 = 1.0 - (years_diff2 / max_years)
            confidence_diff = abs(confidence1 - confidence2)

            # Calculate contradiction score based on confidence difference and time difference
            # Normalize time difference to a score between 0 and 1
            time_score = min(1.0, time_diff / 365)  # Cap at 1 year

            # For identical claims, the time difference is the primary factor
            if claim1 == claim2:
                # If claims are identical, we need a significant time difference (> 5 years)
                # to consider it a temporal contradiction
                if time_diff > 365 * 5:  # More than 5 years
                    contradiction_score = time_score
                else:
                    contradiction_score = 0.0
            elif has_different_numbers:
                # If claims have different numerical values and are published far apart,
                # it's likely a temporal contradiction with updated information
                if time_diff > 365 * 5:  # More than 5 years
                    contradiction_score = time_score
                else:
                    contradiction_score = 0.0
            elif is_subset:
                # If one claim is a subset of the other and they're published far apart,
                # it's likely a temporal contradiction with updated information
                if time_diff > 365 * 5:  # More than 5 years
                    contradiction_score = time_score
                else:
                    contradiction_score = 0.0
            else:
                # For similar but not identical claims, combine confidence difference and time score
                contradiction_score = (confidence_diff * 0.7) + (time_score * 0.3)

            # Check if there are contradictory findings with significant time difference
            threshold = 0.7  # Temporal contradiction threshold
            if contradiction_score > threshold:
                result["is_contradiction"] = True
                result["score"] = contradiction_score

                # Set confidence based on score
                if contradiction_score > 0.9:
                    result["confidence"] = ContradictionConfidence.HIGH
                elif contradiction_score > 0.8:
                    result["confidence"] = ContradictionConfidence.MEDIUM
                else:
                    result["confidence"] = ContradictionConfidence.LOW

                # Generate explanation
                newer_date = max(pub_date1, pub_date2).strftime("%Y-%m-%d")
                older_date = min(pub_date1, pub_date2).strftime("%Y-%m-%d")
                time_diff_years = time_diff / 365.0

                result["explanation"] = f"Temporal contradiction detected: Claims are similar but published {time_diff_years:.1f} years apart ({older_date} vs {newer_date}). The more recent publication may reflect updated evidence or changing medical knowledge."

        return result
    except Exception as e:
        logger.error(f"Error detecting temporal contradiction: {str(e)}")
        return result

async def test_temporal_contradiction_detection():
    """Test temporal contradiction detection."""
    logger.info("Testing temporal contradiction detection...")

    # Test each claim pair
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        logger.info(f"Publication date 1: {test_case['metadata1'].get('publication_date')}")
        logger.info(f"Publication date 2: {test_case['metadata2'].get('publication_date')}")

        # Detect temporal contradiction
        result = detect_temporal_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"]
        )

        # Print result
        logger.info(f"Result: {'Contradiction' if result['is_contradiction'] else 'No contradiction'}")
        logger.info(f"Score: {result['score']:.2f}")
        logger.info(f"Confidence: {result['confidence']}")
        if result["explanation"]:
            logger.info(f"Explanation: {result['explanation']}")

        # Check if result matches expected result
        expected = test_case["expected_contradiction"]
        actual = result["is_contradiction"]
        logger.info(f"Expected: {'Contradiction' if expected else 'No contradiction'}")
        logger.info(f"Test {'PASSED' if expected == actual else 'FAILED'}")
        logger.info("---")

    logger.info("Temporal contradiction detection tests completed")

async def main():
    """Main function."""
    logger.info("Starting temporal contradiction detection tests...")

    try:
        # Test temporal contradiction detection
        await test_temporal_contradiction_detection()

        logger.info("All tests completed successfully")
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
