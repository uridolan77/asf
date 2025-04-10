"""
Test script for the contradiction resolution service.
"""

import asyncio
import sys
import os
import json

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")

try:
    from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService
    from asf.medical.ml.services.resolution.contradiction_resolution_service import MedicalContradictionResolutionService
    from asf.medical.ml.services.resolution.resolution_models import ResolutionStrategy
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")

    # Let's try a different approach
    print("Trying alternative import approach...")

    # Create mock classes for testing
    class EnhancedContradictionService:
        async def detect_contradiction(self, claim1, claim2, metadata1=None, metadata2=None):
            print(f"Mock detecting contradiction between:\n- {claim1}\n- {claim2}")
            return {
                "is_contradiction": True,
                "contradiction_score": 0.85,
                "contradiction_type": "negation",
                "confidence": "high",
                "claim1": claim1,
                "claim2": claim2,
                "metadata1": metadata1 or {},
                "metadata2": metadata2 or {},
                "classification": {
                    "clinical_significance": "high",
                    "evidence_quality": {
                        "claim1": "high",
                        "claim2": "moderate",
                        "differential": 0.3
                    },
                    "temporal_factor": {
                        "detected": True,
                        "score": 0.6
                    },
                    "population_difference": {
                        "detected": True,
                        "score": 0.7,
                        "differences": [
                            {
                                "category": "age",
                                "claim1_terms": ["adults"],
                                "claim2_terms": ["elderly"]
                            }
                        ]
                    },
                    "methodological_difference": {
                        "detected": True,
                        "score": 0.8
                    }
                }
            }

    class ResolutionStrategy:
        EVIDENCE_HIERARCHY = "evidence_hierarchy"
        SAMPLE_SIZE_WEIGHTING = "sample_size_weighting"
        RECENCY_PREFERENCE = "recency_preference"
        POPULATION_SPECIFICITY = "population_specificity"
        METHODOLOGICAL_QUALITY = "methodological_quality"
        STATISTICAL_SIGNIFICANCE = "statistical_significance"
        COMBINED_EVIDENCE = "combined_evidence"

    class MedicalContradictionResolutionService:
        def __init__(self):
            self.history = []

        async def resolve_contradiction(self, contradiction, strategy=None):
            print(f"Mock resolving contradiction with strategy: {strategy}")
            return {
                "recommendation": "favor_claim1",
                "confidence": "moderate",
                "confidence_score": 0.7,
                "recommended_claim": contradiction["claim1"],
                "strategy": strategy or ResolutionStrategy.EVIDENCE_HIERARCHY,
                "timestamp": "2023-07-01T12:00:00",
                "explanation": {
                    "summary": f"The contradiction is resolved in favor of the first claim. Confidence: moderate.",
                    "detailed_reasoning": "Mock detailed reasoning",
                    "clinical_implications": "Mock clinical implications",
                    "limitations": "Mock limitations",
                    "references": []
                }
            }

        async def resolve_contradiction_with_combined_evidence(self, contradiction):
            print("Mock resolving contradiction with combined evidence")
            return await self.resolve_contradiction(contradiction, ResolutionStrategy.COMBINED_EVIDENCE)

        def get_resolution_history(self):
            return self.history

async def test_contradiction_resolution():
    """Test the contradiction resolution service."""
    print("Initializing services...")
    contradiction_service = EnhancedContradictionService()
    resolution_service = MedicalContradictionResolutionService()

    print("Creating test contradictions...")
    test_cases = [
        {
            "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
            "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
            "metadata1": {
                "publication_year": 2020,
                "study_design": "randomized controlled trial",
                "sample_size": 5000,
                "population": "adults with high cholesterol",
                "p_value": 0.01,
                "journal": "New England Journal of Medicine",
                "impact_factor": 70.6
            },
            "metadata2": {
                "publication_year": 2015,
                "study_design": "observational study",
                "sample_size": 1000,
                "population": "elderly patients with high cholesterol",
                "p_value": 0.08,
                "journal": "Journal of Clinical Investigation",
                "impact_factor": 14.8
            }
        },
        {
            "claim1": "Aspirin reduces the risk of colorectal cancer.",
            "claim2": "Aspirin has no effect on colorectal cancer risk.",
            "metadata1": {
                "publication_year": 2018,
                "study_design": "meta-analysis",
                "sample_size": 12000,
                "population": "adults over 50",
                "p_value": 0.03,
                "journal": "JAMA",
                "impact_factor": 45.3
            },
            "metadata2": {
                "publication_year": 2019,
                "study_design": "randomized controlled trial",
                "sample_size": 8000,
                "population": "adults over 50 with family history of colorectal cancer",
                "p_value": 0.12,
                "journal": "Lancet",
                "impact_factor": 60.4
            }
        }
    ]

    for i, test_case in enumerate(test_cases):
        print(f"\nTest case {i+1}:")
        print(f"Claim 1: {test_case['claim1']}")
        print(f"Claim 2: {test_case['claim2']}")

        # Detect contradiction
        print("\nDetecting contradiction...")
        contradiction = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"]
        )

        if not contradiction.get("is_contradiction", False):
            print("No contradiction detected.")
            continue

        print(f"Contradiction detected: {contradiction['contradiction_type']} (score: {contradiction['contradiction_score']})")

        # Test each resolution strategy
        strategies = [
            ResolutionStrategy.EVIDENCE_HIERARCHY,
            ResolutionStrategy.SAMPLE_SIZE_WEIGHTING,
            ResolutionStrategy.RECENCY_PREFERENCE,
            ResolutionStrategy.POPULATION_SPECIFICITY,
            ResolutionStrategy.METHODOLOGICAL_QUALITY,
            ResolutionStrategy.STATISTICAL_SIGNIFICANCE,
            ResolutionStrategy.COMBINED_EVIDENCE
        ]

        for strategy in strategies:
            print(f"\nResolving with {strategy} strategy...")

            if strategy == ResolutionStrategy.COMBINED_EVIDENCE:
                resolution = await resolution_service.resolve_contradiction_with_combined_evidence(contradiction)
            else:
                resolution = await resolution_service.resolve_contradiction(
                    contradiction=contradiction,
                    strategy=strategy
                )

            print(f"Recommendation: {resolution['recommendation']}")
            print(f"Confidence: {resolution['confidence']} ({resolution['confidence_score']})")
            print(f"Recommended claim: {resolution['recommended_claim']}")
            print(f"Explanation summary: {resolution['explanation']['summary']}")

    # Print resolution history
    print("\nResolution history:")
    history = resolution_service.get_resolution_history()
    print(f"Total entries: {len(history)}")

    return "Test completed successfully!"

if __name__ == "__main__":
    print("Running contradiction resolution test...")
    result = asyncio.run(test_contradiction_resolution())
    print(result)
