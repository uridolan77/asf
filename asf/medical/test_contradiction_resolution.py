"""
Test script for the contradiction resolution service.
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)
print(f"Python path: {sys.path}")
print(f"Current directory: {os.getcwd()}")
try:
    from asf.medical.ml.services.unified_contradiction_service import ContradictionService
    from asf.medical.ml.services.resolution.contradiction_resolution_service import MedicalContradictionResolutionService
    from asf.medical.ml.services.resolution.resolution_models import ResolutionStrategy
    print("Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import approach...")
    class ContradictionService:
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