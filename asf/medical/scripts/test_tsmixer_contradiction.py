import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
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
try:
    from asf.medical.ml.services.unified_contradiction_service import ContradictionService
    from asf.medical.ml.services.temporal_service import TemporalService
    from asf.medical.ml.models.biomedlm import BioMedLMService
    USING_REAL_SERVICES = True
except ImportError:
    logger.warning("Failed to import required modules. Using mock implementations.")
    USING_REAL_SERVICES = False
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
        """Initialize the mock TSMixer service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
        Analyze a temporal sequence of claims.
        Args:
            sequence: List of claims with timestamps
            embedding_fn: Function to embed claims
        Returns:
            Analysis results
    def __init__(self):
        """Initialize the mock BioMedLM service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        logger.info("Mock BioMedLM service initialized")
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        if text1 == text2:
            return 1.0
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0
    def encode(self, text):
        """Encode text into a vector.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        import numpy as np
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(768)
class MockTemporalService:
    """Mock temporal service for testing."""
    def __init__(self):
        """Initialize the mock temporal service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    def __init__(self, biomedlm_service=None, temporal_service=None):
        """Initialize the mock contradiction service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        logger.info("Mock contradiction service initialized")
        self.biomedlm_service = biomedlm_service or MockBioMedLMService()
        self.temporal_service = temporal_service or MockTemporalService()
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
    logger.info("Testing TSMixer integration with temporal contradiction detection...")
    biomedlm_service = MockBioMedLMService()
    temporal_service = MockTemporalService()
    if USING_REAL_SERVICES:
        contradiction_service = ContradictionService(
            biomedlm_service=biomedlm_service,
            temporal_service=temporal_service
        )
    else:
        contradiction_service = MockUnifiedUnifiedContradictionService(
            biomedlm_service=biomedlm_service,
            temporal_service=temporal_service
        )
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        logger.info(f"Publication date 1: {test_case['metadata1'].get('publication_date')}")
        logger.info(f"Publication date 2: {test_case['metadata2'].get('publication_date')}")
        result_with_tsmixer = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True,
            use_temporal=True,
            use_tsmixer=True
        )
        result_without_tsmixer = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True,
            use_temporal=True,
            use_tsmixer=False
        )
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
        expected = test_case["expected_contradiction"]
        actual_with_tsmixer = result_with_tsmixer["is_contradiction"]
        actual_without_tsmixer = result_without_tsmixer["is_contradiction"]
        logger.info(f"Expected: {'Contradiction' if expected else 'No contradiction'}")
        logger.info(f"Test with TSMixer: {'PASSED' if expected == actual_with_tsmixer else 'FAILED'}")
        logger.info(f"Test without TSMixer: {'PASSED' if expected == actual_without_tsmixer else 'FAILED'}")
        logger.info("---")
    logger.info("TSMixer contradiction detection tests completed")
async def main():