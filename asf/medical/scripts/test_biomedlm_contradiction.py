import sys
import logging
from pathlib import Path
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
try:
    from asf.medical.ml.models.biomedlm import BioMedLMService
except ImportError:
    logger.warning("BioMedLM service not available, using mock implementation")
    from asf.medical.ml.models.mock_biomedlm import MockBioMedLMService as BioMedLMService
from asf.medical.ml.services.unified_contradiction_service import ContradictionService
logger = logging.getLogger(__name__)
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
    logger.info("Testing contradiction service with BioMedLM integration...")
    biomedlm_service = BioMedLMService()
    contradiction_service = ContradictionService(biomedlm_service=biomedlm_service)
    for i, test_case in enumerate(TEST_CLAIMS):
        logger.info(f"Test case {i+1}:")
        logger.info(f"Claim 1: {test_case['claim1']}")
        logger.info(f"Claim 2: {test_case['claim2']}")
        result_with_biomedlm = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=True
        )
        result_without_biomedlm = await contradiction_service.detect_contradiction(
            claim1=test_case["claim1"],
            claim2=test_case["claim2"],
            metadata1=test_case["metadata1"],
            metadata2=test_case["metadata2"],
            use_biomedlm=False
        )
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