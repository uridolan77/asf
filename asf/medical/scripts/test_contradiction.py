"""
Test script for enhanced contradiction detection.
This script tests the enhanced contradiction detection service.
"""
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from asf.medical.ml.services.unified_contradiction_service import ContradictionService
from asf.medical.ml.services.contradiction_classifier_service import ContradictionType
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.ml.models.tsmixer import TSMixerService
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
TEST_CLAIMS = [
    {
        "claim1": "Statin therapy reduces the risk of cardiovascular events in patients with high cholesterol.",
        "claim2": "Statin therapy does not reduce the risk of cardiovascular events in patients with high cholesterol.",
        "expected_contradiction": True,
        "expected_type": ContradictionType.NEGATION
    },
    {
        "claim1": "Aspirin is effective for preventing heart attacks in high-risk patients.",
        "claim2": "Aspirin increases the risk of bleeding in some patients.",
        "expected_contradiction": False,
        "expected_type": ContradictionType.UNKNOWN
    },
    {
        "claim1": "Vitamin D supplementation improves bone density in postmenopausal women.",
        "claim2": "Vitamin D supplementation has no effect on bone density in postmenopausal women.",
        "expected_contradiction": True,
        "expected_type": ContradictionType.DIRECT
    }
]
async def test_contradiction_detection():
    logger.info("Testing contradiction detection...")
    try:
        biomedlm_service = BioMedLMService()
        logger.info("Initialized BioMedLM service")
    except Exception as e:
        logger.warning(f"Could not initialize BioMedLM service: {str(e)}. Using mock implementation.")
        biomedlm_service = None
    try:
        tsmixer_service = TSMixerService()
        logger.info("Initialized TSMixer service")
    except Exception as e:
        logger.warning(f"Could not initialize TSMixer service: {str(e)}. Temporal contradiction detection will be limited.")
        tsmixer_service = None
    try:
        temporal_service = TemporalService(tsmixer_service=tsmixer_service)
        logger.info("Initialized temporal service")
    except Exception as e:
        logger.warning(f"Could not initialize temporal service: {str(e)}. Temporal contradiction detection will be limited.")
        temporal_service = None
    try:
        shap_explainer = SHAPExplainer()
        logger.info("Initialized SHAP explainer")
    except Exception as e:
        logger.warning(f"Could not initialize SHAP explainer: {str(e)}. Explanations will be limited.")
        shap_explainer = None

    contradiction_service = ContradictionService(
        biomedlm_service=biomedlm_service,
        tsmixer_service=tsmixer_service,
        temporal_service=temporal_service,
        shap_explainer=shap_explainer
    )

    for i, test_case in enumerate(TEST_CLAIMS):
        claim1 = test_case["claim1"]
        claim2 = test_case["claim2"]
        expected_contradiction = test_case["expected_contradiction"]
        expected_type = test_case.get("expected_type")

        logger.info(f"\nTest case {i+1}:")
        logger.info(f"Claim 1: {claim1}")
        logger.info(f"Claim 2: {claim2}")
        logger.info(f"Expected contradiction: {expected_contradiction}")
        logger.info(f"Expected type: {expected_type}")

        result = await contradiction_service.detect_contradiction(
            claim1=claim1,
            claim2=claim2,
            use_biomedlm=True,
            use_tsmixer=True,
            use_lorentz=True,
            use_shap=True
        )

        logger.info("Result:")
        logger.info(f"  Contradiction detected: {result['contradiction_detected']}")
        logger.info(f"  Contradiction score: {result['contradiction_score']}")
        logger.info(f"  Contradiction type: {result['contradiction_type']}")
        logger.info(f"  Confidence: {result['confidence']}")
        logger.info(f"  Explanation: {result['explanation']}")
        logger.info(f"  Models used: {', '.join(result['models_used'])}")

        if result['contradiction_detected'] == expected_contradiction:
            logger.info("✓ Contradiction detection matches expectation")
        else:
            logger.info("✗ Contradiction detection does not match expectation")

        if expected_type and result['contradiction_type'] == expected_type:
            logger.info("✓ Contradiction type matches expectation")
        elif expected_type:
            logger.info("✗ Contradiction type does not match expectation")

TEST_ARTICLES = [
    {
        "pmid": "12345",
        "title": "Statin therapy reduces cardiovascular events",
        "abstract": "Background: Statins are widely used for cholesterol reduction. Methods: We conducted a randomized controlled trial with 1000 patients. Results: Statin therapy reduced cardiovascular events by 30% (p<0.001). Conclusion: Statin therapy is effective for reducing cardiovascular events in patients with high cholesterol.",
        "publication_date": "2020-01-01",
        "study_design": "randomized controlled trial",
        "sample_size": 1000,
        "p_value": 0.001
    },
    {
        "pmid": "67890",
        "title": "No benefit of statin therapy in low-risk patients",
        "abstract": "Background: The benefit of statins in low-risk patients is unclear. Methods: We conducted a randomized controlled trial with 2000 patients. Results: Statin therapy did not reduce cardiovascular events in low-risk patients (p=0.45). Conclusion: Statin therapy does not provide benefit in patients with low cardiovascular risk.",
        "publication_date": "2021-06-15",
        "study_design": "randomized controlled trial",
        "sample_size": 2000,
        "p_value": 0.45
    },
    {
        "pmid": "11111",
        "title": "Statin therapy ineffective in elderly patients",
        "abstract": "Background: The efficacy of statins in elderly patients is debated. Methods: We conducted a randomized controlled trial with 500 patients over 75 years. Results: Statin therapy did not reduce cardiovascular events in elderly patients (p=0.78). Conclusion: Statin therapy does not reduce the risk of cardiovascular events in patients over 75 years with high cholesterol.",
        "publication_date": "2022-03-10",
        "study_design": "randomized controlled trial",
        "sample_size": 500,
        "p_value": 0.78
    }
]
async def test_contradiction_in_articles():
    logger.info("Testing contradiction detection in articles...")
    try:
        biomedlm_service = BioMedLMService()
        logger.info("Initialized BioMedLM service")
    except Exception as e:
        logger.warning(f"Could not initialize BioMedLM service: {str(e)}. Using basic contradiction detection.")
        biomedlm_service = None
    try:
        tsmixer_service = TSMixerService()
        temporal_service = TemporalService(tsmixer_service=tsmixer_service)
        logger.info("Initialized temporal service")
    except Exception as e:
        logger.warning(f"Could not initialize temporal service: {str(e)}. Temporal contradiction detection will be limited.")
        temporal_service = None
    try:
        shap_explainer = SHAPExplainer()
        logger.info("Initialized SHAP explainer")
    except Exception as e:
        logger.warning(f"Could not initialize SHAP explainer: {str(e)}. Explanations will be limited.")
        shap_explainer = None
    contradiction_service = ContradictionService(
        biomedlm_service=biomedlm_service,
        tsmixer_service=tsmixer_service,
        temporal_service=temporal_service,
        shap_explainer=shap_explainer
    )
    contradictions = await contradiction_service.detect_contradictions_in_articles(
        articles=TEST_ARTICLES,
        use_biomedlm=True,
        use_tsmixer=True,
        use_lorentz=True
    )
    logger.info(f"Found {len(contradictions)} contradictions in {len(TEST_ARTICLES)} articles")
    for i, contradiction in enumerate(contradictions):
        logger.info(f"Contradiction {i+1}:")
        logger.info(f"  Article 1: {contradiction['article1']['title']} (ID: {contradiction['article1']['id']})")
        logger.info(f"  Article 2: {contradiction['article2']['title']} (ID: {contradiction['article2']['id']})")
        logger.info(f"  Contradiction score: {contradiction['contradiction_score']}")
        logger.info(f"  Contradiction type: {contradiction['contradiction_type']}")
        logger.info(f"  Confidence: {contradiction['confidence']}")
        logger.info(f"  Explanation: {contradiction['explanation']}")

async def main():
    await test_contradiction_detection()
    await test_contradiction_in_articles()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())