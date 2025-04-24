"""
Pytest configuration file for the Medical Research Synthesizer.

This module provides fixtures and configuration for pytest. It includes fixtures
for sample data, mock services, and real service instances that can be used in
tests throughout the Medical Research Synthesizer codebase.

Fixtures include:
- Sample data: articles, claims, study text
- Mock services: BioMedLM, TSMixer, SHAP explainer
- Service instances: BiasAssessmentService, PRISMAScreeningService, ContradictionService

The module also includes a fixture to clear the cache between tests to ensure
test isolation and prevent test interference.
"""

import os
import sys
import pytest
import logging
import asyncio
from typing import Dict, List, Any, Generator, AsyncGenerator
from unittest.mock import MagicMock

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..ml.services.bias_assessment_service import BiasAssessmentService, BiasRisk, BiasDomain
from ..ml.services.prisma_screening_service import PRISMAScreeningService, ScreeningStage, ScreeningDecision
from ..ml.services.contradiction_service import ContradictionService, ContradictionType, ContradictionConfidence
from ..ml.models.biomedlm import BioMedLMService
from ..ml.models.tsmixer import TSMixerService
from ..ml.models.shap_explainer import SHAPExplainer
from ..ml.services.temporal_service import TemporalService
from ..core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

@pytest.fixture
def sample_articles() -> List[Dict[str, Any]]:
    """
    Sample articles for testing.

    This fixture provides a list of sample medical research articles with
    different study types (RCT, observational study, meta-analysis) and
    varying conclusions about the same medical topic (drug X for condition Y).

    Returns:
        List of dictionaries, each representing an article with PMID, title, and abstract
    """
    return [
        {
            "pmid": "12345",
            "title": "Randomized controlled trial of drug X for condition Y",
            "abstract": "Background: Condition Y affects many patients. Methods: We conducted a double-blind, randomized controlled trial with 500 patients. Results: Drug X showed significant improvement compared to placebo (p<0.001). Conclusion: Drug X is effective for condition Y."
        },
        {
            "pmid": "67890",
            "title": "Observational study of drug X for condition Y",
            "abstract": "Background: Drug X is used for condition Y, but evidence is limited. Methods: We conducted an observational study with 200 patients. Results: Drug X showed some improvement, but results were not statistically significant (p=0.08). Conclusion: More research is needed to establish the efficacy of drug X for condition Y."
        },
        {
            "pmid": "11111",
            "title": "Meta-analysis of interventions for condition Y",
            "abstract": "Background: Multiple interventions exist for condition Y. Methods: We conducted a meta-analysis of 15 studies. Results: Drug Z showed the most consistent benefits. Drug X showed mixed results. Conclusion: Drug Z should be considered first-line therapy for condition Y."
        }
    ]

@pytest.fixture
def sample_claims() -> List[Dict[str, Any]]:
    """
    Sample claims for testing contradiction detection.

    This fixture provides a list of sample medical claims with varying
    contradiction relationships. Some claims directly contradict each other,
    while others do not contradict but present different aspects of the same topic.

    Returns:
        List of dictionaries, each containing a pair of claims, expected contradiction
        status, and expected contradiction type
    """
    return [
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

@pytest.fixture
def sample_study_text() -> str:
    """
    Sample study text for bias assessment.

    This fixture provides a sample text describing a medical study with various
    methodological characteristics that can be assessed for bias, including
    randomization, blinding, allocation concealment, sample size calculation,
    dropout rate, and outcome reporting.

    Returns:
        String containing the sample study text
    """
    return """
    This randomized controlled trial included 100 participants. Patients were randomly assigned to the treatment or placebo group.
    The study did not use blinding for the participants or researchers. Allocation was concealed using sealed envelopes.
    Sample size calculation was performed before the study. There was a 15% dropout rate in the treatment group.
    All pre-specified outcomes were reported in the results.
    """

@pytest.fixture
def mock_biomedlm_service() -> BioMedLMService:
    """
    Mock BioMedLM service for testing.

    This fixture provides a mock implementation of the BioMedLMService that
    returns predefined values for its methods. This allows testing components
    that depend on BioMedLMService without actually loading the large language
    model, making tests faster and more deterministic.

    Returns:
        A mock BioMedLM service with predefined return values
    """
    mock_service = MagicMock(spec=BioMedLMService)
    mock_service.calculate_similarity.return_value = 0.8
    mock_service.detect_contradiction.return_value = (True, 0.85)
    mock_service.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    return mock_service

@pytest.fixture
def mock_tsmixer_service() -> TSMixerService:
    """
    Mock TSMixer service for testing.

    This fixture provides a mock implementation of the TSMixerService that
    returns predefined values for its methods. This allows testing components
    that depend on TSMixerService without actually running the time series
    analysis model, making tests faster and more deterministic.

    Returns:
        A mock TSMixer service with predefined return values for temporal analysis
    """
    mock_service = MagicMock(spec=TSMixerService)

    mock_service.analyze_temporal_sequence.return_value = {
        "contradiction_scores": [0.1, 0.8, 0.3],
        "trend_analysis": {
            "trend_type": "increasing",
            "confidence": 0.7
        }
    }

    return mock_service

@pytest.fixture
def mock_shap_explainer() -> SHAPExplainer:
    """
    Mock SHAP explainer for testing.

    This fixture provides a mock implementation of the SHAPExplainer that
    returns predefined values for its methods. This allows testing components
    that depend on SHAPExplainer without actually computing SHAP values,
    making tests faster and more deterministic.

    Returns:
        A mock SHAP explainer with predefined return values for explanations
    """
    mock_explainer = MagicMock(spec=SHAPExplainer)

    mock_explainer.explain_contradiction.return_value = {
        "explanation": "The claims contradict each other due to the presence of negation.",
        "influential_words": ["reduces", "does not reduce"],
        "visualization_path": "path/to/visualization.html"
    }

    return mock_explainer

@pytest.fixture
def bias_assessment_service() -> BiasAssessmentService:
    """
    BiasAssessmentService instance for testing.

    This fixture provides a real instance of the BiasAssessmentService that
    can be used in tests. The service is initialized with default settings
    and can be used to assess bias in medical studies.

    Returns:
        A BiasAssessmentService instance ready for testing
    """
    return BiasAssessmentService()

@pytest.fixture
def prisma_screening_service(mock_biomedlm_service) -> PRISMAScreeningService:
    """
    PRISMAScreeningService instance for testing.

    This fixture provides a PRISMAScreeningService instance configured with
    the mock_biomedlm_service and predefined screening criteria for different
    stages of the PRISMA process (identification, screening, eligibility).

    Args:
        mock_biomedlm_service: The mock BioMedLM service to use

    Returns:
        A PRISMAScreeningService instance configured for testing
    """
    service = PRISMAScreeningService(biomedlm_service=mock_biomedlm_service)

    service.set_criteria(
        stage=ScreeningStage.IDENTIFICATION,
        include_criteria=["randomized", "controlled trial", "meta-analysis"],
        exclude_criteria=["animal study", "in vitro"]
    )

    service.set_criteria(
        stage=ScreeningStage.SCREENING,
        include_criteria=["condition Y", "drug X"],
        exclude_criteria=["pediatric", "pregnant women"]
    )

    service.set_criteria(
        stage=ScreeningStage.ELIGIBILITY,
        include_criteria=["efficacy", "safety"],
        exclude_criteria=["case report", "sample size < 100"]
    )

    return service

@pytest.fixture
def enhanced_contradiction_service(
    mock_biomedlm_service,
    mock_tsmixer_service,
    mock_shap_explainer
) -> ContradictionService:
    """
    ContradictionService instance for testing.

    This fixture provides a ContradictionService instance configured with
    mock services for BioMedLM, TSMixer, and SHAP explainer. This allows
    testing the contradiction detection functionality without requiring
    the actual ML models.

    Args:
        mock_biomedlm_service: The mock BioMedLM service to use
        mock_tsmixer_service: The mock TSMixer service to use
        mock_shap_explainer: The mock SHAP explainer to use

    Returns:
        A ContradictionService instance configured with mock services
    """
    return ContradictionService(
        biomedlm_service=mock_biomedlm_service,
        tsmixer_service=mock_tsmixer_service,
        shap_explainer=mock_shap_explainer
    )

@pytest.fixture(scope="function", autouse=True)
async def clear_cache():
    """
    Clear the cache between tests.

    This fixture ensures that the cache is cleared between tests to prevent
    test interference. It creates a new event loop for each test and closes
    it after the test completes.

    Yields:
        An asyncio event loop that can be used in the test
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
