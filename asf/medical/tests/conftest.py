"""
Pytest configuration file for the Medical Research Synthesizer.

This module provides fixtures and configuration for pytest.
"""

import os
import sys
import pytest
import logging
import asyncio
from typing import Dict, List, Any, Generator, AsyncGenerator
from unittest.mock import MagicMock

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import components
from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService, BiasRisk, BiasDomain
from asf.medical.ml.services.prisma_screening_service import PRISMAScreeningService, ScreeningStage, ScreeningDecision
from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService, ContradictionType, ContradictionConfidence
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.core.cache import cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test data
@pytest.fixture
def sample_articles() -> List[Dict[str, Any]]:
    """Sample articles for testing."""
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
    """Sample claims for testing contradiction detection."""
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
    """Sample study text for bias assessment."""
    return """
    This randomized controlled trial included 100 participants. Patients were randomly assigned to the treatment or placebo group. 
    The study did not use blinding for the participants or researchers. Allocation was concealed using sealed envelopes.
    Sample size calculation was performed before the study. There was a 15% dropout rate in the treatment group.
    All pre-specified outcomes were reported in the results.
    """

@pytest.fixture
def mock_biomedlm_service() -> BioMedLMService:
    """Mock BioMedLM service for testing."""
    mock_service = MagicMock(spec=BioMedLMService)
    
    # Mock calculate_similarity method
    mock_service.calculate_similarity.return_value = 0.8
    
    # Mock detect_contradiction method
    mock_service.detect_contradiction.return_value = (True, 0.85)
    
    # Mock encode method
    mock_service.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    return mock_service

@pytest.fixture
def mock_tsmixer_service() -> TSMixerService:
    """Mock TSMixer service for testing."""
    mock_service = MagicMock(spec=TSMixerService)
    
    # Mock analyze_temporal_sequence method
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
    """Mock SHAP explainer for testing."""
    mock_explainer = MagicMock(spec=SHAPExplainer)
    
    # Mock explain_contradiction method
    mock_explainer.explain_contradiction.return_value = {
        "explanation": "The claims contradict each other due to the presence of negation.",
        "influential_words": ["reduces", "does not reduce"],
        "visualization_path": "path/to/visualization.html"
    }
    
    return mock_explainer

@pytest.fixture
def bias_assessment_service() -> BiasAssessmentService:
    """BiasAssessmentService instance for testing."""
    return BiasAssessmentService()

@pytest.fixture
def prisma_screening_service(mock_biomedlm_service) -> PRISMAScreeningService:
    """PRISMAScreeningService instance for testing."""
    service = PRISMAScreeningService(biomedlm_service=mock_biomedlm_service)
    
    # Set default criteria
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
) -> EnhancedContradictionService:
    """EnhancedContradictionService instance for testing."""
    return EnhancedContradictionService(
        biomedlm_service=mock_biomedlm_service,
        temporal_service=TemporalService(tsmixer_service=mock_tsmixer_service),
        shap_explainer=mock_shap_explainer
    )

@pytest.fixture
def temporal_service(mock_tsmixer_service) -> TemporalService:
    """TemporalService instance for testing."""
    return TemporalService(tsmixer_service=mock_tsmixer_service)

@pytest.fixture(scope="function", autouse=True)
async def clear_cache():
    """Clear cache before and after each test."""
    await cache_manager.clear()
    yield
    await cache_manager.clear()

# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
