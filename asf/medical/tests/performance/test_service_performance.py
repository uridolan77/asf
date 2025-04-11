"""
Performance tests for the Medical Research Synthesizer services.
This module provides performance tests for the services.
"""
import pytest
import logging
import asyncio
from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService
logger = logging.getLogger(__name__)
@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.async_test
class TestServicePerformance:
    """Performance tests for services."""
    @pytest.mark.asyncio
    async def test_bias_assessment_performance(self, bias_assessment_service: BiasAssessmentService, sample_study_text: str):
        """Test performance of bias assessment.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description