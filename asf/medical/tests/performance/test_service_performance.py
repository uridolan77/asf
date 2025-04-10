"""
Performance tests for the Medical Research Synthesizer services.

This module provides performance tests for the services.
"""

import pytest
import logging
import time
import asyncio
from typing import Dict, Any, List

from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService
from asf.medical.ml.services.prisma_screening_service import PRISMAScreeningService, ScreeningStage
from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService

# Configure logging
logger = logging.getLogger(__name__)

@pytest.mark.performance
@pytest.mark.slow
@pytest.mark.async_test
class TestServicePerformance:
    """Performance tests for services."""

    @pytest.mark.asyncio
    async def test_bias_assessment_performance(self, bias_assessment_service: BiasAssessmentService, sample_study_text: str):
        """Test performance of bias assessment."""
        # Warm-up
        await bias_assessment_service.assess_study(sample_study_text)
        
        # Measure performance
        start_time = time.time()
        iterations = 10
        
        for _ in range(iterations):
            await bias_assessment_service.assess_study(sample_study_text)
        
        elapsed_time = time.time() - start_time
        average_time = elapsed_time / iterations
        
        logger.info(f"Bias assessment performance: {average_time:.4f} seconds per assessment")
        
        # Assertion (adjust threshold as needed)
        assert average_time < 1.0, f"Bias assessment is too slow: {average_time:.4f} seconds per assessment"
    
    @pytest.mark.asyncio
    async def test_prisma_screening_performance(self, prisma_screening_service: PRISMAScreeningService, sample_articles: List[Dict[str, Any]]):
        """Test performance of PRISMA screening."""
        # Warm-up
        await prisma_screening_service.screen_articles(sample_articles, ScreeningStage.SCREENING)
        
        # Measure performance
        start_time = time.time()
        iterations = 10
        
        for _ in range(iterations):
            await prisma_screening_service.screen_articles(sample_articles, ScreeningStage.SCREENING)
        
        elapsed_time = time.time() - start_time
        average_time = elapsed_time / iterations
        average_time_per_article = average_time / len(sample_articles)
        
        logger.info(f"PRISMA screening performance: {average_time:.4f} seconds per batch, {average_time_per_article:.4f} seconds per article")
        
        # Assertion (adjust threshold as needed)
        assert average_time_per_article < 0.5, f"PRISMA screening is too slow: {average_time_per_article:.4f} seconds per article"
    
    @pytest.mark.asyncio
    async def test_contradiction_detection_performance(self, enhanced_contradiction_service: EnhancedContradictionService, sample_claims: List[Dict[str, Any]]):
        """Test performance of contradiction detection."""
        # Warm-up
        await enhanced_contradiction_service.detect_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"]
        )
        
        # Measure performance
        start_time = time.time()
        iterations = 10
        
        for _ in range(iterations):
            await enhanced_contradiction_service.detect_contradiction(
                claim1=sample_claims[0]["claim1"],
                claim2=sample_claims[0]["claim2"]
            )
        
        elapsed_time = time.time() - start_time
        average_time = elapsed_time / iterations
        
        logger.info(f"Contradiction detection performance: {average_time:.4f} seconds per detection")
        
        # Assertion (adjust threshold as needed)
        assert average_time < 1.0, f"Contradiction detection is too slow: {average_time:.4f} seconds per detection"
    
    @pytest.mark.asyncio
    async def test_contradiction_analysis_performance(self, enhanced_contradiction_service: EnhancedContradictionService, sample_articles: List[Dict[str, Any]]):
        """Test performance of contradiction analysis."""
        # Warm-up
        await enhanced_contradiction_service.detect_contradictions_in_articles(
            articles=sample_articles,
            threshold=0.7
        )
        
        # Measure performance
        start_time = time.time()
        iterations = 5
        
        for _ in range(iterations):
            await enhanced_contradiction_service.detect_contradictions_in_articles(
                articles=sample_articles,
                threshold=0.7
            )
        
        elapsed_time = time.time() - start_time
        average_time = elapsed_time / iterations
        
        logger.info(f"Contradiction analysis performance: {average_time:.4f} seconds per analysis")
        
        # Assertion (adjust threshold as needed)
        assert average_time < 5.0, f"Contradiction analysis is too slow: {average_time:.4f} seconds per analysis"
