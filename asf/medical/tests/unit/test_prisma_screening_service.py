"""
Unit tests for the PRISMAScreeningService.

This module provides unit tests for the PRISMAScreeningService class.
"""

import pytest
import logging
from typing import Dict, Any, List
from unittest.mock import MagicMock

from asf.medical.ml.services.prisma_screening_service import (
    PRISMAScreeningService, ScreeningStage, ScreeningDecision
)
from asf.medical.ml.models.biomedlm import BioMedLMService

# Configure logging
logger = logging.getLogger(__name__)

@pytest.mark.unit
@pytest.mark.service
@pytest.mark.async_test
class TestPRISMAScreeningService:
    """Test cases for PRISMAScreeningService."""

    @pytest.mark.asyncio
    async def test_screen_article_identification(self, prisma_screening_service: PRISMAScreeningService, sample_articles: List[Dict[str, Any]]):
        """Test screening an article at the identification stage."""
        # Screen article
        result = await prisma_screening_service.screen_article(
            article=sample_articles[0],
            stage=ScreeningStage.IDENTIFICATION
        )
        
        # Assertions
        assert result["article_id"] == sample_articles[0]["pmid"]
        assert result["title"] == sample_articles[0]["title"]
        assert result["stage"] == ScreeningStage.IDENTIFICATION
        assert result["decision"] in [ScreeningDecision.INCLUDE, ScreeningDecision.EXCLUDE, ScreeningDecision.UNCERTAIN]
        assert isinstance(result["confidence"], float)
        assert isinstance(result["matched_include_criteria"], list)
        assert isinstance(result["matched_exclude_criteria"], list)
        assert isinstance(result["notes"], str)
    
    @pytest.mark.asyncio
    async def test_screen_article_screening(self, prisma_screening_service: PRISMAScreeningService, sample_articles: List[Dict[str, Any]]):
        """Test screening an article at the screening stage."""
        # Screen article
        result = await prisma_screening_service.screen_article(
            article=sample_articles[0],
            stage=ScreeningStage.SCREENING
        )
        
        # Assertions
        assert result["article_id"] == sample_articles[0]["pmid"]
        assert result["title"] == sample_articles[0]["title"]
        assert result["stage"] == ScreeningStage.SCREENING
        assert result["decision"] in [ScreeningDecision.INCLUDE, ScreeningDecision.EXCLUDE, ScreeningDecision.UNCERTAIN]
        assert isinstance(result["confidence"], float)
        assert isinstance(result["matched_include_criteria"], list)
        assert isinstance(result["matched_exclude_criteria"], list)
        assert isinstance(result["notes"], str)
    
    @pytest.mark.asyncio
    async def test_screen_article_eligibility(self, prisma_screening_service: PRISMAScreeningService, sample_articles: List[Dict[str, Any]]):
        """Test screening an article at the eligibility stage."""
        # Screen article
        result = await prisma_screening_service.screen_article(
            article=sample_articles[0],
            stage=ScreeningStage.ELIGIBILITY
        )
        
        # Assertions
        assert result["article_id"] == sample_articles[0]["pmid"]
        assert result["title"] == sample_articles[0]["title"]
        assert result["stage"] == ScreeningStage.ELIGIBILITY
        assert result["decision"] in [ScreeningDecision.INCLUDE, ScreeningDecision.EXCLUDE, ScreeningDecision.UNCERTAIN]
        assert isinstance(result["confidence"], float)
        assert isinstance(result["matched_include_criteria"], list)
        assert isinstance(result["matched_exclude_criteria"], list)
        assert isinstance(result["notes"], str)
    
    @pytest.mark.asyncio
    async def test_screen_article_custom_criteria(self, prisma_screening_service: PRISMAScreeningService, sample_articles: List[Dict[str, Any]]):
        """Test screening an article with custom criteria."""
        # Custom criteria
        custom_criteria = {
            "include": ["randomized controlled trial"],
            "exclude": ["observational study"]
        }
        
        # Screen article
        result = await prisma_screening_service.screen_article(
            article=sample_articles[0],
            stage=ScreeningStage.SCREENING,
            custom_criteria=custom_criteria
        )
        
        # Assertions
        assert result["article_id"] == sample_articles[0]["pmid"]
        assert result["title"] == sample_articles[0]["title"]
        assert result["stage"] == ScreeningStage.SCREENING
        assert result["decision"] in [ScreeningDecision.INCLUDE, ScreeningDecision.EXCLUDE, ScreeningDecision.UNCERTAIN]
        assert isinstance(result["confidence"], float)
        assert isinstance(result["matched_include_criteria"], list)
        assert isinstance(result["matched_exclude_criteria"], list)
        assert isinstance(result["notes"], str)
    
    @pytest.mark.asyncio
    async def test_screen_articles(self, prisma_screening_service: PRISMAScreeningService, sample_articles: List[Dict[str, Any]]):
        """Test screening multiple articles."""
        # Screen articles
        results = await prisma_screening_service.screen_articles(
            articles=sample_articles,
            stage=ScreeningStage.SCREENING
        )
        
        # Assertions
        assert len(results) == len(sample_articles)
        for i, result in enumerate(results):
            assert result["article_id"] == sample_articles[i]["pmid"]
            assert result["title"] == sample_articles[i]["title"]
            assert result["stage"] == ScreeningStage.SCREENING
            assert result["decision"] in [ScreeningDecision.INCLUDE, ScreeningDecision.EXCLUDE, ScreeningDecision.UNCERTAIN]
            assert isinstance(result["confidence"], float)
            assert isinstance(result["matched_include_criteria"], list)
            assert isinstance(result["matched_exclude_criteria"], list)
            assert isinstance(result["notes"], str)
    
    @pytest.mark.asyncio
    async def test_check_criterion_with_biomedlm(self, mock_biomedlm_service: BioMedLMService):
        """Test checking a criterion with BioMedLM."""
        # Create service with mock BioMedLM
        service = PRISMAScreeningService(biomedlm_service=mock_biomedlm_service)
        
        # Check criterion
        result = await service._check_criterion(
            text="This is a randomized controlled trial.",
            criterion="randomized controlled trial"
        )
        
        # Assertions
        assert result is True
        mock_biomedlm_service.calculate_similarity.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_criterion_without_biomedlm(self):
        """Test checking a criterion without BioMedLM."""
        # Create service without BioMedLM
        service = PRISMAScreeningService(biomedlm_service=None)
        
        # Check criterion that matches
        result1 = await service._check_criterion(
            text="This is a randomized controlled trial.",
            criterion="randomized controlled trial"
        )
        
        # Check criterion that doesn't match
        result2 = await service._check_criterion(
            text="This is an observational study.",
            criterion="randomized controlled trial"
        )
        
        # Assertions
        assert result1 is True
        assert result2 is False
    
    def test_set_criteria(self, prisma_screening_service: PRISMAScreeningService):
        """Test setting criteria."""
        # Set criteria
        prisma_screening_service.set_criteria(
            stage=ScreeningStage.IDENTIFICATION,
            include_criteria=["test include"],
            exclude_criteria=["test exclude"]
        )
        
        # Assertions
        assert prisma_screening_service.criteria[ScreeningStage.IDENTIFICATION]["include"] == ["test include"]
        assert prisma_screening_service.criteria[ScreeningStage.IDENTIFICATION]["exclude"] == ["test exclude"]
    
    def test_update_flow_data(self, prisma_screening_service: PRISMAScreeningService):
        """Test updating flow data."""
        # Initial flow data
        initial_identified = prisma_screening_service.flow_data["identification"]["records_identified"]
        initial_removed = prisma_screening_service.flow_data["identification"]["records_removed_before_screening"]
        
        # Update flow data
        prisma_screening_service._update_flow_data(
            stage=ScreeningStage.IDENTIFICATION,
            decision=ScreeningDecision.EXCLUDE
        )
        
        # Assertions
        assert prisma_screening_service.flow_data["identification"]["records_identified"] == initial_identified + 1
        assert prisma_screening_service.flow_data["identification"]["records_removed_before_screening"] == initial_removed + 1
    
    def test_get_flow_data(self, prisma_screening_service: PRISMAScreeningService):
        """Test getting flow data."""
        # Get flow data
        flow_data = prisma_screening_service.get_flow_data()
        
        # Assertions
        assert "identification" in flow_data
        assert "screening" in flow_data
        assert "eligibility" in flow_data
        assert "included" in flow_data
        assert "records_identified" in flow_data["identification"]
        assert "records_removed_before_screening" in flow_data["identification"]
        assert "records_screened" in flow_data["screening"]
        assert "records_excluded" in flow_data["screening"]
        assert "full_text_assessed" in flow_data["eligibility"]
        assert "full_text_excluded" in flow_data["eligibility"]
        assert "exclusion_reasons" in flow_data["eligibility"]
        assert "studies_included" in flow_data["included"]
    
    def test_generate_flow_diagram(self, prisma_screening_service: PRISMAScreeningService):
        """Test generating flow diagram."""
        # Generate flow diagram
        diagram_data = prisma_screening_service.generate_flow_diagram()
        
        # Assertions
        assert "identification" in diagram_data
        assert "screening" in diagram_data
        assert "eligibility" in diagram_data
        assert "included" in diagram_data
        assert "records_identified" in diagram_data["identification"]
        assert "records_removed" in diagram_data["identification"]
        assert "records_remaining" in diagram_data["identification"]
        assert "records_screened" in diagram_data["screening"]
        assert "records_excluded" in diagram_data["screening"]
        assert "records_remaining" in diagram_data["screening"]
        assert "full_text_assessed" in diagram_data["eligibility"]
        assert "full_text_excluded" in diagram_data["eligibility"]
        assert "exclusion_reasons" in diagram_data["eligibility"]
        assert "studies_included" in diagram_data["included"]
