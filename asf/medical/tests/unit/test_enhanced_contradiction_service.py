"""
Unit tests for the EnhancedContradictionService.

This module provides unit tests for the EnhancedContradictionService class.
"""

import pytest
import logging
from typing import Dict, Any, List
from unittest.mock import MagicMock

from asf.medical.ml.services.enhanced_contradiction_service import (
    EnhancedContradictionService, ContradictionType, ContradictionConfidence
)
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.temporal_service import TemporalService

# Configure logging
logger = logging.getLogger(__name__)

@pytest.mark.unit
@pytest.mark.service
@pytest.mark.async_test
class TestEnhancedContradictionService:
    """Test cases for EnhancedContradictionService."""

    @pytest.mark.asyncio
    async def test_detect_contradiction(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting contradiction between two claims."""
        # Detect contradiction
        result = await enhanced_contradiction_service.detect_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"],
            use_all_methods=True
        )
        
        # Assertions
        assert "claim1" in result
        assert "claim2" in result
        assert "is_contradiction" in result
        assert "contradiction_score" in result
        assert "contradiction_type" in result
        assert "confidence" in result
        assert "methods_used" in result
        assert "explanation" in result
        assert "details" in result
        assert isinstance(result["is_contradiction"], bool)
        assert isinstance(result["contradiction_score"], float)
        assert result["contradiction_type"] in [ct.value for ct in ContradictionType]
        assert result["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(result["methods_used"], list)
        assert isinstance(result["explanation"], str)
        assert isinstance(result["details"], dict)
    
    @pytest.mark.asyncio
    async def test_detect_contradictions_in_articles(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_articles: List[Dict[str, Any]]
    ):
        """Test detecting contradictions in articles."""
        # Detect contradictions
        contradictions = await enhanced_contradiction_service.detect_contradictions_in_articles(
            articles=sample_articles,
            threshold=0.7,
            use_all_methods=True
        )
        
        # Assertions
        assert isinstance(contradictions, list)
        if contradictions:
            contradiction = contradictions[0]
            assert "article1" in contradiction
            assert "article2" in contradiction
            assert "contradiction_score" in contradiction
            assert "contradiction_type" in contradiction
            assert "confidence" in contradiction
            assert "explanation" in contradiction
            assert "id" in contradiction["article1"]
            assert "title" in contradiction["article1"]
            assert "claim" in contradiction["article1"]
            assert "id" in contradiction["article2"]
            assert "title" in contradiction["article2"]
            assert "claim" in contradiction["article2"]
            assert isinstance(contradiction["contradiction_score"], float)
            assert contradiction["contradiction_type"] in [ct.value for ct in ContradictionType]
            assert contradiction["confidence"] in [cc.value for cc in ContradictionConfidence]
            assert isinstance(contradiction["explanation"], str)
    
    @pytest.mark.asyncio
    async def test_detect_direct_contradiction(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting direct contradiction."""
        # Detect direct contradiction
        result = await enhanced_contradiction_service._detect_direct_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"]
        )
        
        # Assertions
        assert "is_contradiction" in result
        assert "score" in result
        assert "confidence" in result
        assert "explanation" in result
        assert isinstance(result["is_contradiction"], bool)
        assert isinstance(result["score"], float)
        assert result["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(result["explanation"], str)
    
    @pytest.mark.asyncio
    async def test_detect_negation_contradiction(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting negation contradiction."""
        # Detect negation contradiction
        result = await enhanced_contradiction_service._detect_negation_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"]
        )
        
        # Assertions
        assert "is_contradiction" in result
        assert "score" in result
        assert "confidence" in result
        assert "explanation" in result
        assert isinstance(result["is_contradiction"], bool)
        assert isinstance(result["score"], float)
        assert result["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(result["explanation"], str)
    
    @pytest.mark.asyncio
    async def test_detect_temporal_contradiction(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting temporal contradiction."""
        # Metadata
        metadata1 = {"publication_date": "2020-01-01"}
        metadata2 = {"publication_date": "2021-01-01"}
        
        # Detect temporal contradiction
        result = await enhanced_contradiction_service._detect_temporal_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"],
            metadata1=metadata1,
            metadata2=metadata2
        )
        
        # Assertions
        assert "is_contradiction" in result
        assert "score" in result
        assert "confidence" in result
        assert "explanation" in result
        assert isinstance(result["is_contradiction"], bool)
        assert isinstance(result["score"], float)
        assert result["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(result["explanation"], str)
    
    @pytest.mark.asyncio
    async def test_detect_methodological_contradiction(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting methodological contradiction."""
        # Metadata
        metadata1 = {
            "study_design": "randomized controlled trial",
            "sample_size": 1000,
            "population": "high-risk patients"
        }
        metadata2 = {
            "study_design": "observational study",
            "sample_size": 200,
            "population": "low-risk patients"
        }
        
        # Detect methodological contradiction
        result = await enhanced_contradiction_service._detect_methodological_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"],
            metadata1=metadata1,
            metadata2=metadata2
        )
        
        # Assertions
        assert "is_contradiction" in result
        assert "score" in result
        assert "confidence" in result
        assert "explanation" in result
        assert isinstance(result["is_contradiction"], bool)
        assert isinstance(result["score"], float)
        assert result["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(result["explanation"], str)
    
    @pytest.mark.asyncio
    async def test_detect_statistical_contradiction(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test detecting statistical contradiction."""
        # Metadata
        metadata1 = {
            "p_value": 0.01,
            "confidence_interval": [0.1, 0.5]
        }
        metadata2 = {
            "p_value": 0.2,
            "confidence_interval": [0.6, 1.0]
        }
        
        # Detect statistical contradiction
        result = await enhanced_contradiction_service._detect_statistical_contradiction(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"],
            metadata1=metadata1,
            metadata2=metadata2
        )
        
        # Assertions
        assert "is_contradiction" in result
        assert "score" in result
        assert "confidence" in result
        assert "explanation" in result
        assert isinstance(result["is_contradiction"], bool)
        assert isinstance(result["score"], float)
        assert result["confidence"] in [cc.value for cc in ContradictionConfidence]
        assert isinstance(result["explanation"], str)
    
    @pytest.mark.asyncio
    async def test_generate_explanation(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_claims: List[Dict[str, Any]]
    ):
        """Test generating explanation."""
        # Generate explanation
        explanation = await enhanced_contradiction_service._generate_explanation(
            claim1=sample_claims[0]["claim1"],
            claim2=sample_claims[0]["claim2"],
            contradiction_type=ContradictionType.DIRECT
        )
        
        # Assertions
        assert isinstance(explanation, str)
        assert explanation
    
    def test_extract_main_claim(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_articles: List[Dict[str, Any]]
    ):
        """Test extracting main claim."""
        # Extract main claim
        claim = enhanced_contradiction_service._extract_main_claim(sample_articles[0])
        
        # Assertions
        assert isinstance(claim, str)
        assert claim
    
    def test_extract_metadata(
        self, 
        enhanced_contradiction_service: EnhancedContradictionService, 
        sample_articles: List[Dict[str, Any]]
    ):
        """Test extracting metadata."""
        # Add metadata to article
        article = sample_articles[0].copy()
        article["publication_date"] = "2020-01-01"
        article["study_design"] = "randomized controlled trial"
        article["sample_size"] = 1000
        article["population"] = "high-risk patients"
        article["p_value"] = 0.01
        article["confidence_interval"] = [0.1, 0.5]
        
        # Extract metadata
        metadata = enhanced_contradiction_service._extract_metadata(article)
        
        # Assertions
        assert isinstance(metadata, dict)
        assert "publication_date" in metadata
        assert "study_design" in metadata
        assert "sample_size" in metadata
        assert "population" in metadata
        assert "p_value" in metadata
        assert "confidence_interval" in metadata
        assert metadata["publication_date"] == "2020-01-01"
        assert metadata["study_design"] == "randomized controlled trial"
        assert metadata["sample_size"] == 1000
        assert metadata["population"] == "high-risk patients"
        assert metadata["p_value"] == 0.01
        assert metadata["confidence_interval"] == [0.1, 0.5]
