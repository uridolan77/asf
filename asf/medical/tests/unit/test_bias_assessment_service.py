"""
Unit tests for the BiasAssessmentService.

This module provides unit tests for the BiasAssessmentService class.
"""

import pytest
import logging
from typing import Dict, Any

from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService, BiasRisk, BiasDomain

# Configure logging
logger = logging.getLogger(__name__)

@pytest.mark.unit
@pytest.mark.service
@pytest.mark.async_test
class TestBiasAssessmentService:
    """Test cases for BiasAssessmentService."""

    @pytest.mark.asyncio
    async def test_bias_assessment(self, bias_assessment_service: BiasAssessmentService, sample_study_text: str):
        """Test bias assessment with a sample text."""
        # Sample text with randomization but no blinding
        result = await bias_assessment_service.assess_study(sample_study_text)
        
        # Assertions
        assert result[BiasDomain.RANDOMIZATION]["risk"] == BiasRisk.LOW
        assert result[BiasDomain.BLINDING]["risk"] == BiasRisk.HIGH
        assert any("randomly assigned" in evidence["context"] for evidence in result[BiasDomain.RANDOMIZATION]["evidence"])
        assert "overall" in result
        assert result["overall"]["risk"] in [BiasRisk.LOW, BiasRisk.MODERATE, BiasRisk.HIGH]
    
    @pytest.mark.asyncio
    async def test_assess_randomization(self, bias_assessment_service: BiasAssessmentService):
        """Test assessment of randomization domain."""
        # Text with good randomization
        good_text = "This was a randomized controlled trial with computer-generated randomization."
        good_result = await bias_assessment_service._assess_domain(BiasDomain.RANDOMIZATION, None, good_text)
        
        # Text with poor randomization
        poor_text = "This was a non-randomized study with alternate allocation."
        poor_result = await bias_assessment_service._assess_domain(BiasDomain.RANDOMIZATION, None, poor_text)
        
        # Text with no mention of randomization
        unclear_text = "This was a clinical trial."
        unclear_result = await bias_assessment_service._assess_domain(BiasDomain.RANDOMIZATION, None, unclear_text)
        
        # Assertions
        assert good_result["risk"] == BiasRisk.LOW
        assert poor_result["risk"] == BiasRisk.HIGH
        assert unclear_result["risk"] == BiasRisk.UNCLEAR
    
    @pytest.mark.asyncio
    async def test_assess_blinding(self, bias_assessment_service: BiasAssessmentService):
        """Test assessment of blinding domain."""
        # Text with good blinding
        good_text = "This was a double-blind study where neither participants nor researchers knew the treatment allocation."
        good_result = await bias_assessment_service._assess_domain(BiasDomain.BLINDING, None, good_text)
        
        # Text with poor blinding
        poor_text = "This was an open-label study with no blinding."
        poor_result = await bias_assessment_service._assess_domain(BiasDomain.BLINDING, None, poor_text)
        
        # Text with no mention of blinding
        unclear_text = "This was a clinical trial."
        unclear_result = await bias_assessment_service._assess_domain(BiasDomain.BLINDING, None, unclear_text)
        
        # Assertions
        assert good_result["risk"] == BiasRisk.LOW
        assert poor_result["risk"] == BiasRisk.HIGH
        assert unclear_result["risk"] == BiasRisk.UNCLEAR
    
    @pytest.mark.asyncio
    async def test_assess_allocation_concealment(self, bias_assessment_service: BiasAssessmentService):
        """Test assessment of allocation concealment domain."""
        # Text with good allocation concealment
        good_text = "Allocation was concealed using sealed envelopes."
        good_result = await bias_assessment_service._assess_domain(BiasDomain.ALLOCATION_CONCEALMENT, None, good_text)
        
        # Text with poor allocation concealment
        poor_text = "Allocation was not concealed."
        poor_result = await bias_assessment_service._assess_domain(BiasDomain.ALLOCATION_CONCEALMENT, None, poor_text)
        
        # Text with no mention of allocation concealment
        unclear_text = "This was a clinical trial."
        unclear_result = await bias_assessment_service._assess_domain(BiasDomain.ALLOCATION_CONCEALMENT, None, unclear_text)
        
        # Assertions
        assert good_result["risk"] == BiasRisk.LOW
        assert poor_result["risk"] == BiasRisk.HIGH
        assert unclear_result["risk"] == BiasRisk.UNCLEAR
    
    @pytest.mark.asyncio
    async def test_assess_sample_size(self, bias_assessment_service: BiasAssessmentService):
        """Test assessment of sample size domain."""
        # Text with good sample size
        good_text = "Sample size calculation was performed before the study."
        good_result = await bias_assessment_service._assess_domain(BiasDomain.SAMPLE_SIZE, None, good_text)
        
        # Text with poor sample size
        poor_text = "This was a small pilot study with an underpowered sample size."
        poor_result = await bias_assessment_service._assess_domain(BiasDomain.SAMPLE_SIZE, None, poor_text)
        
        # Text with no mention of sample size
        unclear_text = "This was a clinical trial."
        unclear_result = await bias_assessment_service._assess_domain(BiasDomain.SAMPLE_SIZE, None, unclear_text)
        
        # Assertions
        assert good_result["risk"] == BiasRisk.LOW
        assert poor_result["risk"] == BiasRisk.HIGH
        assert unclear_result["risk"] == BiasRisk.UNCLEAR
    
    @pytest.mark.asyncio
    async def test_assess_attrition(self, bias_assessment_service: BiasAssessmentService):
        """Test assessment of attrition domain."""
        # Text with good attrition handling
        good_text = "There was no loss to follow-up and intention-to-treat analysis was performed."
        good_result = await bias_assessment_service._assess_domain(BiasDomain.ATTRITION, None, good_text)
        
        # Text with poor attrition handling
        poor_text = "There was high dropout rate and per-protocol analysis was used."
        poor_result = await bias_assessment_service._assess_domain(BiasDomain.ATTRITION, None, poor_text)
        
        # Text with no mention of attrition
        unclear_text = "This was a clinical trial."
        unclear_result = await bias_assessment_service._assess_domain(BiasDomain.ATTRITION, None, unclear_text)
        
        # Assertions
        assert good_result["risk"] == BiasRisk.LOW
        assert poor_result["risk"] == BiasRisk.HIGH
        assert unclear_result["risk"] == BiasRisk.UNCLEAR
    
    @pytest.mark.asyncio
    async def test_assess_selective_reporting(self, bias_assessment_service: BiasAssessmentService):
        """Test assessment of selective reporting domain."""
        # Text with good reporting
        good_text = "The study was pre-registered and all outcomes were reported."
        good_result = await bias_assessment_service._assess_domain(BiasDomain.SELECTIVE_REPORTING, None, good_text)
        
        # Text with poor reporting
        poor_text = "Not all outcomes were reported and there was selective reporting."
        poor_result = await bias_assessment_service._assess_domain(BiasDomain.SELECTIVE_REPORTING, None, poor_text)
        
        # Text with no mention of reporting
        unclear_text = "This was a clinical trial."
        unclear_result = await bias_assessment_service._assess_domain(BiasDomain.SELECTIVE_REPORTING, None, unclear_text)
        
        # Assertions
        assert good_result["risk"] == BiasRisk.LOW
        assert poor_result["risk"] == BiasRisk.HIGH
        assert unclear_result["risk"] == BiasRisk.UNCLEAR
    
    @pytest.mark.asyncio
    async def test_assess_studies(self, bias_assessment_service: BiasAssessmentService, sample_articles):
        """Test assessment of multiple studies."""
        # Assess studies
        results = await bias_assessment_service.assess_studies(sample_articles)
        
        # Assertions
        assert len(results) == len(sample_articles)
        for result in results:
            assert "study_id" in result
            assert "title" in result
            assert "assessment" in result
            assert BiasDomain.OVERALL in result["assessment"]
            assert "risk" in result["assessment"][BiasDomain.OVERALL]
            assert result["assessment"][BiasDomain.OVERALL]["risk"] in [BiasRisk.LOW, BiasRisk.MODERATE, BiasRisk.HIGH, BiasRisk.UNCLEAR]
    
    def test_generate_summary(self, bias_assessment_service: BiasAssessmentService):
        """Test generation of summary."""
        # Create a sample assessment
        assessment = {
            BiasDomain.RANDOMIZATION: {
                "risk": BiasRisk.LOW,
                "positive_score": 1.0,
                "negative_score": 0.0,
                "evidence": [
                    {
                        "text": "randomized",
                        "context": "This was a randomized controlled trial.",
                        "type": "positive",
                        "weight": 1.0
                    }
                ]
            },
            BiasDomain.BLINDING: {
                "risk": BiasRisk.HIGH,
                "positive_score": 0.0,
                "negative_score": 1.0,
                "evidence": [
                    {
                        "text": "not blind",
                        "context": "The study did not use blinding.",
                        "type": "negative",
                        "weight": 1.0
                    }
                ]
            },
            BiasDomain.ALLOCATION_CONCEALMENT: {
                "risk": BiasRisk.LOW,
                "positive_score": 1.0,
                "negative_score": 0.0,
                "evidence": [
                    {
                        "text": "sealed envelope",
                        "context": "Allocation was concealed using sealed envelopes.",
                        "type": "positive",
                        "weight": 0.9
                    }
                ]
            },
            BiasDomain.SAMPLE_SIZE: {
                "risk": BiasRisk.LOW,
                "positive_score": 1.0,
                "negative_score": 0.0,
                "evidence": [
                    {
                        "text": "sample size calculation",
                        "context": "Sample size calculation was performed before the study.",
                        "type": "positive",
                        "weight": 1.0
                    }
                ]
            },
            BiasDomain.ATTRITION: {
                "risk": BiasRisk.MODERATE,
                "positive_score": 0.5,
                "negative_score": 0.0,
                "evidence": [
                    {
                        "text": "dropout",
                        "context": "There was a 15% dropout rate in the treatment group.",
                        "type": "positive",
                        "weight": 0.5
                    }
                ]
            },
            BiasDomain.SELECTIVE_REPORTING: {
                "risk": BiasRisk.LOW,
                "positive_score": 1.0,
                "negative_score": 0.0,
                "evidence": [
                    {
                        "text": "all outcomes reported",
                        "context": "All pre-specified outcomes were reported in the results.",
                        "type": "positive",
                        "weight": 0.9
                    }
                ]
            },
            BiasDomain.OVERALL: {
                "risk": BiasRisk.MODERATE,
                "summary": "1 domains at high risk of bias, 0 domains unclear",
                "high_risk_domains": [BiasDomain.BLINDING],
                "unclear_domains": []
            }
        }
        
        # Generate summary
        summary = bias_assessment_service.generate_summary(assessment)
        
        # Assertions
        assert isinstance(summary, str)
        assert "Overall risk of bias: moderate" in summary.lower()
        assert "randomization: low" in summary.lower()
        assert "blinding: high" in summary.lower()
        assert "allocation concealment: low" in summary.lower()
        assert "sample size: low" in summary.lower()
        assert "attrition: moderate" in summary.lower()
        assert "selective reporting: low" in summary.lower()
        assert "randomized" in summary
        assert "not blind" in summary
        assert "sealed envelope" in summary
        assert "sample size calculation" in summary
        assert "dropout" in summary
        assert "all outcomes reported" in summary
