"""
Screening router for the Medical Research Synthesizer API.

This module provides endpoints for PRISMA-guided screening and bias assessment.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from asf.medical.api.dependencies import (
    get_search_service, get_prisma_screening_service, get_bias_assessment_service,
    get_current_active_user
)
from asf.medical.ml.services.prisma_screening_service import (
    PRISMAScreeningService, ScreeningStage, ScreeningDecision
)
from asf.medical.ml.services.bias_assessment_service import (
    BiasAssessmentService, BiasRisk, BiasDomain
)
from asf.medical.services.search_service import SearchService
from asf.medical.storage.models import User

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/screening", tags=["screening"])

# Define request/response models
class ScreeningCriteriaItem(BaseModel):
    """Screening criteria item."""
    text: str = Field(..., description="Criterion text")
    description: Optional[str] = Field(None, description="Criterion description")

class ScreeningCriteria(BaseModel):
    """Screening criteria."""
    include: List[str] = Field(default_factory=list, description="Inclusion criteria")
    exclude: List[str] = Field(default_factory=list, description="Exclusion criteria")

class ScreeningRequest(BaseModel):
    """PRISMA screening request."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(20, description="Maximum number of results to screen")
    stage: ScreeningStage = Field(ScreeningStage.SCREENING, description="Screening stage")
    criteria: Optional[ScreeningCriteria] = Field(None, description="Custom screening criteria")

class BiasAssessmentRequest(BaseModel):
    """Bias assessment request."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(20, description="Maximum number of results to assess")
    domains: Optional[List[BiasDomain]] = Field(None, description="Bias domains to assess")

class ScreeningResult(BaseModel):
    """PRISMA screening result."""
    query: str = Field(..., description="Search query")
    stage: ScreeningStage = Field(..., description="Screening stage")
    total_articles: int = Field(..., description="Total number of articles")
    included: int = Field(..., description="Number of included articles")
    excluded: int = Field(..., description="Number of excluded articles")
    uncertain: int = Field(..., description="Number of uncertain articles")
    results: List[Dict[str, Any]] = Field(..., description="Screening results")
    flow_data: Dict[str, Any] = Field(..., description="PRISMA flow data")

class BiasAssessmentResult(BaseModel):
    """Bias assessment result."""
    query: str = Field(..., description="Search query")
    total_articles: int = Field(..., description="Total number of articles")
    low_risk: int = Field(..., description="Number of articles with low risk of bias")
    moderate_risk: int = Field(..., description="Number of articles with moderate risk of bias")
    high_risk: int = Field(..., description="Number of articles with high risk of bias")
    unclear_risk: int = Field(..., description="Number of articles with unclear risk of bias")
    results: List[Dict[str, Any]] = Field(..., description="Bias assessment results")

@router.post("/prisma", response_model=ScreeningResult)
async def screen_articles(
    request: ScreeningRequest,
    search_service: SearchService = Depends(get_search_service),
    screening_service: PRISMAScreeningService = Depends(get_prisma_screening_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Screen articles according to PRISMA guidelines.
    
    This endpoint searches for articles and screens them according to PRISMA guidelines,
    returning the screening results and PRISMA flow data.
    """
    try:
        logger.info(f"Screening articles for query: {request.query}")
        
        # Search for articles
        search_result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            user_id=current_user.id
        )
        
        articles = search_result.get("results", [])
        
        if not articles:
            logger.warning(f"No articles found for query: {request.query}")
            return {
                "query": request.query,
                "stage": request.stage,
                "total_articles": 0,
                "included": 0,
                "excluded": 0,
                "uncertain": 0,
                "results": [],
                "flow_data": screening_service.get_flow_data()
            }
        
        # Set custom criteria if provided
        if request.criteria:
            screening_service.set_criteria(
                stage=request.stage,
                include_criteria=request.criteria.include,
                exclude_criteria=request.criteria.exclude
            )
        
        # Screen articles
        screening_results = await screening_service.screen_articles(
            articles=articles,
            stage=request.stage
        )
        
        # Count results by decision
        included = sum(1 for r in screening_results if r["decision"] == ScreeningDecision.INCLUDE)
        excluded = sum(1 for r in screening_results if r["decision"] == ScreeningDecision.EXCLUDE)
        uncertain = sum(1 for r in screening_results if r["decision"] == ScreeningDecision.UNCERTAIN)
        
        # Get PRISMA flow data
        flow_data = screening_service.get_flow_data()
        
        logger.info(f"Screening completed: {included} included, {excluded} excluded, {uncertain} uncertain")
        
        return {
            "query": request.query,
            "stage": request.stage,
            "total_articles": len(articles),
            "included": included,
            "excluded": excluded,
            "uncertain": uncertain,
            "results": screening_results,
            "flow_data": flow_data
        }
    except ValueError as e:
        logger.warning(f"Validation error in screening: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error screening articles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to screen articles: {str(e)}"
        )

@router.post("/bias-assessment", response_model=BiasAssessmentResult)
async def assess_bias(
    request: BiasAssessmentRequest,
    search_service: SearchService = Depends(get_search_service),
    bias_service: BiasAssessmentService = Depends(get_bias_assessment_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Assess risk of bias in articles.
    
    This endpoint searches for articles and assesses the risk of bias in each article,
    returning the assessment results.
    """
    try:
        logger.info(f"Assessing bias for query: {request.query}")
        
        # Search for articles
        search_result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            user_id=current_user.id
        )
        
        articles = search_result.get("results", [])
        
        if not articles:
            logger.warning(f"No articles found for query: {request.query}")
            return {
                "query": request.query,
                "total_articles": 0,
                "low_risk": 0,
                "moderate_risk": 0,
                "high_risk": 0,
                "unclear_risk": 0,
                "results": []
            }
        
        # Assess bias
        assessment_results = await bias_service.assess_studies(articles)
        
        # Count results by risk level
        low_risk = sum(1 for r in assessment_results 
                      if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.LOW)
        moderate_risk = sum(1 for r in assessment_results 
                           if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.MODERATE)
        high_risk = sum(1 for r in assessment_results 
                       if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.HIGH)
        unclear_risk = sum(1 for r in assessment_results 
                          if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.UNCLEAR)
        
        logger.info(f"Bias assessment completed: {low_risk} low risk, {moderate_risk} moderate risk, {high_risk} high risk, {unclear_risk} unclear risk")
        
        return {
            "query": request.query,
            "total_articles": len(articles),
            "low_risk": low_risk,
            "moderate_risk": moderate_risk,
            "high_risk": high_risk,
            "unclear_risk": unclear_risk,
            "results": assessment_results
        }
    except ValueError as e:
        logger.warning(f"Validation error in bias assessment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error assessing bias: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess bias: {str(e)}"
        )

@router.get("/flow-diagram")
async def get_flow_diagram(
    screening_service: PRISMAScreeningService = Depends(get_prisma_screening_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get PRISMA flow diagram data.
    
    This endpoint returns the data needed to generate a PRISMA flow diagram.
    """
    try:
        logger.info("Getting PRISMA flow diagram data")
        
        # Generate flow diagram data
        diagram_data = screening_service.generate_flow_diagram()
        
        return diagram_data
    except Exception as e:
        logger.error(f"Error getting flow diagram data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get flow diagram data: {str(e)}"
        )
