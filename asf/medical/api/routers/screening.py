"""Screening router for the Medical Research Synthesizer API.

This module provides endpoints for PRISMA-guided screening and bias assessment.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from asf.medical.api.models.base import APIResponse, ErrorResponse
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
from asf.medical.storage.models import MedicalUser
from asf.medical.core.observability import async_timed, log_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/screening", tags=["screening"])

class ScreeningCriteriaItem(BaseModel):
    """Screening criteria item.

    This model represents a single screening criterion item with text and optional description.
    """
    text: str = Field(..., description="Criterion text")
    description: Optional[str] = Field(None, description="Criterion description")

class ScreeningCriteria(BaseModel):
    """Screening criteria.

    This model represents a set of inclusion and exclusion criteria for screening.
    """
    include: List[str] = Field(default_factory=list, description="Inclusion criteria")
    exclude: List[str] = Field(default_factory=list, description="Exclusion criteria")

class ScreeningRequest(BaseModel):
    """PRISMA screening request.

    This model represents a request for PRISMA-guided screening of articles.
    """
    query: str = Field(..., description="Search query")
    max_results: int = Field(20, description="Maximum number of results to screen")
    stage: ScreeningStage = Field(ScreeningStage.SCREENING, description="Screening stage")
    criteria: Optional[ScreeningCriteria] = Field(None, description="Custom screening criteria")

class BiasAssessmentRequest(BaseModel):
    """Bias assessment request.

    This model represents a request for bias assessment of articles.
    """
    query: str = Field(..., description="Search query")
    max_results: int = Field(20, description="Maximum number of results to assess")
    domains: Optional[List[BiasDomain]] = Field(None, description="Bias domains to assess")

@router.post("/prisma", response_model=APIResponse[Dict[str, Any]])
@async_timed("screen_articles_endpoint")
async def screen_articles(
    request: ScreeningRequest,
    search_service: SearchService = Depends(get_search_service),
    screening_service: PRISMAScreeningService = Depends(get_prisma_screening_service),
    current_user: MedicalUser = Depends(get_current_active_user)
):
    """Screen articles using PRISMA guidelines.

    This endpoint screens articles based on PRISMA guidelines. It first searches for articles
    using the provided query, then applies screening criteria to determine which articles
    should be included or excluded.

    Args:
        request: The screening request containing query and criteria
        search_service: Service for searching articles
        screening_service: Service for screening articles
        current_user: The authenticated user making the request

    Returns:
        APIResponse containing screening results

    Raises:
        HTTPException: If an error occurs during screening
    """
    try:
        logger.info(f"Screening articles for query: {request.query}")

        search_result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            user_id=current_user.id
        )

        articles = search_result.get("results", [])

        if not articles:
            logger.warning(f"No articles found for query: {request.query}")
            return APIResponse(
                success=True,
                message="No articles found for the given query",
                data={
                    "query": request.query,
                    "stage": request.stage,
                    "total_articles": 0,
                    "included": 0,
                    "excluded": 0,
                    "uncertain": 0,
                    "results": [],
                    "flow_data": screening_service.get_flow_data()
                },
                meta={
                    "query": request.query,
                    "max_results": request.max_results,
                    "stage": request.stage
                }
            )

        if request.criteria:
            screening_service.set_criteria(
                stage=request.stage,
                include_criteria=request.criteria.include,
                exclude_criteria=request.criteria.exclude
            )

        screening_results = await screening_service.screen_articles(
            articles=articles,
            stage=request.stage
        )

        included = sum(1 for r in screening_results if r["decision"] == ScreeningDecision.INCLUDE)
        excluded = sum(1 for r in screening_results if r["decision"] == ScreeningDecision.EXCLUDE)
        uncertain = sum(1 for r in screening_results if r["decision"] == ScreeningDecision.UNCERTAIN)

        flow_data = screening_service.get_flow_data()

        logger.info(f"Screening completed: {included} included, {excluded} excluded, {uncertain} uncertain")

        return APIResponse(
            success=True,
            message="Screening completed successfully",
            data={
                "query": request.query,
                "stage": request.stage,
                "total_articles": len(articles),
                "included": included,
                "excluded": excluded,
                "uncertain": uncertain,
                "results": screening_results,
                "flow_data": flow_data
            },
            meta={
                "query": request.query,
                "max_results": request.max_results,
                "stage": request.stage,
                "criteria": request.criteria.model_dump() if request.criteria else None
            }
        )
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in screening: {str(e)}")
        return ErrorResponse(
            message="Invalid screening parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.error(f"Error screening articles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to screen articles: {str(e)}"
        )

@router.post("/bias-assessment", response_model=APIResponse[Dict[str, Any]])
@async_timed("assess_bias_endpoint")
async def assess_bias(
    request: BiasAssessmentRequest,
    search_service: SearchService = Depends(get_search_service),
    bias_service: BiasAssessmentService = Depends(get_bias_assessment_service),
    current_user: MedicalUser = Depends(get_current_active_user)
):
    """Assess bias in articles.

    This endpoint assesses bias in articles based on various domains. It first searches for articles
    using the provided query, then applies bias assessment to determine the risk level for each article.

    Args:
        request: The bias assessment request containing query and domains
        search_service: Service for searching articles
        bias_service: Service for assessing bias
        current_user: The authenticated user making the request

    Returns:
        APIResponse containing bias assessment results

    Raises:
        HTTPException: If an error occurs during bias assessment
    """
    try:
        logger.info(f"Assessing bias for query: {request.query}")

        search_result = await search_service.search(
            query=request.query,
            max_results=request.max_results,
            user_id=current_user.id
        )

        articles = search_result.get("results", [])

        if not articles:
            logger.warning(f"No articles found for query: {request.query}")
            return APIResponse(
                success=True,
                message="No articles found for the given query",
                data={
                    "query": request.query,
                    "total_articles": 0,
                    "low_risk": 0,
                    "moderate_risk": 0,
                    "high_risk": 0,
                    "unclear_risk": 0,
                    "results": []
                },
                meta={
                    "query": request.query,
                    "max_results": request.max_results,
                    "domains": [domain.value for domain in request.domains] if request.domains else None
                }
            )

        assessment_results = await bias_service.assess_studies(articles)

        low_risk = sum(1 for r in assessment_results
                      if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.LOW)
        moderate_risk = sum(1 for r in assessment_results
                           if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.MODERATE)
        high_risk = sum(1 for r in assessment_results
                       if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.HIGH)
        unclear_risk = sum(1 for r in assessment_results
                          if r["assessment"][BiasDomain.OVERALL]["risk"] == BiasRisk.UNCLEAR)

        logger.info(f"Bias assessment completed: {low_risk} low risk, {moderate_risk} moderate risk, {high_risk} high risk, {unclear_risk} unclear risk")

        return APIResponse(
            success=True,
            message="Bias assessment completed successfully",
            data={
                "query": request.query,
                "total_articles": len(articles),
                "low_risk": low_risk,
                "moderate_risk": moderate_risk,
                "high_risk": high_risk,
                "unclear_risk": unclear_risk,
                "results": assessment_results
            },
            meta={
                "query": request.query,
                "max_results": request.max_results,
                "domains": [domain.value for domain in request.domains] if request.domains else None
            }
        )
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in bias assessment: {str(e)}")
        return ErrorResponse(
            message="Invalid bias assessment parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"query": request.query, "user_id": current_user.id})
        logger.error(f"Error assessing bias: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to assess bias: {str(e)}"
        )

@router.get("/flow-diagram", response_model=APIResponse[Dict[str, Any]])
@async_timed("get_flow_diagram_endpoint")
async def get_flow_diagram(
    screening_service: PRISMAScreeningService = Depends(get_prisma_screening_service),
    current_user: MedicalUser = Depends(get_current_active_user)
):
    """Get PRISMA flow diagram data.

    This endpoint retrieves data for generating a PRISMA flow diagram, which shows the flow of
    articles through the screening process.

    Args:
        screening_service: Service for screening articles
        current_user: The authenticated user making the request

    Returns:
        APIResponse containing flow diagram data

    Raises:
        HTTPException: If an error occurs while retrieving flow diagram data
    """
    try:
        logger.info("Getting PRISMA flow diagram data")

        diagram_data = screening_service.generate_flow_diagram()

        return APIResponse(
            success=True,
            message="Flow diagram data retrieved successfully",
            data=diagram_data,
            meta={
                "user_id": current_user.id
            }
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error getting flow diagram data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get flow diagram data: {str(e)}"
        )
