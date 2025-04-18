#!/usr/bin/env python3
"""
Clinical Data API Router

Provides API endpoints for integrated clinical data operations:
- Combined terminology and clinical trials search
- Semantic expansion of clinical terms
- Mapping between SNOMED CT and clinical trials
- Advanced clinical data analytics
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...services.clinical_data_service import ClinicalDataService
from ..dependencies import get_clinical_data_service

router = APIRouter(
    prefix="/clinical-data",
    tags=["clinical-data"],
    responses={404: {"description": "Not found"}},
)


class TrialReference(BaseModel):
    """Clinical trial reference model."""
    nct_id: str
    title: str
    status: Optional[str] = None
    phase: Optional[str] = None
    enrollment_count: Optional[int] = None


class ConceptReference(BaseModel):
    """Terminology concept reference model."""
    concept_id: str
    preferred_term: str
    fsn: Optional[str] = None


class IntegratedSearchResult(BaseModel):
    """Response model for integrated search results."""
    term: str
    concepts: List[Dict]
    trials: List[Dict]


class SemanticSearchResult(BaseModel):
    """Response model for semantic search results."""
    original_term: str
    normalized_term: str
    search_terms_used: List[str]
    confidence: float
    trials: List[Dict]


class ConceptTrialsResult(BaseModel):
    """Response model for concept-based trial search."""
    concept: Dict
    trials: List[Dict]


class ConditionMappingResult(BaseModel):
    """Response model for condition mapping."""
    nct_id: str
    study_title: str
    condition_mappings: Dict[str, List[Dict]]


class TrialSemanticContext(BaseModel):
    """Response model for trial semantic context."""
    nct_id: str
    study_title: str
    condition_mappings: Dict[str, List[Dict]]
    intervention_mappings: Dict[str, List[Dict]]
    hierarchical_context: List[Dict]


@router.get("/search", response_model=IntegratedSearchResult)
async def search_concept_and_trials(
    term: str,
    max_trials: int = Query(10, ge=1, le=50),
    service: ClinicalDataService = Depends(get_clinical_data_service)
):
    """
    Search for a medical term and find related SNOMED CT concepts and clinical trials.
    
    This endpoint performs an integrated search that returns both terminology concepts
    and relevant clinical trials for a given medical term.
    
    Args:
        term: The medical term to search for
        max_trials: Maximum number of trials to return (1-50)
        
    Returns:
        SNOMED CT concepts and clinical trials related to the search term
    """
    try:
        results = service.search_concept_and_trials(term, max_trials=max_trials)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing integrated search: {str(e)}")


@router.get("/concept/{concept_id}/trials", response_model=ConceptTrialsResult)
async def get_trials_by_concept(
    concept_id: str,
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    max_trials: int = Query(10, ge=1, le=50),
    service: ClinicalDataService = Depends(get_clinical_data_service)
):
    """
    Find clinical trials related to a specific medical concept.
    
    Args:
        concept_id: The concept identifier (e.g., SNOMED CT concept ID)
        terminology: The terminology system (currently only SNOMED CT supported)
        max_trials: Maximum number of trials to return (1-50)
        
    Returns:
        Clinical trials related to the specified concept
    """
    try:
        results = service.search_by_concept_id(
            concept_id, 
            terminology=terminology, 
            max_trials=max_trials
        )
        
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
            
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving trials by concept: {str(e)}")


@router.get("/trial/{nct_id}/mapping", response_model=ConditionMappingResult)
async def map_trial_conditions(
    nct_id: str,
    service: ClinicalDataService = Depends(get_clinical_data_service)
):
    """
    Map all conditions in a clinical trial to SNOMED CT concepts.
    
    Args:
        nct_id: The ClinicalTrials.gov identifier (NCT number)
        
    Returns:
        Condition mappings to SNOMED CT concepts
    """
    try:
        mappings = service.map_trial_conditions(nct_id)
        return mappings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mapping trial conditions: {str(e)}")


@router.get("/semantic-search", response_model=SemanticSearchResult)
async def find_trials_with_semantic_expansion(
    term: str,
    include_similar: bool = Query(True, description="Include similar concepts in search"),
    max_trials: int = Query(20, ge=1, le=100),
    service: ClinicalDataService = Depends(get_clinical_data_service)
):
    """
    Find clinical trials with semantic expansion of the search term.
    
    This endpoint normalizes the input term using SNOMED CT and expands
    the search to include semantically similar concepts.
    
    Args:
        term: The medical term to search for
        include_similar: Whether to include similar concepts
        max_trials: Maximum number of trials to return (1-100)
        
    Returns:
        Clinical trials related to the term and semantically similar terms
    """
    try:
        results = service.find_trials_with_semantic_expansion(
            term, 
            include_similar=include_similar,
            max_trials=max_trials
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in semantic search: {str(e)}")


@router.get("/trial/{nct_id}/semantic-context", response_model=TrialSemanticContext)
async def get_trial_semantic_context(
    nct_id: str,
    service: ClinicalDataService = Depends(get_clinical_data_service)
):
    """
    Get semantic context for a clinical trial by mapping its conditions 
    and interventions to SNOMED CT.
    
    Args:
        nct_id: The ClinicalTrials.gov identifier (NCT number)
        
    Returns:
        Semantic context for the trial including condition and intervention mappings
    """
    try:
        context = service.get_trial_semantic_context(nct_id)
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trial semantic context: {str(e)}")


@router.get("/concept/{concept_id}/phase-analysis")
async def analyze_trial_phases_by_concept(
    concept_id: str,
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    include_descendants: bool = Query(True, description="Include descendant concepts"),
    max_results: int = Query(500, ge=1, le=1000),
    service: ClinicalDataService = Depends(get_clinical_data_service)
):
    """
    Analyze clinical trial phases for a medical concept.
    
    Args:
        concept_id: The concept identifier (e.g., SNOMED CT concept ID)
        terminology: The terminology system (currently only SNOMED CT supported)
        include_descendants: Whether to include descendant concepts
        max_results: Maximum number of trials to analyze (1-1000)
        
    Returns:
        Analysis of trial phases for the concept
    """
    try:
        analysis = service.analyze_trial_phases_by_concept(
            concept_id,
            terminology=terminology,
            include_descendants=include_descendants,
            max_results=max_results
        )
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
            
        return analysis
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing trial phases: {str(e)}")