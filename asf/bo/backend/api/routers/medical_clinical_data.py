"""
Medical Clinical Data API router for BO backend.

This router provides endpoints for integrated clinical data operations:
- Combined terminology and clinical trials search
- Semantic expansion of clinical terms
- Mapping between SNOMED CT and clinical trials
- Advanced clinical data analytics
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, Field

from api.services.medical_clinical_data_service import MedicalClinicalDataService, get_medical_clinical_data_service
from models.user import User
from api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/medical/clinical-data",
    tags=["medical-clinical-data"],
    responses={404: {"description": "Not found"}},
)

# Pydantic models for request/response validation
class TrialInfo(BaseModel):
    NCTId: str
    BriefTitle: str
    OverallStatus: Optional[str] = None
    Phase: Optional[str] = None
    EnrollmentCount: Optional[int] = None

class ConceptInfo(BaseModel):
    conceptId: str
    preferredTerm: str
    definition: Optional[str] = None
    synonyms: Optional[List[str]] = None

class IntegratedSearchResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@router.get("/search", response_model=IntegratedSearchResponse)
async def search_concept_and_trials(
    term: str,
    max_trials: int = Query(10, ge=1, le=50),
    clinical_data_service: MedicalClinicalDataService = Depends(get_medical_clinical_data_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Search for a medical term and find related SNOMED CT concepts and clinical trials.
    
    This endpoint performs an integrated search that returns both terminology concepts
    and relevant clinical trials for a given medical term.
    """
    result = clinical_data_service.search_concept_and_trials(term, max_trials=max_trials)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.get("/concept/{concept_id}/trials", response_model=IntegratedSearchResponse)
async def get_trials_by_concept(
    concept_id: str,
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    max_trials: int = Query(10, ge=1, le=50),
    clinical_data_service: MedicalClinicalDataService = Depends(get_medical_clinical_data_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Find clinical trials related to a specific medical concept.
    
    This endpoint searches for clinical trials related to a specific medical concept
    identified by its concept ID (e.g., SNOMED CT concept ID).
    """
    result = clinical_data_service.search_by_concept_id(
        concept_id, 
        terminology=terminology, 
        max_trials=max_trials
    )
    
    if not result["success"]:
        if "not found" in result["message"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
    return result

@router.get("/trial/{nct_id}/mapping", response_model=IntegratedSearchResponse)
async def map_trial_conditions(
    nct_id: str,
    clinical_data_service: MedicalClinicalDataService = Depends(get_medical_clinical_data_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Map all conditions in a clinical trial to SNOMED CT concepts.
    
    This endpoint takes a ClinicalTrials.gov identifier (NCT number) and maps
    all conditions mentioned in the trial to SNOMED CT concepts.
    """
    result = clinical_data_service.map_trial_conditions(nct_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.get("/semantic-search", response_model=IntegratedSearchResponse)
async def find_trials_with_semantic_expansion(
    term: str,
    include_similar: bool = Query(True, description="Include similar concepts in search"),
    max_trials: int = Query(20, ge=1, le=100),
    clinical_data_service: MedicalClinicalDataService = Depends(get_medical_clinical_data_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Find clinical trials with semantic expansion of the search term.
    
    This endpoint normalizes the input term using SNOMED CT and expands
    the search to include semantically similar concepts.
    """
    result = clinical_data_service.find_trials_with_semantic_expansion(
        term,
        include_similar=include_similar,
        max_trials=max_trials
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.get("/trial/{nct_id}/semantic-context", response_model=IntegratedSearchResponse)
async def get_trial_semantic_context(
    nct_id: str,
    clinical_data_service: MedicalClinicalDataService = Depends(get_medical_clinical_data_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Get semantic context for a clinical trial by mapping its conditions 
    and interventions to SNOMED CT.
    
    This endpoint provides a rich semantic context for a clinical trial by mapping
    its conditions and interventions to standardized SNOMED CT concepts.
    """
    result = clinical_data_service.get_trial_semantic_context(nct_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.get("/concept/{concept_id}/phase-analysis", response_model=IntegratedSearchResponse)
async def analyze_trial_phases_by_concept(
    concept_id: str,
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    include_descendants: bool = Query(True, description="Include descendant concepts"),
    max_results: int = Query(500, ge=1, le=1000),
    clinical_data_service: MedicalClinicalDataService = Depends(get_medical_clinical_data_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Analyze clinical trial phases for a medical concept.
    
    This endpoint analyzes the distribution of clinical trial phases for a given
    medical concept, providing insights into the research landscape.
    """
    result = clinical_data_service.analyze_trial_phases_by_concept(
        concept_id,
        terminology=terminology,
        include_descendants=include_descendants,
        max_results=max_results
    )
    
    if not result["success"]:
        if "not found" in result["message"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
    return result
