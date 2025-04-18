"""
Medical Terminology API router for BO backend.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, Field

from api.services.medical_terminology_service import MedicalTerminologyService, get_medical_terminology_service
from models.user import User
from api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/medical/terminology",
    tags=["medical-terminology"],
    responses={404: {"description": "Not found"}},
)

class ECLQueryRequest(BaseModel):
    """ECL query request model."""
    expression: str = Field(..., description="The Expression Constraint Language (ECL) expression")
    max_results: int = Field(200, description="Maximum number of results to return")

@router.get("/normalize")
async def normalize_term(
    term: str = Query(..., description="Clinical term to normalize"),
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Normalize a clinical term to its standard form.
    
    This endpoint converts informal clinical terms (e.g., 'heart attack') to their
    standardized terminology equivalents.
    """
    result = terminology_service.normalize_term(term)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/concept/{code}")
async def get_concept_details(
    code: str = Path(..., description="The concept code"),
    terminology: str = Query("SNOMEDCT", description="Terminology system (currently only SNOMEDCT supported)"),
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Get detailed information about a medical concept.
    
    This endpoint retrieves comprehensive information about a medical concept
    identified by its code in the specified terminology.
    """
    result = terminology_service.get_concept_details(code, terminology)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND if "not found" in result["message"].lower() else status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/search")
async def semantic_search(
    query: str = Query(..., description="Search query for medical concepts"),
    max_results: int = Query(20, description="Maximum number of results to return"),
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Perform a semantic search for medical concepts.
    
    This endpoint searches for medical concepts using natural language queries
    and returns semantically relevant matches from the terminology.
    """
    result = terminology_service.semantic_search(query, max_results)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/hierarchy/{concept_id}/{relationship_type}")
async def get_hierarchical_relationships(
    concept_id: str = Path(..., description="The concept identifier"),
    relationship_type: str = Path(..., description="Type of relationship (parents, children, ancestors, descendants)"),
    terminology: str = Query("SNOMEDCT", description="Terminology system (currently only SNOMEDCT supported)"),
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Get hierarchical relationships for a medical concept.
    
    This endpoint retrieves hierarchical relationships (parents, children, ancestors, or descendants)
    for a given medical concept.
    """
    result = terminology_service.get_hierarchical_relationships(concept_id, relationship_type, terminology)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/relationships/{concept_id}")
async def get_concept_relationships(
    concept_id: str = Path(..., description="The concept identifier"),
    relationship_type: Optional[str] = Query(None, description="Optional relationship type ID to filter by"),
    terminology: str = Query("SNOMEDCT", description="Terminology system (currently only SNOMEDCT supported)"),
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Get relationships for a given concept.
    
    This endpoint retrieves all relationships or filtered relationships for a given
    medical concept.
    """
    result = terminology_service.get_concept_relationships(concept_id, relationship_type, terminology)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/is-a/{concept_id}/{parent_id}")
async def is_a_relationship(
    concept_id: str = Path(..., description="The concept to check"),
    parent_id: str = Path(..., description="The potential parent concept"),
    terminology: str = Query("SNOMEDCT", description="Terminology system (currently only SNOMEDCT supported)"),
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Check if a concept is a subtype of another concept.
    
    This endpoint determines whether one medical concept is a subtype (is-a relationship)
    of another concept.
    """
    result = terminology_service.is_a_relationship(concept_id, parent_id, terminology)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.post("/ecl")
async def evaluate_ecl(
    request: ECLQueryRequest,
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Evaluate an Expression Constraint Language (ECL) expression.
    
    This endpoint evaluates a SNOMED CT Expression Constraint Language query and
    returns matching concepts.
    """
    result = terminology_service.evaluate_ecl(request.expression, request.max_results)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/examples/diabetes-types")
async def get_diabetes_types(
    terminology_service: MedicalTerminologyService = Depends(get_medical_terminology_service)
):
    """
    Find all types of diabetes.
    
    This example endpoint retrieves all subtypes of diabetes mellitus using
    Expression Constraint Language.
    """
    result = terminology_service.find_all_diabetes_types()
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result