#!/usr/bin/env python3
"""
Terminology API Router

Provides API endpoints for terminology operations:
- Term normalization
- Code lookup
- Semantic search
- Hierarchical navigation
- Concept relationships
"""
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...services import TerminologyService
from ..dependencies import get_terminology_service

router = APIRouter(
    prefix="/terminology",
    tags=["terminology"],
    responses={404: {"description": "Not found"}},
)


class NormalizedTerm(BaseModel):
    """Response model for term normalization."""
    original_term: str
    normalized_term: str
    confidence: float
    concepts: List[Dict]


class Concept(BaseModel):
    """Response model for a terminology concept."""
    conceptId: str
    fsn: str
    preferredTerm: str
    active: bool


class ConceptDetail(BaseModel):
    """Response model for detailed concept information."""
    conceptId: str
    fsn: str
    preferredTerm: str
    active: bool
    terms: List[Dict] = []
    parents: List[str] = []
    children: List[str] = []
    relationships: List[Dict] = []


class Relationship(BaseModel):
    """Response model for concept relationships."""
    type: str
    typeId: str
    destinationId: str
    active: bool


@router.get("/normalize/{term}", response_model=NormalizedTerm)
async def normalize_term(
    term: str,
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Normalize a clinical term to its standard form.
    
    Args:
        term: The clinical term to normalize
    
    Returns:
        Normalized term and matching concepts
    """
    try:
        result = service.normalize_clinical_term(term)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error normalizing term: {str(e)}")


@router.get("/search", response_model=List[Concept])
async def search_concepts(
    query: str,
    max_results: int = Query(20, ge=1, le=100),
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    active_only: bool = Query(True, description="Return only active concepts"),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Search for terminology concepts.
    
    Args:
        query: The search term
        max_results: Maximum number of results to return (1-100)
        terminology: The terminology system (currently only SNOMED CT supported)
        active_only: Whether to return only active concepts
    
    Returns:
        List of matching concepts
    """
    try:
        if terminology.upper() != "SNOMEDCT":
            raise HTTPException(status_code=400, detail=f"Unsupported terminology: {terminology}")
        
        results = service.snomed_client.search(query, max_results=max_results, active_only=active_only)
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching concepts: {str(e)}")


@router.get("/concept/{code}", response_model=ConceptDetail)
async def get_concept(
    code: str,
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get detailed information about a concept.
    
    Args:
        code: The concept code
        terminology: The terminology system (currently only SNOMED CT supported)
    
    Returns:
        Detailed concept information
    """
    try:
        concept = service.get_concept_details(code, terminology)
        return concept
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving concept: {str(e)}")


@router.get("/concept/{code}/parents", response_model=List[Concept])
async def get_concept_parents(
    code: str,
    direct_only: bool = Query(True, description="Return only direct parents"),
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get parent concepts.
    
    Args:
        code: The concept code
        direct_only: If True, return only direct parents; if False, return all ancestors
        terminology: The terminology system (currently only SNOMED CT supported)
    
    Returns:
        List of parent concepts
    """
    try:
        if direct_only:
            parents = service.get_parents(code, terminology)
        else:
            parents = service.get_ancestors(code, terminology)
        return parents
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving parents: {str(e)}")


@router.get("/concept/{code}/children", response_model=List[Concept])
async def get_concept_children(
    code: str,
    direct_only: bool = Query(True, description="Return only direct children"),
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get child concepts.
    
    Args:
        code: The concept code
        direct_only: If True, return only direct children; if False, return all descendants
        terminology: The terminology system (currently only SNOMED CT supported)
    
    Returns:
        List of child concepts
    """
    try:
        if direct_only:
            children = service.get_children(code, terminology)
        else:
            children = service.get_descendants(code, terminology)
        return children
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving children: {str(e)}")


@router.get("/concept/{code}/relationships", response_model=List[Relationship])
async def get_concept_relationships(
    code: str,
    relationship_type: Optional[str] = Query(None, description="Relationship type ID to filter by"),
    terminology: str = Query("SNOMEDCT", description="Terminology system"),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get relationships for a concept.
    
    Args:
        code: The concept code
        relationship_type: Optional relationship type ID to filter by
        terminology: The terminology system (currently only SNOMED CT supported)
    
    Returns:
        List of relationships
    """
    try:
        relationships = service.get_relationships(code, relationship_type, terminology)
        return relationships
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving relationships: {str(e)}")


@router.get("/concept/{code}/sites", response_model=List[Concept])
async def get_finding_sites(
    code: str,
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get anatomical sites associated with a diagnosis.
    
    Args:
        code: The SNOMED CT concept ID for the diagnosis
    
    Returns:
        List of anatomical sites
    """
    try:
        sites = service.get_finding_sites(code)
        return sites
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving finding sites: {str(e)}")


@router.get("/concept/{code}/agents", response_model=List[Concept])
async def get_causative_agents(
    code: str,
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get causative agents associated with a diagnosis.
    
    Args:
        code: The SNOMED CT concept ID for the diagnosis
    
    Returns:
        List of causative agents
    """
    try:
        agents = service.get_causative_agents(code)
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving causative agents: {str(e)}")


@router.get("/ecl", response_model=List[Concept])
async def evaluate_ecl_expression(
    expression: str,
    max_results: int = Query(200, ge=1, le=1000),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Evaluate an Expression Constraint Language (ECL) expression.
    
    Args:
        expression: The ECL expression
        max_results: Maximum number of results to return (1-1000)
    
    Returns:
        List of matching concepts
    """
    try:
        results = service.evaluate_ecl(expression, max_results=max_results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error evaluating ECL expression: {str(e)}")


@router.get("/refset/{refset_id}/members", response_model=List[Dict])
async def get_refset_members(
    refset_id: str,
    max_results: int = Query(100, ge=1, le=500),
    service: TerminologyService = Depends(get_terminology_service)
):
    """
    Get members of a reference set.
    
    Args:
        refset_id: The reference set identifier
        max_results: Maximum number of results to return (1-500)
    
    Returns:
        List of reference set members
    """
    try:
        members = service.get_reference_set_members(refset_id, max_results=max_results)
        return members
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving reference set members: {str(e)}")