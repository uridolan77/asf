"""
Search router for the Medical Research Synthesizer API.

This module provides endpoints for searching medical literature.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from asf.medical.api.models import QueryRequest, SearchResponse, PICORequest
from asf.medical.api.dependencies import get_synthesizer, get_current_user
from asf.medical.api.auth import User
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from asf.medical.data_ingestion_layer.query_builder import MedicalCondition, MedicalIntervention, OutcomeMetric, StudyDesign

# Initialize router
router = APIRouter(prefix="/v1/search", tags=["Search"])

# In-memory storage for results (will be replaced with database in Phase 2)
result_storage: Dict[str, Any] = {}
query_storage: Dict[str, Any] = {}

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/", response_model=SearchResponse)
async def search(
    request: QueryRequest,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Search PubMed with the given query and return enriched results.

    This endpoint performs a search using the enhanced NCBIClient and enriches
    the results with metadata such as impact factors, authority scores,
    and standardized dates.
    """
    try:
        logger.info(f"Executing search: {request.query} (max_results={request.max_results})")
        results = synthesizer.search_and_enrich(query=request.query, max_results=request.max_results)
        
        # Store results for later use
        result_id = str(uuid.uuid4())
        result_storage[result_id] = {
            'query': request.query,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        logger.info(f"Search completed: {len(results)} results found (result_id={result_id})")
        
        return {
            "query": request.query,
            "total_count": len(results),
            "results": results,
            "result_id": result_id
        }
    except Exception as e:
        logger.error(f"Error executing search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pico", response_model=SearchResponse)
async def pico_search(
    request: PICORequest,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Search using PICO (Population, Intervention, Comparison, Outcome) framework.

    This endpoint builds a structured query using the PICO framework and returns
    enriched search results.
    """
    try:
        logger.info(f"Executing PICO search: condition={request.condition}, interventions={request.interventions}")
        
        # Create condition
        condition = MedicalCondition(request.condition)
        
        # Create interventions
        interventions = [MedicalIntervention(i) for i in request.interventions]
        
        # Create outcomes
        outcomes = [OutcomeMetric(o) for o in request.outcomes]
        
        # Create population and study design if provided
        population = None
        if request.population:
            population = request.population
            
        study_design = None
        if request.study_design:
            study_design = StudyDesign(request.study_design)
        
        # Build query
        builder = synthesizer.create_pico_query(
            condition=condition,
            interventions=interventions,
            outcomes=outcomes,
            population=population,
            study_design=study_design,
            years=request.years
        )
        
        # Store the query and builder
        query_id = str(uuid.uuid4())
        query = builder.build_pico_query(use_mesh=True)
        query_storage[query_id] = {
            'query': query,
            'builder': builder,
            'created_at': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        # Execute search
        results = synthesizer.search_and_enrich(query_builder=builder, max_results=request.max_results)
        
        # Store results
        result_id = str(uuid.uuid4())
        result_storage[result_id] = {
            'query': query,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        logger.info(f"PICO search completed: {len(results)} results found (result_id={result_id})")
        
        return {
            "query": query,
            "total_count": len(results),
            "results": results,
            "result_id": result_id
        }
    except Exception as e:
        logger.error(f"Error executing PICO search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/template/{template_id}")
async def get_query_from_template(
    template_id: str,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Create a query from a template.
    
    This endpoint creates a structured query from a predefined template.
    """
    try:
        logger.info(f"Creating query from template: {template_id}")
        builder = synthesizer.create_query_from_template(template_id)
        
        # Build the query
        query = builder.build_pico_query(use_mesh=True)
        
        # Store the query and builder
        query_id = str(uuid.uuid4())
        query_storage[query_id] = {
            'query': query,
            'builder': builder,
            'created_at': datetime.now().isoformat(),
            'user': current_user.email,
            'template': template_id
        }
        
        # Get explanation for this query
        explanation = synthesizer.query_interface.explain_query(query_type='pico', use_mesh=True)
        
        logger.info(f"Query created from template: {template_id} (query_id={query_id})")
        
        return {
            'query_id': query_id,
            'query': query,
            'components': explanation['components']
        }
    except Exception as e:
        logger.error(f"Error creating query from template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
