"""
Analysis router for the Medical Research Synthesizer API.

This module provides endpoints for analyzing medical literature,
including contradiction detection and specialized analyses.
"""

import uuid
import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status

from asf.medical.api.models import ContradictionAnalysisRequest, ContradictionAnalysisResponse
from asf.medical.api.dependencies import get_synthesizer, get_current_user
from asf.medical.api.auth_unified import User
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer

# Initialize router
router = APIRouter(prefix="/v1/analysis", tags=["Analysis"])

# In-memory storage for results (will be replaced with database in Phase 2)
result_storage: Dict[str, Any] = {}

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/contradictions", response_model=ContradictionAnalysisResponse)
async def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Analyze contradictions in literature matching the query.

    This endpoint searches for and identifies potential contradictions in the
    medical literature, comparing the authority of contradictory findings.

    It can use BioMedLM for more accurate contradiction detection if enabled.
    """
    try:
        logger.info(f"Analyzing contradictions: {request.query} (max_results={request.max_results}, use_biomedlm={request.use_biomedlm})")
        
        analysis = synthesizer.search_and_analyze_contradictions(
            query=request.query,
            max_results=request.max_results,
            use_biomedlm=request.use_biomedlm,
            threshold=request.threshold
        )
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'query': request.query,
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        logger.info(f"Contradiction analysis completed: {analysis['num_contradictions']} contradictions found (analysis_id={analysis_id})")
        
        # Add analysis_id to the response
        analysis['analysis_id'] = analysis_id
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing contradictions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cap")
async def cap_analysis(
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Perform specialized analysis of CAP literature.

    This endpoint executes a predefined analysis of contradictions in
    Community-Acquired Pneumonia treatment literature.
    """
    try:
        logger.info("Executing CAP analysis")
        
        analysis = synthesizer.search_cap_contradictory_treatments()
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        logger.info(f"CAP analysis completed: {analysis['num_contradictions']} contradictions found (analysis_id={analysis_id})")
        
        return {
            'analysis_id': analysis_id,
            'total_articles': analysis['total_articles'],
            'num_contradictions': analysis['num_contradictions'],
            'contradictions_by_intervention': {k: len(v) for k, v in analysis['contradictions_by_intervention'].items()},
            'authority_analysis': analysis['authority_analysis']
        }
    except Exception as e:
        logger.error(f"Error analyzing CAP treatments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cap/detailed")
async def cap_analysis_detailed(
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Detailed analysis of CAP treatment duration vs agent.

    This endpoint performs a specialized analysis comparing treatment duration
    and agent effectiveness for Community-Acquired Pneumonia.
    """
    try:
        logger.info("Executing detailed CAP analysis")
        
        analysis = synthesizer.cap_duration_vs_agent_analysis()
        
        # Store the analysis
        analysis_id = str(uuid.uuid4())
        result_storage[analysis_id] = {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        logger.info(f"Detailed CAP analysis completed (analysis_id={analysis_id})")
        
        return {
            'analysis_id': analysis_id,
            'duration_analysis': analysis['duration_analysis'],
            'agent_analysis': analysis['agent_analysis'],
            'cross_analysis': analysis['cross_analysis']
        }
    except Exception as e:
        logger.error(f"Error performing detailed CAP analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
