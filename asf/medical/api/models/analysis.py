"""
Analysis models for the Medical Research Synthesizer API.

This module defines the Pydantic models for analysis requests and responses.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
class ContradictionAnalysisRequest(BaseModel):
    """
    Request model for the contradiction analysis endpoint.
    """
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to analyze", ge=1, le=100)
    threshold: float = Field(0.7, description="Contradiction detection threshold", ge=0.0, le=1.0)
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    use_tsmixer: bool = Field(False, description="Whether to use TSMixer for temporal contradiction detection")
    use_lorentz: bool = Field(False, description="Whether to use Lorentz embeddings for hierarchical contradiction detection")

class ContradictionAnalysisResponse(BaseModel):
    """
    Response model for the contradiction analysis endpoint.
    """
    query: str = Field(..., description="The search query")
    total_articles: int = Field(..., description="Total number of articles analyzed")
    contradictions: List[Dict[str, Any]] = Field(..., description="Detected contradictions")
    analysis_id: str = Field(..., description="Unique ID for this analysis")
    detection_method: str = Field(..., description="Method used for contradiction detection")
