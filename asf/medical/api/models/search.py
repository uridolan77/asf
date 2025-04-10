"""
Search models for the Medical Research Synthesizer API.

This module defines the Pydantic models for search requests and responses.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    """Request model for the search endpoint."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)

class SearchResponse(BaseModel):
    """Response model for the search endpoint."""
    query: str = Field(..., description="The search query")
    total_count: int = Field(..., description="Total number of results")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    result_id: str = Field(..., description="Unique ID for this search result")

class PICORequest(BaseModel):
    """Request model for the PICO search endpoint."""
    condition: str = Field(..., description="Medical condition")
    interventions: List[str] = Field([], description="List of interventions")
    outcomes: List[str] = Field([], description="List of outcomes")
    population: Optional[str] = Field(None, description="Target population")
    study_design: Optional[str] = Field(None, description="Study design")
    years: int = Field(5, description="Number of years to search back", ge=1, le=50)
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)
