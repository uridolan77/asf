Pydantic models for the Medical Research Synthesizer API.

This module defines the request and response models for the API endpoints.

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    Request model for the search endpoint.
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)

class SearchResponse(BaseModel):
    Response model for the search endpoint.
    query: str = Field(..., description="The search query")
    total_count: int = Field(..., description="Total number of results")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    result_id: Optional[str] = Field(None, description="ID for retrieving these results later")

class PICORequest(BaseModel):
    Request model for the PICO search endpoint.
    condition: str = Field(..., description="Medical condition")
    interventions: List[str] = Field([], description="List of interventions")
    outcomes: List[str] = Field([], description="List of outcomes")
    population: Optional[str] = Field(None, description="Target population")
    study_design: Optional[str] = Field(None, description="Study design")
    years: int = Field(5, description="Number of years to search back", ge=1, le=50)
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)

# Contradiction analysis models
class ContradictionAnalysisRequest(BaseModel):
    Request model for the contradiction analysis endpoint.
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to analyze", ge=1, le=100)
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    threshold: float = Field(0.7, description="Threshold for contradiction detection", ge=0.0, le=1.0)

class ContradictionAnalysisResponse(BaseModel):
    Response model for the contradiction analysis endpoint.
        valid_schedules = ["daily", "weekly", "monthly"]
        if v not in valid_schedules:
            raise ValueError(f'Schedule must be one of: {", ".join(valid_schedules)}')
        return v

class KnowledgeBaseResponse(BaseModel):
    Response model for knowledge base operations.
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Search query for the knowledge base")
    kb_file: str = Field(..., description="Path to the knowledge base file")
    initial_results: int = Field(..., description="Number of initial results")
    update_schedule: str = Field(..., description="Update schedule")
    created_date: str = Field(..., description="Date the knowledge base was created")

# Export models
class ExportRequest(BaseModel):
    Request model for exporting results.
        if 'result_id' not in values and 'query' not in values:
            raise ValueError('Either result_id or query must be provided')
        return v

class UserRegistrationRequest(BaseModel):
    Request model for user registration.
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
