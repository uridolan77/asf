"""
Pydantic models for the Medical Research Synthesizer API.

This module defines the request and response models for the API endpoints.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, validator

class QueryRequest(BaseModel):
    """Request model for the search endpoint."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)

class SearchResponse(BaseModel):
    """Response model for the search endpoint."""
    query: str = Field(..., description="The search query")
    total_count: int = Field(..., description="Total number of results")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    result_id: Optional[str] = Field(None, description="ID for retrieving these results later")

class PICORequest(BaseModel):
    """Request model for the PICO search endpoint."""
    condition: str = Field(..., description="Medical condition")
    interventions: List[str] = Field([], description="List of interventions")
    outcomes: List[str] = Field([], description="List of outcomes")
    population: Optional[str] = Field(None, description="Target population")
    study_design: Optional[str] = Field(None, description="Study design")
    years: int = Field(5, description="Number of years to search back", ge=1, le=50)
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)

# Contradiction analysis models
class ContradictionAnalysisRequest(BaseModel):
    """Request model for the contradiction analysis endpoint."""
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to analyze", ge=1, le=100)
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    threshold: float = Field(0.7, description="Threshold for contradiction detection", ge=0.0, le=1.0)

class ContradictionAnalysisResponse(BaseModel):
    """Response model for the contradiction analysis endpoint."""
    total_articles: int = Field(..., description="Total number of articles analyzed")
    num_contradictions: int = Field(..., description="Number of contradictions found")
    contradictions: List[Dict[str, Any]] = Field(..., description="List of contradictions")
    by_topic: Dict[str, List[Dict[str, Any]]] = Field(..., description="Contradictions grouped by topic")
    detection_method: str = Field("keyword", description="Method used for contradiction detection")
    analysis_id: Optional[str] = Field(None, description="ID for retrieving this analysis later")

# Knowledge base models
class KnowledgeBaseRequest(BaseModel):
    """Request model for creating a knowledge base."""
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Search query for the knowledge base")
    schedule: str = Field("weekly", description="Update schedule (daily, weekly, monthly)")
    max_results: int = Field(100, description="Maximum number of results to include", ge=1, le=500)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.isalnum() and not all(c.isalnum() or c == '_' for c in v):
            raise ValueError('Name must contain only alphanumeric characters and underscores')
        return v
    
    @validator('schedule')
    def validate_schedule(cls, v):
        valid_schedules = ["daily", "weekly", "monthly"]
        if v not in valid_schedules:
            raise ValueError(f'Schedule must be one of: {", ".join(valid_schedules)}')
        return v

class KnowledgeBaseResponse(BaseModel):
    """Response model for knowledge base operations."""
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Search query for the knowledge base")
    kb_file: str = Field(..., description="Path to the knowledge base file")
    initial_results: int = Field(..., description="Number of initial results")
    update_schedule: str = Field(..., description="Update schedule")
    created_date: str = Field(..., description="Date the knowledge base was created")

# Export models
class ExportRequest(BaseModel):
    """Request model for exporting results."""
    result_id: Optional[str] = Field(None, description="ID of previously stored results")
    query: Optional[str] = Field(None, description="Search query (if result_id is not provided)")
    max_results: int = Field(20, description="Maximum number of results (if query is provided)", ge=1, le=100)
    
    @validator('result_id', 'query')
    def validate_result_id_or_query(cls, v, values):
        if 'result_id' not in values and 'query' not in values:
            raise ValueError('Either result_id or query must be provided')
        return v

class UserRegistrationRequest(BaseModel):
    """Request model for user registration."""
    email: str = Field(..., description="User email")
    password: str = Field(..., description="User password", min_length=8)
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v
