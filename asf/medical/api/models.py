"""
Pydantic models for the Medical Research Synthesizer API.

This module defines the request and response models for the API endpoints,
providing validation and documentation for the API data structures.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    """
    Request model for the search endpoint.
    """
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)


class SearchResponse(BaseModel):
    """
    Response model for the search endpoint.
    """
    query: str = Field(..., description="The search query")
    total_count: int = Field(..., description="Total number of results")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    result_id: Optional[str] = Field(None, description="ID for retrieving these results later")


class PICORequest(BaseModel):
    """
    Request model for the PICO search endpoint.
    """
    condition: str = Field(..., description="Medical condition")
    interventions: List[str] = Field([], description="List of interventions")
    outcomes: List[str] = Field([], description="List of outcomes")
    population: Optional[str] = Field(None, description="Target population")
    study_design: Optional[str] = Field(None, description="Study design")
    years: int = Field(5, description="Number of years to search back", ge=1, le=50)
    max_results: int = Field(20, description="Maximum number of results to return", ge=1, le=100)


# Contradiction analysis models
class ContradictionAnalysisRequest(BaseModel):
    """
    Request model for the contradiction analysis endpoint.
    """
    query: str = Field(..., description="The search query")
    max_results: int = Field(20, description="Maximum number of results to analyze", ge=1, le=100)
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for contradiction detection")
    use_tsmixer: bool = Field(True, description="Whether to use TSMixer for time-series analysis")
    use_lorentz: bool = Field(True, description="Whether to use Lorentz transformations")
    threshold: float = Field(0.7, description="Threshold for contradiction detection", ge=0.0, le=1.0)


class ContradictionAnalysisResponse(BaseModel):
    """
    Response model for the contradiction analysis endpoint.
    """
    query: str = Field(..., description="The search query")
    contradictions: List[Dict[str, Any]] = Field(..., description="Detected contradictions")
    total_articles: int = Field(..., description="Total number of articles analyzed")
    total_contradictions: int = Field(..., description="Total number of contradictions found")
    analysis_id: Optional[str] = Field(None, description="ID for retrieving this analysis later")
    
    
# Knowledge base models
class KnowledgeBaseRequest(BaseModel):
    """
    Request model for creating a knowledge base.
    """
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Search query for the knowledge base")
    update_schedule: str = Field("weekly", description="Update schedule (daily, weekly, monthly)")
    
    @validator('update_schedule')
    def validate_schedule(cls, v):
        valid_schedules = ["daily", "weekly", "monthly"]
        if v not in valid_schedules:
            raise ValueError(f'Schedule must be one of: {", ".join(valid_schedules)}')
        return v


class KnowledgeBaseResponse(BaseModel):
    """
    Response model for knowledge base operations.
    """
    kb_id: str = Field(..., description="ID of the knowledge base")
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Search query for the knowledge base")
    kb_file: str = Field(..., description="Path to the knowledge base file")
    initial_results: int = Field(..., description="Number of initial results")
    update_schedule: str = Field(..., description="Update schedule")
    created_at: datetime = Field(..., description="Date the knowledge base was created")
    updated_at: Optional[datetime] = Field(None, description="Date the knowledge base was last updated")
    user_id: str = Field(..., description="ID of the user who created the knowledge base")


# Export models
class ExportRequest(BaseModel):
    """
    Request model for exporting results.
    """
    result_id: Optional[str] = Field(None, description="ID of the result to export")
    query: Optional[str] = Field(None, description="Search query to execute and export")
    max_results: int = Field(20, description="Maximum number of results to export", ge=1, le=100)
    
    @validator('max_results')
    def validate_export_request(cls, v, values):
        if 'result_id' not in values and 'query' not in values:
            raise ValueError('Either result_id or query must be provided')
        return v


class UserRegistrationRequest(BaseModel):
    """
    Request model for user registration.
    """
    username: str = Field(..., description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password", min_length=8)
    full_name: Optional[str] = Field(None, description="Full name")
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v


# Pagination model for all paginated requests
class PaginationParams(BaseModel):
    """
    Pagination parameters for paginated requests.
    """
    page: int = Field(1, description="Page number", ge=1)
    page_size: int = Field(10, description="Items per page", ge=1, le=100)


# Base response model with pagination information
class PaginatedResponse(BaseModel):
    """
    Base model for paginated responses.
    """
    pagination: Dict[str, Any] = Field(..., description="Pagination information")
    total_count: int = Field(..., description="Total count of items")
    results: List[Dict[str, Any]] = Field(..., description="Results for the current page")
