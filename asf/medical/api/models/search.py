"""
Search models for the Medical Research Synthesizer API.

This module defines the Pydantic models for search requests and responses.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator

class PaginationParams(BaseModel):
    """Pagination parameters for search requests."""
    page: int = Field(1, description="Page number (1-based)", ge=1)
    page_size: int = Field(20, description="Number of results per page", ge=1, le=100)

class QueryRequest(BaseModel):
    """
    Request model for the search endpoint.

    This model defines the parameters for searching medical literature using
    various search methods, including PubMed, ClinicalTrials.gov, and GraphRAG.

    GraphRAG (Graph-based Retrieval-Augmented Generation) combines vector search
    and graph traversal to find relevant articles, potentially providing more
    comprehensive and contextually relevant results than traditional search methods.
        if not v or not v.strip():
            raise ValueError('Search query cannot be empty')
        return v.strip()

    @field_validator('use_vector_search', 'use_graph_search')
    def validate_graph_search_params(cls, v, info):
        """Validate that at least one search method is enabled when using GraphRAG.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        values = info.data
        if 'use_graph_rag' in values and values['use_graph_rag']:
            if 'use_vector_search' in values and not values['use_vector_search'] and not v:
                raise ValueError('At least one of use_vector_search or use_graph_search must be True when using GraphRAG')
        return v

class PaginationMetadata(BaseModel):
    """Pagination metadata for search responses."""
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of results per page")
    total_pages: int = Field(..., description="Total number of pages")
    total_count: int = Field(..., description="Total number of results")
    has_previous: bool = Field(..., description="Whether there is a previous page")
    has_next: bool = Field(..., description="Whether there is a next page")

class SearchResponse(BaseModel):
    """Response model for the search endpoint."""
    query: str = Field(..., description="The search query")
    total_count: int = Field(..., description="Total number of results")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    result_id: str = Field(..., description="Unique ID for this search result")
    pagination: PaginationMetadata = Field(..., description="Pagination metadata")

class PICORequest(BaseModel):
    """Request model for the PICO search endpoint."""
    condition: str = Field(..., description="Medical condition")
    interventions: List[str] = Field([], description="List of interventions")
    outcomes: List[str] = Field([], description="List of outcomes")
    population: Optional[str] = Field(None, description="Target population")
    study_design: Optional[str] = Field(None, description="Study design")
    years: int = Field(5, description="Number of years to search back", ge=1, le=50)
    max_results: int = Field(100, description="Maximum number of results to return", ge=1, le=500)
    pagination: PaginationParams = Field(default_factory=PaginationParams, description="Pagination parameters")
