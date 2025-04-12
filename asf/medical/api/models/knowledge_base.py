"""Knowledge base models for the Medical Research Synthesizer API.

This module defines the Pydantic models for knowledge base requests and responses.
"""

from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

class KnowledgeBaseRequest(BaseModel):
    """Request model for the knowledge base creation endpoint."""
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Query to build the knowledge base")
    update_schedule: str = Field("weekly", description="Update schedule (daily, weekly, monthly)")

class KnowledgeBaseResponse(BaseModel):
    """Response model for the knowledge base endpoints."""
    kb_id: str = Field(..., description="Unique ID for the knowledge base")
    name: str = Field(..., description="Name of the knowledge base")
    query: str = Field(..., description="Query used to build the knowledge base")
    file_path: str = Field(..., description="Path to the knowledge base file")
    update_schedule: str = Field(..., description="Update schedule")
    last_updated: Optional[datetime] = Field(None, description="Last update timestamp")
    next_update: Optional[datetime] = Field(None, description="Next scheduled update timestamp")
    initial_results: int = Field(..., description="Number of initial results")
    created_at: datetime = Field(..., description="Creation timestamp")
