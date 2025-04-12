"""Export models for the Medical Research Synthesizer API.

This module defines the Pydantic models for export requests and responses.
"""

from typing import Optional
from pydantic import BaseModel, Field

class ExportRequest(BaseModel):
    """Request model for the export endpoint."""
    result_id: str = Field(..., description="ID of the result to export")
    format: str = Field(..., description="Export format (json, csv, excel, pdf)")
    include_abstracts: bool = Field(True, description="Whether to include abstracts in the export")
    include_metadata: bool = Field(True, description="Whether to include metadata in the export")

class ExportResponse(BaseModel):
    """Response model for the export endpoint."""
    file_url: str = Field(..., description="URL to download the exported file")
    file_name: str = Field(..., description="Name of the exported file")
    format: str = Field(..., description="Format of the exported file")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp for the download URL")
