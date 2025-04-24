"""
Base models for the Medical Research Synthesizer API.

This module provides base models for consistent API responses.
"""

from typing import Optional, List, Dict, Any, TypeVar, Generic
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar('T')


class APIResponse(GenericModel, Generic[T]):
    """
    Base model for API responses.
    
    This model provides a consistent structure for API responses.
    """
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: Optional[T] = Field(None, description="Response data")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    meta: Optional[Dict[str, Any]] = Field(None, description="Metadata")


class PaginatedResponse(GenericModel, Generic[T]):
    """
    Base model for paginated API responses.
    
    This model provides a consistent structure for paginated API responses.
    """
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    data: List[T] = Field(..., description="Response data")
    meta: Dict[str, Any] = Field(..., description="Metadata")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Page size")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")


class ErrorResponse(BaseModel):
    """
    Model for error responses.
    
    This model provides a consistent structure for error responses.
    """
    success: bool = Field(False, description="Whether the request was successful")
    message: str = Field(..., description="Error message")
    errors: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")
    code: Optional[str] = Field(None, description="Error code")


class HealthResponse(BaseModel):
    """
    Model for health check responses.
    
    This model provides a consistent structure for health check responses.
    """
    status: str = Field(..., description="Health status (ok, warning, error)")
    checks: Dict[str, Any] = Field(..., description="Health check details")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Timestamp of the health check")
