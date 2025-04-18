"""
Models for LLM API.

This module provides Pydantic models for LLM-related API endpoints.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime


# Progress tracking models
class ProgressStepResponse(BaseModel):
    """Response model for a progress step."""
    
    step_number: int = Field(..., description="Step number (1-based)")
    message: str = Field(..., description="Progress message")
    timestamp: str = Field(..., description="Timestamp of this step")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details about this step")


class ProgressOperationResponse(BaseModel):
    """Response model for a progress operation."""
    
    operation_id: str = Field(..., description="Unique identifier for the operation")
    operation_type: str = Field(..., description="Type of operation")
    status: str = Field(..., description="Current status (pending, running, completed, failed)")
    percent_complete: float = Field(..., description="Percentage of completion (0-100)")
    message: str = Field(..., description="Current progress message")
    start_time: str = Field(..., description="Start time of the operation")
    end_time: Optional[str] = Field(None, description="End time of the operation (if completed)")


class ProgressDetailsResponse(BaseModel):
    """Response model for progress details."""
    
    operation_id: str = Field(..., description="Unique identifier for the operation")
    operation_type: str = Field(..., description="Type of operation")
    total_steps: int = Field(..., description="Total number of steps in the operation")
    current_step: int = Field(..., description="Current step number (0-based)")
    status: str = Field(..., description="Current status (pending, running, completed, failed)")
    message: str = Field(..., description="Current progress message")
    percent_complete: float = Field(..., description="Percentage of completion (0-100)")
    start_time: str = Field(..., description="Start time of the operation")
    end_time: Optional[str] = Field(None, description="End time of the operation (if completed)")
    elapsed_time: float = Field(..., description="Elapsed time in seconds")
    estimated_time_remaining: Optional[float] = Field(None, description="Estimated time remaining in seconds")
    steps: List[ProgressStepResponse] = Field(..., description="List of progress steps")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the operation")


class ProgressSummaryResponse(BaseModel):
    """Response model for progress summary."""
    
    total: int = Field(..., description="Total number of operations")
    active: int = Field(..., description="Number of active operations")
    completed: int = Field(..., description="Number of completed operations")
    failed: int = Field(..., description="Number of failed operations")
    by_status: Dict[str, int] = Field(..., description="Count of operations by status")
    by_type: Dict[str, int] = Field(..., description="Count of operations by type")


# Gateway models
class ProviderStatus(BaseModel):
    """Status of a provider."""
    
    provider_id: str = Field(..., description="Provider ID")
    status: str = Field(..., description="Status of the provider")
    provider_type: str = Field(..., description="Type of provider")
    display_name: Optional[str] = Field(None, description="Display name of the provider")
    models: List[str] = Field(default_factory=list, description="List of models supported by the provider")
    checked_at: str = Field(..., description="Timestamp of the status check")
    message: Optional[str] = Field(None, description="Status message")


class GatewayStatus(BaseModel):
    """Status of the LLM Gateway."""
    
    gateway_id: str = Field(..., description="Gateway ID")
    status: str = Field(..., description="Status of the gateway")
    version: str = Field(..., description="Version of the gateway")
    default_provider: str = Field(..., description="Default provider")
    active_providers: List[ProviderStatus] = Field(..., description="List of active providers")
    config_path: str = Field(..., description="Path to the gateway configuration file")
    checked_at: str = Field(..., description="Timestamp of the status check")


class ModelInfo(BaseModel):
    """Information about a model."""
    
    model_id: str = Field(..., description="Model ID")
    provider_id: str = Field(..., description="Provider ID")
    display_name: str = Field(..., description="Display name of the model")
    model_type: Optional[str] = Field(None, description="Type of model")
    context_window: Optional[int] = Field(None, description="Context window size")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")
    enabled: bool = Field(True, description="Whether the model is enabled")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional information about the model")


class ModelCreateRequest(BaseModel):
    """Request to create a model."""
    
    model_id: str = Field(..., description="Model ID")
    provider_id: str = Field(..., description="Provider ID")
    display_name: str = Field(..., description="Display name of the model")
    model_type: Optional[str] = Field(None, description="Type of model")
    context_window: Optional[int] = Field(None, description="Context window size")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens")
    enabled: bool = Field(True, description="Whether the model is enabled")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional information about the model")


class ProviderUpdateRequest(BaseModel):
    """Request to update a provider."""
    
    display_name: Optional[str] = Field(None, description="Display name of the provider")
    connection_params: Optional[Dict[str, Any]] = Field(None, description="Connection parameters")
    models: Optional[Dict[str, Dict[str, Any]]] = Field(None, description="Models supported by the provider")
    enabled: bool = Field(True, description="Whether the provider is enabled")


class LLMRequestModel(BaseModel):
    """Request to generate a response from an LLM."""
    
    prompt: str = Field(..., description="Prompt to send to the LLM")
    model: Optional[str] = Field(None, description="Model to use")
    provider: Optional[str] = Field(None, description="Provider to use")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, description="Temperature for generation")
    stream: bool = Field(False, description="Whether to stream the response")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")


class LLMResponseModel(BaseModel):
    """Response from an LLM."""
    
    request_id: str = Field(..., description="Request ID")
    text: str = Field(..., description="Generated text")
    finish_reason: str = Field(..., description="Reason for finishing generation")
    model: str = Field(..., description="Model used")
    provider: str = Field(..., description="Provider used")
    usage: Dict[str, int] = Field(..., description="Token usage information")
    elapsed_ms: float = Field(..., description="Elapsed time in milliseconds")
    additional_info: Dict[str, Any] = Field(default_factory=dict, description="Additional information")
