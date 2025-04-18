"""
Models for progress tracking in the LLM Gateway.

This module defines the data models used for progress tracking, including
progress details, operation types, and progress updates.
"""

import enum
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class OperationType(str, enum.Enum):
    """Types of operations that can be tracked."""
    
    # LLM operations
    LLM_REQUEST = "llm_request"
    LLM_STREAMING = "llm_streaming"
    LLM_BATCH = "llm_batch"
    LLM_FINE_TUNING = "llm_fine_tuning"
    LLM_EMBEDDING = "llm_embedding"
    
    # Provider operations
    PROVIDER_CONNECTION = "provider_connection"
    PROVIDER_INITIALIZATION = "provider_initialization"
    
    # Session operations
    SESSION_CREATION = "session_creation"
    SESSION_MANAGEMENT = "session_management"
    
    # Cache operations
    CACHE_OPERATION = "cache_operation"
    
    # General operations
    GENERAL = "general"
    CUSTOM = "custom"


class ProgressStep(BaseModel):
    """A step in a progress tracking operation."""
    
    step_number: int = Field(..., description="Step number (1-based)")
    message: str = Field(..., description="Progress message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of this step")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional details about this step")


class ProgressDetails(BaseModel):
    """Detailed information about a progress tracking operation."""
    
    operation_id: str = Field(..., description="Unique identifier for the operation")
    operation_type: OperationType = Field(..., description="Type of operation")
    total_steps: int = Field(..., description="Total number of steps in the operation")
    current_step: int = Field(0, description="Current step number (0-based)")
    status: str = Field("pending", description="Current status (pending, running, completed, failed)")
    message: str = Field("", description="Current progress message")
    percent_complete: float = Field(0.0, description="Percentage of completion (0-100)")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Start time of the operation")
    end_time: Optional[datetime] = Field(None, description="End time of the operation (if completed)")
    elapsed_time: float = Field(0.0, description="Elapsed time in seconds")
    estimated_time_remaining: Optional[float] = Field(None, description="Estimated time remaining in seconds")
    steps: List[ProgressStep] = Field(default_factory=list, description="List of progress steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the operation")
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }


class ProgressUpdate(BaseModel):
    """Update to a progress tracking operation."""
    
    operation_id: str = Field(..., description="Unique identifier for the operation")
    step: int = Field(..., description="Current step number (0-based)")
    message: str = Field(..., description="Progress message")
    status: Optional[str] = Field(None, description="New status (if changed)")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details about this update")
