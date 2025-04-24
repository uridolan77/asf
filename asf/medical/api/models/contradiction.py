"""API models for contradiction detection.

This module provides Pydantic models for the contradiction detection API endpoints.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ContradictionRequest(BaseModel):
    """Request model for contradiction detection."""
    claim1: str = Field(..., description="First claim to compare")
    claim2: str = Field(..., description="Second claim to compare")
    metadata1: Optional[Dict[str, Any]] = Field(None, description="Metadata for first claim (publication date, study design, sample size, p-value, effect size)")
    metadata2: Optional[Dict[str, Any]] = Field(None, description="Metadata for second claim (publication date, study design, sample size, p-value, effect size)")
    threshold: float = Field(0.7, description="Contradiction detection threshold", ge=0.0, le=1.0)
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for semantic analysis")
    use_temporal: bool = Field(True, description="Whether to use temporal analysis for contradiction detection")
    use_tsmixer: bool = Field(True, description="Whether to use TSMixer for temporal sequence analysis")

class ContradictionDetail(BaseModel):
    """Model for contradiction detection details."""
    is_contradiction: bool = Field(..., description="Whether a contradiction was detected")
    score: float = Field(..., description="Contradiction score")
    confidence: str = Field(..., description="Confidence level (high, medium, low, unknown)")
    explanation: Optional[str] = Field(None, description="Explanation of the contradiction")

class ContradictionResponse(BaseModel):
    """Response model for contradiction detection."""
    is_contradiction: bool = Field(..., description="Whether a contradiction was detected")
    contradiction_score: float = Field(..., description="Overall contradiction score")
    contradiction_type: str = Field(..., description="Type of contradiction (direct, negation, statistical, methodological, temporal, none, unknown)")
    confidence: str = Field(..., description="Confidence level (high, medium, low, unknown)")
    explanation: Optional[str] = Field(None, description="Explanation of the contradiction")
    methods_used: List[str] = Field(..., description="Methods used for contradiction detection")
    details: Dict[str, ContradictionDetail] = Field(..., description="Detailed results for each detection method")

class BatchContradictionRequest(BaseModel):
    """Request model for batch contradiction detection."""
    articles: List[Dict[str, Any]] = Field(..., description="List of articles to analyze for contradictions")
    threshold: float = Field(0.7, description="Contradiction detection threshold", ge=0.0, le=1.0)
    use_biomedlm: bool = Field(True, description="Whether to use BioMedLM for semantic analysis")
    use_temporal: bool = Field(True, description="Whether to use temporal analysis for contradiction detection")
    use_tsmixer: bool = Field(True, description="Whether to use TSMixer for temporal sequence analysis")

class ArticleContradiction(BaseModel):
    """Model for contradiction between two articles."""
    article1: Dict[str, Any] = Field(..., description="First article")
    article2: Dict[str, Any] = Field(..., description="Second article")
    contradiction: ContradictionResponse = Field(..., description="Contradiction detection result")

class BatchContradictionResponse(BaseModel):
    """Response model for batch contradiction detection."""
    contradictions: List[ArticleContradiction] = Field(..., description="List of detected contradictions")
    total_articles: int = Field(..., description="Total number of articles analyzed")
    total_contradictions: int = Field(..., description="Total number of contradictions detected")
    threshold: float = Field(..., description="Contradiction detection threshold used")
