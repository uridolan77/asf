"""
Dependency injection for the Medical Research Synthesizer API.

This module provides dependencies that can be injected into API endpoints
using FastAPI's dependency injection system.
"""

from fastapi import Depends, HTTPException, status
from typing import Optional

from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
from asf.medical.api.auth import get_current_user, User

# Global instance of the synthesizer
_synthesizer = None

def get_synthesizer() -> EnhancedMedicalResearchSynthesizer:
    """
    Get or create a singleton instance of the EnhancedMedicalResearchSynthesizer.
    
    Returns:
        EnhancedMedicalResearchSynthesizer: The synthesizer instance
    """
    global _synthesizer
    
    if _synthesizer is None:
        import os
        _synthesizer = EnhancedMedicalResearchSynthesizer(
            email=os.getenv("NCBI_EMAIL", "your.email@example.com"),
            api_key=os.getenv("NCBI_API_KEY"),
            impact_factor_source=os.getenv("IMPACT_FACTOR_SOURCE", "journal_impact_factors.csv")
        )
    
    return _synthesizer

def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Verify that the current user has admin privileges.
    
    Args:
        current_user: The current authenticated user
        
    Returns:
        User: The current user if they have admin privileges
        
    Raises:
        HTTPException: If the user does not have admin privileges
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user
