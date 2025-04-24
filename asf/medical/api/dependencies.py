"""
Dependencies for the Medical Research Synthesizer API.

This module provides dependency injection functions for the FastAPI application.
"""

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from asf.medical.storage.database import get_db_session
from asf.medical.storage.models import MedicalUser as User
# Import our demo mode authentication functions
from .auth import get_current_active_user_demo_mode, get_admin_user_demo_mode

# Import services
from asf.medical.services.search_service import SearchService
from asf.medical.ml.services.prisma_screening_service import PRISMAScreeningService
from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService

# Use demo mode authentication instead of regular authentication
get_current_active_user = get_current_active_user_demo_mode
get_admin_user = get_admin_user_demo_mode

# Services dependencies
def get_search_service(db: AsyncSession = Depends(get_db_session)) -> SearchService:
    """
    Get a search service instance.
    
    Args:
        db: Database session
        
    Returns:
        SearchService: A search service instance
    """
    return SearchService(db)

def get_prisma_screening_service(db: AsyncSession = Depends(get_db_session)) -> PRISMAScreeningService:
    """
    Get a PRISMA screening service instance.
    
    Args:
        db: Database session
        
    Returns:
        PRISMAScreeningService: A PRISMA screening service instance
    """
    return PRISMAScreeningService(db)

def get_bias_assessment_service(db: AsyncSession = Depends(get_db_session)) -> BiasAssessmentService:
    """
    Get a bias assessment service instance.
    
    Args:
        db: Database session
        
    Returns:
        BiasAssessmentService: A bias assessment service instance
    """
    return BiasAssessmentService(db)
