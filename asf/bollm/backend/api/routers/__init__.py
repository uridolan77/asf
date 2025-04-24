"""
API routers package for BO backend.

This package contains all API routers for the BO backend.
"""

from fastapi import APIRouter

# Create main router
router = APIRouter()

# Import sub-routers
from .llm.main import router as llm_router
from .auth import router as auth_router

# Include sub-routers
router.include_router(llm_router)
router.include_router(auth_router)
