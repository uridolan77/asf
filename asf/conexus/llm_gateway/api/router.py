"""
Main API router for the Conexus LLM Gateway.

This module combines all the individual API routers and endpoints
to provide a complete REST API for the LLM Gateway.
"""

import logging
from fastapi import APIRouter

# Import individual routers
from asf.conexus.llm_gateway.api.models_router import router as models_router
from asf.conexus.llm_gateway.api.providers_router import router as providers_router
from asf.conexus.llm_gateway.api.completion_router import router as completion_router
from asf.conexus.llm_gateway.api.progress_router import router as progress_router
from asf.conexus.llm_gateway.api.health_router import router as health_router
from asf.conexus.llm_gateway.api.admin_router import router as admin_router

logger = logging.getLogger(__name__)

# Create main API router
api_router = APIRouter()

# Include all individual routers
api_router.include_router(health_router)
api_router.include_router(models_router)
api_router.include_router(providers_router)
api_router.include_router(completion_router)
api_router.include_router(progress_router)
api_router.include_router(admin_router)