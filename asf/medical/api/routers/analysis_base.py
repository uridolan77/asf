"""Base module for analysis routers in the Medical Research Synthesizer API.

This module provides the base router and common utilities for analysis endpoints.
"""

import logging
from fastapi import APIRouter, Depends

from ..models.base import APIResponse, ErrorResponse
from ..dependencies import get_analysis_service
from ..auth import get_current_active_user
from ...services.analysis_service import AnalysisService
from ...storage.models import User
from ...core.observability import async_timed, log_error

# Create the base router
router = APIRouter(prefix="/analysis", tags=["analysis"])

# Set up logging
logger = logging.getLogger(__name__)
