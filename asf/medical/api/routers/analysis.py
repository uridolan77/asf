"""Analysis router for the Medical Research Synthesizer API.

This module provides endpoints for analyzing medical literature,
including contradiction detection and specialized analyses.
It combines multiple analysis-related routers into a single router.
"""

# Import the base router
from .analysis_base import router

# Import all analysis-related endpoints to register them with the router
# Use absolute imports to avoid circular dependencies
from asf.medical.api.routers import contradiction_analysis  # noqa: F401
from asf.medical.api.routers import cap_analysis  # noqa: F401
from asf.medical.api.routers import analysis_retrieval  # noqa: F401

# Export the router
__all__ = ["router"]
