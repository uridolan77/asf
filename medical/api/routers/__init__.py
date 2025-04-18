"""
Router modules for the Medical Research Synthesizer API.

This package contains all the router modules for the API, organized by functionality.
"""

# Analysis-related routers
from .analysis import router as analysis_router

# Export the main routers for use in the API
__all__ = [
    "analysis_router",
]

