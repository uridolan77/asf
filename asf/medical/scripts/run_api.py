"""
Script to run the API.

This script starts the FastAPI application using uvicorn.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import uvicorn
from asf.medical.core.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the API."""
    logger.info(f"Starting API in {'debug' if settings.DEBUG else 'production'} mode")
    
    # Run the API
    uvicorn.run(
        "asf.medical.api.main_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )

if __name__ == "__main__":
    main()
