Script to run the API.

This script starts the FastAPI application using uvicorn.

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import uvicorn
from asf.medical.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the API.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
    logger.info(f"Starting API in {'debug' if settings.DEBUG else 'production'} mode")

    uvicorn.run(
        "asf.medical.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )

if __name__ == "__main__":
    main()
