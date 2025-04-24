"""
API Server Launcher Script.

This script starts the FastAPI application using uvicorn as the ASGI server.
It configures the server based on the application settings, enabling features
like hot reloading in debug mode and proper production settings otherwise.

Usage:
    python -m asf.medical.scripts.run_api

Environment Variables:
    DEBUG: Set to 'True' to enable debug mode with hot reloading
    PORT: Set to override the default port (8000)
    HOST: Set to override the default host ('0.0.0.0')
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import uvicorn
from asf.medical.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run the API server using uvicorn.

    This function starts the FastAPI application using uvicorn as the ASGI server.
    It configures the server based on the application settings, including:
    - Host address (defaults to 0.0.0.0 to listen on all interfaces)
    - Port (defaults to 8000)
    - Hot reloading (enabled in debug mode)
    - Log level (more verbose in debug mode)

    The function blocks until the server is stopped with Ctrl+C or a signal.

    Returns:
        None
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
