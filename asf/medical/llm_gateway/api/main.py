"""
Main application for LLM Gateway API.

This module provides the main FastAPI application for the LLM Gateway API.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from asf.medical.llm_gateway.api.routes.providers import router as providers_router
from asf.medical.llm_gateway.db.session import init_db

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="LLM Gateway API",
    description="API for managing LLM providers and models",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(providers_router)

@app.on_event("startup")
async def startup_event():
    """
    Initialize the database on startup.
    """
    try:
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

@app.get("/", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    This endpoint returns the status of the API.
    """
    return {
        "status": "ok",
        "message": "LLM Gateway API is running",
        "version": "1.0.0"
    }

def start():
    """
    Start the FastAPI application with Uvicorn.
    """
    # Get host and port from environment variables
    host = os.environ.get("LLM_GATEWAY_API_HOST", "0.0.0.0")
    port = int(os.environ.get("LLM_GATEWAY_API_PORT", "8000"))
    
    # Start Uvicorn server
    uvicorn.run(
        "asf.medical.llm_gateway.api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    start()
