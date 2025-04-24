"""
LLM Gateway API Server

This module provides the FastAPI application for the LLM Gateway API.
It initializes the API routes, middleware, and cache system.
"""

import os
import logging
import asyncio
from typing import Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asf.medical.llm_gateway.config.loader import load_config
from asf.medical.llm_gateway.api.initialize_cache import setup_cache, shutdown_cache
from asf.medical.llm_gateway.api.routes.cache import router as cache_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This handles initializing components on startup and cleaning up on shutdown.
    
    Args:
        app: The FastAPI application
    """
    logger.info("Initializing LLM Gateway server...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration for gateway: {config.get('gateway_id', 'default')}")
    
    # Initialize cache system
    await setup_cache(config)
    logger.info("Cache system initialized")
    
    # Application is now ready to serve requests
    logger.info("LLM Gateway server started")
    
    yield
    
    # Shutdown operations
    logger.info("Shutting down LLM Gateway server...")
    
    # Shutdown cache system
    await shutdown_cache()
    logger.info("Cache system shut down")
    
    logger.info("LLM Gateway server shutdown complete")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="LLM Gateway API",
        description="API for the LLM Gateway service",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Update this for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    app.include_router(cache_router, tags=["cache"])
    
    @app.get("/", tags=["status"])
    async def root():
        """Root endpoint that returns basic service information."""
        return {
            "service": "LLM Gateway",
            "version": "1.0.0",
            "status": "running"
        }
    
    @app.get("/health", tags=["status"])
    async def health():
        """Health check endpoint."""
        # TODO: Add more comprehensive health checks
        return {"status": "ok"}
    
    return app

app = create_app()

# For direct execution
if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("LLM_GATEWAY_PORT", 8000))
    
    # Start server
    uvicorn.run(
        "asf.medical.llm_gateway.api.server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )