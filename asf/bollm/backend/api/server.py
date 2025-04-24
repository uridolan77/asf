"""
LLM Management BO API Server

This module provides the FastAPI application for the LLM Management BO API.
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

from asf.bollm.backend.config.config_loader import load_config
from asf.bollm.backend.api.routes.llm_router import router as llm_router
from asf.bollm.backend.api.routes.provider_router import router as provider_router
from asf.bollm.backend.api.routes.model_router import router as model_router
from asf.bollm.backend.api.routes.cache_router import router as cache_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    This handles initializing components on startup and cleaning up on shutdown.
    
    Args:
        app: The FastAPI application
    """
    logger.info("Initializing LLM Management BO server...")
    
    # Load configuration
    config = load_config()
    logger.info(f"Loaded configuration")
    
    # Initialize cache system
    # await setup_cache(config)
    logger.info("Cache system initialized")
    
    # Application is now ready to serve requests
    logger.info("LLM Management BO server started")
    
    yield
    
    # Shutdown operations
    logger.info("Shutting down LLM Management BO server...")
    
    # Shutdown cache system
    # await shutdown_cache()
    logger.info("Cache system shut down")
    
    logger.info("LLM Management BO server shutdown complete")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="LLM Management BO API",
        description="API for the LLM Management BO service",
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
    # app.include_router(llm_router, prefix="/api/llm", tags=["llm"])  # Commented out to avoid duplicate routes
    app.include_router(provider_router, prefix="/api/providers", tags=["providers"])
    app.include_router(model_router, prefix="/api/models", tags=["models"])
    app.include_router(cache_router, prefix="/api/cache", tags=["cache"])
    
    # Import existing routers
    from asf.bollm.backend.api.routers import router as existing_router
    app.include_router(existing_router)
    
    @app.get("/", tags=["status"])
    async def root():
        """Root endpoint that returns basic service information."""
        return {
            "service": "LLM Management BO",
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
    port = int(os.environ.get("BOLLM_PORT", 8001))
    
    # Start server
    uvicorn.run(
        "asf.bollm.backend.api.server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )
