"""
RESTful API for Medical Research Synthesizer

This module provides a comprehensive API for accessing all features of the
enhanced medical research synthesizer.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from asf.medical.api.routers import search_new, analysis_new, kb_new
from asf.medical.api.routers.auth import router as auth_router
from asf.medical.api.routers.export import router as export_router
from asf.medical.storage.database import init_db
from asf.medical.core.config import settings
from asf.medical.core.cache import cache_manager
from asf.medical.ml.model_registry import model_registry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize the API
app = FastAPI(
    title="Medical Research Synthesizer API",
    description="API for searching, analyzing and synthesizing medical research literature",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our routers
app.include_router(auth_router, prefix="/v1")
app.include_router(search_new.router)
app.include_router(analysis_new.router)
app.include_router(kb_new.router)
app.include_router(export_router, prefix="/v1")

# Custom exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Custom handler for HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Custom handler for validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Initializing database...")
    init_db()
    
    logger.info("Initializing cache...")
    await cache_manager.init()
    
    logger.info("Initializing model registry...")
    model_registry.init()
    
    logger.info("API startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Closing cache connections...")
    await cache_manager.close()
    
    logger.info("Unloading models...")
    model_registry.unload_all_models()
    
    logger.info("API shutdown complete")

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with basic API information."""
    return {
        "message": "Welcome to the Medical Research Synthesizer API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "auth": "/v1/auth/login, /v1/auth/register",
            "search": "/v1/search, /v1/search/pico",
            "analysis": "/v1/analysis/contradictions, /v1/analysis/cap",
            "knowledge_base": "/v1/kb",
            "export": "/v1/export/{format}"
        }
    }

# Run with: uvicorn asf.medical.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("asf.medical.api.main:app", host="0.0.0.0", port=8000, reload=True)
