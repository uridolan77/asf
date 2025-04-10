"""
Unified FastAPI application for the Medical Research Synthesizer API.

This module initializes the FastAPI application and includes all routers.
It provides a comprehensive API for searching, analyzing, and synthesizing medical research literature.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager

from asf.medical.core.logging_config import get_logger

from asf.medical.api.middleware import MonitoringMiddleware
from asf.medical.core.monitoring import setup_monitoring, get_metrics, run_health_checks, export_metrics_to_json

from asf.medical.api.routers.auth import router as auth_router
from asf.medical.api.routers.search import router as search_router
# Removed old contradiction router import
from asf.medical.api.routers.enhanced_contradiction import router as enhanced_contradiction_router
from asf.medical.api.routers.contradiction_resolution import router as contradiction_resolution_router
from asf.medical.api.routers.contradiction import router as contradiction_router
from asf.medical.api.routers.screening import router as screening_router
from asf.medical.api.routers.export import router as export_router
from asf.medical.api.routers.analysis import router as analysis_router
from asf.medical.api.routers.knowledge_base import router as knowledge_base_router
from asf.medical.api.routers.async_ml import router as async_ml_router
from asf.medical.api.routers.model_cache import router as model_cache_router
from asf.medical.api.routers.task_management import router as task_management_router
from asf.medical.api.routers.resource_monitoring import router as resource_monitoring_router
from asf.medical.core.config import settings
from asf.medical.core.cache import cache_manager
from asf.medical.storage.database import init_db
from asf.medical.ml.model_registry import model_registry

# Get logger
logger = get_logger(__name__)

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.

    This function handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting application in {settings.ENVIRONMENT} environment")

    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")

        # Initialize cache manager with Redis if configured
        if settings.REDIS_URL:
            # Re-initialize cache manager with Redis URL
            cache_manager.__init__(
                max_size=10000,  # Increase cache size for production
                redis_url=settings.REDIS_URL,
                default_ttl=settings.CACHE_TTL,
                namespace="asf:medical:"
            )
            logger.info(f"Cache manager initialized with Redis: {settings.REDIS_URL}")
        else:
            logger.info("Cache manager initialized with local LRU cache only")

        # Initialize model registry
        model_registry.initialize(use_gpu=settings.USE_GPU)
        logger.info(f"Model registry initialized with GPU support: {settings.USE_GPU}")

        # Set up monitoring
        setup_monitoring()
        logger.info("Monitoring initialized")

        # Log application info
        logger.info("Application startup complete")
        logger.info("API documentation available at: /docs and /redoc")
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

    yield

    # Shutdown
    logger.info("Application shutdown initiated")

    try:
        # Clear cache
        await cache_manager.clear()
        logger.info("Cache cleared")

        # Unload models
        model_registry.unload_all()
        logger.info("Models unloaded")

        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")
        # Don't re-raise here to ensure all cleanup attempts are made

# Initialize the API
app = FastAPI(
    title="Medical Research Synthesizer API",
    description="API for searching, analyzing and synthesizing medical research literature",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add custom components
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    # Add security schemes
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}

    # Define Bearer token security scheme
    openapi_schema["components"]["securitySchemes"]["Bearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter JWT token",
    }

    # Apply security globally
    openapi_schema["security"] = [{"Bearer": []}]

    # Add custom tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "auth",
            "description": "Authentication operations",
        },
        {
            "name": "search",
            "description": "Search operations for medical literature",
        },
        {
            "name": "analysis",
            "description": "Analysis operations for medical research",
        },
        {
            "name": "contradiction",
            "description": "Basic contradiction detection between research claims",
        },
        {
            "name": "enhanced-contradiction",
            "description": "Enhanced contradiction detection between research claims",
        },
        {
            "name": "contradiction-resolution",
            "description": "Resolution strategies for contradictions",
        },
        {
            "name": "screening",
            "description": "Literature screening pipeline",
        },
        {
            "name": "export",
            "description": "Export operations for results",
        },
        {
            "name": "knowledge_base",
            "description": "Knowledge base management",
        },
        {
            "name": "async-ml",
            "description": "Asynchronous ML inference operations",
        },
        {
            "name": "admin",
            "description": "Administrative operations",
        },
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add monitoring middleware
app.add_middleware(MonitoringMiddleware)

# Include routers
app.include_router(auth_router)
app.include_router(search_router, prefix=settings.API_V1_STR, tags=["search"])
# Removed old contradiction router
app.include_router(contradiction_router, prefix=settings.API_V1_STR, tags=["contradiction"])
app.include_router(enhanced_contradiction_router, prefix=settings.API_V1_STR, tags=["enhanced-contradiction"])
app.include_router(contradiction_resolution_router, prefix=settings.API_V1_STR, tags=["contradiction-resolution"])
app.include_router(screening_router, prefix=settings.API_V1_STR, tags=["screening"])
app.include_router(export_router, prefix=settings.API_V1_STR, tags=["export"])
app.include_router(analysis_router, prefix=settings.API_V1_STR, tags=["analysis"])
app.include_router(knowledge_base_router, prefix=settings.API_V1_STR, tags=["knowledge_base"])
app.include_router(async_ml_router, prefix=settings.API_V1_STR, tags=["async-ml"])
app.include_router(model_cache_router)
app.include_router(task_management_router)
app.include_router(resource_monitoring_router)

# Root endpoint
@app.get("/", tags=["status"])
async def root():
    """Root endpoint for health check."""
    return {
        "message": "Welcome to the Medical Research Synthesizer API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "healthy"
    }

# Health check endpoint
@app.get("/health", tags=["status"])
async def health():
    """Health check endpoint."""
    health_checks = run_health_checks()
    all_ok = all(check.get("status") == "ok" for check in health_checks.values())

    if all_ok:
        return {"status": "ok", "checks": health_checks}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "checks": health_checks}
        )

# Cache stats endpoint
@app.get("/cache/stats", tags=["admin"])
async def cache_stats():
    """Get cache statistics."""
    return await cache_manager.get_stats()

# Clear cache endpoint
@app.post("/cache/clear", tags=["admin"])
async def clear_cache(namespace: str = None):
    """Clear the cache."""
    await cache_manager.clear(namespace)
    return {"status": "ok", "message": f"Cache cleared for namespace: {namespace if namespace else 'all'}"}

# Metrics endpoint
@app.get("/metrics", tags=["admin"])
async def metrics():
    """Get metrics."""
    return get_metrics()

# Export metrics endpoint
@app.post("/metrics/export", tags=["admin"])
async def export_metrics(file_path: str = "logs/metrics.json"):
    """Export metrics to a JSON file."""
    export_metrics_to_json(file_path)
    return {"status": "ok", "message": f"Metrics exported to {file_path}"}

# Custom Swagger UI
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI."""
    from fastapi.templating import Jinja2Templates
    from fastapi.responses import HTMLResponse
    from fastapi.requests import Request
    import os

    # Check if templates directory exists
    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    if os.path.exists(os.path.join(templates_dir, "custom_swagger.html")):
        # Use custom template
        templates = Jinja2Templates(directory=templates_dir)
        return templates.TemplateResponse(
            "custom_swagger.html",
            {
                "request": Request,
                "openapi_url": app.openapi_url,
                "title": f"{app.title} - API Documentation",
                "csrf_token": "",  # Add CSRF token if needed
            },
        )
    else:
        # Fallback to default Swagger UI
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
        )

# Custom ReDoc
@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """ReDoc UI."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

# Custom OpenAPI schema
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """Custom OpenAPI schema."""
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=[
            {"name": "auth", "description": "Authentication endpoints"},
            {"name": "search", "description": "Search endpoints for medical literature"},
            {"name": "analysis", "description": "Analysis endpoints for medical literature"},
            {"name": "knowledge_base", "description": "Knowledge base management endpoints"},
            {"name": "export", "description": "Export endpoints for data export"},
            {"name": "screening", "description": "PRISMA-guided screening and bias assessment endpoints"},
            # Contradiction endpoints
            {"name": "contradiction", "description": "Basic contradiction detection endpoints"},
            {"name": "enhanced-contradiction", "description": "Enhanced multi-dimensional contradiction classification endpoints"},
            {"name": "contradiction-resolution", "description": "Evidence-based contradiction resolution endpoints"},
            {"name": "async-ml", "description": "Asynchronous ML model inference endpoints"},
            {"name": "status", "description": "Status endpoints"},
            {"name": "admin", "description": "Admin endpoints"}
        ]
    )

# Run with: uvicorn asf.medical.api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("asf.medical.api.main:app", host="0.0.0.0", port=8000, reload=settings.DEBUG)
