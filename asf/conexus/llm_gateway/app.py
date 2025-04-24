"""
Main application for the Conexus LLM Gateway.

This module creates and configures the FastAPI application
for the domain-agnostic LLM Gateway.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from asf.conexus.llm_gateway.api.router import api_router
from asf.conexus.llm_gateway.config.settings import get_settings, Settings
from asf.conexus.llm_gateway.core.client import get_client, initialize_client
from asf.conexus.llm_gateway.db.database import initialize_database
from asf.conexus.llm_gateway.observability.metrics import configure_metrics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("conexus.llm_gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    
    Handles startup and shutdown events for the gateway.
    """
    # Get settings
    settings = get_settings()
    
    # Initialize metrics
    configure_metrics(
        enable_metrics=settings.metrics.enable_metrics,
        enable_prometheus=settings.metrics.enable_prometheus,
        prometheus_port=settings.metrics.prometheus_port,
        service_name="conexus-llm-gateway"
    )
    logger.info("Metrics configured")
    
    # Initialize database
    if settings.database.enabled:
        logger.info(f"Initializing database at {settings.database.url}")
        await initialize_database()
    
    # Initialize cache manager
    from asf.conexus.llm_gateway.cache.cache_manager import get_cache_manager, initialize_cache_manager
    logger.info("Initializing cache manager")
    await initialize_cache_manager()
    cache_manager = get_cache_manager()
    logger.info(f"Cache manager initialized with {len(await cache_manager.semantic_cache.list_entries()) if cache_manager.semantic_cache else 0} entries")
    
    # Initialize client
    logger.info("Initializing LLM Gateway client")
    await initialize_client()
    
    # Load providers
    client = get_client()
    try:
        provider_count = await client.load_providers()
        logger.info(f"Loaded {provider_count} LLM providers")
    except Exception as e:
        logger.error(f"Error loading providers: {e}")
    
    # Application is now ready to handle requests
    logger.info("LLM Gateway startup complete")
    yield
    
    # Shutdown logic
    logger.info("Shutting down LLM Gateway")
    
    # Close cache manager
    logger.info("Closing cache manager")
    try:
        cache_manager = get_cache_manager()
        await cache_manager.close()
    except Exception as e:
        logger.error(f"Error closing cache manager: {e}")
    
    # Clean up client resources
    client = get_client()
    try:
        await client.cleanup()
    except Exception as e:
        logger.error(f"Error during client cleanup: {e}")
        
    # Close any other resources if needed
    # For example, close database connections
    # if settings.database.enabled:
    #     await close_database_connections()


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        The configured FastAPI application
    """
    # Get settings
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title="Conexus LLM Gateway",
        description="A domain-agnostic API gateway for LLM providers",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configure middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.allow_origins,
        allow_credentials=settings.cors.allow_credentials,
        allow_methods=settings.cors.allow_methods,
        allow_headers=settings.cors.allow_headers,
    )
    
    # Add request ID middleware
    @app.middleware("http")
    async def add_request_id(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            import uuid
            request_id = str(uuid.uuid4())
        
        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    
    # Add error handler for unhandled exceptions
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        # Log the exception
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        # Return a generic error response
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred",
                "type": type(exc).__name__
            }
        )
    
    # Include API router
    app.include_router(api_router)
    
    return app


# Main application instance
app = create_application()


def main():
    """Run the application using Uvicorn."""
    settings = get_settings()
    
    # Configure logging level
    log_level = settings.logging.level.upper()
    
    # Run with uvicorn
    uvicorn.run(
        "asf.conexus.llm_gateway.app:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=log_level.lower(),
        reload=settings.server.reload
    )


if __name__ == "__main__":
    main()