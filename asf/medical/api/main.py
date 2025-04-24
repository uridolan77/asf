"""
Unified FastAPI application for the Medical Research Synthesizer API.

This module initializes the FastAPI application and includes all routers.
It provides a comprehensive API for searching, analyzing, and synthesizing medical research literature.
"""

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_redoc_html
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager

from ..storage.models import User
from .dependencies import get_admin_user

from ..core.logging_config import get_logger
from .middleware import MonitoringMiddleware
from ..core.observability import setup_monitoring, get_metrics, run_health_checks
from ..core.service_initialization import initialize_services
from ..core.redis_event_broker import initialize_event_system, shutdown_event_system
from ..core.messaging.initialization import initialize_messaging_system, shutdown_messaging_system
logger = get_logger(__name__)

# Import demo mode flag
from .auth import DEMO_MODE_ENABLED

# Check if middleware modules are available
try:
    # Import middleware modules to check availability
    from .middleware.admin_middleware import add_admin_middleware
    from .middleware.login_rate_limit_middleware import add_login_rate_limit_middleware
    from .middleware.csrf_middleware import add_csrf_middleware
    middleware_available = True
except ImportError as e:
    logger.warning(f"Middleware modules not found: {str(e)}. Some security features will be disabled.")
    middleware_available = False
    # Define dummy functions to avoid errors
    def add_admin_middleware(app, **kwargs):
        pass

    def add_login_rate_limit_middleware(app, **kwargs):
        pass

    def add_csrf_middleware(app, **kwargs):
        pass

from routers.auth import router as auth_router
from routers.search import router as search_router
from .routers.contradiction import router as contradiction_router
from .routers.contradiction_resolution import router as contradiction_resolution_router
from .routers.screening import router as screening_router
from .routers.export import router as export_router
from .routers.analysis import router as analysis_router
from .routers.knowledge_base import router as knowledge_base_router
from .routers.terminology import router as terminology_router
from .routers.clinical_data import router as clinical_data_router
from .routers.async_ml import router as async_ml_router
from .routers.model_cache import router as model_cache_router
from .routers.task_management import router as task_management_router
from .routers.resource_monitoring import router as resource_monitoring_router
from .routers.messaging_tasks import router as messaging_tasks_router
from .routers.websockets import router as websockets_router
from .routers.task_repository_api import router as task_repository_api_router
from .routers.messaging_metrics import router as messaging_metrics_router
from .static import router as static_router
from ..core.config import settings
from ..core.enhanced_cache import EnhancedCacheManager

# Import our ML endpoints and LLM management modules
from .routers import ml_endpoints
from .routers.llm_management import router as llm_management_router

# Create a global cache manager instance
cache_manager = EnhancedCacheManager()
from ..storage.database import init_db
from ..ml.model_registry import model_registry
from ..core.exceptions import DatabaseError

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Application lifespan context manager.

    This function handles the startup and shutdown of the application.
    It initializes all required components during startup and
    properly shuts them down when the application is terminated.

    Args:
        _: FastAPI application instance

    Yields:
        None

    Raises:
        DatabaseError: If there's an error during application startup
    """
    logger.info(f"Starting application in {settings.ENVIRONMENT} environment")

    try:
        # Initialize database
        init_db()
        logger.info("Database initialized successfully")

        # Initialize cache
        if settings.REDIS_URL:
            cache_manager.__init__(
                max_size=10000,  # Increase cache size for production
                redis_url=settings.REDIS_URL,
                default_ttl=settings.CACHE_TTL,
                namespace="asf:medical:"
            )
            logger.info(f"Cache manager initialized with Redis: {settings.REDIS_URL}")
        else:
            logger.info("Cache manager initialized with local LRU cache only")

        # Initialize service registry
        initialize_services()
        logger.info("Service registry initialized")

        # Initialize event system
        await initialize_event_system()
        logger.info("Event system initialized")

        # Initialize messaging system if RabbitMQ is enabled
        if settings.RABBITMQ_ENABLED:
            await initialize_messaging_system()
            logger.info("Messaging system initialized")
        else:
            logger.info("RabbitMQ messaging is disabled")

        # Initialize model registry
        model_registry.initialize(use_gpu=settings.USE_GPU)
        logger.info(f"Model registry initialized with GPU support: {settings.USE_GPU}")

        # Setup monitoring
        setup_monitoring()
        logger.info("Monitoring initialized")

        # Setup exception handlers will be implemented later
        logger.info("Exception handlers will be implemented later")

        logger.info("Application startup complete")
        logger.info("API documentation available at: /docs and /redoc")
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise DatabaseError(f"Error during application startup: {str(e)}")

    yield

    logger.info("Application shutdown initiated")

    try:
        # Shutdown messaging system if RabbitMQ is enabled
        if settings.RABBITMQ_ENABLED:
            await shutdown_messaging_system()
            logger.info("Messaging system shutdown")

        # Shutdown event system
        await shutdown_event_system()
        logger.info("Event system shutdown")

        # Clear cache
        await cache_manager.clear()
        logger.info("Cache cleared")

        # Unload models
        model_registry.unload_all()
        logger.info("Models unloaded")

        logger.info("Application shutdown completed successfully")
    except Exception as e:
        logger.error(f"Error during application shutdown: {str(e)}")
        # Log error but don't raise during shutdown

app = FastAPI(
    title="Medical Research Synthesizer API",
    description="API for searching, analyzing and synthesizing medical research literature",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/openapi.json",
    lifespan=lifespan
)

def custom_openapi():
    """Generate a custom OpenAPI schema for the application.

    This function extends the default OpenAPI schema with additional
    information, such as security schemes and tags. It is used by
    the FastAPI application to generate the OpenAPI schema.

    Returns:
        dict: The OpenAPI schema
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    if "components" not in openapi_schema:
        openapi_schema["components"] = {}

    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}

    openapi_schema["components"]["securitySchemes"]["Bearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter JWT token",
    }

    openapi_schema["security"] = [{"Bearer": []}]

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
            "name": "terminology",
            "description": "Medical terminology operations including SNOMED CT access",
        },
        {
            "name": "clinical-data",
            "description": "Integrated clinical data operations connecting terminology with clinical trials",
        },
        {
            "name": "contradiction",
            "description": "Contradiction detection between research claims",
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(MonitoringMiddleware)

if middleware_available:
    add_admin_middleware(app, admin_path_patterns=[
        "/cache/",
        "/metrics",
        "/model-cache/",
        "/task-management/"
    ])

    add_login_rate_limit_middleware(
        app,
        login_path="/v1/auth/token",
        rate=5,  # 5 attempts per minute
        burst=3,  # 3 attempts in a burst
        window=60,  # 1 minute window
        block_time=300  # 5 minutes block time after too many attempts
    )

    add_csrf_middleware(
        app,
        cookie_name="csrf_token",
        header_name="X-CSRF-Token",
        cookie_secure=not settings.DEBUG,  # Secure in production, not in development
        cookie_httponly=False,  # Must be accessible by JavaScript
        cookie_samesite="lax",
        exempt_paths=[
            "/docs",
            "/redoc",
            "/openapi.json",
            "/health",
            "/v1/auth/token"  # Exempt login endpoint
        ]
    )
else:
    logger.warning("Security middleware not available. Running with reduced security.")

app.include_router(auth_router)
app.include_router(search_router, prefix=settings.API_V1_STR, tags=["search"])
app.include_router(contradiction_router, prefix=settings.API_V1_STR, tags=["contradiction"])
app.include_router(contradiction_resolution_router, prefix=settings.API_V1_STR, tags=["contradiction-resolution"])
app.include_router(screening_router, prefix=settings.API_V1_STR, tags=["screening"])
app.include_router(export_router, prefix=settings.API_V1_STR, tags=["export"])
app.include_router(analysis_router, prefix=settings.API_V1_STR, tags=["analysis"])
app.include_router(knowledge_base_router, prefix=settings.API_V1_STR, tags=["knowledge_base"])
app.include_router(terminology_router, prefix=settings.API_V1_STR, tags=["terminology"])
app.include_router(clinical_data_router, prefix=settings.API_V1_STR, tags=["clinical-data"])
app.include_router(async_ml_router, prefix=settings.API_V1_STR, tags=["async-ml"])
app.include_router(model_cache_router)
app.include_router(task_management_router)
app.include_router(resource_monitoring_router)
app.include_router(messaging_tasks_router)
app.include_router(websockets_router)
app.include_router(task_repository_api_router)
app.include_router(messaging_metrics_router)
app.include_router(static_router)

# Register the ML endpoints and LLM management routers with the app
app.include_router(ml_endpoints.router)
app.include_router(ml_endpoints.medical_router)  # Register the new medical ML router
app.include_router(llm_management_router, tags=["llm-management"])

@app.get("/", tags=["status"])
async def root():
    """Root endpoint that returns the API health status.

    This endpoint performs health checks on various components of the system
    and returns their status. If all checks pass, it returns a 200 OK response.
    If any check fails, it returns a 503 Service Unavailable response.

    Returns:
        JSON response with health check results
    """
    if DEMO_MODE_ENABLED:
        # Special demo mode indicator HTML
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
            <head>
                <title>Medical Research Synthesizer API - DEMO MODE</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        margin: 0;
                        padding: 2rem;
                        background-color: #f5f5f5;
                    }
                    .container {
                        max-width: 800px;
                        margin: 0 auto;
                        background-color: white;
                        padding: 2rem;
                        border-radius: 0.5rem;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    }
                    .demo-badge {
                        display: inline-block;
                        background-color: #ff4500;
                        color: white;
                        padding: 0.5rem 1rem;
                        border-radius: 0.25rem;
                        font-weight: bold;
                        margin-bottom: 1rem;
                    }
                    h1 {
                        color: #333;
                        margin-top: 0;
                    }
                    a {
                        color: #0066cc;
                        text-decoration: none;
                    }
                    a:hover {
                        text-decoration: underline;
                    }
                    .button {
                        display: inline-block;
                        background-color: #0066cc;
                        color: white;
                        padding: 0.75rem 1.5rem;
                        border-radius: 0.25rem;
                        text-decoration: none;
                        margin-top: 1rem;
                        font-weight: bold;
                    }
                    .button:hover {
                        background-color: #0052a3;
                        text-decoration: none;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="demo-badge">DEMO MODE</div>
                    <h1>Medical Research Synthesizer API</h1>
                    <p>Welcome to the Medical Research Synthesizer API. This API is currently running in DEMO MODE, which means:</p>
                    <ul>
                        <li>Authentication is bypassed - you are automatically authenticated as an admin user</li>
                        <li>All endpoints are accessible without login credentials</li>
                        <li>This is for demonstration purposes only</li>
                    </ul>
                    <p>You can access the API documentation and explore the available endpoints:</p>
                    <a href="/docs" class="button">API Documentation</a>
                </div>
            </body>
        </html>
        """, status_code=200)
    else:
        health_checks = run_health_checks()
        all_ok = all(check.get("status") == "ok" for check in health_checks.values())

        if all_ok:
            return {"status": "ok", "checks": health_checks}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "checks": health_checks}
            )

@app.get("/cache/stats", tags=["admin"])
async def cache_stats(_: User = Depends(get_admin_user)):
    """Get cache statistics.

    This endpoint returns statistics about the cache, such as hit rate,
    miss rate, and size. It requires admin privileges to access.

    Args:
        _: User object (admin user required)

    Returns:
        JSON response with cache statistics
    """
    stats = await cache_manager.get_stats()
    return {"status": "ok", "stats": stats}

@app.get("/metrics", tags=["admin"])
async def metrics(_: User = Depends(get_admin_user)):
    """Get system metrics.

    This endpoint returns various metrics about the system, such as
    request counts, response times, and error rates. It requires
    admin privileges to access.

    Args:
        _: User object (admin user required)

    Returns:
        JSON response with system metrics
    """
    metrics_data = get_metrics()
    return {"status": "ok", "metrics": metrics_data}

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint.

    This endpoint serves the ReDoc UI for API documentation.
    It is not included in the OpenAPI schema.

    Returns:
        HTML response with ReDoc UI
    """
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """OpenAPI schema endpoint.

    This endpoint returns the OpenAPI schema for the API.
    It is not included in the OpenAPI schema itself.

    Returns:
        JSON response with OpenAPI schema
    """
    return app.openapi()