"""
Run the backend server with service configurations API enabled.

This script runs the FastAPI server with the service configurations API enabled.
"""

import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Import the service router and database initialization
from asf.bollm.backend.api.routers.llm.service import router as service_router
from asf.bollm.backend.init_service_config_db import init_db

# Import database connection
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Depends
from sqlalchemy.orm import Session

# Get database URL from environment variables or use defaults
DB_USER = os.getenv('BO_DB_USER', 'root')
DB_PASSWORD = os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I')
DB_HOST = os.getenv('BO_DB_HOST', 'localhost')
DB_PORT = os.getenv('BO_DB_PORT', '3306')
DB_NAME = os.getenv('BO_DB_NAME', 'bo_admin')

# Create database URL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create engine and session
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get database session
def get_db():
    """
    Get a database session.
    
    Yields:
        SQLAlchemy session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Service Configurations API",
        description="API for managing LLM service configurations",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods
        allow_headers=["*"],  # Allow all headers
    )
    
    # Include the service router
    app.include_router(service_router, prefix="/api/llm")
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "service": "LLM Service Configurations API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": [
                "/api/llm/service/configurations",
                "/api/llm/service/configurations/{config_id}",
                "/api/llm/service/configurations/{config_id}/apply",
                "/api/llm/service/config",
                "/api/llm/service/health",
                "/api/llm/service/stats",
                "/api/llm/service/cache/clear",
                "/api/llm/service/resilience/reset-circuit-breakers"
            ]
        }
    
    @app.get("/health")
    async def health_check(db: Session = Depends(get_db)):
        """Health check endpoint that verifies database connection."""
        try:
            # Test database connection
            db.execute("SELECT 1")
            db_status = "connected"
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            db_status = f"error: {str(e)}"
        
        return {
            "status": "ok",
            "database": db_status,
            "service": "LLM Service Configurations API"
        }
    
    return app

if __name__ == "__main__":
    # Initialize the database
    init_db()
    
    # Create the app
    app = create_app()
    
    # Get port from environment or use default
    port = int(os.environ.get("SERVICE_CONFIG_PORT", 8000))
    
    # Run the server
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
