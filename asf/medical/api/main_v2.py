"""
Main FastAPI application for the Medical Research Synthesizer API.

This module initializes the FastAPI application and includes all routers.
"""

import os
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

from asf.medical.api.auth_unified import (
    Token, User, authenticate_user, create_access_token, 
    get_current_active_user, register_user, users_db,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from asf.medical.api.models import UserRegistrationRequest
from asf.medical.api.routers import search, analysis, knowledge_base, export
from asf.medical.api.dependencies import get_admin_user
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("medical_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("medical_research_api")

# Initialize the API
app = FastAPI(
    title="Medical Research Synthesizer API",
    description="API for searching, analyzing and synthesizing medical research literature",
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable default redoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include our routers
app.include_router(search.router)
app.include_router(analysis.router)
app.include_router(knowledge_base.router)
app.include_router(export.router)

# Authentication routes
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login endpoint."""
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.post("/register", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register_new_user(
    request: UserRegistrationRequest,
    current_user: User = Depends(get_admin_user)
):
    """
    User registration endpoint (only available to admins).
    
    This endpoint allows administrators to register new users.
    """
    if register_user(request.email, request.password):
        return {"message": "User registered successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered"
        )

# Custom docs routes
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    Custom Swagger UI route that requires authentication.
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

# Root endpoint
@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint with basic API information"""
    return {
        "message": "Welcome to the Medical Research Synthesizer API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "search": "/v1/search",
            "pico-search": "/v1/search/pico",
            "analyze-contradictions": "/v1/analysis/contradictions",
            "knowledge-base": "/v1/knowledge-base",
            "export": "/v1/export/{format}"
        }
    }

# Health check endpoint
@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v2:app", host="0.0.0.0", port=8000, reload=True)
