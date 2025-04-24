from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Depends, HTTPException, status, Query, Body, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
import jwt
import os
import sys
import json
import uuid
import random
import logging
import httpx

# Create router
logger = logging.getLogger(__name__)

# API URLs
MEDICAL_API_URL = os.getenv("MEDICAL_API_URL", "http://localhost:8000")
router = APIRouter()

# Add the project root directory to sys.path to import the asf module
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import routers
from api.routers.medical_kb import router as medical_kb_router
from api.routers.medical_search import router as medical_search_router
from api.routers.medical_contradiction import router as medical_contradiction_router
from api.routers.medical_terminology import router as medical_terminology_router
from api.routers.enhanced_medical_contradiction import router as enhanced_medical_contradiction_router
from api.routers.medical_clinical_data import router as medical_clinical_data_router
from api.routers.llm_gateway import router as llm_gateway_router
from api.routers.document_processing import router as document_processing_router
from api.routers.dspy import router as dspy_router  # Import the DSPy router
from api.routers.llm.mcp import router as mcp_router  # Import the MCP router
from api.routers.llm.service import router as service_router  # Import the Service router
from api.websockets import router as websocket_router  # Import the WebSocket router

# Import config routers
from api.routers.config.provider_router import router as provider_router
from api.routers.config.configuration_router import router as configuration_router
from api.routers.config.user_provider_router import router as user_provider_router

# Import clients router
from api.clients import router as clients_router

# Import repositories
from repositories.user_repository import UserRepository

from asf.bollm.backend.models.user import User, Role
from asf.bollm.backend.models.base import Base
from config.config import SessionLocal, engine

SECRET_KEY = os.getenv('BO_SECRET_KEY', 'your-secret-key')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

app = FastAPI()

# Configure CORS - Allow specific origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:5175", "http://localhost:57104", "http://localhost:57054", "http://10.100.102.28:57104", "http://10.100.102.28:57054"],  # Include frontend URLs
    allow_credentials=True,  # Set to True to allow credentials
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(medical_kb_router)
app.include_router(medical_search_router)
app.include_router(medical_contradiction_router)
app.include_router(medical_terminology_router)
app.include_router(enhanced_medical_contradiction_router)
app.include_router(medical_clinical_data_router)
app.include_router(clients_router)
app.include_router(llm_gateway_router)
app.include_router(document_processing_router)
app.include_router(dspy_router)
app.include_router(mcp_router, prefix="/api/llm")
app.include_router(service_router, prefix="/api/llm")
app.include_router(websocket_router)

# Include config routers
app.include_router(provider_router)
app.include_router(configuration_router)
app.include_router(user_provider_router)

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role_id: int

class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role_id: int
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class UserUpdate(BaseModel):
    username: str
    email: str
    current_password: str = None
    new_password: str = None

class SystemSettings(BaseModel):
    sessionTimeout: int
    enableNotifications: bool
    darkMode: bool
    language: str

# Global variable to store system settings (in a real app, this would be in the database)
SYSTEM_SETTINGS = {
    "sessionTimeout": 60,
    "enableNotifications": True,
    "darkMode": False,
    "language": "en"
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_user_by_email(db: Session, email: str):
    user_repo = UserRepository(db)
    return user_repo.get_user_by_email(email)

def authenticate_user(db: Session, email: str, password: str):
    user_repo = UserRepository(db)
    return user_repo.authenticate_user(email, password, pwd_context)

@app.post("/api/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        user_repo = UserRepository(db)
        db_user = user_repo.get_user_by_email(user.email)
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        user_data = {
            "username": user.username,
            "email": user.email,
            "password": user.password,
            "role_id": user.role_id
        }
        new_user = user_repo.create_user(user_data, pwd_context)
        return new_user
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/login", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    print(f"Login attempt with username: {form_data.username}, password: {'*' * len(form_data.password)}")

    try:
        # Use the UserRepository for authentication
        user_repo = UserRepository(db)
        user = user_repo.get_user_by_email(form_data.username)
        
        # Debug output to see what user was found
        if user:
            print(f"Found user in database: {user.username}, email: {user.email}")
        else:
            print(f"No user found in database with email: {form_data.username}")
            
        # Authenticate the user
        authenticated_user = user_repo.authenticate_user(form_data.username, form_data.password, pwd_context)
        
        if not authenticated_user:
            # No fallback to mock users - strict database authentication only
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token = create_access_token(data={"sub": str(authenticated_user.id)})
        return {"access_token": access_token, "token_type": "bearer"}
    except Exception as e:
        print(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.get("/api/me", response_model=dict)
async def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    # Get user from database
    user_repo = UserRepository(db)
    user = user_repo.get_user_by_id(int(user_id))
    
    if not user:
        # Fallback to mock users for development
        import os
        if os.getenv('ENVIRONMENT') == 'development':
            if user_id == "1":
                return {
                    "id": 1,
                    "username": "admin",
                    "email": "admin@example.com",
                    "role_id": 1
                }
            elif user_id == "2":
                return {
                    "id": 2,
                    "username": "user",
                    "email": "user@example.com",
                    "role_id": 2
                }
        
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role_id": user.role_id
    }

@app.put("/api/me", response_model=UserOut)
def update_profile(user_data: UserUpdate, token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Update the current user's profile."""
    # Get current user
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user_repo = UserRepository(db)
    current_user = user_repo.get_user_by_id(int(user_id))
    if current_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if email is being changed to an existing email
    if user_data.email != current_user.email:
        existing_user = user_repo.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

    # Prepare update data
    update_data = {
        "username": user_data.username,
        "email": user_data.email
    }
    
    # Password change logic
    if user_data.new_password:
        if not user_data.current_password:
            raise HTTPException(status_code=400, detail="Current password is required to set a new password")

        # Verify current password
        if not verify_password(user_data.current_password, current_user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Add new password to update data
        update_data["password"] = user_data.new_password

    # Update user
    updated_user = user_repo.update_user(int(user_id), update_data, pwd_context)
    return updated_user

@app.get("/api/roles")
def get_roles(db: Session = Depends(get_db)):
    """Get all available roles."""
    user_repo = UserRepository(db)
    roles = user_repo.get_all_roles()
    return [{"id": role.id, "name": role.name} for role in roles]

# Function to check if user is admin
def get_current_admin_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    user_repo = UserRepository(db)
    user = user_repo.get_user_by_id(int(user_id))
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if user is admin (role_id = 2)
    if user.role_id != 2:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    return user

@app.get("/api/users", response_model=list[UserOut])
def get_users(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get all users (admin only)."""
    user_repo = UserRepository(db)
    users = user_repo.get_all_users()
    return users

@app.post("/api/users", response_model=UserOut)
def create_user(user: UserCreate, admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Create a new user (admin only)."""
    user_repo = UserRepository(db)
    db_user = user_repo.get_user_by_email(user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_data = {
        "username": user.username,
        "email": user.email,
        "password": user.password,
        "role_id": user.role_id
    }
    new_user = user_repo.create_user(user_data, pwd_context)
    return new_user

@app.delete("/api/users/{user_id}", response_model=dict)
def delete_user(user_id: int, admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Delete a user (admin only)."""
    # Check if user trying to delete themselves
    if admin_user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user_repo = UserRepository(db)
    success = user_repo.delete_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")

    return {"message": "User deleted successfully"}

@app.get("/api/stats")
def get_dashboard_stats(current_user: User = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get basic dashboard statistics."""
    # Mock data for dashboard stats
    user_count = 15  # In a real app, this would come from the database
    active_sessions = 8
    system_status = "Operational"

    # Generate mock monthly data
    monthly_data = [
        {"month": "Jan", "searches": 45, "analyses": 18},
        {"month": "Feb", "searches": 52, "analyses": 22},
        {"month": "Mar", "searches": 49, "analyses": 25},
        {"month": "Apr", "searches": 63, "analyses": 30},
        {"month": "May", "searches": 55, "analyses": 24}
    ]

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "system_status": system_status,
        "monthly_data": monthly_data
    }

@app.get("/api/admin/stats")
def get_admin_stats(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get detailed system statistics (admin only)."""
    # Count all users
    user_count = db.query(User).count()

    # For demo purposes, we'll simulate active sessions
    # In a real app, you'd track this in a sessions table or using Redis
    import random
    active_sessions = random.randint(1, user_count)

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "admin_data": {
            "api_calls_today": random.randint(500, 2000),
            "avg_response_time": round(random.uniform(0.1, 0.5), 2),
            "errors_today": random.randint(0, 50),
            "server_load": round(random.uniform(10, 60), 1)
        }
    }

@app.get("/api/settings")
def get_system_settings(admin_user: User = Depends(get_current_admin_user)):
    """Get system settings (admin only)."""
    return SYSTEM_SETTINGS

@app.put("/api/settings")
def update_system_settings(settings: SystemSettings, admin_user: User = Depends(get_current_admin_user)):
    """Update system settings (admin only)."""
    # Update global settings
    SYSTEM_SETTINGS["sessionTimeout"] = settings.sessionTimeout
    SYSTEM_SETTINGS["enableNotifications"] = settings.enableNotifications
    SYSTEM_SETTINGS["darkMode"] = settings.darkMode
    SYSTEM_SETTINGS["language"] = settings.language

    # In a real application, this would be saved to a database

    return SYSTEM_SETTINGS

# Add a test endpoint that doesn't require authentication
@app.get("/api/dashboard-test")
def get_dashboard_test_data():
    """Get test dashboard data without authentication."""
    # Mock data for dashboard stats
    user_count = 15
    active_sessions = 8
    system_status = "Operational"

    # Generate mock monthly data
    monthly_data = [
        {"month": "Jan", "searches": 45, "analyses": 18},
        {"month": "Feb", "searches": 52, "analyses": 22},
        {"month": "Mar", "searches": 49, "analyses": 25},
        {"month": "Apr", "searches": 63, "analyses": 30},
        {"month": "May", "searches": 55, "analyses": 24}
    ]

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "system_status": system_status,
        "monthly_data": monthly_data
    }

@app.get("/api/research-metrics")
def get_research_metrics(current_user: User = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get research metrics for dashboard charts."""
    # Mock data for research metrics
    metrics = [
        {"category": "Clinical Trials", "count": 145},
        {"category": "Meta-analyses", "count": 72},
        {"category": "Systematic Reviews", "count": 98},
        {"category": "Cohort Studies", "count": 123},
        {"category": "Case Reports", "count": 86}
    ]

    return metrics

@app.get("/api/recent-updates")
def get_recent_updates(current_user: User = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get recent research updates for the dashboard."""
    # Mock data for recent updates
    updates = {
        "last_updated": datetime.utcnow().isoformat(),
        "items": [
            {
                "title": "Procalcitonin-guided antibiotic therapy in CAP shows promising results",
                "date": "Apr 2025",
                "link": "/analysis/123"
            },
            {
                "title": "New data on antibiotic resistance patterns in Streptococcus pneumoniae",
                "date": "Mar 2025",
                "link": "/knowledge-base/456"
            },
            {
                "title": "Post-COVID patterns in respiratory infections suggest modified treatment approaches",
                "date": "Feb 2025",
                "link": "/clinical-data/789"
            },
            {
                "title": "Machine learning model improves early detection of sepsis in pneumonia patients",
                "date": "Feb 2025",
                "link": "/ml-services/012"
            },
            {
                "title": "Updated guidelines for CAP management published by international consortium",
                "date": "Jan 2025",
                "link": "/knowledge-base/345"
            }
        ]
    }

    return updates

@app.get("/api/llm-usage")
async def get_llm_usage(current_user: User = Depends(oauth2_scheme)):
    """Get LLM usage statistics for the dashboard."""
    try:
        # Call the medical API to get the LLM usage statistics
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/llm/usage",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get LLM usage from API: {response.status_code}")
                # Return fallback mock data with generic model names
                usage = [
                    { "model": "provider1-model1", "usage_count": 2580 },
                    { "model": "provider2-model1", "usage_count": 1420 },
                    { "model": "provider3-model1", "usage_count": 3850 },
                    { "model": "provider4-model1", "usage_count": 980 }
                ]
                return {"usage": usage}
            
            # Process the response to match the expected format
            data = response.json()
            if isinstance(data, list):
                # If the API returns a list directly, use it
                return {"usage": data}
            elif "usage" in data:
                # If the API returns a dict with a usage key, use that
                return data
            else:
                # Try to extract model usage from the response
                usage = []
                for model, count in data.get("components", {}).get("gateway", {}).get("models", {}).items():
                    usage.append({"model": model, "usage_count": count})
                return {"usage": usage}
    except Exception as e:
        logger.error(f"Error getting LLM usage: {str(e)}")
        # Return fallback mock data with generic model names
        usage = [
            { "model": "provider1-model1", "usage_count": 2580 },
            { "model": "provider2-model1", "usage_count": 1420 },
            { "model": "provider3-model1", "usage_count": 3850 },
            { "model": "provider4-model1", "usage_count": 980 }
        ]
        return {"usage": usage}

@app.get("/api/public/stats")
def get_public_dashboard_stats():
    """Get basic dashboard statistics without requiring authentication."""
    # Mock data for dashboard stats
    user_count = 15
    active_sessions = 8
    system_status = "Operational"

    # Generate mock monthly data
    monthly_data = [
        {"month": "Jan", "searches": 45, "analyses": 18},
        {"month": "Feb", "searches": 52, "analyses": 22},
        {"month": "Mar", "searches": 49, "analyses": 25},
        {"month": "Apr", "searches": 63, "analyses": 30},
        {"month": "May", "searches": 55, "analyses": 24}
    ]

    return {
        "user_count": user_count,
        "active_sessions": active_sessions,
        "system_status": system_status,
        "monthly_data": monthly_data
    }

@app.get("/api/public/research-metrics")
def get_public_research_metrics():
    """Get research metrics for dashboard charts without requiring authentication."""
    # Mock data for research metrics
    metrics = [
        {"category": "Clinical Trials", "count": 145},
        {"category": "Meta-analyses", "count": 72},
        {"category": "Systematic Reviews", "count": 98},
        {"category": "Cohort Studies", "count": 123},
        {"category": "Case Reports", "count": 86}
    ]

    return metrics

@app.get("/api/public/recent-updates")
def get_public_recent_updates():
    """Get recent research updates for the dashboard without requiring authentication."""
    # Mock data for recent updates
    updates = {
        "last_updated": datetime.utcnow().isoformat(),
        "items": [
            {
                "title": "Procalcitonin-guided antibiotic therapy in CAP shows promising results",
                "date": "Apr 2025",
                "link": "/analysis/123"
            },
            {
                "title": "New data on antibiotic resistance patterns in Streptococcus pneumoniae",
                "date": "Mar 2025",
                "link": "/knowledge-base/456"
            },
            {
                "title": "Post-COVID patterns in respiratory infections suggest modified treatment approaches",
                "date": "Feb 2025",
                "link": "/clinical-data/789"
            },
            {
                "title": "Machine learning model improves early detection of sepsis in pneumonia patients",
                "date": "Feb 2025",
                "link": "/ml-services/012"
            },
            {
                "title": "Updated guidelines for CAP management published by international consortium",
                "date": "Jan 2025",
                "link": "/knowledge-base/345"
            }
        ]
    }

    return updates

# Models for Medical Research API
class PICOSearchRequest(BaseModel):
    condition: str
    interventions: List[str]
    outcomes: List[str]
    population: Optional[str] = None
    study_design: Optional[str] = None
    years: Optional[int] = None
    max_results: int = 20

class ContradictionAnalysisRequest(BaseModel):
    query: str
    max_results: int = 20
    threshold: float = 0.7
    use_biomedlm: bool = True
    use_tsmixer: bool = False
    use_lorentz: bool = False

class ScreeningRequest(BaseModel):
    query: str
    max_results: int = 20
    stage: str = "screening"
    criteria: Optional[Dict[str, List[str]]] = None

class BiasAssessmentRequest(BaseModel):
    query: str
    max_results: int = 20
    domains: Optional[List[str]] = None

class KnowledgeBaseRequest(BaseModel):
    name: str
    query: str
    update_schedule: str = "weekly"

class ExportRequest(BaseModel):
    result_id: Optional[str] = None
    query: Optional[str] = None
    max_results: int = 20

# Medical API endpoints
@app.post("/api/medical/search")
def search_medical(
    query: str = Body(...),
    max_results: int = Body(20),
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Search for medical research based on a query.
    """
    try:
        # Import the NCBI client to perform real searches
        from medical.clients.ncbi import NCBIClient

        # Create a client instance
        ncbi_client = NCBIClient()

        # Perform the actual search using the client
        # This assumes there's a search method in the NCBIClient class
        search_results = ncbi_client.search(query, max_results=max_results)

        # Process and format the results
        articles = []

        for result in search_results:
            article = {
                "id": result.get("id", f"ncbi_{len(articles)}"),
                "title": result.get("title", "Untitled"),
                "authors": result.get("authors", []),
                "journal": result.get("journal", "Unknown Journal"),
                "year": result.get("year", datetime.now().year),
                "abstract": result.get("abstract", "No abstract available"),
                "relevance_score": result.get("score", 0.8),
                "source": "NCBI"
            }
            articles.append(article)

        return {
            "success": True,
            "message": f"Found {len(articles)} results for query: {query}",
            "data": {
                "articles": articles,
                "query": query,
                "total_results": len(articles)
            }
        }
    except ImportError as e:
        print(f"Import error: {str(e)}")
        # Fallback to mock data if the client import fails
        articles = [
            {
                "id": f"article_{i}",
                "title": f"Research on {query} - Part {i}",
                "authors": ["Author A", "Author B"],
                "journal": "Journal of Medical Research",
                "year": 2023 - i,
                "abstract": f"This study investigates {query} and its implications for healthcare.",
                "relevance_score": round(random.uniform(0.5, 0.99), 2),
                "source": "Mock Data (Client import failed)"
            }
            for i in range(1, min(max_results + 1, 21))
        ]

        return {
            "success": True,
            "message": f"Found {len(articles)} results for query: {query} (using mock data due to import error: {str(e)})",
            "data": {
                "articles": articles,
                "query": query,
                "total_results": len(articles)
            }
        }
    except Exception as e:
        print(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error performing medical search: {str(e)}"
        )

@app.post("/api/medical/search/pico")
def search_pico(
    request: PICOSearchRequest,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Search for medical research using PICO framework.
    """
    # Simulate searching using PICO components
    pico_description = f"P: {request.population or request.condition}, " \
                       f"I: {', '.join(request.interventions)}, " \
                       f"O: {', '.join(request.outcomes)}"

    # In a real implementation, this would call the medical API
    # For now, we'll just return mock data

    # Save search to history
    search_id = str(uuid.uuid4())

    # Return mock results
    return {
        "success": True,
        "message": f"Found 5 results for PICO search: {pico_description}",
        "data": {
            "total_count": 5,
            "page": 1,
            "page_size": 20,
            "results": [
                {
                    "id": f"result_{i}",
                    "title": f"PICO Study {i} on {request.condition}",
                    "authors": ["Author A", "Author B"],
                    "journal": "Journal of Medical Research",
                    "year": 2025 - i,
                    "abstract": f"This study examines {request.condition} with {', '.join(request.interventions)}...",
                    "relevance_score": 0.95 - (i * 0.1)
                } for i in range(5)
            ]
        }
    }

@app.get("/api/medical/search/history")
async def get_search_history(current_user: User = Depends(oauth2_scheme)):
    """
    Get search history for the current user.
    """
    try:
        # Call the medical API to get the search history
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{MEDICAL_API_URL}/api/search/history",
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to get search history from medical API: {response.status_code}")
                # Return fallback mock data
                searches = [
                    {
                        "id": "search_1",
                        "query": "pneumonia treatment",
                        "type": "standard",
                        "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                        "result_count": 42
                    },
                    {
                        "id": "search_2",
                        "query": "P: adults with hypertension, I: ACE inhibitors, C: ARBs, O: blood pressure reduction",
                        "type": "pico",
                        "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                        "result_count": 15
                    },
                    {
                        "id": "search_3",
                        "query": "diabetes management",
                        "type": "standard",
                        "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                        "result_count": 78
                    }
                ]
                return {"searches": searches}

            return response.json()
    except Exception as e:
        logger.error(f"Error getting search history: {str(e)}")
        # Return fallback mock data
        searches = [
            {
                "id": "search_1",
                "query": "pneumonia treatment",
                "type": "standard",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "result_count": 42
            },
            {
                "id": "search_2",
                "query": "P: adults with hypertension, I: ACE inhibitors, C: ARBs, O: blood pressure reduction",
                "type": "pico",
                "timestamp": (datetime.now() - timedelta(days=3)).isoformat(),
                "result_count": 15
            },
            {
                "id": "search_3",
                "query": "diabetes management",
                "type": "standard",
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                "result_count": 78
            }
        ]
        return {"searches": searches}

@app.post("/api/medical/search/history")
async def save_search_history(search: dict = Body(...), current_user: User = Depends(oauth2_scheme)):
    """
    Save a search to the user's history.
    """
    try:
        # Call the medical API to save the search history
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{MEDICAL_API_URL}/api/search/history",
                json=search,
                headers={"Authorization": f"Bearer {current_user.token}"}
            )

            if response.status_code != 200:
                logger.warning(f"Failed to save search history to medical API: {response.status_code}")
                # Return fallback mock response
                search_id = str(uuid.uuid4())
                return {
                    "success": True,
                    "message": "Search saved to history (local)",
                    "data": {
                        "id": search_id,
                        "query": search.get("query", ""),
                        "type": search.get("type", "standard"),
                        "timestamp": search.get("timestamp", datetime.now(datetime.timezone.utc).isoformat())
                    }
                }

            return response.json()
    except Exception as e:
        logger.error(f"Error saving search history: {str(e)}")
        # Return fallback mock response
        search_id = str(uuid.uuid4())
        return {
            "success": True,
            "message": "Search saved to history (local)",
            "data": {
                "id": search_id,
                "query": search.get("query", ""),
                "type": search.get("type", "standard"),
                "timestamp": search.get("timestamp", datetime.now(datetime.timezone.utc).isoformat())
            }
        }

@app.post("/api/medical/analysis/contradictions")
def analyze_contradictions(
    request: ContradictionAnalysisRequest,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Analyze contradictions in medical research for a given query.
    """
    # Mock data for contradictions analysis
    models_used = []
    if request.use_biomedlm:
        models_used.append("BioMedLM")
    if request.use_tsmixer:
        models_used.append("TSMixer")
    if request.use_lorentz:
        models_used.append("Lorentz")

    contradiction_pairs = [
        {
            "article1": {
                "id": f"article_a_{i}",
                "title": f"Study supporting {request.query} - Part {i}",
                "claim": f"Evidence shows that {request.query} is effective for treating certain conditions."
            },
            "article2": {
                "id": f"article_b_{i}",
                "title": f"Study refuting {request.query} - Part {i}",
                "claim": f"No significant evidence was found to support {request.query} as an effective treatment."
            },
            "contradiction_score": round(random.uniform(request.threshold, 0.99), 2),
            "explanation": f"These studies present contradictory findings about the efficacy of {request.query}."
        }
        for i in range(1, min(request.max_results + 1, 11))
    ]

    return {
        "success": True,
        "message": f"Identified {len(contradiction_pairs)} contradiction pairs",
        "data": {
            "contradiction_pairs": contradiction_pairs,
            "query": request.query,
            "threshold": request.threshold,
            "models_used": models_used
        }
    }

@app.get("/api/medical/analysis/cap")
def analyze_cap(
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get analysis specific to Community Acquired Pneumonia (CAP).
    """
    # Mock data for CAP analysis
    treatments = [
        {
            "name": "Amoxicillin",
            "efficacy_score": 0.85,
            "recommendation_level": "Strong",
            "patient_groups": ["Adults", "Children > 5 years"],
            "contraindications": ["Penicillin allergy"]
        },
        {
            "name": "Azithromycin",
            "efficacy_score": 0.78,
            "recommendation_level": "Moderate",
            "patient_groups": ["Adults", "Children > 2 years"],
            "contraindications": ["Macrolide allergy", "Certain cardiac conditions"]
        },
        {
            "name": "Respiratory support",
            "efficacy_score": 0.92,
            "recommendation_level": "Strong",
            "patient_groups": ["All patients with respiratory distress"],
            "contraindications": []
        },
        {
            "name": "Corticosteroids",
            "efficacy_score": 0.65,
            "recommendation_level": "Conditional",
            "patient_groups": ["Severe cases", "Patients with inflammatory response"],
            "contraindications": ["Active untreated infections"]
        }
    ]

    diagnostic_criteria = [
        {
            "criterion": "Chest X-ray confirmation",
            "sensitivity": 0.87,
            "specificity": 0.83,
            "recommendation": "Strongly recommended for diagnosis"
        },
        {
            "criterion": "Clinical symptoms (fever, cough, dyspnea)",
            "sensitivity": 0.92,
            "specificity": 0.61,
            "recommendation": "Essential for initial assessment"
        },
        {
            "criterion": "Sputum culture",
            "sensitivity": 0.65,
            "specificity": 0.95,
            "recommendation": "Recommended for pathogen identification"
        },
        {
            "criterion": "Blood tests (WBC count, CRP)",
            "sensitivity": 0.81,
            "specificity": 0.74,
            "recommendation": "Recommended to assess severity"
        }
    ]

    recent_findings = [
        {
            "title": "Antibiotic resistance trends in CAP",
            "summary": "Increasing resistance to macrolides observed in Streptococcus pneumoniae isolates.",
            "year": 2024,
            "impact": "High",
            "source": "International Journal of Antimicrobial Agents"
        },
        {
            "title": "Procalcitonin-guided therapy in CAP",
            "summary": "Procalcitonin-guided antibiotic therapy reduced antibiotic exposure without affecting outcomes.",
            "year": 2024,
            "impact": "Moderate",
            "source": "American Journal of Respiratory and Critical Care Medicine"
        },
        {
            "title": "CAP in the post-COVID era",
            "summary": "Changes in pathogen distribution and disease severity noted since the COVID-19 pandemic.",
            "year": 2023,
            "impact": "High",
            "source": "Lancet Respiratory Medicine"
        }
    ]

    return {
        "success": True,
        "message": "Retrieved CAP analysis data",
        "data": {
            "treatments": treatments,
            "diagnostic_criteria": diagnostic_criteria,
            "recent_findings": recent_findings,
            "meta": {
                "last_updated": "2025-04-15",
                "guidelines_source": "Infectious Diseases Society of America / American Thoracic Society"
            }
        }
    }

@app.post("/api/medical/screening/prisma")
def screen_articles(
    request: ScreeningRequest,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Screen articles using PRISMA framework.
    """
    # Mock data for article screening
    stages = {
        "identification": "Initial database search",
        "screening": "Title/abstract screening",
        "eligibility": "Full-text assessment",
        "included": "Final inclusion"
    }

    articles = [
        {
            "id": f"screening_{i}",
            "title": f"Screening Study on {request.query} - Part {i}",
            "source": "PubMed",
            "year": 2023 - i % 5,
            "stage": request.stage,
            "status": random.choice(["included", "excluded"]),
            "exclusion_reason": None if random.random() > 0.4 else random.choice([
                "Wrong population", "Wrong intervention", "Wrong outcome", "Wrong study design"
            ]),
            "screening_score": round(random.uniform(0.5, 0.99), 2)
        }
        for i in range(1, min(request.max_results + 1, 21))
    ]

    # Filter out excluded articles if at the included stage
    if request.stage == "included":
        articles = [a for a in articles if a["status"] == "included"]

    prisma_stats = {
        "identification": random.randint(200, 500),
        "screening": random.randint(100, 200),
        "eligibility": random.randint(30, 100),
        "included": len([a for a in articles if a["status"] == "included"])
    }

    return {
        "success": True,
        "message": f"Screened articles at {stages.get(request.stage, request.stage)} stage",
        "data": {
            "articles": articles,
            "query": request.query,
            "stage": request.stage,
            "prisma_stats": prisma_stats,
            "total_results": len(articles)
        }
    }

@app.post("/api/medical/screening/bias-assessment")
def assess_bias(
    request: BiasAssessmentRequest,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Assess bias in medical research articles.
    """
    # Default domains if not provided
    domains = request.domains or [
        "Selection bias",
        "Performance bias",
        "Detection bias",
        "Attrition bias",
        "Reporting bias"
    ]

    articles = [
        {
            "id": f"bias_{i}",
            "title": f"Bias Assessment Study on {request.query} - Part {i}",
            "journal": "Journal of Evidence-Based Medicine",
            "year": 2023 - i % 5,
            "bias_assessment": {
                domain: {
                    "risk": random.choice(["Low", "Moderate", "High"]),
                    "explanation": f"Assessment of {domain.lower()} for this study."
                }
                for domain in domains
            },
            "overall_risk": random.choice(["Low", "Moderate", "High"]),
            "assessment_tool": "Cochrane Risk of Bias Tool"
        }
        for i in range(1, min(request.max_results + 1, 21))
    ]

    summary = {
        domain: {
            "Low": len([a for a in articles if a["bias_assessment"][domain]["risk"] == "Low"]),
            "Moderate": len([a for a in articles if a["bias_assessment"][domain]["risk"] == "Moderate"]),
            "High": len([a for a in articles if a["bias_assessment"][domain]["risk"] == "High"])
        }
        for domain in domains
    }

    return {
        "success": True,
        "message": f"Assessed bias in {len(articles)} articles",
        "data": {
            "articles": articles,
            "query": request.query,
            "domains": domains,
            "summary": summary,
            "total_results": len(articles)
        }
    }

@app.post("/api/medical/knowledge-base")
def create_knowledge_base(
    request: KnowledgeBaseRequest,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Create a new knowledge base (admin only).
    """
    kb_id = str(uuid.uuid4())

    # In a real implementation, this would save to a database
    # For now, we'll just return a mock response
    return {
        "success": True,
        "message": f"Created knowledge base: {request.name}",
        "data": {
            "id": kb_id,
            "name": request.name,
            "query": request.query,
            "update_schedule": request.update_schedule,
            "created_at": datetime.utcnow().isoformat(),
            "created_by": admin_user.username,
            "article_count": 0
        }
    }

@app.get("/api/medical/knowledge-base")
def list_knowledge_bases(
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    List all knowledge bases.
    """
    # Mock data for knowledge bases
    knowledge_bases = [
        {
            "id": str(uuid.uuid4()),
            "name": f"Knowledge Base {i}",
            "query": f"Medical topic {i}",
            "update_schedule": random.choice(["daily", "weekly", "monthly"]),
            "created_at": (datetime.utcnow() - timedelta(days=i*10)).isoformat(),
            "article_count": random.randint(10, 100),
            "last_updated": (datetime.utcnow() - timedelta(days=i)).isoformat()
        }
        for i in range(1, 6)
    ]

    # Include a CAP-specific knowledge base
    cap_kb = {
        "id": str(uuid.uuid4()),
        "name": "Community Acquired Pneumonia Research",
        "query": "community acquired pneumonia treatment diagnosis",
        "update_schedule": "weekly",
        "created_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
        "article_count": 87,
        "last_updated": (datetime.utcnow() - timedelta(days=2)).isoformat()
    }
    knowledge_bases.append(cap_kb)

    return {
        "success": True,
        "message": f"Retrieved {len(knowledge_bases)} knowledge bases",
        "data": {
            "knowledge_bases": knowledge_bases,
            "total": len(knowledge_bases)
        }
    }

@app.get("/api/medical/knowledge-base/{kb_id}")
def get_knowledge_base(
    kb_id: str,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Get details of a specific knowledge base.
    """
    # For demonstration, create a mock knowledge base
    # In real implementation, this would fetch from a database
    is_cap = "cap" in kb_id.lower()

    kb = {
        "id": kb_id,
        "name": "Community Acquired Pneumonia Research" if is_cap else f"Knowledge Base {kb_id[:4]}",
        "query": "community acquired pneumonia treatment diagnosis" if is_cap else f"Medical topic {kb_id[:4]}",
        "update_schedule": "weekly",
        "created_at": (datetime.utcnow() - timedelta(days=15)).isoformat(),
        "article_count": 87 if is_cap else random.randint(10, 100),
        "last_updated": (datetime.utcnow() - timedelta(days=2)).isoformat(),
        "articles": [
            {
                "id": f"kb_article_{i}",
                "title": f"{'CAP Study ' if is_cap else 'Study '} {i}",
                "journal": "Journal of Respiratory Medicine" if is_cap else "Journal of Medicine",
                "year": 2023 - i % 5,
                "relevance_score": round(random.uniform(0.7, 0.99), 2)
            }
            for i in range(1, 11)
        ],
        "concepts": [
            {
                "id": f"concept_{i}",
                "name": f"{'Pneumonia ' if is_cap else 'Medical '} Concept {i}",
                "related_articles": random.randint(3, 8)
            }
            for i in range(1, 6)
        ]
    }

    # Add CAP-specific concepts if relevant
    if is_cap:
        kb["concepts"].extend([
            {
                "id": "concept_strep_pneumo",
                "name": "Streptococcus pneumoniae",
                "related_articles": 27
            },
            {
                "id": "concept_antibiotics",
                "name": "Antibiotic therapy",
                "related_articles": 42
            },
            {
                "id": "concept_diagnosis",
                "name": "Diagnostic criteria",
                "related_articles": 35
            }
        ])

    return {
        "success": True,
        "message": f"Retrieved knowledge base: {kb['name']}",
        "data": kb
    }

@app.post("/api/medical/knowledge-base/{kb_id}/update")
def update_knowledge_base(
    kb_id: str,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Trigger an update of a knowledge base (admin only).
    """
    # This would trigger a background job in a real application
    return {
        "success": True,
        "message": f"Update triggered for knowledge base ID: {kb_id}",
        "data": {
            "id": kb_id,
            "update_started": datetime.utcnow().isoformat(),
            "status": "in_progress",
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
    }

@app.delete("/api/medical/knowledge-base/{kb_id}")
def delete_knowledge_base(
    kb_id: str,
    admin_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """
    Delete a knowledge base (admin only).
    """
    return {
        "success": True,
        "message": f"Deleted knowledge base ID: {kb_id}",
        "data": {
            "id": kb_id,
            "deleted_at": datetime.utcnow().isoformat()
        }
    }

@app.post("/api/medical/export/{format}")
def export_results(
    format: str,
    request: ExportRequest,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Export search results in various formats.

    Formats supported:
    - csv: Comma-separated values
    - json: JSON format
    - pdf: PDF report (mock)
    - xlsx: Excel spreadsheet (mock)
    """
    if format not in ["csv", "json", "pdf", "xlsx"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

    if not request.result_id and not request.query:
        raise HTTPException(status_code=400, detail="Either result_id or query must be provided")

    # Generate mock export data
    result_id = request.result_id or f"query_{uuid.uuid4()}"
    query = request.query or "Exported search results"

    if format == "csv":
        content = "title,authors,journal,year,abstract\n"
        for i in range(1, min(request.max_results + 1, 11)):
            content += f"Research Article {i},Author A|Author B,Journal of Medicine,{2023-i},Abstract text for article {i}...\n"
        content_type = "text/csv"

    elif format == "json":
        content = json.dumps({
            "query": query,
            "exported_at": datetime.utcnow().isoformat(),
            "total_results": min(request.max_results, 10),
            "articles": [
                {
                    "title": f"Research Article {i}",
                    "authors": ["Author A", "Author B"],
                    "journal": "Journal of Medicine",
                    "year": 2023 - i,
                    "abstract": f"Abstract text for article {i}..."
                }
                for i in range(1, min(request.max_results + 1, 11))
            ]
        })
        content_type = "application/json"

    else:  # pdf or xlsx (mock)
        content = f"This would be a {format.upper()} file in a real implementation"
        content_type = "text/plain"  # Mock

    return {
        "success": True,
        "message": f"Exported results in {format} format",
        "data": {
            "result_id": result_id,
            "query": query,
            "format": format,
            "content_type": content_type,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "exported_at": datetime.utcnow().isoformat(),
            "download_url": f"/api/medical/download/{format}/{result_id}"  # This would be a real URL in implementation
        }
    }

# Client management endpoints
@app.get("/api/medical/clients")
def get_medical_clients(current_user: User = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get all medical clients."""
    # Instead of mock data, use the real client implementations
    try:
        # Import client modules
        from medical.clients.ncbi import NCBIClient
        from medical.clients.umls import UMLSClient
        from medical.clients.clinical_trials_gov import ClinicalTrialsClient
        from medical.clients.cochrane import CochraneClient
        from medical.clients.crossref import CrossrefClient
        from medical.clients.snomed import SnomedClient

        clients = []

        # NCBI client
        try:
            ncbi_client = NCBIClient()
            status = "connected"
            error_message = None
            response_time = 0.8  # This would be measured in a real implementation
        except Exception as e:
            status = "error"
            error_message = str(e)
            response_time = None

        clients.append({
            "client_id": "ncbi",
            "name": "NCBI",
            "description": "National Center for Biotechnology Information",
            "status": status,
            "api_version": ncbi_client.version if status == "connected" else None,
            "last_checked": datetime.utcnow().isoformat(),
            "response_time": response_time,
            "error_message": error_message
        })

        # UMLS client
        try:
            umls_client = UMLSClient()
            status = "connected"
            error_message = None
            response_time = 1.2  # This would be measured in a real implementation
        except Exception as e:
            status = "error"
            error_message = str(e)
            response_time = None

        clients.append({
            "client_id": "umls",
            "name": "UMLS",
            "description": "Unified Medical Language System",
            "status": status,
            "api_version": umls_client.version if status == "connected" else None,
            "last_checked": datetime.utcnow().isoformat(),
            "response_time": response_time,
            "error_message": error_message
        })

        # Add similar implementations for other clients
        # ClinicalTrials.gov, Cochrane, Crossref, SNOMED CT

        return clients
    except ImportError as e:
        print(f"Import error: {str(e)}")
        # Fallback to static data if imports fail
        return [
            {
                "client_id": "ncbi",
                "name": "NCBI",
                "description": "National Center for Biotechnology Information",
                "status": "unknown",
                "api_version": None,
                "last_checked": datetime.utcnow().isoformat(),
                "error_message": f"Failed to import client module: {str(e)}"
            },
            # Other clients with unknown status
        ]
    except Exception as e:
        print(f"Error getting clients: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving medical clients: {str(e)}"
        )

@app.get("/api/medical/clients/{client_id}")
def get_medical_client(client_id: str, current_user: User = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get a specific medical client by ID."""
    # Mock client data based on client_id
    client_data = {
        "ncbi": {
            "client_id": "ncbi",
            "name": "NCBI",
            "description": "National Center for Biotechnology Information",
            "status": "connected",
            "api_version": "2.0.1",
            "last_checked": datetime.utcnow().isoformat(),
            "response_time": 0.82,
            "api_key": "********",
            "base_url": "https://api.ncbi.nlm.nih.gov",
            "rate_limit": 3,
            "rate_limit_period": "second",
            "timeout": 30,
            "retry_count": 3
        },
        "umls": {
            "client_id": "umls",
            "name": "UMLS",
            "description": "Unified Medical Language System",
            "status": "connected",
            "api_version": "3.5.0",
            "last_checked": datetime.utcnow().isoformat(),
            "response_time": 1.24,
            "api_key": "********",
            "base_url": "https://uts-ws.nlm.nih.gov/api",
            "rate_limit": 20,
            "rate_limit_period": "second",
            "timeout": 30,
            "retry_count": 3
        },
        "clinical_trials": {
            "client_id": "clinical_trials",
            "name": "ClinicalTrials.gov",
            "description": "Clinical trials database",
            "status": "connected",
            "api_version": "1.8.5",
            "last_checked": datetime.utcnow().isoformat(),
            "response_time": 1.56,
            "base_url": "https://clinicaltrials.gov/api",
            "rate_limit": 5,
            "rate_limit_period": "second",
            "timeout": 30,
            "retry_count": 3
        },
        "cochrane": {
            "client_id": "cochrane",
            "name": "Cochrane Library",
            "description": "Systematic reviews database",
            "status": "error",
            "api_version": "2.1.0",
            "last_checked": datetime.utcnow().isoformat(),
            "error_message": "API rate limit exceeded",
            "response_time": 3.45,
            "api_key": "********",
            "base_url": "https://www.cochranelibrary.com/api",
            "rate_limit": 2,
            "rate_limit_period": "second",
            "timeout": 30,
            "retry_count": 3
        },
        "crossref": {
            "client_id": "crossref",
            "name": "Crossref",
            "description": "DOI registration agency",
            "status": "connected",
            "api_version": "1.2.3",
            "last_checked": datetime.utcnow().isoformat(),
            "response_time": 0.95,
            "email": "your.email@example.com",
            "base_url": "https://api.crossref.org",
            "rate_limit": 50,
            "rate_limit_period": "second",
            "timeout": 30,
            "retry_count": 3
        },
        "snomed": {
            "client_id": "snomed",
            "name": "SNOMED CT",
            "description": "Clinical terminology",
            "status": "disconnected",
            "last_checked": datetime.utcnow().isoformat(),
            "api_key": "",
            "base_url": "https://browser.ihtsdotools.org/api",
            "rate_limit": 10,
            "rate_limit_period": "second",
            "timeout": 30,
            "retry_count": 3
        }
    }

    if client_id not in client_data:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")

    return client_data[client_id]

@app.put("/api/medical/clients/{client_id}")
def update_medical_client(
    client_id: str,
    client_config: dict = Body(...),
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Update a medical client configuration."""
    # Check if client exists
    if client_id not in ["ncbi", "umls", "clinical_trials", "cochrane", "crossref", "snomed"]:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")

    # In a real app, this would update the client configuration in the database
    # Here we'll just return a mock updated client
    updated_client = {
        "client_id": client_id,
        "name": client_config.get("name", f"Unknown Client {client_id}"),
        "description": client_config.get("description", ""),
        "status": "connected",  # Assume connection is successful after update
        "api_version": client_config.get("api_version", "1.0.0"),
        "last_checked": datetime.utcnow().isoformat(),
        "response_time": random.uniform(0.5, 2.0),
        **client_config
    }

    return updated_client

@app.post("/api/medical/clients/{client_id}/test")
def test_medical_client_connection(
    client_id: str,
    current_user: User = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """Test connection to a medical client."""
    # Check if client exists
    if client_id not in ["ncbi", "umls", "clinical_trials", "cochrane", "crossref", "snomed"]:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")

    # In a real app, this would actually test the connection to the client
    # For now, we'll return mock results with some randomized outcomes

    # Randomly determine success/failure (weighted towards success)
    success = random.random() > 0.2

    result = {
        "success": success,
        "message": f"Successfully connected to {client_id}" if success else f"Connection to {client_id} failed",
        "status": "connected" if success else "error",
        "response_time": round(random.uniform(0.5, 3.0), 2)
    }

    if success:
        result["api_version"] = f"{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    else:
        error_messages = [
            "API key invalid or expired",
            "Connection timeout",
            "Rate limit exceeded",
            "Server returned 503 Service Unavailable",
            "Network error"
        ]
        result["error_message"] = random.choice(error_messages)

    return result
