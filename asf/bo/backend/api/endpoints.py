from typing import List, Dict, Optional, Any
from fastapi import FastAPI, Depends, HTTPException, status, Query, Body
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

from models.user import User, Role, Base
from config.config import SessionLocal, engine

SECRET_KEY = os.getenv('BO_SECRET_KEY', 'your-secret-key')
ALGORITHM = 'HS256'
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(medical_kb_router)
app.include_router(medical_search_router)
app.include_router(medical_contradiction_router)
app.include_router(medical_terminology_router)
app.include_router(enhanced_medical_contradiction_router)
app.include_router(medical_clinical_data_router)

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
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, email: str, password: str):
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.password_hash):
        return None
    return user

@app.post("/api/register", response_model=UserOut)
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = get_user_by_email(db, user.email)
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")

        hashed_password = get_password_hash(user.password)
        new_user = User(
            username=user.username,
            email=user.email,
            password_hash=hashed_password,
            role_id=user.role_id
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return new_user
    except Exception as e:
        print(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@app.post("/api/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/me", response_model=UserOut)
def read_users_me(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

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

    # Get user from database
    current_user = db.query(User).filter(User.id == int(user_id)).first()
    if current_user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if email is being changed to an existing email
    if user_data.email != current_user.email:
        existing_user = get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already registered")

    # Password change logic
    if user_data.new_password:
        if not user_data.current_password:
            raise HTTPException(status_code=400, detail="Current password is required to set a new password")

        # Verify current password
        if not verify_password(user_data.current_password, current_user.password_hash):
            raise HTTPException(status_code=400, detail="Current password is incorrect")

        # Update password hash
        current_user.password_hash = get_password_hash(user_data.new_password)

    # Update other fields
    current_user.username = user_data.username
    current_user.email = user_data.email
    current_user.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(current_user)

    return current_user

@app.get("/api/roles")
def get_roles(db: Session = Depends(get_db)):
    """Get all available roles."""
    roles = db.query(Role).all()
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

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    # Check if user is admin (role_id = 2)
    if user.role_id != 2:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    return user

@app.get("/api/users", response_model=list[UserOut])
def get_users(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get all users (admin only)."""
    users = db.query(User).all()
    return users

@app.post("/api/users", response_model=UserOut)
def create_user(user: UserCreate, admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Create a new user (admin only)."""
    db_user = get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_password = get_password_hash(user.password)
    new_user = User(
        username=user.username,
        email=user.email,
        password_hash=hashed_password,
        role_id=user.role_id
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.delete("/api/users/{user_id}", response_model=dict)
def delete_user(user_id: int, admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Delete a user (admin only)."""
    # Check if user trying to delete themselves
    if admin_user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

@app.get("/api/stats")
def get_stats(admin_user: User = Depends(get_current_admin_user), db: Session = Depends(get_db)):
    """Get system statistics (admin only)."""
    # Count all users
    user_count = db.query(User).count()

    # For demo purposes, we'll simulate active sessions
    # In a real app, you'd track this in a sessions table or using Redis
    import random
    active_sessions = random.randint(1, user_count)

    return {
        "user_count": user_count,
        "active_sessions": active_sessions
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
    # In a real implementation, this would call your medical research search service
    # For now, we'll return mock data
    articles = [
        {
            "id": f"article_{i}",
            "title": f"Research on {query} - Part {i}",
            "authors": ["Author A", "Author B"],
            "journal": "Journal of Medical Research",
            "year": 2023 - i,
            "abstract": f"This study investigates {query} and its implications for healthcare.",
            "relevance_score": round(random.uniform(0.5, 0.99), 2)
        }
        for i in range(1, min(max_results + 1, 21))
    ]

    return {
        "success": True,
        "message": f"Found {len(articles)} results for query: {query}",
        "data": {
            "articles": articles,
            "query": query,
            "total_results": len(articles)
        }
    }

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

    articles = [
        {
            "id": f"pico_{i}",
            "title": f"PICO Study on {request.condition} - Part {i}",
            "authors": ["PICO Author A", "PICO Author B"],
            "journal": "PICO Medical Journal",
            "year": 2023 - i,
            "abstract": f"Using the PICO framework, this study examines {request.condition} " \
                       f"with interventions like {', '.join(request.interventions[:1])} " \
                       f"measuring outcomes including {', '.join(request.outcomes[:1])}.",
            "relevance_score": round(random.uniform(0.7, 0.99), 2)
        }
        for i in range(1, min(request.max_results + 1, 21))
    ]

    return {
        "success": True,
        "message": f"Found {len(articles)} results for PICO query",
        "data": {
            "articles": articles,
            "pico_query": pico_description,
            "total_results": len(articles)
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
