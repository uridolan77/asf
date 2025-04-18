from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

# User models
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

# Medical search models
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
