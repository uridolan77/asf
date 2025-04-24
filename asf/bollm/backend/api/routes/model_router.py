"""
Model Router - Handles LLM model management operations
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

router = APIRouter()

class ModelBase(BaseModel):
    """Base model for LLM model information"""
    name: str
    display_name: str
    provider_id: str
    description: Optional[str] = None
    model_type: str  # e.g., "chat", "completion", "embedding"
    context_window: Optional[int] = None
    is_active: bool = True

class ModelCreate(ModelBase):
    """Model for creating a new LLM model"""
    pricing: Optional[Dict[str, float]] = None
    capabilities: Optional[List[str]] = None
    additional_config: Optional[Dict[str, Any]] = None

class Model(ModelBase):
    """Model for LLM model information responses"""
    id: str
    provider_name: str
    capabilities: List[str]
    created_at: str
    updated_at: str

    class Config:
        orm_mode = True

class ModelUsage(BaseModel):
    """Model for LLM model usage statistics"""
    model_id: str
    total_requests: int
    total_tokens: int
    average_tokens_per_request: float
    last_used: str

from asf.bollm.backend.repositories.model_repository import ModelRepository
from asf.bollm.backend.config.config import SessionLocal

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.get("/", response_model=List[Model])
async def get_models():
    """
    Get all available LLM models
    """
    # Use the ModelRepository to get models from the database
    db = SessionLocal()
    try:
        repo = ModelRepository(db)
        models = repo.get_all_models()
        
        # Convert to response model
        result = []
        for model in models:
            result.append(Model(
                id=model.id,
                name=model.model_id,
                display_name=model.display_name,
                provider_id=model.provider_id,
                provider_name=model.provider_id,  # We don't have provider_name in our model
                description="",  # We don't have description in our model
                model_type=model.model_type,
                context_window=model.context_window,
                is_active=True,  # We don't have is_active in our model
                capabilities=model.capabilities or [],
                created_at=model.created_at.isoformat() if model.created_at else "",
                updated_at=model.updated_at.isoformat() if model.updated_at else ""
            ))
        
        return result
    finally:
        db.close()

@router.get("/{model_id}", response_model=Model)
async def get_model(model_id: str, provider_id: Optional[str] = None):
    """
    Get details of a specific LLM model
    """
    db = SessionLocal()
    try:
        repo = ModelRepository(db)
        
        # If provider_id is not provided, try to find the model by model_id only
        if provider_id:
            model = repo.get_model_by_id(model_id, provider_id)
        else:
            # Get all models and find the one with matching model_id
            all_models = repo.get_all_models()
            model = next((m for m in all_models if m.model_id == model_id), None)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return Model(
            id=model.id,
            name=model.model_id,
            display_name=model.display_name,
            provider_id=model.provider_id,
            provider_name=model.provider_id,  # We don't have provider_name in our model
            description="",  # We don't have description in our model
            model_type=model.model_type,
            context_window=model.context_window,
            is_active=True,  # We don't have is_active in our model
            capabilities=model.capabilities or [],
            created_at=model.created_at.isoformat() if model.created_at else "",
            updated_at=model.updated_at.isoformat() if model.updated_at else ""
        )
    finally:
        db.close()

@router.get("/provider/{provider_id}", response_model=List[Model])
async def get_models_by_provider(provider_id: str):
    """
    Get all models for a specific provider
    """
    db = SessionLocal()
    try:
        repo = ModelRepository(db)
        models = repo.get_all_models(provider_id=provider_id)
        
        # Convert to response model
        result = []
        for model in models:
            result.append(Model(
                id=model.id,
                name=model.model_id,
                display_name=model.display_name,
                provider_id=model.provider_id,
                provider_name=model.provider_id,  # We don't have provider_name in our model
                description="",  # We don't have description in our model
                model_type=model.model_type,
                context_window=model.context_window,
                is_active=True,  # We don't have is_active in our model
                capabilities=model.capabilities or [],
                created_at=model.created_at.isoformat() if model.created_at else "",
                updated_at=model.updated_at.isoformat() if model.updated_at else ""
            ))
        
        return result
    finally:
        db.close()

@router.post("/", response_model=Model)
async def create_model(model: ModelCreate):
    """
    Create a new LLM model
    """
    # TODO: Implement model creation
    
    # Placeholder response
    return Model(
        id="new-model-1",
        name=model.name,
        display_name=model.display_name,
        provider_id=model.provider_id,
        provider_name="unknown",  # This should be retrieved from the provider ID
        description=model.description,
        model_type=model.model_type,
        context_window=model.context_window,
        is_active=model.is_active,
        capabilities=model.capabilities or [],
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )

@router.put("/{model_id}", response_model=Model)
async def update_model(model_id: str, model_update: ModelCreate):
    """
    Update an existing LLM model
    """
    # TODO: Implement model update
    
    # Placeholder response
    return Model(
        id=model_id,
        name=model_update.name,
        display_name=model_update.display_name,
        provider_id=model_update.provider_id,
        provider_name="unknown",  # This should be retrieved from the provider ID
        description=model_update.description,
        model_type=model_update.model_type,
        context_window=model_update.context_window,
        is_active=model_update.is_active,
        capabilities=model_update.capabilities or [],
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T00:00:00Z"
    )

@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """
    Delete an LLM model
    """
    # TODO: Implement model deletion
    
    return {"status": "success", "message": f"Model {model_id} deleted"}

@router.get("/{model_id}/usage", response_model=ModelUsage)
async def get_model_usage(model_id: str):
    """
    Get usage statistics for a specific model
    """
    # TODO: Implement usage statistics retrieval
    
    # Placeholder response
    return ModelUsage(
        model_id=model_id,
        total_requests=1250,
        total_tokens=125000,
        average_tokens_per_request=100.0,
        last_used="2023-01-15T12:30:45Z"
    )
    return ModelUsage(
        model_id=model_id,
        total_requests=1250,
        total_tokens=125000,
        average_tokens_per_request=100.0,
        last_used="2023-01-15T12:30:45Z"
    )
