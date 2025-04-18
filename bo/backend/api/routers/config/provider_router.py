from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
import sys
import os

# Add the backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Use absolute imports
from config.database import get_db
from services.provider_service import ProviderService
from utils.crypto import generate_key
from api.auth import get_current_user
from models.user import User
from schemas.provider import (
    ProviderCreate,
    ProviderUpdate,
    ProviderResponse,
    ApiKeyCreate,
    ApiKeyResponse,
    ConnectionParameterCreate,
    ConnectionParameterResponse
)

router = APIRouter(
    prefix="/api/providers",
    tags=["providers"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Get encryption key (in production, this should be loaded from a secure source)
ENCRYPTION_KEY = generate_key()

# Provider endpoints

@router.get("/", response_model=List[ProviderResponse])
async def get_providers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all providers."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    return service.get_all_providers()

@router.get("/{provider_id}", response_model=ProviderResponse)
async def get_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a provider by ID."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    provider = service.get_provider_by_id(provider_id)
    if not provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with ID {provider_id} not found"
        )
    return provider

@router.post("/", response_model=ProviderResponse, status_code=status.HTTP_201_CREATED)
async def create_provider(
    provider: ProviderCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    return service.create_provider(provider.dict())

@router.put("/{provider_id}", response_model=ProviderResponse)
async def update_provider(
    provider_id: str,
    provider: ProviderUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    updated_provider = service.update_provider(provider_id, provider.dict(exclude_unset=True))
    if not updated_provider:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with ID {provider_id} not found"
        )
    return updated_provider

@router.delete("/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    result = service.delete_provider(provider_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider with ID {provider_id} not found"
        )
    return None

# API Key endpoints

@router.get("/{provider_id}/api-keys", response_model=List[ApiKeyResponse])
async def get_api_keys(
    provider_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all API keys for a provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    return service.get_api_keys_by_provider_id(provider_id)

@router.post("/{provider_id}/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    provider_id: str,
    api_key: ApiKeyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new API key for a provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    api_key_data = api_key.dict()
    api_key_data["provider_id"] = provider_id
    return service.create_api_key(api_key_data)

@router.get("/{provider_id}/api-keys/{key_id}/value")
async def get_api_key_value(
    provider_id: str,
    key_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get the actual API key value (decrypted if necessary)."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    key_value = service.get_api_key_value(key_id)
    if not key_value:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"API key with ID {key_id} not found"
        )
    return {"key_value": key_value}

# Connection Parameter endpoints

@router.get("/{provider_id}/connection-params")
async def get_connection_params(
    provider_id: str,
    include_sensitive: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all connection parameters for a provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    return service.get_provider_connection_params(provider_id, include_sensitive)

@router.post("/{provider_id}/connection-params", response_model=ConnectionParameterResponse, status_code=status.HTTP_201_CREATED)
async def set_connection_param(
    provider_id: str,
    param: ConnectionParameterCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Set a connection parameter for a provider."""
    service = ProviderService(db, ENCRYPTION_KEY, current_user.id)
    param_data = param.dict()
    param_data["provider_id"] = provider_id
    return service.set_connection_parameter(param_data)
