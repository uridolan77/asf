"""
Provider API routes for LLM Gateway.

This module provides API endpoints for managing LLM providers, including
CRUD operations for providers, models, API keys, and connection parameters.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body, Query, Path
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
from datetime import datetime

from asf.medical.llm_gateway.services.provider_service import ProviderService
from asf.medical.llm_gateway.api.dependencies import get_db, get_current_user, get_encryption_key

# Define response models
from pydantic import BaseModel, Field

class ProviderBase(BaseModel):
    provider_id: str
    display_name: str
    provider_type: str
    description: Optional[str] = None
    enabled: bool = True
    connection_params: Optional[Dict[str, Any]] = None
    request_settings: Optional[Dict[str, Any]] = None

class ProviderCreate(ProviderBase):
    models: Optional[List[Dict[str, Any]]] = None
    connection_parameters: Optional[List[Dict[str, Any]]] = None
    api_key: Optional[Dict[str, Any]] = None

class ProviderUpdate(BaseModel):
    display_name: Optional[str] = None
    provider_type: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    connection_params: Optional[Dict[str, Any]] = None
    request_settings: Optional[Dict[str, Any]] = None

class ModelBase(BaseModel):
    model_id: str
    provider_id: str
    display_name: str
    model_type: Optional[str] = "chat"
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    enabled: bool = True
    capabilities: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None

class ModelCreate(ModelBase):
    pass

class ModelUpdate(BaseModel):
    display_name: Optional[str] = None
    model_type: Optional[str] = None
    context_window: Optional[int] = None
    max_tokens: Optional[int] = None
    enabled: Optional[bool] = None
    capabilities: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None

class ApiKeyCreate(BaseModel):
    provider_id: str
    key_value: str
    is_encrypted: bool = True
    environment: str = "development"
    expires_at: Optional[datetime] = None

class ConnectionParameterCreate(BaseModel):
    provider_id: str
    param_name: str
    param_value: str
    is_sensitive: bool = False
    environment: str = "development"

# Create router
router = APIRouter(prefix="/providers", tags=["providers"])

logger = logging.getLogger(__name__)

# Provider endpoints

@router.get("/", response_model=List[Dict[str, Any]])
async def get_providers(
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all providers.
    
    This endpoint returns a list of all providers with their models and connection parameters.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.get_all_providers()
    except Exception as e:
        logger.error(f"Error getting providers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get providers: {str(e)}"
        )

@router.get("/{provider_id}", response_model=Dict[str, Any])
async def get_provider(
    provider_id: str = Path(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get a provider by ID.
    
    This endpoint returns a provider with its models and connection parameters.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        provider = service.get_provider_by_id(provider_id)
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )
        return provider
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get provider '{provider_id}': {str(e)}"
        )

@router.post("/", response_model=Dict[str, Any])
async def create_provider(
    provider_data: ProviderCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new provider.
    
    This endpoint creates a new provider with its models and connection parameters.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.create_provider(provider_data.dict())
    except Exception as e:
        logger.error(f"Error creating provider: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create provider: {str(e)}"
        )

@router.put("/{provider_id}", response_model=Dict[str, Any])
async def update_provider(
    provider_id: str = Path(..., description="Provider ID"),
    provider_data: ProviderUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update a provider.
    
    This endpoint updates a provider.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        provider = service.update_provider(provider_id, provider_data.dict(exclude_unset=True))
        if not provider:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )
        return provider
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update provider '{provider_id}': {str(e)}"
        )

@router.delete("/{provider_id}", response_model=Dict[str, Any])
async def delete_provider(
    provider_id: str = Path(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a provider.
    
    This endpoint deletes a provider.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        result = service.delete_provider(provider_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found"
            )
        return {"provider_id": provider_id, "message": "Provider deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete provider '{provider_id}': {str(e)}"
        )

# Model endpoints

@router.get("/{provider_id}/models", response_model=List[Dict[str, Any]])
async def get_models(
    provider_id: str = Path(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all models for a provider.
    
    This endpoint returns a list of all models for a provider.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.get_models_by_provider_id(provider_id)
    except Exception as e:
        logger.error(f"Error getting models for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models for provider '{provider_id}': {str(e)}"
        )

@router.get("/{provider_id}/models/{model_id}", response_model=Dict[str, Any])
async def get_model(
    provider_id: str = Path(..., description="Provider ID"),
    model_id: str = Path(..., description="Model ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get a model by ID.
    
    This endpoint returns a model.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        model = service.get_model_by_id(model_id, provider_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model '{model_id}' for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model '{model_id}' for provider '{provider_id}': {str(e)}"
        )

@router.post("/{provider_id}/models", response_model=Dict[str, Any])
async def create_model(
    provider_id: str = Path(..., description="Provider ID"),
    model_data: ModelCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new model.
    
    This endpoint creates a new model for a provider.
    """
    try:
        # Ensure provider_id in path matches provider_id in body
        if model_data.provider_id != provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider ID in path '{provider_id}' does not match provider ID in body '{model_data.provider_id}'"
            )
        
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.create_model(model_data.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating model for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model for provider '{provider_id}': {str(e)}"
        )

@router.put("/{provider_id}/models/{model_id}", response_model=Dict[str, Any])
async def update_model(
    provider_id: str = Path(..., description="Provider ID"),
    model_id: str = Path(..., description="Model ID"),
    model_data: ModelUpdate = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Update a model.
    
    This endpoint updates a model.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        model = service.update_model(model_id, provider_id, model_data.dict(exclude_unset=True))
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating model '{model_id}' for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model '{model_id}' for provider '{provider_id}': {str(e)}"
        )

@router.delete("/{provider_id}/models/{model_id}", response_model=Dict[str, Any])
async def delete_model(
    provider_id: str = Path(..., description="Provider ID"),
    model_id: str = Path(..., description="Model ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a model.
    
    This endpoint deletes a model.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        result = service.delete_model(model_id, provider_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_id}' not found for provider '{provider_id}'"
            )
        return {"provider_id": provider_id, "model_id": model_id, "message": "Model deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model '{model_id}' for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model '{model_id}' for provider '{provider_id}': {str(e)}"
        )

# API Key endpoints

@router.get("/{provider_id}/api-keys", response_model=List[Dict[str, Any]])
async def get_api_keys(
    provider_id: str = Path(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all API keys for a provider.
    
    This endpoint returns a list of all API keys for a provider (without the actual key values).
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.get_api_keys_by_provider_id(provider_id)
    except Exception as e:
        logger.error(f"Error getting API keys for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API keys for provider '{provider_id}': {str(e)}"
        )

@router.post("/{provider_id}/api-keys", response_model=Dict[str, Any])
async def create_api_key(
    provider_id: str = Path(..., description="Provider ID"),
    api_key_data: ApiKeyCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Create a new API key.
    
    This endpoint creates a new API key for a provider.
    """
    try:
        # Ensure provider_id in path matches provider_id in body
        if api_key_data.provider_id != provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider ID in path '{provider_id}' does not match provider ID in body '{api_key_data.provider_id}'"
            )
        
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.create_api_key(api_key_data.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key for provider '{provider_id}': {str(e)}"
        )

@router.get("/{provider_id}/api-keys/{key_id}/value", response_model=Dict[str, Any])
async def get_api_key_value(
    provider_id: str = Path(..., description="Provider ID"),
    key_id: int = Path(..., description="API Key ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the actual API key value.
    
    This endpoint returns the actual API key value (decrypted if necessary).
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        key_value = service.get_api_key_value(key_id)
        if not key_value:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"API key '{key_id}' not found for provider '{provider_id}'"
            )
        return {"key_id": key_id, "provider_id": provider_id, "key_value": key_value}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key value for key '{key_id}' of provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get API key value for key '{key_id}' of provider '{provider_id}': {str(e)}"
        )

# Connection Parameter endpoints

@router.get("/{provider_id}/connection-params", response_model=Dict[str, Any])
async def get_connection_params(
    provider_id: str = Path(..., description="Provider ID"),
    include_sensitive: bool = Query(False, description="Whether to include sensitive parameters"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get all connection parameters for a provider.
    
    This endpoint returns a dictionary of all connection parameters for a provider.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.get_provider_connection_params(provider_id, include_sensitive)
    except Exception as e:
        logger.error(f"Error getting connection parameters for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get connection parameters for provider '{provider_id}': {str(e)}"
        )

@router.post("/{provider_id}/connection-params", response_model=Dict[str, Any])
async def set_connection_param(
    provider_id: str = Path(..., description="Provider ID"),
    param_data: ConnectionParameterCreate = Body(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Set a connection parameter.
    
    This endpoint creates or updates a connection parameter for a provider.
    """
    try:
        # Ensure provider_id in path matches provider_id in body
        if param_data.provider_id != provider_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Provider ID in path '{provider_id}' does not match provider ID in body '{param_data.provider_id}'"
            )
        
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        return service.set_connection_parameter(param_data.dict())
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting connection parameter for provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set connection parameter for provider '{provider_id}': {str(e)}"
        )

# Test endpoints

@router.post("/{provider_id}/test", response_model=Dict[str, Any])
async def test_provider(
    provider_id: str = Path(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Test a provider connection.
    
    This endpoint tests the connection to a provider.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        result = await service.test_provider_connection(provider_id)
        return result
    except Exception as e:
        logger.error(f"Error testing provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to test provider '{provider_id}': {str(e)}"
        )

@router.post("/{provider_id}/sync", response_model=Dict[str, Any])
async def sync_provider(
    provider_id: str = Path(..., description="Provider ID"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Synchronize a provider with the LLM Gateway configuration.
    
    This endpoint synchronizes a provider with the LLM Gateway configuration.
    """
    try:
        encryption_key = get_encryption_key()
        service = ProviderService(db, encryption_key, current_user.get("id"))
        result = service.sync_provider_with_gateway_config(provider_id)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Provider '{provider_id}' not found or synchronization failed"
            )
        return {"provider_id": provider_id, "message": "Provider synchronized successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error synchronizing provider '{provider_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to synchronize provider '{provider_id}': {str(e)}"
        )
