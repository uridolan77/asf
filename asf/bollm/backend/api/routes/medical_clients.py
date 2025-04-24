"""
Medical Clients API Routes

This module provides API endpoints for managing medical clients,
their configurations, status, and usage statistics.
"""

from fastapi import APIRouter, Depends, HTTPException, Body, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import logging

from ...database.session import get_db
from ...services.medical_client_service import MedicalClientService
from ..auth import get_current_user, User

router = APIRouter(prefix="/api/medical/clients", tags=["medical-clients"])
logger = logging.getLogger(__name__)

@router.get("/")
async def get_all_clients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all medical clients with their status.
    
    Args:
        current_user: The authenticated user
        db: Database session
        
    Returns:
        List of medical clients
    """
    service = MedicalClientService(db)
    clients = service.get_all_clients()
    return clients

@router.get("/{client_id}")
async def get_client(
    client_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific medical client by ID.
    
    Args:
        client_id: The client ID
        current_user: The authenticated user
        db: Database session
        
    Returns:
        Medical client details
    """
    service = MedicalClientService(db)
    client = service.get_client(client_id)
    
    if not client:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")
    
    return client

@router.put("/{client_id}")
async def update_client_config(
    client_id: str,
    config_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update a medical client configuration.
    
    Args:
        client_id: The client ID
        config_data: The configuration data
        current_user: The authenticated user
        db: Database session
        
    Returns:
        Updated client configuration
    """
    service = MedicalClientService(db)
    updated_config = service.update_client_config(client_id, config_data)
    
    if not updated_config:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")
    
    return {"message": "Client configuration updated successfully", "config": updated_config}

@router.post("/{client_id}/test")
async def test_client_connection(
    client_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Test connection to a medical client.
    
    Args:
        client_id: The client ID
        current_user: The authenticated user
        db: Database session
        
    Returns:
        Test result
    """
    service = MedicalClientService(db)
    result = await service.test_client_connection(client_id)
    
    return result

@router.get("/{client_id}/usage")
async def get_client_usage(
    client_id: str,
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get usage statistics for a client.
    
    Args:
        client_id: The client ID
        days: Number of days to retrieve stats for
        current_user: The authenticated user
        db: Database session
        
    Returns:
        Usage statistics
    """
    service = MedicalClientService(db)
    stats = service.get_client_usage(client_id, days)
    
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")
    
    return stats

@router.get("/{client_id}/status-history")
async def get_client_status_history(
    client_id: str,
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get status history for a client.
    
    Args:
        client_id: The client ID
        limit: Maximum number of records to retrieve
        current_user: The authenticated user
        db: Database session
        
    Returns:
        Status history
    """
    service = MedicalClientService(db)
    history = service.get_client_status_history(client_id, limit)
    
    if history is None:
        raise HTTPException(status_code=404, detail=f"Medical client not found: {client_id}")
    
    return history
