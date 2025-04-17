"""
DSPy Client Routes

This module provides API routes for DSPy client management.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from sqlalchemy.orm import Session
from datetime import date, datetime
from typing import List, Dict, Any, Optional

from ..db.database import get_db
from ..services.medical_client_service import DSPyClientService
from ..models.dspy_client import DSPyClient, DSPyClientConfig, DSPyModule

router = APIRouter(
    prefix="/api/dspy",
    tags=["dspy"],
    responses={404: {"description": "Not found"}},
)


@router.get("/clients", response_model=List[Dict[str, Any]])
def get_all_dspy_clients(db: Session = Depends(get_db)):
    """Get all DSPy clients"""
    service = DSPyClientService(db)
    return service.get_all_dspy_clients()


@router.get("/clients/{client_id}", response_model=Dict[str, Any])
def get_dspy_client(client_id: str, db: Session = Depends(get_db)):
    """Get a specific DSPy client by ID"""
    service = DSPyClientService(db)
    client = service.get_dspy_client(client_id)
    if not client:
        raise HTTPException(status_code=404, detail=f"DSPy client not found: {client_id}")
    return client


@router.post("/clients", response_model=Dict[str, Any])
def create_dspy_client(client_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create a new DSPy client"""
    try:
        # Check required fields
        if "name" not in client_data or not client_data["name"]:
            raise HTTPException(status_code=400, detail="Client name is required")
        
        if "base_url" not in client_data or not client_data["base_url"]:
            raise HTTPException(status_code=400, detail="Base URL is required")
        
        # Create client
        client = DSPyClient(
            name=client_data["name"],
            description=client_data.get("description", ""),
            base_url=client_data["base_url"]
        )
        
        db.add(client)
        db.commit()
        db.refresh(client)
        
        # Create config if provided
        if "config" in client_data and isinstance(client_data["config"], dict):
            config_data = client_data["config"]
            config = DSPyClientConfig(client_id=client.client_id)
            
            # Update config fields
            for key, value in config_data.items():
                if hasattr(config, key) and key not in ['config_id', 'client_id', 'created_at', 'updated_at']:
                    setattr(config, key, value)
            
            # Handle additional_config as JSON
            if 'additional_config' in config_data and isinstance(config_data['additional_config'], dict):
                config.additional_config = config_data['additional_config']
            
            db.add(config)
            db.commit()
        
        # Return the created client
        service = DSPyClientService(db)
        return service.get_dspy_client(client.client_id)
    
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error creating DSPy client: {str(e)}")


@router.put("/clients/{client_id}", response_model=Dict[str, Any])
def update_dspy_client(client_id: str, client_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Update a DSPy client"""
    try:
        # Check if client exists
        client = db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail=f"DSPy client not found: {client_id}")
        
        # Update client fields
        if "name" in client_data:
            client.name = client_data["name"]
        
        if "description" in client_data:
            client.description = client_data["description"]
        
        if "base_url" in client_data:
            client.base_url = client_data["base_url"]
        
        db.commit()
        
        # Update config if provided
        if "config" in client_data and isinstance(client_data["config"], dict):
            service = DSPyClientService(db)
            service.update_dspy_client_config(client_id, client_data["config"])
        
        # Return the updated client
        service = DSPyClientService(db)
        return service.get_dspy_client(client_id)
    
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error updating DSPy client: {str(e)}")


@router.delete("/clients/{client_id}", response_model=Dict[str, Any])
def delete_dspy_client(client_id: str, db: Session = Depends(get_db)):
    """Delete a DSPy client"""
    try:
        # Check if client exists
        client = db.query(DSPyClient).filter(DSPyClient.client_id == client_id).first()
        if not client:
            raise HTTPException(status_code=404, detail=f"DSPy client not found: {client_id}")
        
        # Delete client
        db.delete(client)
        db.commit()
        
        return {"success": True, "message": f"DSPy client deleted: {client_id}"}
    
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error deleting DSPy client: {str(e)}")


@router.post("/clients/{client_id}/test-connection", response_model=Dict[str, Any])
async def test_dspy_client_connection(client_id: str, db: Session = Depends(get_db)):
    """Test connection to a DSPy client"""
    service = DSPyClientService(db)
    result = await service.test_dspy_client_connection(client_id)
    return result


@router.put("/clients/{client_id}/config", response_model=Dict[str, Any])
def update_dspy_client_config(client_id: str, config_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Update a DSPy client configuration"""
    service = DSPyClientService(db)
    config = service.update_dspy_client_config(client_id, config_data)
    if not config:
        raise HTTPException(status_code=404, detail=f"DSPy client not found: {client_id}")
    return config


@router.get("/clients/{client_id}/modules", response_model=List[Dict[str, Any]])
def get_dspy_modules(client_id: str, db: Session = Depends(get_db)):
    """Get all modules for a DSPy client"""
    service = DSPyClientService(db)
    return service.get_dspy_modules(client_id)


@router.get("/modules/{module_id}", response_model=Dict[str, Any])
def get_dspy_module(module_id: str, db: Session = Depends(get_db)):
    """Get a specific DSPy module by ID"""
    service = DSPyClientService(db)
    module = service.get_dspy_module(module_id)
    if not module:
        raise HTTPException(status_code=404, detail=f"DSPy module not found: {module_id}")
    return module


@router.post("/clients/{client_id}/sync-modules", response_model=Dict[str, Any])
def sync_dspy_modules(client_id: str, db: Session = Depends(get_db)):
    """Synchronize modules from a DSPy client"""
    service = DSPyClientService(db)
    result = service.sync_dspy_modules(client_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["message"])
    return result


@router.post("/modules", response_model=Dict[str, Any])
def create_dspy_module(module_data: Dict[str, Any], db: Session = Depends(get_db)):
    """Create a new DSPy module"""
    try:
        # Check required fields
        if "client_id" not in module_data or not module_data["client_id"]:
            raise HTTPException(status_code=400, detail="Client ID is required")
        
        if "name" not in module_data or not module_data["name"]:
            raise HTTPException(status_code=400, detail="Module name is required")
        
        # Check if client exists
        client = db.query(DSPyClient).filter(DSPyClient.client_id == module_data["client_id"]).first()
        if not client:
            raise HTTPException(status_code=404, detail=f"DSPy client not found: {module_data['client_id']}")
        
        # Create module
        module = DSPyModule(
            client_id=module_data["client_id"],
            name=module_data["name"],
            description=module_data.get("description", ""),
            module_type=module_data.get("module_type", ""),
            class_name=module_data.get("class_name", "")
        )
        
        db.add(module)
        db.commit()
        db.refresh(module)
        
        return module.to_dict()
    
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error creating DSPy module: {str(e)}")


@router.get("/clients/{client_id}/usage", response_model=List[Dict[str, Any]])
def get_dspy_client_usage(
    client_id: str, 
    days: int = Query(30, description="Number of days to include in the statistics"),
    db: Session = Depends(get_db)
):
    """Get usage statistics for a DSPy client"""
    service = DSPyClientService(db)
    stats = service.get_dspy_client_usage(client_id, days=days)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"DSPy client not found: {client_id}")
    return stats


@router.get("/clients/{client_id}/status-history", response_model=List[Dict[str, Any]])
def get_dspy_client_status_history(
    client_id: str, 
    limit: int = Query(100, description="Maximum number of log entries to return"),
    db: Session = Depends(get_db)
):
    """Get status history for a DSPy client"""
    service = DSPyClientService(db)
    logs = service.get_dspy_client_status_history(client_id, limit=limit)
    if logs is None:
        raise HTTPException(status_code=404, detail=f"DSPy client not found: {client_id}")
    return logs


@router.get("/clients/{client_id}/audit-logs", response_model=List[Dict[str, Any]])
def get_dspy_audit_logs(
    client_id: str = Path(..., description="ID of the DSPy client"), 
    module_id: Optional[str] = Query(None, description="Filter by module ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    start_date: Optional[date] = Query(None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: Optional[date] = Query(None, description="Filter by end date (YYYY-MM-DD)"),
    limit: int = Query(100, description="Maximum number of log entries to return"),
    offset: int = Query(0, description="Number of log entries to skip"),
    db: Session = Depends(get_db)
):
    """Get audit logs for a DSPy client with flexible filtering options"""
    service = DSPyClientService(db)
    logs = service.get_dspy_audit_logs(
        client_id=client_id,
        module_id=module_id,
        event_type=event_type,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    return logs


@router.get("/clients/{client_id}/circuit-breakers", response_model=List[Dict[str, Any]])
def get_circuit_breaker_status(client_id: str, db: Session = Depends(get_db)):
    """Get circuit breaker status for a DSPy client"""
    service = DSPyClientService(db)
    return service.get_circuit_breaker_status(client_id)