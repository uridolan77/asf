"""
LLM Service Abstraction Layer API router.

This module provides API endpoints for managing the LLM Service Abstraction Layer.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Mock EnhancedLLMService for development
class EnhancedLLMService:
    def __init__(self, config=None):
        self.config = config or {}

    async def initialize(self):
        pass

    def update_config(self, config):
        self.config.update(config)

    def get_config(self):
        return self.config

    async def get_health(self):
        return {
            "service_id": self.config.get("service_id", "mock-service"),
            "status": "operational",
            "components": {}
        }

from ...dependencies import get_db
from api.auth import get_current_user, User
from repositories.service_config_repository import ServiceConfigRepository

# Create router
router = APIRouter(
    prefix="/service",
    tags=["llm-service"],
    responses={404: {"description": "Not found"}},
    dependencies=[]  # Remove any dependencies that might require authentication
)

# Service instance (initialized in startup event)
_service_instance: Optional[EnhancedLLMService] = None

# Import service configuration utilities
from .service_config_utils import (
    get_service_config_from_db, list_service_configs,
    get_active_service_config, apply_service_config
)

# Pydantic Models
class CachingConfigModel(BaseModel):
    """Caching configuration model."""
    similarity_threshold: float = 0.92
    max_entries: int = 10000
    ttl_seconds: int = 3600
    persistence_type: str = "disk"
    persistence_config: Optional[Dict[str, Any]] = None

class ResilienceConfigModel(BaseModel):
    """Resilience configuration model."""
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_timeout: int = 30
    timeout_seconds: float = 30.0

class ObservabilityConfigModel(BaseModel):
    """Observability configuration model."""
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    logging_level: str = "INFO"
    export_metrics: bool = False
    metrics_export_url: Optional[str] = None

class EventsConfigModel(BaseModel):
    """Events configuration model."""
    max_event_history: int = 100
    publish_to_external: bool = False
    external_event_url: Optional[str] = None
    event_types_filter: Optional[List[str]] = None

class ProgressTrackingConfigModel(BaseModel):
    """Progress tracking configuration model."""
    max_active_operations: int = 100
    operation_ttl_seconds: int = 3600
    publish_updates: bool = True

class ServiceConfigModel(BaseModel):
    """Service configuration model."""
    service_id: str
    name: str
    description: Optional[str] = None
    enable_caching: bool = True
    enable_resilience: bool = True
    enable_observability: bool = True
    enable_events: bool = True
    enable_progress_tracking: bool = True
    is_public: bool = False
    config: Dict[str, Any] = {
        "caching": {},
        "resilience": {},
        "observability": {},
        "events": {},
        "progress_tracking": {}
    }

class ServiceConfigCreateModel(BaseModel):
    """Service configuration create model."""
    service_id: str
    name: str
    description: Optional[str] = None
    enable_caching: bool = True
    enable_resilience: bool = True
    enable_observability: bool = True
    enable_events: bool = True
    enable_progress_tracking: bool = True
    is_public: bool = False
    caching: Optional[CachingConfigModel] = None
    resilience: Optional[ResilienceConfigModel] = None
    observability: Optional[ObservabilityConfigModel] = None
    events: Optional[EventsConfigModel] = None
    progress_tracking: Optional[ProgressTrackingConfigModel] = None

class ServiceConfigUpdateModel(BaseModel):
    """Service configuration update model."""
    name: Optional[str] = None
    description: Optional[str] = None
    enable_caching: Optional[bool] = None
    enable_resilience: Optional[bool] = None
    enable_observability: Optional[bool] = None
    enable_events: Optional[bool] = None
    enable_progress_tracking: Optional[bool] = None
    is_public: Optional[bool] = None
    caching: Optional[CachingConfigModel] = None
    resilience: Optional[ResilienceConfigModel] = None
    observability: Optional[ObservabilityConfigModel] = None
    events: Optional[EventsConfigModel] = None
    progress_tracking: Optional[ProgressTrackingConfigModel] = None

class ServiceHealth(BaseModel):
    """Service health model."""
    service_id: str
    status: str
    components: Dict[str, Any]

class ServiceStats(BaseModel):
    """Service statistics model."""
    service_id: str
    metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    active_operations: Dict[str, Any]
    recent_events: List[Dict[str, Any]]

# Helper functions
async def get_service() -> EnhancedLLMService:
    """Get the service instance."""
    global _service_instance
    if _service_instance is None:
        # Load configuration from database
        config = get_active_service_config()

        # Initialize service with configuration from database
        _service_instance = EnhancedLLMService(config=config)
        await _service_instance.initialize()
    return _service_instance

def get_service_config_from_db(db: Session, config_id: int, user: User = None):
    """Get a service configuration from the database."""
    repo = ServiceConfigRepository(db)
    if user:
        return repo.get_configuration_by_id(config_id, user.id)
    else:
        return repo.get_configuration_by_id(config_id)

def apply_service_config(service: EnhancedLLMService, config) -> None:
    """Apply a service configuration to the service instance."""
    # Convert DB model to dict for service
    service_config = {
        "service_id": config.service_id,
        "enable_caching": config.enable_caching,
        "enable_resilience": config.enable_resilience,
        "enable_observability": config.enable_observability,
        "enable_events": config.enable_events,
        "enable_progress_tracking": config.enable_progress_tracking,
        "config": {}
    }

    # Add component configs
    if config.caching_config:
        service_config["config"]["cache"] = {
            "similarity_threshold": config.caching_config.similarity_threshold,
            "max_entries": config.caching_config.max_entries,
            "ttl_seconds": config.caching_config.ttl_seconds,
            "persistence_type": config.caching_config.persistence_type,
            "persistence_config": config.caching_config.persistence_config
        }

    # Update service configuration
    service.update_config(service_config)

# Routes - Service Configuration Management
@router.post("/configurations", response_model=ServiceConfigModel, status_code=status.HTTP_201_CREATED)
async def create_service_configuration(
    config_data: ServiceConfigCreateModel,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new service configuration.
    """
    try:
        # Create configuration using repository
        repo = ServiceConfigRepository(db)

        # Prepare configuration data
        config_dict = {
            "service_id": config_data.service_id,
            "name": config_data.name,
            "description": config_data.description,
            "enable_caching": config_data.enable_caching,
            "enable_resilience": config_data.enable_resilience,
            "enable_observability": config_data.enable_observability,
            "enable_events": config_data.enable_events,
            "enable_progress_tracking": config_data.enable_progress_tracking,
            "is_public": config_data.is_public,
            "created_by_user_id": current_user.id
        }

        # Add component configurations if provided
        if config_data.caching:
            config_dict["caching"] = config_data.caching.dict()

        if config_data.resilience:
            config_dict["resilience"] = config_data.resilience.dict()

        if config_data.observability:
            config_dict["observability"] = config_data.observability.dict()

        if config_data.events:
            config_dict["events"] = config_data.events.dict()

        if config_data.progress_tracking:
            config_dict["progress_tracking"] = config_data.progress_tracking.dict()

        # Create configuration
        service_config = repo.create_configuration(config_dict)

        if not service_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create service configuration"
            )

        return service_config.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating service configuration: {str(e)}"
        )

@router.get("/configurations", response_model=List[ServiceConfigModel])
async def list_service_configurations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    include_public: bool = Query(True, description="Include public configurations"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100)
):
    """
    List service configurations.
    """
    try:
        # Use repository to get configurations
        repo = ServiceConfigRepository(db)
        configs = repo.get_all_configurations(
            user_id=current_user.id,
            include_public=include_public,
            skip=skip,
            limit=limit
        )

        # Convert to response model
        return [config.to_dict() for config in configs]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing service configurations: {str(e)}"
        )

@router.get("/configurations/{config_id}", response_model=ServiceConfigModel)
async def get_service_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get a service configuration by ID.
    """
    # Get configuration
    config = get_service_config_from_db(db, config_id, current_user)

    # Check if configuration exists
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service configuration not found"
        )

    return config.to_dict()

@router.put("/configurations/{config_id}", response_model=ServiceConfigModel)
async def update_service_configuration(
    config_id: int,
    config_data: ServiceConfigUpdateModel,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Update a service configuration.
    """
    try:
        # Use repository to get and update configuration
        repo = ServiceConfigRepository(db)

        # Get configuration
        config = repo.get_configuration_by_id(config_id)

        # Check if configuration exists
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service configuration not found"
            )

        # Check ownership
        if config.created_by_user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this configuration"
            )

        # Prepare update data
        update_data = {}
        if config_data.name is not None:
            update_data["name"] = config_data.name
        if config_data.description is not None:
            update_data["description"] = config_data.description
        if config_data.enable_caching is not None:
            update_data["enable_caching"] = config_data.enable_caching
        if config_data.enable_resilience is not None:
            update_data["enable_resilience"] = config_data.enable_resilience
        if config_data.enable_observability is not None:
            update_data["enable_observability"] = config_data.enable_observability
        if config_data.enable_events is not None:
            update_data["enable_events"] = config_data.enable_events
        if config_data.enable_progress_tracking is not None:
            update_data["enable_progress_tracking"] = config_data.enable_progress_tracking
        if config_data.is_public is not None:
            update_data["is_public"] = config_data.is_public

        # Add component configurations if provided
        if config_data.caching:
            update_data["caching"] = config_data.caching.dict()

        if config_data.resilience:
            update_data["resilience"] = config_data.resilience.dict()

        if config_data.observability:
            update_data["observability"] = config_data.observability.dict()

        if config_data.events:
            update_data["events"] = config_data.events.dict()

        if config_data.progress_tracking:
            update_data["progress_tracking"] = config_data.progress_tracking.dict()

        # Update configuration
        updated_config = repo.update_configuration(config_id, update_data)

        if not updated_config:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update service configuration"
            )

        return updated_config.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating service configuration: {str(e)}"
        )

@router.delete("/configurations/{config_id}", response_model=Dict[str, Any])
async def delete_service_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a service configuration.
    """
    try:
        # Use repository to get and delete configuration
        repo = ServiceConfigRepository(db)

        # Get configuration
        config = repo.get_configuration_by_id(config_id)

        # Check if configuration exists
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service configuration not found"
            )

        # Check ownership
        if config.created_by_user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this configuration"
            )

        # Delete configuration
        success = repo.delete_configuration(config_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete service configuration"
            )

        return {"success": True, "message": "Configuration deleted successfully"}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting service configuration: {str(e)}"
        )

@router.post("/configurations/{config_id}/apply", response_model=ServiceConfigModel)
async def apply_service_configuration(
    config_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Apply a service configuration to the service instance.
    """
    # Get configuration
    config = get_service_config_from_db(db, config_id, current_user)

    # Check if configuration exists
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service configuration not found"
        )

    # Apply configuration
    apply_service_config(service, config)

    return config.to_dict()

# Routes - Service Operations
@router.get("/config", response_model=ServiceConfigModel)
async def get_service_config(
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Get the current service configuration.
    """
    # Get configuration
    config = service.get_config()

    # Convert to response model
    return {
        "service_id": config["service_id"],
        "name": "Current Active Configuration",
        "description": "The currently active service configuration",
        "enable_caching": config["enable_caching"],
        "enable_resilience": config["enable_resilience"],
        "enable_observability": config["enable_observability"],
        "enable_events": config["enable_events"],
        "enable_progress_tracking": config["enable_progress_tracking"],
        "is_public": True,
        "config": config.get("config", {})
    }

@router.put("/config", response_model=ServiceConfigModel)
async def update_service_config(
    config: ServiceConfigModel,
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Update the service configuration.
    """
    # Update configuration
    service.update_config(config.dict())

    # Return updated configuration
    return get_service_config(service)

@router.get("/health", response_model=ServiceHealth)
async def get_service_health(
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Get the service health status.
    """
    # Get health
    health = await service.get_health()
    return health

@router.get("/stats", response_model=ServiceStats)
async def get_service_stats(
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Get service statistics.
    """
    # Get statistics
    stats = await service.get_stats()
    return stats

@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_cache(
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Clear the service cache.
    """
    # Clear cache
    await service.clear_cache()

    # Return success
    return {"success": True, "message": "Cache cleared successfully"}

@router.post("/resilience/reset-circuit-breakers", response_model=Dict[str, Any])
async def reset_circuit_breakers(
    service: EnhancedLLMService = Depends(get_service)
):
    """
    Reset all circuit breakers.
    """
    # Reset circuit breakers
    service.resilience.reset_all_circuit_breakers()

    # Return success
    return {"success": True, "message": "Circuit breakers reset successfully"}

# Startup and shutdown events
async def startup_event():
    """Initialize service on startup."""
    await get_service()

async def shutdown_event():
    """Shut down service on shutdown."""
    global _service_instance
    if _service_instance is not None:
        await _service_instance.shutdown()
        _service_instance = None
