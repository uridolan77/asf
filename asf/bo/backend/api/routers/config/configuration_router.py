from fastapi import APIRouter, Depends, HTTPException, status, Request, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
import sys
import os

# Add the backend directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# Use absolute imports
from config.database import get_db
from services.configuration_service import ConfigurationService
from api.auth import get_current_user
from models.user import User
from schemas.configuration import (
    ConfigurationCreate,
    ConfigurationUpdate,
    ConfigurationResponse,
    UserSettingCreate,
    UserSettingUpdate,
    UserSettingResponse
)

router = APIRouter(
    prefix="/api/config",
    tags=["configuration"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)

# Global Configuration endpoints

@router.get("/", response_model=List[ConfigurationResponse])
async def get_configurations(
    environment: str = "development",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all configurations for an environment."""
    service = ConfigurationService(db, current_user.id)
    return service.get_all_configurations(environment)

@router.get("/{config_key}", response_model=ConfigurationResponse)
async def get_configuration(
    config_key: str,
    environment: str = "development",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a configuration by key."""
    service = ConfigurationService(db, current_user.id)
    config = service.get_configuration_by_key(config_key, environment)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with key {config_key} not found in environment {environment}"
        )
    return config

@router.post("/", response_model=ConfigurationResponse, status_code=status.HTTP_201_CREATED)
async def create_configuration(
    config: ConfigurationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new configuration."""
    service = ConfigurationService(db, current_user.id)
    return service.set_configuration(config.dict())

@router.put("/{config_key}", response_model=ConfigurationResponse)
async def update_configuration(
    config_key: str,
    config: ConfigurationUpdate,
    environment: str = "development",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a configuration."""
    service = ConfigurationService(db, current_user.id)

    # Check if configuration exists
    existing_config = service.get_configuration_by_key(config_key, environment)
    if not existing_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with key {config_key} not found in environment {environment}"
        )

    # Update configuration
    config_data = config.dict(exclude_unset=True)
    config_data["config_key"] = config_key
    config_data["environment"] = environment

    return service.set_configuration(config_data)

@router.delete("/{config_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_configuration(
    config_key: str,
    environment: str = "development",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a configuration."""
    service = ConfigurationService(db, current_user.id)
    result = service.delete_configuration(config_key, environment)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration with key {config_key} not found in environment {environment}"
        )
    return None

# User Settings endpoints

@router.get("/user/settings", response_model=List[UserSettingResponse])
async def get_user_settings(
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get all settings for a user."""
    service = ConfigurationService(db, current_user.id)

    # Only allow admins to get settings for other users
    if user_id is not None and user_id != current_user.id and current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access settings for other users"
        )

    return service.get_user_settings(user_id)

@router.get("/user/settings/{setting_key}", response_model=UserSettingResponse)
async def get_user_setting(
    setting_key: str,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get a user setting."""
    service = ConfigurationService(db, current_user.id)

    # Only allow admins to get settings for other users
    if user_id is not None and user_id != current_user.id and current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access settings for other users"
        )

    setting = service.get_user_setting(setting_key, user_id)
    if not setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting with key {setting_key} not found for user"
        )
    return setting

@router.post("/user/settings", response_model=UserSettingResponse, status_code=status.HTTP_201_CREATED)
async def create_user_setting(
    setting: UserSettingCreate,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a new user setting."""
    service = ConfigurationService(db, current_user.id)

    # Only allow admins to create settings for other users
    if user_id is not None and user_id != current_user.id and current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to create settings for other users"
        )

    setting_data = setting.dict()
    if user_id is not None:
        setting_data["user_id"] = user_id

    return service.set_user_setting(setting_data)

@router.put("/user/settings/{setting_key}", response_model=UserSettingResponse)
async def update_user_setting(
    setting_key: str,
    setting: UserSettingUpdate,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update a user setting."""
    service = ConfigurationService(db, current_user.id)

    # Only allow admins to update settings for other users
    if user_id is not None and user_id != current_user.id and current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update settings for other users"
        )

    # Check if setting exists
    existing_setting = service.get_user_setting(setting_key, user_id)
    if not existing_setting:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting with key {setting_key} not found for user"
        )

    # Update setting
    setting_data = setting.dict(exclude_unset=True)
    setting_data["setting_key"] = setting_key
    if user_id is not None:
        setting_data["user_id"] = user_id

    return service.set_user_setting(setting_data)

@router.delete("/user/settings/{setting_key}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_setting(
    setting_key: str,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete a user setting."""
    service = ConfigurationService(db, current_user.id)

    # Only allow admins to delete settings for other users
    if user_id is not None and user_id != current_user.id and current_user.role.name != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete settings for other users"
        )

    result = service.delete_user_setting(setting_key, user_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Setting with key {setting_key} not found for user"
        )
    return None
