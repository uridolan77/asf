from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
from repositories.configuration_repository import ConfigurationRepository
from repositories.audit_repository import AuditRepository
from models.configuration import Configuration, UserSetting

logger = logging.getLogger(__name__)

class ConfigurationService:
    def __init__(self, db: Session, current_user_id: Optional[int] = None):
        self.db = db
        self.current_user_id = current_user_id
        self.config_repo = ConfigurationRepository(db)
        self.audit_repo = AuditRepository(db)
    
    # Global Configuration methods
    
    def get_all_configurations(self, environment: str = "development") -> List[Dict[str, Any]]:
        """Get all configurations for an environment."""
        configs = self.config_repo.get_all_configurations(environment)
        result = []
        
        for config in configs:
            result.append({
                "config_id": config.config_id,
                "config_key": config.config_key,
                "config_value": config.config_value,
                "config_type": config.config_type,
                "description": config.description,
                "environment": config.environment,
                "created_at": config.created_at,
                "updated_at": config.updated_at
            })
        
        return result
    
    def get_configuration_by_key(self, config_key: str, environment: str = "development") -> Optional[Dict[str, Any]]:
        """Get a configuration by key."""
        config = self.config_repo.get_configuration_by_key(config_key, environment)
        if not config:
            return None
        
        return {
            "config_id": config.config_id,
            "config_key": config.config_key,
            "config_value": config.config_value,
            "config_type": config.config_type,
            "description": config.description,
            "environment": config.environment,
            "created_at": config.created_at,
            "updated_at": config.updated_at
        }
    
    def get_configuration_value(self, config_key: str, environment: str = "development", default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config_repo.get_configuration_value(config_key, environment, default)
    
    def set_configuration(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set a configuration (create or update)."""
        # Add current user ID if available
        if self.current_user_id and "created_by_user_id" not in config_data:
            config_data["created_by_user_id"] = self.current_user_id
        
        # Get the configuration before update for audit
        old_config = self.config_repo.get_configuration_by_key(
            config_data["config_key"],
            config_data.get("environment", "development")
        )
        
        # Set configuration
        config = self.config_repo.set_configuration(config_data)
        
        # Log audit
        if old_config:
            # Update
            self.audit_repo.create_audit_log({
                "table_name": "configurations",
                "record_id": str(config.config_id),
                "action": "update",
                "changed_by_user_id": self.current_user_id,
                "old_values": {
                    "config_value": old_config.config_value,
                    "config_type": old_config.config_type,
                    "description": old_config.description
                },
                "new_values": {
                    "config_value": config.config_value,
                    "config_type": config.config_type,
                    "description": config.description
                }
            })
        else:
            # Create
            self.audit_repo.create_audit_log({
                "table_name": "configurations",
                "record_id": str(config.config_id),
                "action": "create",
                "changed_by_user_id": self.current_user_id,
                "new_values": {
                    "config_key": config.config_key,
                    "config_value": config.config_value,
                    "config_type": config.config_type,
                    "environment": config.environment
                }
            })
        
        # Return the configuration
        return {
            "config_id": config.config_id,
            "config_key": config.config_key,
            "config_value": config.config_value,
            "config_type": config.config_type,
            "description": config.description,
            "environment": config.environment,
            "created_at": config.created_at,
            "updated_at": config.updated_at
        }
    
    def delete_configuration(self, config_key: str, environment: str = "development") -> bool:
        """Delete a configuration."""
        # Get the configuration before delete for audit
        old_config = self.config_repo.get_configuration_by_key(config_key, environment)
        if not old_config:
            return False
        
        # Delete configuration
        result = self.config_repo.delete_configuration(config_key, environment)
        if not result:
            return False
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "configurations",
            "record_id": str(old_config.config_id),
            "action": "delete",
            "changed_by_user_id": self.current_user_id,
            "old_values": {
                "config_key": old_config.config_key,
                "config_value": old_config.config_value,
                "config_type": old_config.config_type,
                "environment": old_config.environment
            }
        })
        
        return True
    
    # User Settings methods
    
    def get_user_settings(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all settings for a user."""
        # Use current user ID if not provided
        if user_id is None:
            user_id = self.current_user_id
        
        if not user_id:
            return []
        
        settings = self.config_repo.get_user_settings(user_id)
        result = []
        
        for setting in settings:
            result.append({
                "setting_id": setting.setting_id,
                "user_id": setting.user_id,
                "setting_key": setting.setting_key,
                "setting_value": setting.setting_value,
                "created_at": setting.created_at,
                "updated_at": setting.updated_at
            })
        
        return result
    
    def get_user_setting(self, setting_key: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get a user setting."""
        # Use current user ID if not provided
        if user_id is None:
            user_id = self.current_user_id
        
        if not user_id:
            return None
        
        setting = self.config_repo.get_user_setting(user_id, setting_key)
        if not setting:
            return None
        
        return {
            "setting_id": setting.setting_id,
            "user_id": setting.user_id,
            "setting_key": setting.setting_key,
            "setting_value": setting.setting_value,
            "created_at": setting.created_at,
            "updated_at": setting.updated_at
        }
    
    def get_user_setting_value(self, setting_key: str, default: Any = None, user_id: Optional[int] = None) -> Any:
        """Get a user setting value."""
        # Use current user ID if not provided
        if user_id is None:
            user_id = self.current_user_id
        
        if not user_id:
            return default
        
        return self.config_repo.get_user_setting_value(user_id, setting_key, default)
    
    def set_user_setting(self, setting_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set a user setting (create or update)."""
        # Use current user ID if not provided
        if "user_id" not in setting_data:
            setting_data["user_id"] = self.current_user_id
        
        if not setting_data.get("user_id"):
            raise ValueError("User ID is required")
        
        # Get the setting before update for audit
        old_setting = self.config_repo.get_user_setting(
            setting_data["user_id"],
            setting_data["setting_key"]
        )
        
        # Set setting
        setting = self.config_repo.set_user_setting(setting_data)
        
        # Log audit
        if old_setting:
            # Update
            self.audit_repo.create_audit_log({
                "table_name": "user_settings",
                "record_id": str(setting.setting_id),
                "action": "update",
                "changed_by_user_id": self.current_user_id,
                "old_values": {
                    "setting_value": old_setting.setting_value
                },
                "new_values": {
                    "setting_value": setting.setting_value
                }
            })
        else:
            # Create
            self.audit_repo.create_audit_log({
                "table_name": "user_settings",
                "record_id": str(setting.setting_id),
                "action": "create",
                "changed_by_user_id": self.current_user_id,
                "new_values": {
                    "user_id": setting.user_id,
                    "setting_key": setting.setting_key,
                    "setting_value": setting.setting_value
                }
            })
        
        # Return the setting
        return {
            "setting_id": setting.setting_id,
            "user_id": setting.user_id,
            "setting_key": setting.setting_key,
            "setting_value": setting.setting_value,
            "created_at": setting.created_at,
            "updated_at": setting.updated_at
        }
    
    def delete_user_setting(self, setting_key: str, user_id: Optional[int] = None) -> bool:
        """Delete a user setting."""
        # Use current user ID if not provided
        if user_id is None:
            user_id = self.current_user_id
        
        if not user_id:
            return False
        
        # Get the setting before delete for audit
        old_setting = self.config_repo.get_user_setting(user_id, setting_key)
        if not old_setting:
            return False
        
        # Delete setting
        result = self.config_repo.delete_user_setting(user_id, setting_key)
        if not result:
            return False
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "user_settings",
            "record_id": str(old_setting.setting_id),
            "action": "delete",
            "changed_by_user_id": self.current_user_id,
            "old_values": {
                "user_id": old_setting.user_id,
                "setting_key": old_setting.setting_key,
                "setting_value": old_setting.setting_value
            }
        })
        
        return True
