from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
from models.configuration import Configuration, UserSetting

logger = logging.getLogger(__name__)

class ConfigurationRepository:
    def __init__(self, db: Session):
        self.db = db
    
    # Global Configuration methods
    
    def get_all_configurations(self, environment: str = "development") -> List[Configuration]:
        """Get all configurations for an environment."""
        return self.db.query(Configuration).filter(Configuration.environment == environment).all()
    
    def get_configuration_by_key(self, config_key: str, environment: str = "development") -> Optional[Configuration]:
        """Get a configuration by key."""
        return self.db.query(Configuration).filter(
            Configuration.config_key == config_key,
            Configuration.environment == environment
        ).first()
    
    def get_configuration_value(self, config_key: str, environment: str = "development", default: Any = None) -> Any:
        """Get a configuration value by key."""
        config = self.get_configuration_by_key(config_key, environment)
        if not config:
            return default
        
        # Convert value based on type
        if config.config_type == "integer":
            try:
                return int(config.config_value)
            except (ValueError, TypeError):
                return default
        elif config.config_type == "float":
            try:
                return float(config.config_value)
            except (ValueError, TypeError):
                return default
        elif config.config_type == "boolean":
            return config.config_value.lower() in ("true", "yes", "1")
        else:
            return config.config_value
    
    def set_configuration(self, config_data: Dict[str, Any]) -> Configuration:
        """Set a configuration (create or update)."""
        try:
            # Check if configuration already exists
            config = self.get_configuration_by_key(
                config_data["config_key"],
                config_data.get("environment", "development")
            )
            
            if config:
                # Update existing configuration
                config.config_value = str(config_data["config_value"])
                config.config_type = config_data.get("config_type", config.config_type)
                config.description = config_data.get("description", config.description)
                if "created_by_user_id" in config_data:
                    config.created_by_user_id = config_data["created_by_user_id"]
            else:
                # Create new configuration
                config = Configuration(
                    config_key=config_data["config_key"],
                    config_value=str(config_data["config_value"]),
                    config_type=config_data.get("config_type", "string"),
                    description=config_data.get("description"),
                    environment=config_data.get("environment", "development"),
                    created_by_user_id=config_data.get("created_by_user_id")
                )
                self.db.add(config)
            
            self.db.commit()
            self.db.refresh(config)
            return config
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error setting configuration: {e}")
            raise
    
    def delete_configuration(self, config_key: str, environment: str = "development") -> bool:
        """Delete a configuration."""
        try:
            config = self.get_configuration_by_key(config_key, environment)
            if not config:
                return False
            
            self.db.delete(config)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting configuration: {e}")
            raise
    
    # User Settings methods
    
    def get_user_settings(self, user_id: int) -> List[UserSetting]:
        """Get all settings for a user."""
        return self.db.query(UserSetting).filter(UserSetting.user_id == user_id).all()
    
    def get_user_setting(self, user_id: int, setting_key: str) -> Optional[UserSetting]:
        """Get a user setting."""
        return self.db.query(UserSetting).filter(
            UserSetting.user_id == user_id,
            UserSetting.setting_key == setting_key
        ).first()
    
    def get_user_setting_value(self, user_id: int, setting_key: str, default: Any = None) -> Any:
        """Get a user setting value."""
        setting = self.get_user_setting(user_id, setting_key)
        return setting.setting_value if setting else default
    
    def set_user_setting(self, setting_data: Dict[str, Any]) -> UserSetting:
        """Set a user setting (create or update)."""
        try:
            # Check if setting already exists
            setting = self.get_user_setting(
                setting_data["user_id"],
                setting_data["setting_key"]
            )
            
            if setting:
                # Update existing setting
                setting.setting_value = setting_data["setting_value"]
            else:
                # Create new setting
                setting = UserSetting(
                    user_id=setting_data["user_id"],
                    setting_key=setting_data["setting_key"],
                    setting_value=setting_data["setting_value"]
                )
                self.db.add(setting)
            
            self.db.commit()
            self.db.refresh(setting)
            return setting
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error setting user setting: {e}")
            raise
    
    def delete_user_setting(self, user_id: int, setting_key: str) -> bool:
        """Delete a user setting."""
        try:
            setting = self.get_user_setting(user_id, setting_key)
            if not setting:
                return False
            
            self.db.delete(setting)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting user setting: {e}")
            raise
