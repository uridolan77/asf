"""
Configuration loader for LLM Gateway.

This module provides functions for loading and saving configuration
from both file and database.
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, Union
from sqlalchemy.orm import Session
import copy

# Set up logging
logger = logging.getLogger(__name__)

# Default configuration path
DEFAULT_CONFIG_PATH = os.environ.get("LLM_GATEWAY_CONFIG_PATH", "config/gateway_config.yaml")

class ConfigLoader:
    """
    Configuration loader for LLM Gateway.
    
    This class provides methods for loading and saving configuration
    from both file and database.
    """
    
    def __init__(self, db: Optional[Session] = None, config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            db: Optional SQLAlchemy database session
            config_path: Optional path to the configuration file
        """
        self.db = db
        self.config_path = config_path or DEFAULT_CONFIG_PATH
    
    def load_config(self, provider_id: Optional[str] = None, use_db: bool = True) -> Dict[str, Any]:
        """
        Load configuration from file or database.
        
        Args:
            provider_id: Optional provider ID to load configuration for
            use_db: Whether to use the database for configuration
            
        Returns:
            Configuration dictionary
        """
        # Load from file first
        config = self.load_from_yaml()
        
        # If database is available and use_db is True, load from database
        if self.db is not None and use_db:
            try:
                # Import here to avoid circular imports
                from asf.medical.llm_gateway.services.provider_service import ProviderService
                
                # Create provider service
                service = ProviderService(self.db)
                
                # If provider_id is provided, load only that provider
                if provider_id:
                    provider = service.get_provider_by_id(provider_id)
                    if provider:
                        # Update provider configuration
                        if "additional_config" not in config:
                            config["additional_config"] = {}
                        
                        if "providers" not in config["additional_config"]:
                            config["additional_config"]["providers"] = {}
                        
                        # Create provider config
                        provider_config = {
                            "provider_type": provider["provider_type"],
                            "display_name": provider["display_name"],
                            "models": {}
                        }
                        
                        # Add connection parameters
                        if "connection_params" in provider:
                            provider_config.update(provider["connection_params"])
                        
                        # Add models
                        for model in provider["models"]:
                            if model["enabled"]:
                                model_config = {
                                    "display_name": model["display_name"],
                                    "context_window": model["context_window"],
                                    "max_tokens": model["max_tokens"]
                                }
                                
                                # Add model parameters
                                if "parameters" in model:
                                    model_config.update(model["parameters"])
                                
                                provider_config["models"][model["model_id"]] = model_config
                        
                        # Update the gateway configuration
                        config["additional_config"]["providers"][provider_id] = provider_config
                        
                        # Add to allowed providers if not already there
                        if "allowed_providers" not in config:
                            config["allowed_providers"] = []
                        
                        if provider_id not in config["allowed_providers"] and provider["enabled"]:
                            config["allowed_providers"].append(provider_id)
                        elif provider_id in config["allowed_providers"] and not provider["enabled"]:
                            config["allowed_providers"].remove(provider_id)
                else:
                    # Load all providers
                    providers = service.get_all_providers()
                    
                    # Update provider configurations
                    if "additional_config" not in config:
                        config["additional_config"] = {}
                    
                    if "providers" not in config["additional_config"]:
                        config["additional_config"]["providers"] = {}
                    
                    # Add to allowed providers if not already there
                    if "allowed_providers" not in config:
                        config["allowed_providers"] = []
                    
                    for provider in providers:
                        provider_id = provider["provider_id"]
                        
                        # Create provider config
                        provider_config = {
                            "provider_type": provider["provider_type"],
                            "display_name": provider["display_name"],
                            "models": {}
                        }
                        
                        # Add connection parameters
                        if "connection_params" in provider:
                            provider_config.update(provider["connection_params"])
                        
                        # Add models
                        for model in provider["models"]:
                            if model["enabled"]:
                                model_config = {
                                    "display_name": model["display_name"],
                                    "context_window": model["context_window"],
                                    "max_tokens": model["max_tokens"]
                                }
                                
                                # Add model parameters
                                if "parameters" in model:
                                    model_config.update(model["parameters"])
                                
                                provider_config["models"][model["model_id"]] = model_config
                        
                        # Update the gateway configuration
                        config["additional_config"]["providers"][provider_id] = provider_config
                        
                        # Add to allowed providers if not already there
                        if provider_id not in config["allowed_providers"] and provider["enabled"]:
                            config["allowed_providers"].append(provider_id)
                        elif provider_id in config["allowed_providers"] and not provider["enabled"]:
                            config["allowed_providers"].remove(provider_id)
            except Exception as e:
                logger.error(f"Error loading configuration from database: {e}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], use_db: bool = True) -> bool:
        """
        Save configuration to file and optionally to database.
        
        Args:
            config: Configuration dictionary
            use_db: Whether to use the database for configuration
            
        Returns:
            True if saved successfully, False otherwise
        """
        # Save to file
        result = self.save_to_yaml(config)
        
        # If database is available and use_db is True, save to database
        if self.db is not None and use_db:
            try:
                # Import here to avoid circular imports
                from asf.medical.llm_gateway.services.provider_service import ProviderService
                
                # Create provider service
                service = ProviderService(self.db)
                
                # Get providers from config
                providers = config.get("additional_config", {}).get("providers", {})
                allowed_providers = config.get("allowed_providers", [])
                
                # Update providers in database
                for provider_id, provider_config in providers.items():
                    # Check if provider exists
                    provider = service.get_provider_by_id(provider_id)
                    
                    # Create provider data
                    provider_data = {
                        "provider_id": provider_id,
                        "provider_type": provider_config.get("provider_type", "unknown"),
                        "display_name": provider_config.get("display_name", provider_id),
                        "enabled": provider_id in allowed_providers,
                        "connection_params": {k: v for k, v in provider_config.items() if k not in ["provider_type", "display_name", "models"]},
                        "models": []
                    }
                    
                    # Add models
                    for model_id, model_config in provider_config.get("models", {}).items():
                        model_data = {
                            "model_id": model_id,
                            "provider_id": provider_id,
                            "display_name": model_config.get("display_name", model_id),
                            "context_window": model_config.get("context_window"),
                            "max_tokens": model_config.get("max_tokens"),
                            "enabled": True,
                            "parameters": {k: v for k, v in model_config.items() if k not in ["display_name", "context_window", "max_tokens"]}
                        }
                        provider_data["models"].append(model_data)
                    
                    if provider:
                        # Update provider
                        service.update_provider(provider_id, provider_data)
                    else:
                        # Create provider
                        service.create_provider(provider_data)
            except Exception as e:
                logger.error(f"Error saving configuration to database: {e}")
                return False
        
        return result
    
    def load_from_yaml(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
        """
        try:
            # Check if file exists
            if not os.path.exists(self.config_path):
                logger.warning(f"Configuration file '{self.config_path}' not found, creating default configuration")
                return self._create_default_config()
            
            # Load configuration from file
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            
            # Validate configuration
            if not config:
                logger.warning(f"Configuration file '{self.config_path}' is empty, creating default configuration")
                return self._create_default_config()
            
            return config
        except Exception as e:
            logger.error(f"Error loading configuration from file: {e}")
            return self._create_default_config()
    
    def save_to_yaml(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            # Save configuration to file
            with open(self.config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            
            return True
        except Exception as e:
            logger.error(f"Error saving configuration to file: {e}")
            return False
    
    def sync_db_with_yaml(self) -> bool:
        """
        Synchronize database with YAML configuration.
        
        Returns:
            True if synchronized successfully, False otherwise
        """
        try:
            # Load configuration from file
            config = self.load_from_yaml()
            
            # Save configuration to database
            return self.save_config(config, use_db=True)
        except Exception as e:
            logger.error(f"Error synchronizing database with YAML configuration: {e}")
            return False
    
    def sync_yaml_with_db(self) -> bool:
        """
        Synchronize YAML configuration with database.
        
        Returns:
            True if synchronized successfully, False otherwise
        """
        try:
            # Load configuration from database
            config = self.load_config(use_db=True)
            
            # Save configuration to file
            return self.save_to_yaml(config)
        except Exception as e:
            logger.error(f"Error synchronizing YAML configuration with database: {e}")
            return False
    
    def get_provider_config(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_id: Provider ID
            
        Returns:
            Provider configuration dictionary or None if not found
        """
        try:
            # Load configuration
            config = self.load_config(provider_id=provider_id)
            
            # Get provider configuration
            provider_config = config.get("additional_config", {}).get("providers", {}).get(provider_id)
            
            if not provider_config:
                return None
            
            # Add provider ID
            provider_config["provider_id"] = provider_id
            
            # Add enabled flag
            provider_config["enabled"] = provider_id in config.get("allowed_providers", [])
            
            return provider_config
        except Exception as e:
            logger.error(f"Error getting provider configuration: {e}")
            return None
    
    def update_provider_config(self, provider_id: str, provider_config: Dict[str, Any]) -> bool:
        """
        Update configuration for a specific provider.
        
        Args:
            provider_id: Provider ID
            provider_config: Provider configuration dictionary
            
        Returns:
            True if updated successfully, False otherwise
        """
        try:
            # Load configuration
            config = self.load_config()
            
            # Update provider configuration
            if "additional_config" not in config:
                config["additional_config"] = {}
            
            if "providers" not in config["additional_config"]:
                config["additional_config"]["providers"] = {}
            
            # Create provider config
            new_provider_config = {
                "provider_type": provider_config.get("provider_type", "unknown"),
                "display_name": provider_config.get("display_name", provider_id),
                "models": provider_config.get("models", {})
            }
            
            # Add connection parameters
            for key, value in provider_config.items():
                if key not in ["provider_id", "provider_type", "display_name", "models", "enabled"]:
                    new_provider_config[key] = value
            
            # Update the gateway configuration
            config["additional_config"]["providers"][provider_id] = new_provider_config
            
            # Add to allowed providers if not already there
            if "allowed_providers" not in config:
                config["allowed_providers"] = []
            
            if provider_config.get("enabled", True):
                if provider_id not in config["allowed_providers"]:
                    config["allowed_providers"].append(provider_id)
            else:
                if provider_id in config["allowed_providers"]:
                    config["allowed_providers"].remove(provider_id)
            
            # Save configuration
            return self.save_config(config)
        except Exception as e:
            logger.error(f"Error updating provider configuration: {e}")
            return False
    
    def load_config_as_object(self) -> 'GatewayConfig':
        """
        Load configuration as a GatewayConfig object.
        
        Returns:
            GatewayConfig object
        """
        try:
            # Import here to avoid circular imports
            from asf.medical.llm_gateway.core.models import GatewayConfig
            
            # Load configuration
            config_dict = self.load_config()
            
            # Create GatewayConfig object
            return GatewayConfig(**config_dict)
        except Exception as e:
            logger.error(f"Error loading configuration as object: {e}")
            # Import here to avoid circular imports
            from asf.medical.llm_gateway.core.models import GatewayConfig
            
            # Create default GatewayConfig object
            return GatewayConfig(**self._create_default_config())
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "gateway_id": "default",
            "default_provider": "openai",
            "allowed_providers": ["openai"],
            "additional_config": {
                "providers": {
                    "openai": {
                        "provider_type": "openai",
                        "display_name": "OpenAI",
                        "api_base": "https://api.openai.com/v1",
                        "models": {
                            "gpt-4-turbo-preview": {
                                "display_name": "GPT-4 Turbo",
                                "context_window": 128000,
                                "max_tokens": 4096
                            },
                            "gpt-3.5-turbo": {
                                "display_name": "GPT-3.5 Turbo",
                                "context_window": 16385,
                                "max_tokens": 4096
                            }
                        }
                    }
                }
            }
        }
