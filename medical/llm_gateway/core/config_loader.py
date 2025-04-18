"""
Configuration loader for the LLM Gateway.

This module provides functionality to load configuration from the database.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Configuration loader for the LLM Gateway.

    This class provides functionality to load configuration from the database
    and fall back to YAML files if database configuration is not available.
    """

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize the configuration loader.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.config_cache = {}

    def load_from_db(self, provider_id: str = None) -> Dict[str, Any]:
        """
        Load configuration from the database.

        Args:
            provider_id: Optional provider ID to load configuration for

        Returns:
            Configuration dictionary
        """
        if not self.db:
            logger.warning("Database session not provided, falling back to YAML configuration")
            return self.load_from_yaml()

        try:
            # Import here to avoid circular imports
            from asf.bo.backend.repositories.provider_repository import ProviderRepository
            from asf.bo.backend.repositories.configuration_repository import ConfigurationRepository
            from asf.bo.backend.utils.crypto import generate_key

            # Get encryption key (in production, this should be loaded from a secure source)
            encryption_key = generate_key()

            # Initialize repositories
            provider_repo = ProviderRepository(self.db, encryption_key)
            config_repo = ConfigurationRepository(self.db)

            # Build configuration dictionary
            config = {
                "gateway_id": "llm_gateway",
                "version": "1.0",
                "additional_config": {
                    "providers": {}
                }
            }

            # Get global configurations
            global_configs = config_repo.get_all_configurations()
            for global_config in global_configs:
                if global_config.config_key.startswith("llm_gateway."):
                    key = global_config.config_key.replace("llm_gateway.", "")
                    value = config_repo.get_configuration_value(global_config.config_key)
                    config[key] = value

            # Get providers
            if provider_id:
                providers = [provider_repo.get_provider_by_id(provider_id)]
                if not providers[0]:
                    logger.warning(f"Provider {provider_id} not found in database, falling back to YAML configuration")
                    return self.load_from_yaml(provider_id)
            else:
                providers = provider_repo.get_all_providers()

            # Build provider configurations
            for provider in providers:
                if not provider.enabled:
                    continue

                provider_config = {
                    "provider_type": provider.provider_type,
                    "display_name": provider.display_name,
                    "connection_params": {},
                    "models": {}
                }

                # Get connection parameters
                connection_params = provider_repo.get_connection_parameters_by_provider_id(provider.provider_id)
                for param in connection_params:
                    provider_config["connection_params"][param.param_name] = param.param_value

                # Get API key
                api_keys = provider_repo.get_api_keys_by_provider_id(provider.provider_id)
                if api_keys:
                    # Use the first active API key
                    api_key = api_keys[0]
                    key_value = provider_repo.get_decrypted_api_key(api_key.key_id)
                    if key_value:
                        provider_config["connection_params"]["api_key"] = key_value

                # Get models
                models = provider_repo.get_models_by_provider_id(provider.provider_id)
                for model in models:
                    if not model.enabled:
                        continue

                    provider_config["models"][model.model_id] = {
                        "display_name": model.display_name,
                        "model_type": model.model_type,
                        "context_window": model.context_window,
                        "max_tokens": model.max_tokens
                    }

                config["additional_config"]["providers"][provider.provider_id] = provider_config

            return config

        except Exception as e:
            logger.error(f"Error loading configuration from database: {e}")
            logger.info("Falling back to YAML configuration")
            return self.load_from_yaml(provider_id)

    def load_from_yaml(self, provider_id: str = None) -> Dict[str, Any]:
        """
        Load configuration from YAML files.

        Args:
            provider_id: Optional provider ID to load configuration for

        Returns:
            Configuration dictionary
        """
        try:
            # Get configuration directory
            config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                                    "bo", "backend", "config", "llm")
            gateway_config_path = os.path.join(config_dir, "llm_gateway_config.yaml")
            local_config_path = os.path.join(config_dir, "llm_gateway_config.local.yaml")

            # Load base configuration
            with open(gateway_config_path, 'r') as f:
                config = yaml.safe_load(f)

            # Ensure gateway_id is present
            if 'gateway_id' not in config:
                config['gateway_id'] = 'llm_gateway'

            # Check if local configuration exists
            if os.path.exists(local_config_path):
                try:
                    with open(local_config_path, 'r') as f:
                        local_config = yaml.safe_load(f)

                    # Merge configurations
                    if local_config:
                        logger.info(f"Merging local configuration from {local_config_path}")
                        config = self._deep_merge(config, local_config)
                except Exception as local_e:
                    logger.warning(f"Error loading local configuration: {str(local_e)}")

            # Filter by provider ID if specified
            if provider_id and "additional_config" in config and "providers" in config["additional_config"]:
                providers = config["additional_config"]["providers"]
                if provider_id in providers:
                    filtered_providers = {provider_id: providers[provider_id]}
                    config["additional_config"]["providers"] = filtered_providers
                else:
                    logger.warning(f"Provider {provider_id} not found in YAML configuration")

            return config

        except Exception as e:
            logger.error(f"Error loading configuration from YAML: {e}")
            return {}

    def load_config(self, provider_id: str = None, use_db: bool = True) -> Dict[str, Any]:
        """
        Load configuration from the database or YAML files.

        Args:
            provider_id: Optional provider ID to load configuration for
            use_db: Whether to try loading from the database first

        Returns:
            Configuration dictionary
        """
        # Check cache first
        cache_key = f"{provider_id}_{use_db}"
        if cache_key in self.config_cache:
            return self.config_cache[cache_key]

        # Load configuration
        if use_db and self.db:
            config = self.load_from_db(provider_id)
        else:
            config = self.load_from_yaml(provider_id)

        # Cache configuration
        self.config_cache[cache_key] = config

        return config

    def clear_cache(self):
        """Clear the configuration cache."""
        self.config_cache = {}

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.

        The override dictionary will be merged into the base dictionary.
        If a key exists in both dictionaries, the value from the override dictionary will be used,
        unless both values are dictionaries, in which case they will be merged recursively.

        Args:
            base: Base dictionary
            override: Override dictionary

        Returns:
            Merged dictionary
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # If both values are dictionaries, merge them recursively
                result[key] = self._deep_merge(result[key], value)
            else:
                # Otherwise, use the value from the override dictionary
                result[key] = value

        return result
