"""
Configuration management for MCP Provider.

This module provides a centralized configuration management system
with support for various sources (files, environment, secrets) and
validation using the Pydantic models.
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import structlog
import yaml
from pydantic import BaseModel, ValidationError

from .models import MCPConnectionConfig

# For type hints with generic return type
T = TypeVar('T', bound=BaseModel)

logger = structlog.get_logger("mcp_config.manager")


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""
    pass


class ConfigManager:
    """
    Configuration management system for MCP Provider.

    Features:
    - Configuration from multiple sources (files, env, secrets)
    - Validation using Pydantic models
    - Environment variable interpolation
    - Secret management integration
    - Default configuration profiles
    """

    def __init__(
        self,
        config_dir: Optional[str] = None,
        env_prefix: str = "MCP_",
        secrets_provider: Optional[str] = None,
        default_profile: str = "default"
    ):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
            env_prefix: Prefix for environment variables
            secrets_provider: Secret provider (vault, aws, none)
            default_profile: Default configuration profile to use
        """
        self.config_dir = config_dir or os.environ.get("MCP_CONFIG_DIR", "./config")
        self.env_prefix = env_prefix
        self.secrets_provider = secrets_provider or os.environ.get("MCP_SECRETS_PROVIDER", "none")
        self.default_profile = default_profile

        # Create config dir if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)

        # Initialize secret provider if specified
        self._init_secrets_provider()

        self.logger = logger.bind(
            config_dir=self.config_dir,
            secrets_provider=self.secrets_provider
        )

        self.logger.info(
            "Initialized configuration manager",
            env_prefix=env_prefix,
            default_profile=default_profile
        )

    def _init_secrets_provider(self) -> None:
        """Initialize the secrets provider based on configuration."""
        if self.secrets_provider.lower() == "vault":
            try:
                import hvac
                self.secrets_client = hvac.Client(
                    url=os.environ.get("VAULT_ADDR", "http://127.0.0.1:8200"),
                    token=os.environ.get("VAULT_TOKEN")
                )
                self.logger.info("Initialized HashiCorp Vault secrets provider")
            except (ImportError, Exception) as e:
                self.logger.warning(
                    "Failed to initialize Vault secrets provider, falling back to none",
                    error=str(e)
                )
                self.secrets_provider = "none"
                self.secrets_client = None

        elif self.secrets_provider.lower() == "aws":
            try:
                import boto3
                self.secrets_client = boto3.client("secretsmanager")
                self.logger.info("Initialized AWS Secrets Manager provider")
            except (ImportError, Exception) as e:
                self.logger.warning(
                    "Failed to initialize AWS secrets provider, falling back to none",
                    error=str(e)
                )
                self.secrets_provider = "none"
                self.secrets_client = None

        else:
            self.secrets_provider = "none"
            self.secrets_client = None
            self.logger.info("Using no secrets provider (secrets from environment only)")

    def load_config(
        self,
        model_class: Type[T],
        profile: Optional[str] = None,
        config_file: Optional[str] = None,
        allow_env_override: bool = True,
        allow_secrets: bool = True
    ) -> T:
        """
        Load and validate configuration.

        Args:
            model_class: Pydantic model class for validation
            profile: Configuration profile name (default if None)
            config_file: Specific configuration file to load
            allow_env_override: Whether to allow environment override
            allow_secrets: Whether to resolve secrets

        Returns:
            Validated configuration object

        Raises:
            ConfigurationError: On configuration loading or validation error
        """
        # Determine profile and config file path
        profile = profile or self.default_profile

        if config_file is None:
            # Use model class name to determine config file
            model_name = model_class.__name__
            # Convert CamelCase to snake_case for filename
            config_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_name).lower()
            config_file = f"{config_name}.yaml"

        # Build absolute path
        config_path = os.path.join(self.config_dir, config_file)

        # Load raw configuration
        try:
            raw_config = self._load_raw_config(config_path, profile)

            # Process environment overrides if allowed
            if allow_env_override:
                raw_config = self._apply_env_overrides(raw_config)

            # Resolve secrets if allowed
            if allow_secrets:
                raw_config = self._resolve_secrets(raw_config)

            # Interpolate environment variables in string values
            raw_config = self._interpolate_env_vars(raw_config)

            # Validate using model
            validated_config = model_class.parse_obj(raw_config)

            return validated_config

        except (FileNotFoundError, ValidationError, ValueError, json.JSONDecodeError, yaml.YAMLError) as e:
            self.logger.error(
                "Failed to load configuration",
                config_file=config_file,
                profile=profile,
                error=str(e),
                exc_info=True
            )
            raise ConfigurationError(f"Failed to load configuration from {config_file}: {str(e)}")

    def load_mcp_config(
        self,
        provider_id: str,
        profile: Optional[str] = None,
        allow_env_override: bool = True,
        allow_secrets: bool = True
    ) -> MCPConnectionConfig:
        """
        Load MCP connection configuration.

        This is a convenience method specifically for loading the
        standard MCP connection configuration.

        Args:
            provider_id: Provider ID to load config for
            profile: Configuration profile name (default if None)
            allow_env_override: Whether to allow environment override
            allow_secrets: Whether to resolve secrets

        Returns:
            Validated MCP connection configuration
        """
        # Try provider-specific config first
        provider_config_file = f"mcp_{provider_id}.yaml"
        provider_config_path = os.path.join(self.config_dir, provider_config_file)

        if os.path.exists(provider_config_path):
            return self.load_config(
                MCPConnectionConfig,
                profile=profile,
                config_file=provider_config_file,
                allow_env_override=allow_env_override,
                allow_secrets=allow_secrets
            )
        else:
            # Fall back to generic config
            return self.load_config(
                MCPConnectionConfig,
                profile=profile,
                config_file="mcp_connection_config.yaml",
                allow_env_override=allow_env_override,
                allow_secrets=allow_secrets
            )

    def _load_raw_config(self, config_path: str, profile: str) -> Dict[str, Any]:
        """
        Load raw configuration from file.

        Args:
            config_path: Path to configuration file
            profile: Profile name to load

        Returns:
            Raw configuration dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If profile doesn't exist in file
        """
        if not os.path.exists(config_path):
            self.logger.warning(
                "Configuration file not found",
                config_path=config_path
            )
            return {}  # Return empty config

        # Load file based on extension
        config_ext = os.path.splitext(config_path)[1].lower()

        try:
            with open(config_path, 'r') as f:
                if config_ext in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_ext == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_ext}")

            # Handle profile-based configuration
            if isinstance(config_data, dict) and "profiles" in config_data:
                profiles = config_data["profiles"]
                if profile in profiles:
                    config = profiles[profile]

                    # Apply inheritance if specified
                    if "inherits" in config:
                        parent_profile = config.pop("inherits")
                        if parent_profile in profiles:
                            # Deep merge parent and child configs
                            parent_config = dict(profiles[parent_profile])
                            self._deep_merge(parent_config, config)
                            config = parent_config
                        else:
                            self.logger.warning(
                                "Parent profile not found",
                                profile=profile,
                                parent_profile=parent_profile
                            )

                    return config
                else:
                    self.logger.warning(
                        "Profile not found in config",
                        config_path=config_path,
                        profile=profile,
                        available_profiles=list(profiles.keys())
                    )
                    # Return empty if profile not found
                    return {}
            else:
                # Single configuration (no profiles)
                return config_data

        except (yaml.YAMLError, json.JSONDecodeError) as e:
            self.logger.error(
                "Failed to parse configuration file",
                config_path=config_path,
                error=str(e)
            )
            raise

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.

        Args:
            config: Raw configuration dictionary

        Returns:
            Configuration with environment overrides applied
        """
        # Create a copy to avoid modifying original
        result = dict(config)

        # Look for environment variables with the specified prefix
        for env_key, env_value in os.environ.items():
            if env_key.startswith(self.env_prefix):
                # Convert environment variable name to config path
                # Example: MCP_TRANSPORT_TYPE -> transport_type
                config_path = env_key[len(self.env_prefix):].lower()
                config_keys = config_path.split('_')

                # Try to convert value to appropriate type
                try:
                    # Try as JSON first
                    value = json.loads(env_value)
                except json.JSONDecodeError:
                    # If not valid JSON, use as string
                    value = env_value

                # Apply override using config path
                self._set_nested_value(result, config_keys, value)

        return result

    def _resolve_secrets(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve secret references in configuration.

        Args:
            config: Raw configuration dictionary

        Returns:
            Configuration with secrets resolved
        """
        if self.secrets_provider == "none" or self.secrets_client is None:
            return config

        # Create a deep copy to avoid modifying original
        result = self._deep_copy(config)

        # Process secret references
        return self._process_secret_refs(result)

    def _process_secret_refs(self, obj: Any) -> Any:
        """
        Recursively process secret references in an object.

        Args:
            obj: Object to process

        Returns:
            Object with secrets resolved
        """
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                if key == "_secret" and isinstance(value, str):
                    # This is a secret reference
                    return self._fetch_secret(value)
                else:
                    obj[key] = self._process_secret_refs(value)
            return obj
        elif isinstance(obj, list):
            return [self._process_secret_refs(item) for item in obj]
        else:
            return obj

    def _fetch_secret(self, secret_ref: str) -> Any:
        """
        Fetch a secret value from the configured secrets provider.

        Args:
            secret_ref: Secret reference string

        Returns:
            Secret value
        """
        try:
            if self.secrets_provider == "vault":
                # Parse vault path: "vault:path/to/secret:key"
                parts = secret_ref.split(":", 2)
                if len(parts) != 3 or parts[0] != "vault":
                    raise ValueError(f"Invalid Vault secret reference: {secret_ref}")

                path, key = parts[1], parts[2]
                response = self.secrets_client.secrets.kv.v2.read_secret_version(path=path)
                if response and "data" in response and "data" in response["data"]:
                    data = response["data"]["data"]
                    if key in data:
                        return data[key]
                    else:
                        raise ValueError(f"Secret key '{key}' not found in '{path}'")
                else:
                    raise ValueError(f"Secret not found at path '{path}'")

            elif self.secrets_provider == "aws":
                # Parse AWS path: "aws:secret-name:key"
                parts = secret_ref.split(":", 2)
                if len(parts) != 3 or parts[0] != "aws":
                    raise ValueError(f"Invalid AWS secret reference: {secret_ref}")

                secret_name, key = parts[1], parts[2]
                response = self.secrets_client.get_secret_value(SecretId=secret_name)

                if "SecretString" in response:
                    secret_data = json.loads(response["SecretString"])
                    if key in secret_data:
                        return secret_data[key]
                    else:
                        raise ValueError(f"Secret key '{key}' not found in '{secret_name}'")
                else:
                    raise ValueError(f"Secret not found: '{secret_name}'")

            else:
                raise ValueError(f"Unsupported secrets provider: {self.secrets_provider}")

        except Exception as e:
            self.logger.error(
                "Failed to fetch secret",
                secret_ref=secret_ref,
                provider=self.secrets_provider,
                error=str(e)
            )
            # Return placeholder to avoid exposing errors
            return f"<secret:{secret_ref}>"

    def _interpolate_env_vars(self, obj: Any) -> Any:
        """
        Interpolate environment variables in string values.

        Replaces ${ENV_VAR} or $ENV_VAR with environment variable values.

        Args:
            obj: Object to process

        Returns:
            Object with environment variables interpolated
        """
        if isinstance(obj, str):
            # Pattern for ${VAR} and $VAR
            pattern = r'\${([^}]+)}|\$([a-zA-Z0-9_]+)'

            def replace_env_var(match):
                env_var = match.group(1) or match.group(2)
                return os.environ.get(env_var, match.group(0))

            return re.sub(pattern, replace_env_var, obj)
        elif isinstance(obj, dict):
            return {key: self._interpolate_env_vars(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._interpolate_env_vars(item) for item in obj]
        else:
            return obj

    def _set_nested_value(self, obj: Dict[str, Any], keys: List[str], value: Any) -> None:
        """
        Set a nested value in a dictionary using a list of keys.

        Args:
            obj: Dictionary to modify
            keys: List of keys forming the path
            value: Value to set
        """
        if not keys:
            return

        if len(keys) == 1:
            obj[keys[0]] = value
            return

        if keys[0] not in obj or not isinstance(obj[keys[0]], dict):
            obj[keys[0]] = {}

        self._set_nested_value(obj[keys[0]], keys[1:], value)

    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep merge two dictionaries, modifying target in-place.

        Args:
            target: Target dictionary to merge into
            source: Source dictionary to merge from
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = self._deep_copy(value)

    def _deep_copy(self, obj: Any) -> Any:
        """
        Deep copy an object with basic types.

        Args:
            obj: Object to copy

        Returns:
            Deep copy of the object
        """
        if isinstance(obj, dict):
            return {key: self._deep_copy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj

    def save_config(
        self,
        config: BaseModel,
        config_file: str,
        profile: Optional[str] = None
    ) -> None:
        """
        Save configuration to file.

        Args:
            config: Configuration to save
            config_file: File to save to
            profile: Profile name (if None, saves as single config)

        Raises:
            ConfigurationError: On save error
        """
        # Build absolute path
        config_path = os.path.join(self.config_dir, config_file)

        # Convert to dictionary
        config_dict = config.dict()

        # Wrap in profile if specified
        if profile is not None:
            config_dict = {
                "profiles": {
                    profile: config_dict
                }
            }

        # Save file based on extension
        config_ext = os.path.splitext(config_path)[1].lower()

        try:
            with open(config_path, 'w') as f:
                if config_ext in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif config_ext == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {config_ext}")

            self.logger.info(
                "Saved configuration",
                config_file=config_file,
                profile=profile
            )

        except Exception as e:
            self.logger.error(
                "Failed to save configuration",
                config_file=config_file,
                profile=profile,
                error=str(e),
                exc_info=True
            )
            raise ConfigurationError(f"Failed to save configuration to {config_file}: {str(e)}")

    def get_available_profiles(self, config_file: str) -> List[str]:
        """
        Get available profiles in a configuration file.

        Args:
            config_file: Configuration file name

        Returns:
            List of profile names

        Raises:
            ConfigurationError: On file loading error
        """
        config_path = os.path.join(self.config_dir, config_file)

        if not os.path.exists(config_path):
            return []

        try:
            config_ext = os.path.splitext(config_path)[1].lower()

            with open(config_path, 'r') as f:
                if config_ext in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                elif config_ext == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_ext}")

            if isinstance(config_data, dict) and "profiles" in config_data:
                return list(config_data["profiles"].keys())
            else:
                return []

        except Exception as e:
            self.logger.error(
                "Failed to get available profiles",
                config_file=config_file,
                error=str(e),
                exc_info=True
            )
            raise ConfigurationError(f"Failed to read profiles from {config_file}: {str(e)}")