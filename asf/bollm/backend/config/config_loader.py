"""
Configuration Loader for LLM Management BO

This module handles loading and managing configuration settings from various sources
including environment variables, configuration files, and default values.
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "gateway_id": "bollm_default",
    "api": {
        "host": "0.0.0.0",
        "port": 8001
    },
    "cache": {
        "enabled": True,
        "storage_type": "memory",
        "ttl_seconds": 86400,  # 24 hours
        "max_size_bytes": 1073741824  # 1GB
    },
    "providers": {
        "openai": {
            "api_type": "openai",
            "api_base": "https://api.openai.com/v1"
        },
        "azure": {
            "api_type": "azure",
            "api_version": "2023-05-15"
        },
        "anthropic": {
            "api_type": "anthropic",
            "api_base": "https://api.anthropic.com"
        }
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    }
}

def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file (JSON or YAML).
    
    Args:
        file_path: Path to the configuration file
        
    Returns:
        Dict containing the configuration
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.warning(f"Config file {file_path} does not exist, using defaults")
        return {}
    
    try:
        if path.suffix.lower() in ('.yaml', '.yml'):
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        else:  # Default to JSON
            with open(path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config from {file_path}: {e}")
        return {}

def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with values from overlay taking precedence.
    
    Args:
        base: Base dictionary
        overlay: Dictionary to overlay on base
        
    Returns:
        Merged dictionary
    """
    result = base.copy()
    
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def load_config() -> Dict[str, Any]:
    """
    Load and merge configuration from different sources.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Default values
    
    Returns:
        Dict containing the merged configuration
    """
    # Start with default config
    config = DEFAULT_CONFIG.copy()
    
    # Load from config file if specified
    config_file = os.environ.get("BOLLM_CONFIG")
    if config_file:
        file_config = load_config_from_file(config_file)
        config = deep_merge(config, file_config)
    
    # Override with environment variables
    # API settings
    if "BOLLM_HOST" in os.environ:
        config["api"]["host"] = os.environ["BOLLM_HOST"]
    
    if "BOLLM_PORT" in os.environ:
        config["api"]["port"] = int(os.environ["BOLLM_PORT"])
    
    # Cache settings
    if "BOLLM_CACHE_ENABLED" in os.environ:
        config["cache"]["enabled"] = os.environ["BOLLM_CACHE_ENABLED"].lower() == "true"
    
    if "BOLLM_CACHE_DIR" in os.environ:
        config["cache"]["storage_path"] = os.environ["BOLLM_CACHE_DIR"]
    
    # Provider API keys
    for provider in ["OPENAI", "AZURE", "ANTHROPIC"]:
        env_key = f"{provider}_API_KEY"
        if env_key in os.environ and provider.lower() in config["providers"]:
            config["providers"][provider.lower()]["api_key"] = os.environ[env_key]
    
    # Logging settings
    if "BOLLM_LOG_LEVEL" in os.environ:
        config["logging"]["level"] = os.environ["BOLLM_LOG_LEVEL"].upper()
    
    logger.info(f"Configuration loaded successfully for gateway ID: {config.get('gateway_id')}")
    return config