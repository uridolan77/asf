"""
Utility functions for LLM API routers.

This module provides utility functions for LLM-related API endpoints.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Union
from fastapi import HTTPException, status
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                         "config", "llm")
GATEWAY_CONFIG_PATH = os.path.join(CONFIG_DIR, "llm_gateway_config.yaml")
GATEWAY_LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "llm_gateway_config.local.yaml")
DSPY_CONFIG_PATH = os.path.join(CONFIG_DIR, "dspy_config.yaml")
DSPY_LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "dspy_config.local.yaml")
BIOMEDLM_CONFIG_PATH = os.path.join(CONFIG_DIR, "biomedlm_config.yaml")
BIOMEDLM_LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "biomedlm_config.local.yaml")

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    If a local configuration file exists (with .local.yaml extension),
    it will be loaded and merged with the base configuration.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary

    Raises:
        HTTPException: If the configuration file cannot be loaded
    """
    try:
        # Load base configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Check if local configuration exists
        local_config_path = config_path.replace('.yaml', '.local.yaml')
        if os.path.exists(local_config_path):
            try:
                with open(local_config_path, 'r') as f:
                    local_config = yaml.safe_load(f)

                # Merge configurations
                if local_config:
                    logger.info(f"Merging local configuration from {local_config_path}")
                    config = deep_merge(config, local_config)
            except Exception as local_e:
                logger.warning(f"Error loading local configuration from {local_config_path}: {str(local_e)}")

        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load configuration: {str(e)}"
        )

def save_config(config_path: str, config: Dict[str, Any]) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config_path: Path to the configuration file
        config: Configuration dictionary

    Raises:
        HTTPException: If the configuration file cannot be saved
    """
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save configuration: {str(e)}"
        )

def format_timestamp() -> str:
    """
    Get the current timestamp in ISO format.

    Returns:
        Current timestamp in ISO format
    """
    return datetime.now(timezone.utc).isoformat()

def check_module_availability(module_name: str) -> bool:
    """
    Check if a Python module is available.

    Args:
        module_name: Name of the module to check

    Returns:
        True if the module is available, False otherwise
    """
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def get_llm_gateway_availability() -> bool:
    """
    Check if the LLM Gateway is available.

    Returns:
        True if the LLM Gateway is available, False otherwise
    """
    return check_module_availability("asf.medical.llm_gateway.core.client")

def get_dspy_availability() -> bool:
    """
    Check if DSPy is available.

    Returns:
        True if DSPy is available, False otherwise
    """
    return check_module_availability("asf.medical.ml.dspy.dspy_client")

def get_biomedlm_availability() -> bool:
    """
    Check if BiomedLM is available.

    Returns:
        True if BiomedLM is available, False otherwise
    """
    return check_module_availability("asf.medical.ml.models.biomedlm_adapter")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
            result[key] = deep_merge(result[key], value)
        else:
            # Otherwise, use the value from the override dictionary
            result[key] = value

    return result
