#!/usr/bin/env python
# test_local_config.py

"""
Test script to verify that the local configuration is being used.
"""

import os
import sys
import yaml
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    logger.info(f"Added {project_root} to Python path")

def load_config(config_path, local_config_path=None):
    """
    Load configuration from a YAML file.
    
    If a local configuration file exists, it will be loaded and merged with the base configuration.
    
    Args:
        config_path: Path to the configuration file
        local_config_path: Path to the local configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Load base configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check if local configuration exists
        if local_config_path and os.path.exists(local_config_path):
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
        return None

def deep_merge(base, override):
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

def main():
    """Main function."""
    # Load configuration
    config_dir = os.path.join(project_root, "bo", "backend", "config", "llm")
    config_path = os.path.join(config_dir, "llm_gateway_config.yaml")
    local_config_path = os.path.join(config_dir, "llm_gateway_config.local.yaml")
    
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path, local_config_path)
    
    if not config:
        logger.error("Failed to load configuration")
        return 1
    
    # Check for API key in configuration
    providers = config.get("additional_config", {}).get("providers", {})
    openai_provider = providers.get("openai_gpt4_default", {})
    api_key = openai_provider.get("connection_params", {}).get("api_key")
    
    if api_key:
        logger.info(f"Found API key in configuration: {api_key[:5]}...{api_key[-4:]}")
        return 0
    else:
        logger.error("API key not found in configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
