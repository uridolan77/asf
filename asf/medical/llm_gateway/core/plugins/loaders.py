"""
Plugin Loaders for LLM Gateway.

This module provides utilities for loading plugins from various sources,
including configuration files, directories, and entry points.
"""

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import BasePlugin
from .registry import get_registry, PluginRegistry

logger = logging.getLogger(__name__)


async def discover_plugins(config_path: Optional[Union[str, Path]] = None) -> List[BasePlugin]:
    """
    Discover and load plugins from all available sources.
    
    Args:
        config_path: Path to the plugins config file (YAML)
        
    Returns:
        List of initialized plugin instances
    """
    registry = get_registry()
    discovered_plugins = []
    
    # Load plugins from config file if provided
    if config_path:
        plugins_config = load_plugins_config(config_path)
        if plugins_config:
            logger.info(f"Discovering plugins from config: {config_path}")
            config_plugins = await registry.discover_plugins_from_config(plugins_config)
            discovered_plugins.extend(config_plugins)
    
    # Load plugins from entry points
    logger.info("Discovering plugins from entry points")
    entry_point_plugins = await registry.discover_plugins_from_entry_points()
    discovered_plugins.extend(entry_point_plugins)
    
    # Load plugins from standard plugin directories
    plugin_dirs = [
        Path(__file__).parent.parent.parent / "plugins",  # llm_gateway/plugins
    ]
    
    for plugin_dir in plugin_dirs:
        if plugin_dir.exists() and plugin_dir.is_dir():
            logger.info(f"Discovering plugins from directory: {plugin_dir}")
            dir_plugins = await registry.discover_plugins_from_directory(plugin_dir)
            discovered_plugins.extend(dir_plugins)
    
    logger.info(f"Discovered {len(discovered_plugins)} plugins in total")
    return discovered_plugins


def load_plugins_config(config_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load plugin configuration from a YAML file.
    
    Args:
        config_path: Path to the plugins config file
        
    Returns:
        List of plugin configurations
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        plugins_config = config.get('plugins', [])
        if not isinstance(plugins_config, list):
            logger.error(f"Invalid plugins config in {config_path}: 'plugins' must be a list")
            return []
        
        return plugins_config
    
    except (yaml.YAMLError, IOError) as e:
        logger.error(f"Failed to load plugins config from {config_path}: {str(e)}")
        return []


async def register_plugin(plugin: BasePlugin, registry: Optional[PluginRegistry] = None) -> bool:
    """
    Register a plugin instance directly with the registry.
    
    Args:
        plugin: The plugin instance to register
        registry: Optional registry instance (uses global registry if not provided)
        
    Returns:
        True if registration was successful, False otherwise
    """
    if registry is None:
        registry = get_registry()
    
    try:
        await registry.register_plugin(plugin.__class__, plugin.config)
        return True
    
    except Exception as e:
        logger.error(f"Failed to register plugin {plugin.name}: {str(e)}")
        return False