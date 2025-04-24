"""
Plugin System for LLM Gateway.

This module provides a plugin architecture for extending the LLM Gateway
with custom functionality through plugins.
"""

from .base import (
    BasePlugin, PluginCategory, PluginEventType,
    PluginError, PluginNotFoundError, PluginInitializationError, PluginExecutionError
)
from .registry import get_registry, PluginRegistry
from .loaders import discover_plugins, register_plugin, load_plugins_config
from .adapters import (
    InterventionPluginAdapter, adapt_intervention, adapt_interventions
)

__all__ = [
    # Base plugin components
    'BasePlugin',
    'PluginCategory',
    'PluginEventType',
    'PluginError',
    'PluginNotFoundError',
    'PluginInitializationError',
    'PluginExecutionError',
    
    # Registry
    'get_registry',
    'PluginRegistry',
    
    # Loaders
    'discover_plugins',
    'register_plugin',
    'load_plugins_config',
    
    # Adapters
    'InterventionPluginAdapter',
    'adapt_intervention',
    'adapt_interventions',
]