"""
Plugin system for the Conexus LLM Gateway.

This module provides the infrastructure for creating and managing plugins
that can extend the functionality of the LLM Gateway.
"""

import abc
import logging
from enum import Enum
from typing import Any, Dict, Optional, Set, List, Type, Union, Callable

logger = logging.getLogger(__name__)


class PluginCategory(str, Enum):
    """Categories of plugins for the LLM Gateway."""
    REQUEST_PROCESSING = "request_processing"  # Modifies requests before reaching providers
    RESPONSE_PROCESSING = "response_processing"  # Modifies responses after receiving from providers
    PROVIDER = "provider"  # Affects provider selection or behavior
    OBSERVABILITY = "observability"  # Monitoring, logging, metrics
    SECURITY = "security"  # Security, auth, permissions
    GENERAL = "general"  # General purpose
    SYSTEM = "system"  # System-level functionality


class PluginEventType(str, Enum):
    """Event types that plugins can respond to."""
    GATEWAY_INIT = "gateway_init"
    GATEWAY_SHUTDOWN = "gateway_shutdown"
    REQUEST_START = "request_start"
    REQUEST_END = "request_end"
    RESPONSE_START = "response_start"
    RESPONSE_END = "response_end"
    STREAM_START = "stream_start" 
    STREAM_CHUNK = "stream_chunk"
    STREAM_END = "stream_end"
    ERROR = "error"
    PROVIDER_SELECTED = "provider_selected"
    MODEL_SELECTED = "model_selected"
    CUSTOM = "custom"


class BasePlugin(abc.ABC):
    """
    Base class for all LLM Gateway plugins.
    
    Plugins can hook into various events in the request/response
    lifecycle to add functionality like caching, logging, rate limiting, etc.
    """
    
    # Class-level attributes to be defined by subclasses
    name: str = "base_plugin"  # Unique identifier for the plugin
    display_name: str = "Base Plugin"  # Human-readable name
    category: PluginCategory = PluginCategory.GENERAL  # Plugin category
    priority: int = 50  # Execution priority (0-100, lower runs first)
    description: str = "Base plugin implementation"  # Plugin description
    version: str = "1.0.0"  # Plugin version
    author: str = "Conexus Team"  # Plugin author
    tags: Set[str] = {"base"}  # Tags for categorization
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        self.config = config or {}
        self._enabled = self.config.get("enabled", True)
        logger.debug(f"Initialized plugin: {self.name} (enabled: {self._enabled})")
    
    async def initialize(self) -> None:
        """
        Initialize the plugin. Called when the gateway starts.
        Override in subclasses as needed.
        """
        pass
    
    @property
    def enabled(self) -> bool:
        """Check if the plugin is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the enabled state of the plugin."""
        self._enabled = value
        logger.info(f"Plugin {self.name} {'enabled' if value else 'disabled'}")
    
    async def on_event(self, event_type: PluginEventType, payload: Any) -> Any:
        """
        Handle an event in the gateway.
        
        Args:
            event_type: The type of event
            payload: Event data (depends on event type)
            
        Returns:
            Modified payload or None
        """
        # Default implementation does nothing
        return payload
    
    async def shutdown(self) -> None:
        """
        Clean up plugin resources. Called when the gateway shuts down.
        Override in subclasses as needed.
        """
        pass


class PluginRegistry:
    """
    Registry for managing LLM Gateway plugins.
    
    The registry maintains a list of available plugins,
    handles their initialization, and dispatches events to them.
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._plugins: Dict[str, BasePlugin] = {}
        self._enabled_plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        logger.info("Plugin registry initialized")
    
    def register_plugin_class(self, plugin_class: Type[BasePlugin]) -> None:
        """
        Register a plugin class in the registry.
        
        Args:
            plugin_class: The plugin class to register
        """
        plugin_name = plugin_class.name
        if plugin_name in self._plugin_classes:
            logger.warning(f"Plugin class {plugin_name} already registered, overwriting")
        
        self._plugin_classes[plugin_name] = plugin_class
        logger.debug(f"Registered plugin class: {plugin_name}")
    
    def register_plugin(self, plugin: BasePlugin) -> None:
        """
        Register a plugin instance in the registry.
        
        Args:
            plugin: The plugin instance to register
        """
        plugin_name = plugin.name
        if plugin_name in self._plugins:
            logger.warning(f"Plugin {plugin_name} already registered, overwriting")
        
        self._plugins[plugin_name] = plugin
        if plugin.enabled:
            self._enabled_plugins[plugin_name] = plugin
            
        logger.debug(f"Registered plugin instance: {plugin_name} (enabled: {plugin.enabled})")
    
    def create_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> BasePlugin:
        """
        Create a plugin instance from a registered class.
        
        Args:
            plugin_name: Name of the plugin class to instantiate
            config: Configuration for the plugin
            
        Returns:
            The created plugin instance
            
        Raises:
            ValueError: If the plugin class is not registered
        """
        if plugin_name not in self._plugin_classes:
            raise ValueError(f"Plugin class {plugin_name} not registered")
        
        plugin_class = self._plugin_classes[plugin_name]
        plugin = plugin_class(config)
        self.register_plugin(plugin)
        return plugin
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """
        Get a registered plugin by name.
        
        Args:
            plugin_name: Name of the plugin to retrieve
            
        Returns:
            The plugin instance or None if not found
        """
        return self._plugins.get(plugin_name)
    
    def get_plugins_by_category(self, category: PluginCategory) -> List[BasePlugin]:
        """
        Get all registered plugins in a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of plugin instances in the category
        """
        return [p for p in self._plugins.values() if p.category == category]
    
    def get_plugins_by_tag(self, tag: str) -> List[BasePlugin]:
        """
        Get all registered plugins with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of plugin instances with the tag
        """
        return [p for p in self._plugins.values() if tag in p.tags]
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """
        Enable a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to enable
            
        Returns:
            True if successful, False if plugin not found
        """
        plugin = self._plugins.get(plugin_name)
        if plugin is None:
            return False
        
        plugin.enabled = True
        self._enabled_plugins[plugin_name] = plugin
        return True
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """
        Disable a plugin by name.
        
        Args:
            plugin_name: Name of the plugin to disable
            
        Returns:
            True if successful, False if plugin not found
        """
        plugin = self._plugins.get(plugin_name)
        if plugin is None:
            return False
        
        plugin.enabled = False
        if plugin_name in self._enabled_plugins:
            del self._enabled_plugins[plugin_name]
        return True
    
    async def dispatch_event(self, event_type: PluginEventType, payload: Any) -> Any:
        """
        Dispatch an event to all enabled plugins.
        
        Plugins are called in order of priority (lowest to highest).
        
        Args:
            event_type: The type of event
            payload: Event data (depends on event type)
            
        Returns:
            The potentially modified payload after all plugins have processed it
        """
        # Sort plugins by priority (lower numbers run first)
        sorted_plugins = sorted(
            self._enabled_plugins.values(),
            key=lambda p: p.priority
        )
        
        current_payload = payload
        
        for plugin in sorted_plugins:
            try:
                result = await plugin.on_event(event_type, current_payload)
                # Update payload if plugin returned something (not None)
                if result is not None:
                    current_payload = result
            except Exception as e:
                logger.error(f"Error in plugin {plugin.name} during {event_type} event: {e}", exc_info=True)
                # Continue with other plugins despite errors
        
        return current_payload
    
    async def initialize_plugins(self) -> None:
        """Initialize all registered plugins."""
        for plugin_name, plugin in self._plugins.items():
            try:
                await plugin.initialize()
                logger.debug(f"Initialized plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin_name}: {e}", exc_info=True)
                # Disable plugin if initialization fails
                plugin.enabled = False
                if plugin_name in self._enabled_plugins:
                    del self._enabled_plugins[plugin_name]
    
    async def shutdown_plugins(self) -> None:
        """Shut down all registered plugins."""
        for plugin_name, plugin in self._plugins.items():
            try:
                await plugin.shutdown()
                logger.debug(f"Shut down plugin: {plugin_name}")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin_name}: {e}", exc_info=True)


# Singleton instance for global access
_plugin_registry = None


def get_plugin_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    Returns:
        The plugin registry singleton
    """
    global _plugin_registry
    if _plugin_registry is None:
        _plugin_registry = PluginRegistry()
    return _plugin_registry