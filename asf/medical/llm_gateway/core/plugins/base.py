"""
Base Plugin Interface for LLM Gateway.

This module defines the base interface that all plugins must implement to be
compatible with the LLM Gateway plugin registry.
"""

import abc
import logging
from typing import Any, Dict, List, Literal, Optional, Set, Union

logger = logging.getLogger(__name__)

# Define possible plugin categories
PluginCategory = Literal[
    "intervention", 
    "transport", 
    "provider", 
    "metric", 
    "resilience", 
    "observability", 
    "extension"
]

# Define event types that plugins can respond to
PluginEventType = Literal[
    "startup", 
    "shutdown", 
    "request_start",
    "request_end", 
    "response_start",
    "response_end",
    "stream_chunk",
    "error",
    "custom"
]

class BasePlugin(abc.ABC):
    """
    Abstract base class for all plugins in the LLM Gateway.
    
    All plugins must subclass this and implement the required methods.
    Plugins can be discovered, loaded, and managed by the PluginRegistry.
    """
    
    # Plugin metadata (should be overridden by subclasses)
    name: str = "base_plugin"             # Unique identifier
    display_name: str = "Base Plugin"     # Human-readable name
    category: PluginCategory = "extension"  # Plugin category
    priority: int = 100                   # Lower numbers run earlier
    description: str = ""                 # Plugin description
    version: str = "0.1.0"                # Plugin version
    author: str = ""                      # Plugin author
    tags: Set[str] = set()                # Tags for plugin filtering
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the plugin with optional configuration.
        
        Args:
            config: Dictionary containing configuration for this plugin,
                   loaded from the gateway configuration.
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        logger.info(f"Initialized plugin: {self.name} (Category: {self.category}, Priority: {self.priority})")
        
    async def initialize(self) -> None:
        """
        Initialize the plugin. Called once when the plugin is loaded.
        
        This is where you should perform any asynchronous setup like
        loading models, connecting to services, etc.
        """
        pass
        
    @abc.abstractmethod
    async def on_event(self, event_type: PluginEventType, payload: Any) -> Any:
        """
        Process an event from the gateway.
        
        This is the main entrypoint for plugin execution. The gateway will
        call this method with various event types throughout its lifecycle.
        
        Args:
            event_type: The type of event being processed
            payload: Event-specific data (request, response, error, etc.)
            
        Returns:
            Varies by event type and plugin category. May be ignored or 
            used to modify the payload, depending on the plugin's purpose.
        """
        pass
    
    async def shutdown(self) -> None:
        """
        Clean up resources. Called when the plugin is being unloaded.
        
        Override this to release resources, close connections, etc.
        """
        pass


class PluginError(Exception):
    """Base exception for plugin-related errors."""
    pass


class PluginNotFoundError(PluginError):
    """Raised when a plugin could not be found."""
    pass


class PluginInitializationError(PluginError):
    """Raised when a plugin fails to initialize."""
    pass


class PluginExecutionError(PluginError):
    """Raised when a plugin fails during execution."""
    pass