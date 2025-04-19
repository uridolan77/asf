"""
Plugin Registry for LLM Gateway.

This module provides the central registry for managing plugins in the LLM Gateway.
It handles plugin discovery, registration, initialization, and execution.
"""

import asyncio
import importlib
import inspect
import logging
import threading
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from importlib.metadata import entry_points

from .base import (
    BasePlugin, PluginCategory, PluginEventType,
    PluginError, PluginNotFoundError, PluginInitializationError, PluginExecutionError
)

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BasePlugin)

# Global registry instance
_REGISTRY = None

def get_registry() -> 'PluginRegistry':
    """Get the global plugin registry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PluginRegistry()
    return _REGISTRY


class PluginRegistry:
    """
    Central registry for managing plugins in the LLM Gateway.
    
    This class provides methods for discovering, registering, and executing plugins.
    It also manages plugin lifecycle (initialization and shutdown).
    """
    
    def __init__(self):
        """Initialize the plugin registry."""
        self._lock = threading.RLock()
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_classes: Dict[str, Type[BasePlugin]] = {}
        self._startup_complete = False
        self._event_metrics: Dict[str, Dict[str, float]] = {}  # Track event timing
        self._plugin_failure_counts: Dict[str, int] = {}  # Track plugin failures
        logger.info("Plugin registry initialized")
    
    def register_plugin_class(self, plugin_class: Type[BasePlugin]) -> None:
        """
        Register a plugin class with the registry.
        
        Args:
            plugin_class: The plugin class to register
            
        Raises:
            PluginError: If a plugin with the same name is already registered
        """
        with self._lock:
            name = plugin_class.name
            if name in self._plugin_classes:
                raise PluginError(f"Plugin class '{name}' is already registered")
            
            self._plugin_classes[name] = plugin_class
            logger.info(f"Registered plugin class: {name} ({plugin_class.__module__}.{plugin_class.__name__})")
    
    async def register_plugin(self, 
                              plugin_class: Type[BasePlugin], 
                              config: Optional[Dict[str, Any]] = None) -> BasePlugin:
        """
        Register and initialize a plugin instance.
        
        Args:
            plugin_class: The plugin class to instantiate
            config: Optional configuration dictionary for the plugin
            
        Returns:
            The initialized plugin instance
            
        Raises:
            PluginError: If a plugin with the same name is already registered
            PluginInitializationError: If the plugin fails to initialize
        """
        with self._lock:
            name = plugin_class.name
            if name in self._plugins:
                raise PluginError(f"Plugin '{name}' is already registered")
            
            try:
                logger.info(f"Instantiating plugin: {name}")
                instance = plugin_class(config=config)
                
                # Initialize the plugin
                logger.info(f"Initializing plugin: {name}")
                await instance.initialize()
                
                self._plugins[name] = instance
                self._plugin_failure_counts[name] = 0  # Initialize failure count
                
                logger.info(f"Successfully registered plugin: {name}")
                return instance
            
            except Exception as e:
                logger.error(f"Failed to initialize plugin '{name}': {str(e)}", exc_info=True)
                raise PluginInitializationError(f"Failed to initialize plugin '{name}': {str(e)}") from e
    
    async def unregister_plugin(self, name: str) -> None:
        """
        Unregister a plugin, shutting it down gracefully.
        
        Args:
            name: Name of the plugin to unregister
            
        Raises:
            PluginNotFoundError: If the plugin is not registered
        """
        with self._lock:
            if name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{name}' is not registered")
            
            plugin = self._plugins[name]
            
            try:
                logger.info(f"Shutting down plugin: {name}")
                await plugin.shutdown()
                del self._plugins[name]
                logger.info(f"Unregistered plugin: {name}")
            
            except Exception as e:
                logger.error(f"Error shutting down plugin '{name}': {str(e)}", exc_info=True)
                # Still remove the plugin even if shutdown fails
                del self._plugins[name]
                logger.info(f"Forcibly unregistered plugin: {name}")
    
    def get_plugin(self, name: str) -> BasePlugin:
        """
        Get a registered plugin instance by name.
        
        Args:
            name: Name of the plugin to retrieve
            
        Returns:
            The plugin instance
            
        Raises:
            PluginNotFoundError: If the plugin is not registered
        """
        with self._lock:
            if name not in self._plugins:
                raise PluginNotFoundError(f"Plugin '{name}' is not registered")
            
            return self._plugins[name]
    
    def get_plugins_by_category(self, category: PluginCategory) -> List[BasePlugin]:
        """
        Get all registered plugins of a specific category.
        
        Args:
            category: The category to filter by
            
        Returns:
            List of plugin instances, sorted by priority
        """
        with self._lock:
            plugins = [p for p in self._plugins.values() 
                      if p.category == category and p.enabled]
            
            # Sort by priority (lower values first)
            return sorted(plugins, key=lambda p: p.priority)
    
    def get_plugins_by_tag(self, tag: str) -> List[BasePlugin]:
        """
        Get all registered plugins that have a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            List of plugin instances, sorted by priority
        """
        with self._lock:
            plugins = [p for p in self._plugins.values() 
                      if tag in p.tags and p.enabled]
            
            # Sort by priority (lower values first)
            return sorted(plugins, key=lambda p: p.priority)
    
    def get_all_plugins(self) -> List[BasePlugin]:
        """
        Get all registered plugins.
        
        Returns:
            List of all plugin instances
        """
        with self._lock:
            return list(self._plugins.values())
    
    async def discover_plugins_from_config(self, plugins_config: List[Dict[str, Any]]) -> List[BasePlugin]:
        """
        Discover plugins based on configuration.
        
        Args:
            plugins_config: List of plugin configurations with 'name', 'module', etc.
            
        Returns:
            List of initialized plugin instances
        """
        discovered_plugins = []
        
        for plugin_config in plugins_config:
            if not plugin_config.get("enabled", True):
                logger.info(f"Skipping disabled plugin: {plugin_config.get('name', 'unknown')}")
                continue
            
            try:
                # Get plugin class from module path
                if "module" in plugin_config:
                    module_path = plugin_config["module"]
                    class_name = plugin_config.get("class_name")
                    plugin_class = self._import_plugin_class(module_path, class_name)
                
                # Or get plugin class from entry point
                elif "entry_point" in plugin_config:
                    plugin_class = self._get_plugin_class_from_entry_point(plugin_config["entry_point"])
                
                else:
                    logger.error(f"Invalid plugin configuration: {plugin_config}")
                    continue
                
                # Register the plugin class
                self.register_plugin_class(plugin_class)
                
                # Create and initialize plugin instance
                instance = await self.register_plugin(
                    plugin_class, 
                    config=plugin_config.get("config", {})
                )
                
                discovered_plugins.append(instance)
                
            except Exception as e:
                logger.error(f"Failed to discover plugin {plugin_config.get('name', 'unknown')}: {str(e)}", exc_info=True)
        
        return discovered_plugins
    
    async def discover_plugins_from_entry_points(self, group: str = "llm_gateway.plugins") -> List[BasePlugin]:
        """
        Discover plugins from entry points.
        
        Args:
            group: Entry point group name to look for plugins
            
        Returns:
            List of initialized plugin instances
        """
        discovered_plugins = []
        
        try:
            # Use importlib.metadata to get entry points
            eps = entry_points()
            
            # Get entry points from the specified group
            if hasattr(eps, "select"):  # Python 3.10+ API
                plugin_entry_points = eps.select(group=group)
            else:  # Earlier Python versions
                plugin_entry_points = eps.get(group, [])
            
            for entry_point in plugin_entry_points:
                try:
                    # Load the plugin class
                    plugin_class = entry_point.load()
                    
                    if not inspect.isclass(plugin_class) or not issubclass(plugin_class, BasePlugin):
                        logger.warning(f"Entry point '{entry_point.name}' does not point to a BasePlugin subclass")
                        continue
                    
                    # Register the plugin class
                    self.register_plugin_class(plugin_class)
                    
                    # Create and initialize plugin instance
                    instance = await self.register_plugin(plugin_class)
                    discovered_plugins.append(instance)
                    
                except Exception as e:
                    logger.error(f"Failed to load plugin from entry point '{entry_point.name}': {str(e)}", exc_info=True)
            
        except Exception as e:
            logger.error(f"Failed to discover plugins from entry points: {str(e)}", exc_info=True)
        
        return discovered_plugins
    
    async def discover_plugins_from_directory(self, directory: Union[str, Path]) -> List[BasePlugin]:
        """
        Discover plugins from Python files in a directory.
        
        Args:
            directory: Directory to search for plugin modules
            
        Returns:
            List of initialized plugin instances
        """
        discovered_plugins = []
        directory = Path(directory)
        
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Plugin directory does not exist or is not a directory: {directory}")
            return []
        
        # Find all Python files in the directory
        for file_path in directory.rglob("*.py"):
            if file_path.name.startswith("_"):
                continue  # Skip __init__.py, etc.
            
            # Convert file path to module path
            rel_path = file_path.relative_to(directory.parent)
            module_path = str(rel_path).replace("/", ".").replace("\\", ".")[:-3]  # Remove .py extension
            
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Find all plugin classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BasePlugin) and obj != BasePlugin:
                        # Register the plugin class
                        try:
                            self.register_plugin_class(obj)
                            
                            # Create and initialize plugin instance
                            instance = await self.register_plugin(obj)
                            discovered_plugins.append(instance)
                            
                        except Exception as e:
                            logger.error(f"Failed to register plugin class '{name}' from {module_path}: {str(e)}", exc_info=True)
                
            except Exception as e:
                logger.error(f"Failed to import module {module_path}: {str(e)}", exc_info=True)
        
        return discovered_plugins
    
    async def dispatch_event(self, 
                             event_type: PluginEventType, 
                             payload: Any, 
                             category: Optional[PluginCategory] = None) -> List[Any]:
        """
        Dispatch an event to all relevant plugins.
        
        Args:
            event_type: Type of event to dispatch
            payload: Event-specific data
            category: Optional category to filter plugins
            
        Returns:
            List of plugin return values
        """
        start_time = time.time()
        results = []
        
        # Get plugins to notify (either all enabled plugins or filtered by category)
        if category:
            plugins = self.get_plugins_by_category(category)
        else:
            plugins = [p for p in self.get_all_plugins() if p.enabled]
        
        # Log the event dispatch
        logger.debug(f"Dispatching {event_type} event to {len(plugins)} plugins" + 
                    (f" of category {category}" if category else ""))
        
        # Process each plugin
        for plugin in plugins:
            plugin_start = time.time()
            plugin_name = plugin.name
            
            try:
                # Execute the plugin
                result = await plugin.on_event(event_type, payload)
                results.append(result)
                
                # Record successful execution metrics
                plugin_duration = (time.time() - plugin_start) * 1000  # ms
                self._record_plugin_execution(plugin_name, event_type, plugin_duration)
                
                # Clear failure count on success
                self._plugin_failure_counts[plugin_name] = 0
                
            except Exception as e:
                # Record failed execution
                self._plugin_failure_counts[plugin_name] = self._plugin_failure_counts.get(plugin_name, 0) + 1
                failure_count = self._plugin_failure_counts[plugin_name]
                
                logger.error(f"Plugin '{plugin_name}' failed on {event_type} event (failure #{failure_count}): {str(e)}", 
                             exc_info=True)
                
                # Disable plugin if it has failed too many times
                if failure_count >= 3:  # Threshold for disabling
                    logger.warning(f"Disabling plugin '{plugin_name}' after {failure_count} consecutive failures")
                    plugin.enabled = False
        
        # Record overall event metrics
        total_duration = (time.time() - start_time) * 1000  # ms
        event_key = f"{event_type}_{category if category else 'all'}"
        with self._lock:
            if event_key not in self._event_metrics:
                self._event_metrics[event_key] = {"count": 0, "total_ms": 0.0}
            
            self._event_metrics[event_key]["count"] += 1
            self._event_metrics[event_key]["total_ms"] += total_duration
        
        return results
    
    def _record_plugin_execution(self, 
                                plugin_name: str, 
                                event_type: PluginEventType, 
                                duration_ms: float) -> None:
        """Record metrics for a plugin execution."""
        key = f"{plugin_name}_{event_type}"
        
        with self._lock:
            if key not in self._event_metrics:
                self._event_metrics[key] = {
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": float('inf'),
                    "max_ms": 0.0
                }
            
            metrics = self._event_metrics[key]
            metrics["count"] += 1
            metrics["total_ms"] += duration_ms
            metrics["min_ms"] = min(metrics["min_ms"], duration_ms)
            metrics["max_ms"] = max(metrics["max_ms"], duration_ms)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about plugin execution."""
        with self._lock:
            metrics = {
                "registered_plugins": len(self._plugins),
                "registered_plugin_classes": len(self._plugin_classes),
                "event_metrics": {}
            }
            
            # Calculate averages
            for key, data in self._event_metrics.items():
                avg_ms = data["total_ms"] / data["count"] if data["count"] > 0 else 0
                
                if "min_ms" in data:
                    metrics["event_metrics"][key] = {
                        "count": data["count"],
                        "avg_ms": avg_ms,
                        "min_ms": data["min_ms"] if data["min_ms"] != float('inf') else 0,
                        "max_ms": data["max_ms"]
                    }
                else:
                    metrics["event_metrics"][key] = {
                        "count": data["count"],
                        "avg_ms": avg_ms
                    }
            
            return metrics
    
    async def shutdown_all(self) -> None:
        """Shut down all plugins gracefully."""
        logger.info("Shutting down all plugins")
        
        plugin_names = list(self._plugins.keys())
        for name in plugin_names:
            try:
                await self.unregister_plugin(name)
            except Exception as e:
                logger.error(f"Error unregistering plugin '{name}' during shutdown: {str(e)}", exc_info=True)
    
    def _import_plugin_class(self, 
                            module_path: str, 
                            class_name: Optional[str] = None) -> Type[BasePlugin]:
        """
        Import a plugin class from a module path.
        
        Args:
            module_path: Import path to the module
            class_name: Optional class name (if not provided, searches for BasePlugin subclasses)
            
        Returns:
            The plugin class
            
        Raises:
            ImportError: If the module or class cannot be imported
            TypeError: If the class is not a BasePlugin subclass
        """
        try:
            module = importlib.import_module(module_path)
            
            if class_name:
                # Get the specific class
                if not hasattr(module, class_name):
                    raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")
                
                plugin_class = getattr(module, class_name)
                
                if not inspect.isclass(plugin_class) or not issubclass(plugin_class, BasePlugin):
                    raise TypeError(f"Class '{module_path}.{class_name}' is not a BasePlugin subclass")
                
                return plugin_class
            
            else:
                # Find the first BasePlugin subclass
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, BasePlugin) and obj != BasePlugin:
                        return obj
                
                raise ImportError(f"No BasePlugin subclass found in module '{module_path}'")
            
        except ImportError as e:
            logger.error(f"Failed to import module '{module_path}': {str(e)}", exc_info=True)
            raise
        
        except TypeError as e:
            logger.error(f"Invalid plugin class in '{module_path}': {str(e)}", exc_info=True)
            raise
    
    def _get_plugin_class_from_entry_point(self, entry_point_name: str) -> Type[BasePlugin]:
        """
        Get a plugin class from an entry point.
        
        Args:
            entry_point_name: Name of the entry point
            
        Returns:
            The plugin class
            
        Raises:
            ImportError: If the entry point cannot be found
            TypeError: If the entry point does not point to a BasePlugin subclass
        """
        eps = entry_points()
        
        # Python 3.10+ API
        if hasattr(eps, "select"):
            matching_eps = [ep for group in ["llm_gateway.plugins"] 
                           for ep in eps.select(group=group) 
                           if ep.name == entry_point_name]
        # Earlier Python versions
        else:
            matching_eps = []
            for group in ["llm_gateway.plugins"]:
                for ep in eps.get(group, []):
                    if ep.name == entry_point_name:
                        matching_eps.append(ep)
        
        if not matching_eps:
            raise ImportError(f"Entry point '{entry_point_name}' not found")
        
        # Use the first matching entry point
        plugin_class = matching_eps[0].load()
        
        if not inspect.isclass(plugin_class) or not issubclass(plugin_class, BasePlugin):
            raise TypeError(f"Entry point '{entry_point_name}' does not point to a BasePlugin subclass")
        
        return plugin_class