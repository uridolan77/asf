"""
Service Registry for ASF Medical Research Synthesizer.

This module provides a centralized registry for managing service dependencies.
It handles service instantiation, dependency resolution, and lifecycle management.
"""

import inspect
from typing import Dict, Type, TypeVar, Generic, Optional, Any, Callable, List, Set
from fastapi import Depends

from asf.medical.core.logging_config import get_logger

T = TypeVar('T')
logger = get_logger(__name__)

class ServiceRegistry:
    """
    Registry for managing service dependencies.
    
    This class provides a centralized registry for all services in the application.
    It handles service instantiation, dependency resolution, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize an empty service registry."""
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable[..., Any]] = {}
        self._singletons: Dict[Type, bool] = {}
        self._dependencies: Dict[Type, Set[Type]] = {}
        self._initialized = False
        
    def register(self, service_type: Type[T], instance: T) -> None:
        """
        Register an existing service instance.
        
        Args:
            service_type: The service type (usually an interface or class)
            instance: The service instance
        """
        self._services[service_type] = instance
        self._singletons[service_type] = True
        logger.debug(f"Registered service instance: {service_type.__name__}")
        
    def register_factory(
        self, 
        service_type: Type[T], 
        factory: Callable[..., T], 
        singleton: bool = True,
        dependencies: Optional[List[Type]] = None
    ) -> None:
        """
        Register a factory function for creating service instances.
        
        Args:
            service_type: The service type (usually an interface or class)
            factory: A function that creates instances of the service
            singleton: Whether to cache and reuse the instance (True) or create new instances (False)
            dependencies: Optional list of explicit dependencies for this service
        """
        self._factories[service_type] = factory
        self._singletons[service_type] = singleton
        
        # Register dependencies
        if dependencies:
            self._dependencies[service_type] = set(dependencies)
        else:
            # Try to infer dependencies from factory signature
            self._dependencies[service_type] = set()
            sig = inspect.signature(factory)
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty and param.annotation != Any:
                    self._dependencies[service_type].add(param.annotation)
        
        logger.debug(
            f"Registered factory for {service_type.__name__}",
            extra={
                "singleton": singleton,
                "dependencies": [dep.__name__ for dep in self._dependencies.get(service_type, set())]
            }
        )
    
    def get(self, service_type: Type[T]) -> T:
        """
        Get a service instance by type.
        
        Args:
            service_type: The service type to retrieve
            
        Returns:
            An instance of the requested service
            
        Raises:
            KeyError: If the service type is not registered
        """
        # Check if we already have an instance
        if service_type in self._services:
            return self._services[service_type]
        
        # Check if we have a factory
        if service_type in self._factories:
            factory = self._factories[service_type]
            
            # Resolve dependencies for the factory
            sig = inspect.signature(factory)
            kwargs = {}
            
            for param_name, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty and param.annotation != Any:
                    # Try to get the dependency from the registry
                    try:
                        kwargs[param_name] = self.get(param.annotation)
                    except KeyError:
                        # If not found, leave it to be injected by FastAPI
                        pass
            
            # Create the instance
            instance = factory(**kwargs)
            
            # Cache if it's a singleton
            if self._singletons.get(service_type, True):
                self._services[service_type] = instance
            
            logger.debug(f"Created instance of {service_type.__name__}")
            return instance
        
        raise KeyError(f"Service type {service_type.__name__} not registered")
    
    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._dependencies.clear()
        self._initialized = False
        logger.debug("Service registry cleared")
    
    def initialize(self) -> None:
        """
        Initialize all singleton services.
        
        This method creates instances of all registered singleton services
        to ensure they are ready for use.
        """
        if self._initialized:
            logger.warning("Service registry already initialized")
            return
        
        # Find all service types that are singletons
        singleton_types = [
            service_type for service_type, is_singleton in self._singletons.items()
            if is_singleton and service_type not in self._services
        ]
        
        # Sort by dependencies to initialize in the correct order
        sorted_types = self._topological_sort(singleton_types)
        
        # Initialize each service
        for service_type in sorted_types:
            try:
                self.get(service_type)
            except Exception as e:
                logger.error(
                    f"Error initializing service {service_type.__name__}",
                    extra={"error": str(e)},
                    exc_info=e
                )
                raise
        
        self._initialized = True
        logger.info(f"Initialized {len(sorted_types)} services")
    
    def _topological_sort(self, service_types: List[Type]) -> List[Type]:
        """
        Sort service types by dependencies.
        
        Args:
            service_types: List of service types to sort
            
        Returns:
            Sorted list of service types
        """
        # Build a graph of dependencies
        graph = {service_type: self._dependencies.get(service_type, set()) for service_type in service_types}
        
        # Perform topological sort
        result = []
        visited = set()
        temp_visited = set()
        
        def visit(node):
            if node in temp_visited:
                # Circular dependency detected
                cycle = " -> ".join(t.__name__ for t in temp_visited) + " -> " + node.__name__
                logger.warning(f"Circular dependency detected: {cycle}")
                return
            
            if node not in visited:
                temp_visited.add(node)
                for dependency in graph.get(node, set()):
                    if dependency in graph:
                        visit(dependency)
                temp_visited.remove(node)
                visited.add(node)
                result.append(node)
        
        for node in graph:
            if node not in visited:
                visit(node)
        
        return result


# Global registry instance
_registry = ServiceRegistry()

def get_registry() -> ServiceRegistry:
    """Get the global service registry."""
    return _registry

def get_service(service_type: Type[T]) -> Callable[..., T]:
    """
    Create a FastAPI dependency provider for a service.
    
    Args:
        service_type: The service type to provide
        
    Returns:
        A dependency function that returns the service instance
    """
    def _get_service(registry: ServiceRegistry = Depends(get_registry)) -> T:
        return registry.get(service_type)
    
    return _get_service
