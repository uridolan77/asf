"""
Service Registry module for the Medical Research Synthesizer.

This module provides a central registry for all services in the application,
enabling dependency injection and service lifecycle management.

Classes:
    ServiceRegistry: Central registry for managing application services.
"""

import inspect
from typing import Dict, Type, TypeVar, Optional, Any, Callable, List, Set
from fastapi import Depends
from .logging_config import get_logger

T = TypeVar('T')
logger = get_logger(__name__)

class ServiceRegistry:
    """
    Central registry for managing application services.
    
    This class provides methods for registering, retrieving, and managing the lifecycle
    of services throughout the application, supporting dependency injection.
    
    Attributes:
        _factories (Dict[Type, Callable]): Service factory functions.
        _services (Dict[Type, Any]): Initialized service instances.
        _dependencies (Dict[Type, List[Type]]): Service dependencies.
        _initialized (bool): Whether the registry has been initialized.
    """
    def __init__(self):
        """
        Initialize the ServiceRegistry.
        """
        self._services = {}
        self._factories = {}
        self._singletons = {}
        self._dependencies = {}

    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """
        Register an existing service instance.
        
        Args:
            service_type (Type): The service type (usually an interface or class).
            instance (Any): The service instance.
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
        Register a service factory.
        
        Args:
            service_type (Type): The type of service to register.
            factory (Callable): Factory function that creates the service.
            singleton (bool): Whether to cache and reuse the instance (True) or create new instances (False).
            dependencies (List[Type], optional): Dependencies required by the service. Defaults to None.
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
        Get an instance of a service.
        
        Args:
            service_type (Type): The type of service to get.
            
        Returns:
            Any: The service instance.
            
        Raises:
            KeyError: If the service type is not registered.
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
        """
        Reset the registry, clearing all services.
        """
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._dependencies.clear()

    def _sort_by_dependencies(self, service_types: List[Type]) -> List[Type]:
        """
        Sort service types by dependencies.
        
        Args:
            service_types (List[Type]): List of service types to sort.
            
        Returns:
            List[Type]: Sorted list of service types.
        """
        # Build a graph of dependencies
        graph = {service_type: self._dependencies.get(service_type, set()) for service_type in service_types}
        # Perform topological sort
        result = []
        visited = set()
        temp_visited = set()

        def visit(node):
            """
            Visit a node in the dependency graph.
            
            Args:
                node (Type): The service type node to visit.
            """
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
    """
    Get the global service registry.
    
    Returns:
        ServiceRegistry: The global service registry instance.
    """
    return _registry

def create_dependency(service_type: Type[T]) -> Callable[..., T]:
    """
    Create a FastAPI dependency provider for a service.
    
    Args:
        service_type (Type): The service type to provide.
        
    Returns:
        Callable[..., T]: A dependency function that returns the service instance.
    """
    def _get_service(registry: ServiceRegistry = Depends(get_registry)) -> T:
        return registry.get(service_type)
    return _get_service