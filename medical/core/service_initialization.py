"""
Service Initialization module for the Medical Research Synthesizer.

This module provides functionality for initializing and configuring services
within the application, ensuring proper startup and shutdown procedures.

Functions:
    initialize_service: Initialize a service with proper configuration.
    initialize_all_services: Initialize all required services for the application.
    shutdown_service: Properly shutdown a service.
    shutdown_all_services: Shutdown all running services.
    setup_dependencies: Set up service dependencies.
    check_service_health: Check the health status of a service.
"""

from typing import Any, Dict, List
from .service_registry import _registry as registry
from .logging_config import get_logger

logger = get_logger(__name__)

def initialize_services() -> None:
    """
    Initialize all services in the registry.
    
    This function registers all services with the registry and initializes them.
    It should be called during application startup.
    
    Returns:
        None
    """
    logger.info("Initializing services...")

    # Register repositories
    _register_repositories()

    # Register services
    _register_services()

    # Register ML services
    _register_ml_services()

    # Initialize all services
    registry.initialize()

    logger.info("Services initialized successfully")

def _register_repositories():
    """
    Register all repositories with the service registry.

    This function registers repository classes as factories in the service registry.
    Repositories are responsible for data access and storage operations.
    """
    from ..storage.repositories.user_repository import UserRepository
    from ..storage.repositories.query_repository import QueryRepository
    from ..storage.repositories.result_repository import ResultRepository
    from ..storage.repositories.kb_repository import KnowledgeBaseRepository

    # Register repositories as factories (they don't have dependencies)
    registry.register_factory(UserRepository, lambda: UserRepository())
    registry.register_factory(QueryRepository, lambda: QueryRepository())
    registry.register_factory(ResultRepository, lambda: ResultRepository())
    registry.register_factory(KnowledgeBaseRepository, lambda: KnowledgeBaseRepository())

    logger.debug("Repositories registered")

def _register_services():
    """
    Register all application services with the service registry.

    This function registers service classes with their dependencies in the service registry.
    Services provide the core business logic of the application.
    """
    from ..services.auth_service import AuthService
    from ..services.search_service import SearchService
    from ..services.analysis_service import AnalysisService
    from ..services.export_service import ExportService
    from ..services.knowledge_base_service import KnowledgeBaseService
    from ..storage.repositories.user_repository import UserRepository
    from ..storage.repositories.query_repository import QueryRepository
    from ..storage.repositories.result_repository import ResultRepository
    from ..storage.repositories.kb_repository import KnowledgeBaseRepository

    # Register services with their dependencies
    registry.register_factory(
        AuthService,
        lambda user_repo: AuthService(user_repo),
        dependencies=[UserRepository]
    )

    registry.register_factory(
        SearchService,
        lambda query_repo, result_repo: SearchService(query_repo, result_repo),
        dependencies=[QueryRepository, ResultRepository]
    )

    registry.register_factory(
        AnalysisService,
        lambda result_repo: AnalysisService(result_repo),
        dependencies=[ResultRepository]
    )

    registry.register_factory(
        ExportService,
        lambda: ExportService()
    )

    registry.register_factory(
        KnowledgeBaseService,
        lambda kb_repo: KnowledgeBaseService(kb_repo),
        dependencies=[KnowledgeBaseRepository]
    )

    logger.debug("Application services registered")

def _register_ml_services():
    """
    Register all ML services with the service registry.

    This function registers machine learning service classes in the service registry.
    ML services provide advanced data processing and analysis capabilities.
    """
    try:
        from ..ml.services.contradiction_service import ContradictionService
        from ..ml.services.prisma_screening_service import PRISMAScreeningService
        from ..ml.services.bias_assessment_service import BiasAssessmentService

        # Register ML services
        registry.register_factory(
            ContradictionService,
            lambda: ContradictionService()
        )

        registry.register_factory(
            PRISMAScreeningService,
            lambda: PRISMAScreeningService()
        )

        registry.register_factory(
            BiasAssessmentService,
            lambda: BiasAssessmentService()
        )

        logger.debug("ML services registered")
    except ImportError as e:
        logger.warning(f"Could not register ML services: {str(e)}")

def get_all_services():
    """
    Get a list of all registered services.

    This function retrieves all service types currently registered in the service registry.

    Returns:
        List[str]: List of service types registered in the registry.
    """
    return list(registry._factories.keys()) + list(registry._services.keys())

def initialize_service(service_name: str, config: Dict[str, Any] = None) -> bool:
    """
    Initialize a service with proper configuration.

    This function initializes a specific service with the provided configuration,
    performing any necessary setup steps to make it ready for use.

    Args:
        service_name (str): Name of the service to initialize.
        config (Dict[str, Any], optional): Configuration parameters for the service. Defaults to None.

    Returns:
        bool: True if initialization was successful, False otherwise.

    Raises:
        ServiceInitializationError: If initialization fails.
    """
    # Implementation goes here

def initialize_all_services(config: Dict[str, Any] = None) -> Dict[str, bool]:
    """
    Initialize all required services for the application.

    This function initializes all services required by the application
    with appropriate configurations and in the proper order.

    Args:
        config (Dict[str, Any], optional): Configuration parameters for all services. Defaults to None.

    Returns:
        Dict[str, bool]: Dictionary mapping service names to initialization success status.

    Raises:
        ServiceInitializationError: If critical services fail to initialize.
    """
    # Implementation goes here

def shutdown_service(service_name: str) -> bool:
    """
    Properly shutdown a service.

    This function performs a clean shutdown of a specific service,
    ensuring all resources are properly released.

    Args:
        service_name (str): Name of the service to shutdown.

    Returns:
        bool: True if shutdown was successful, False otherwise.
    """
    # Implementation goes here

def shutdown_all_services() -> Dict[str, bool]:
    """
    Shutdown all running services.

    This function performs a clean shutdown of all running services
    in the proper order to prevent dependency issues.

    Returns:
        Dict[str, bool]: Dictionary mapping service names to shutdown success status.
    """
    # Implementation goes here

def setup_dependencies(service_name: str, dependencies: List[str]) -> bool:
    """
    Set up service dependencies.

    This function ensures that all dependencies for a service are properly
    initialized before the service itself is initialized.

    Args:
        service_name (str): Name of the service.
        dependencies (List[str]): List of dependency service names.

    Returns:
        bool: True if all dependencies were set up successfully, False otherwise.

    Raises:
        DependencyError: If a required dependency cannot be set up.
    """
    # Implementation goes here

def check_service_health(service_name: str) -> Dict[str, Any]:
    """
    Check the health status of a service.

    This function checks the health status of a specific service,
    returning detailed information about its current state.

    Args:
        service_name (str): Name of the service to check.

    Returns:
        Dict[str, Any]: Dictionary containing health status information.
    """
    # Implementation goes here
