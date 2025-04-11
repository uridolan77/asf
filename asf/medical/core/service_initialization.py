"""
Service Initialization for ASF Medical Research Synthesizer.

This module initializes all services in the application using the service registry.
It handles dependency resolution and ensures services are created in the correct order.
"""

from asf.medical.core.service_registry import _registry as registry
from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

def initialize_services() -> None:
    """
    Initialize all services in the registry.
    
    This function registers all services with the registry and initializes them.
    It should be called during application startup.
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
    """Register all repositories with the service registry."""
    from asf.medical.storage.repositories.user_repository import UserRepository
    from asf.medical.storage.repositories.query_repository import QueryRepository
    from asf.medical.storage.repositories.result_repository import ResultRepository
    from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository
    
    # Register repositories as factories (they don't have dependencies)
    registry.register_factory(UserRepository, lambda: UserRepository())
    registry.register_factory(QueryRepository, lambda: QueryRepository())
    registry.register_factory(ResultRepository, lambda: ResultRepository())
    registry.register_factory(KnowledgeBaseRepository, lambda: KnowledgeBaseRepository())
    
    logger.debug("Repositories registered")

def _register_services():
    """Register all application services with the service registry."""
    from asf.medical.services.auth_service import AuthService
    from asf.medical.services.search_service import SearchService
    from asf.medical.services.analysis_service import AnalysisService
    from asf.medical.services.export_service import ExportService
    from asf.medical.services.knowledge_base_service import KnowledgeBaseService
    from asf.medical.storage.repositories.user_repository import UserRepository
    from asf.medical.storage.repositories.query_repository import QueryRepository
    from asf.medical.storage.repositories.result_repository import ResultRepository
    from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository
    
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
    """Register all ML services with the service registry."""
    try:
        from asf.medical.ml.services.unified_contradiction_service import UnifiedContradictionService
        from asf.medical.ml.services.screening_service import PrismaScreeningService
        from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService
        
        # Register ML services
        registry.register_factory(
            UnifiedContradictionService,
            lambda: UnifiedContradictionService()
        )
        
        registry.register_factory(
            PrismaScreeningService,
            lambda: PrismaScreeningService()
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
    
    Returns:
        List of service types registered in the registry
    """
    return list(registry._factories.keys()) + list(registry._services.keys())
