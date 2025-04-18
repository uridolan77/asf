"""
Dependency injection for the Medical Research Synthesizer API.

This module provides dependencies that can be injected into API endpoints
using FastAPI's dependency injection system. It uses the service registry
for managing dependencies and ensuring proper instantiation order.
"""

from typing import Optional, TypeVar
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, WebSocket, Query

from ..auth import get_current_user
from ..core.config import settings
from ..storage.models import User
from ..core.service_registry import get_service
from ..core.logging_config import get_logger

# Import service and repository types for type annotations
from ..storage.repositories.user_repository import UserRepository
from ..storage.repositories.query_repository import QueryRepository
from ..storage.repositories.result_repository import ResultRepository
from ..storage.repositories.kb_repository import KnowledgeBaseRepository

# Service imports
from ..services.search_service import SearchService
from ..services.analysis_service import AnalysisService
from ..services.knowledge_base_service import KnowledgeBaseService
from ..services.export_service import ExportService
from ..services.auth_service import AuthService
from ..services.terminology_service import TerminologyService
from ..services.clinical_data_service import ClinicalDataService

# ML service imports
from ..ml.services.contradiction_service import ContradictionService
from ..ml.services.prisma_screening_service import PrismaScreeningService
from ..ml.services.bias_assessment_service import BiasAssessmentService

# Client imports
from ..clients.ncbi.ncbi_client import NCBIClient
from ..clients.clinical_trials_gov.clinical_trials_client import ClinicalTrialsClient
from ..clients.clinical_trials_gov.clinical_trials_client import ClinicalTrialsClient as CTGovClient

# Graph imports
from ..graph.graph_service import GraphService
from ..graph.graph_rag import GraphRAG
from ..core.exceptions import DatabaseError, OperationError


# Set up logging
logger = get_logger(__name__)

# Type variable for generic service dependencies
T = TypeVar('T')


# WebSocket authentication dependency
async def get_current_user_ws(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db = Depends(get_service(UserRepository))
) -> Optional[User]:
    """
    Get the current user from a WebSocket connection.

    Args:
        websocket: WebSocket connection
        token: JWT token
        db: User repository

    Returns:
        User or None if not authenticated
    """
    if not token:
        return None

    try:
        # Decode the token
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        email: str = payload.get("sub")

        if email is None:
            return None

        # Get the user
        user = await db.get_by_email_async(None, email)

        if user is None or not user.is_active:
            return None

        return user
    except JWTError:
        logger.error(f"Error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error authenticating WebSocket user: {str(e)}", exc_info=e)
        raise DatabaseError(f"Database operation failed: {str(e)}")
        return None


# Admin user dependency
async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get the current admin user.

    Args:
        current_user: Current active user

    Returns:
        Current admin user

    Raises:
        HTTPException: If the user is not an admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


# Repository dependencies using service registry
def get_user_repository() -> UserRepository:
    """
    Get the user repository.

    Returns:
        UserRepository: The user repository
    """
    return get_service(UserRepository)()


def get_query_repository() -> QueryRepository:
    """
    Get the query repository.

    Returns:
        QueryRepository: The query repository
    """
    return get_service(QueryRepository)()


def get_result_repository() -> ResultRepository:
    """
    Get the result repository.

    Returns:
        ResultRepository: The result repository
    """
    return get_service(ResultRepository)()


def get_kb_repository() -> KnowledgeBaseRepository:
    """
    Get the knowledge base repository.

    Returns:
        KnowledgeBaseRepository: The knowledge base repository
    """
    return get_service(KnowledgeBaseRepository)()


# Client dependencies
def get_ncbi_client() -> NCBIClient:
    """
    Get the NCBI client.

    Returns:
        NCBIClient: The NCBI client
    """
    return get_service(NCBIClient)()


def get_clinical_trials_client() -> ClinicalTrialsClient:
    """
    Get the ClinicalTrials.gov client.

    Returns:
        ClinicalTrialsClient: The ClinicalTrials.gov client
    """
    return get_service(ClinicalTrialsClient)()


# Service dependencies
def get_auth_service() -> AuthService:
    """
    Get the authentication service.

    Returns:
        AuthService: The authentication service
    """
    return get_service(AuthService)()


def get_terminology_service() -> TerminologyService:
    """
    Get the terminology service.

    Returns:
        TerminologyService: The terminology service
    """
    return get_service(TerminologyService)(
        snomed_access_mode="umls",
        snomed_api_key=settings.UMLS_API_KEY,
        snomed_cache_dir=settings.TERMINOLOGY_CACHE_DIR
    )


def get_ct_gov_client() -> CTGovClient:
    """
    Get the ClinicalTrials.gov API client.

    Returns:
        CTGovClient: The ClinicalTrials.gov client
    """
    return get_service(CTGovClient)(
        cache_dir=settings.CLINICAL_TRIALS_CACHE_DIR
    )


def get_clinical_data_service() -> ClinicalDataService:
    """
    Get the clinical data service that integrates terminology and clinical trials data.

    Returns:
        ClinicalDataService: The clinical data service
    """
    terminology_service = get_terminology_service()
    ct_gov_client = get_ct_gov_client()
    
    return get_service(ClinicalDataService)(
        terminology_service=terminology_service,
        clinical_trials_client=ct_gov_client
    )


def get_search_service() -> SearchService:
    """
    Get the search service.

    Returns:
        SearchService: The search service
    """
    return get_service(SearchService)()


def get_analysis_service() -> AnalysisService:
    """
    Get the analysis service.

    Returns:
        AnalysisService: The analysis service
    """
    return get_service(AnalysisService)()


def get_knowledge_base_service() -> KnowledgeBaseService:
    """
    Get the knowledge base service.

    Returns:
        KnowledgeBaseService: The knowledge base service
    """
    return get_service(KnowledgeBaseService)()


def get_export_service() -> ExportService:
    """
    Get the export service.

    Returns:
        ExportService: The export service
    """
    return get_service(ExportService)()


# ML service dependencies
def get_contradiction_service() -> ContradictionService:
    """
    Get the contradiction service.

    Returns:
        ContradictionService: The contradiction service
    """
    return get_service(ContradictionService)()


def get_prisma_screening_service() -> PrismaScreeningService:
    """
    Get the PRISMA screening service.

    Returns:
        PrismaScreeningService: The PRISMA screening service
    """
    return get_service(PrismaScreeningService)()


def get_bias_assessment_service() -> BiasAssessmentService:
    """
    Get the bias assessment service.

    Returns:
        BiasAssessmentService: The bias assessment service
    """
    return get_service(BiasAssessmentService)()


# Graph dependencies
def get_graph_service() -> GraphService:
    """
    Get the graph service.

    Returns:
        GraphService: The graph service
    """
    return get_service(GraphService)()


def get_graph_rag() -> Optional[GraphRAG]:
    """
    Get the GraphRAG service.

    This function attempts to create a GraphRAG service instance.
    If initialization fails, it returns None instead of raising an exception,
    allowing the application to continue without GraphRAG.

    Returns:
        GraphRAG: The GraphRAG service, or None if initialization fails
    """
    try:
        return get_service(GraphRAG)()
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {str(e)}", exc_info=e)
        raise OperationError(f"Operation failed: {str(e)}")
        return None


# Combined service dependency for the synthesizer
def get_synthesizer():
    """
    Get the synthesizer service that combines search and analysis capabilities.

    Returns:
        A service combining search and analysis functionality
    """
    from ..services.synthesizer_service import SynthesizerService
    return get_service(SynthesizerService)()
