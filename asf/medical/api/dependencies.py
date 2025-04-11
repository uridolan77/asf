"""
Dependency injection for the Medical Research Synthesizer API.

This module provides dependencies that can be injected into API endpoints
using FastAPI's dependency injection system.
"""

import logging
import traceback
from typing import Optional

from fastapi import Depends

# Import only what we need
# Uncomment when needed
# from asf.medical.api.auth import get_current_user
from asf.medical.storage.repositories.user_repository import UserRepository
from asf.medical.storage.repositories.query_repository import QueryRepository
from asf.medical.storage.repositories.result_repository import ResultRepository
from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository
from asf.medical.clients.ncbi_client import NCBIClient
from asf.medical.clients.clinical_trials_client import ClinicalTrialsClient
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.lorentz_embeddings import LorentzEmbeddingService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
# Use only the enhanced contradiction service
from asf.medical.ml.services.enhanced_contradiction_service import EnhancedContradictionService
from asf.medical.ml.services.temporal_service import TemporalService
# TODO: Fix this import when the module is available
# from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer
# Using a placeholder class for now
class EnhancedMedicalResearchSynthesizer:
    """Placeholder for EnhancedMedicalResearchSynthesizer."""
    pass
from asf.medical.ml.services.prisma_screening_service import PRISMAScreeningService
from asf.medical.ml.services.bias_assessment_service import BiasAssessmentService
from asf.medical.services.search_service import SearchService
from asf.medical.services.analysis_service import AnalysisService
from asf.medical.services.knowledge_base_service import KnowledgeBaseService
from asf.medical.services.export_service import ExportService
from asf.medical.graph.graph_rag import GraphRAG
from asf.medical.graph.graph_service import GraphService
from asf.medical.core.config import settings
from asf.medical.ml.model_registry import model_registry
# Import only if needed
# from asf.medical.core.cache import cache_manager

# Set up logging
logger = logging.getLogger(__name__)

# Repository dependencies
async def get_user_repository() -> UserRepository:
    """
    Get the user repository.

    Returns:
        UserRepository: The user repository
    """
    return UserRepository()

async def get_query_repository() -> QueryRepository:
    """
    Get the query repository.

    Returns:
        QueryRepository: The query repository
    """
    return QueryRepository()

async def get_result_repository() -> ResultRepository:
    """
    Get the result repository.

    Returns:
        ResultRepository: The result repository
    """
    return ResultRepository()

async def get_kb_repository() -> KnowledgeBaseRepository:
    """
    Get the knowledge base repository.

    Returns:
        KnowledgeBaseRepository: The knowledge base repository
    """
    return KnowledgeBaseRepository()

# Client dependencies
async def get_ncbi_client() -> NCBIClient:
    """
    Get the NCBI client.

    Returns:
        NCBIClient: The NCBI client
    """
    return NCBIClient(
        email=settings.NCBI_EMAIL,
        api_key=settings.NCBI_API_KEY.get_secret_value() if settings.NCBI_API_KEY else None
    )

async def get_clinical_trials_client() -> ClinicalTrialsClient:
    """
    Get the ClinicalTrials.gov client.

    Returns:
        ClinicalTrialsClient: The ClinicalTrials.gov client
    """
    return ClinicalTrialsClient()

# ML model dependencies
async def get_biomedlm_service() -> BioMedLMService:
    """
    Get the BioMedLM service.

    Returns:
        BioMedLMService: The BioMedLM service
    """
    # Use the model registry to get or create the service
    if not model_registry.is_model_registered("biomedlm"):
        model_registry.register_model_factory(
            "biomedlm",
            lambda: BioMedLMService(model_name=settings.BIOMEDLM_MODEL),
            BioMedLMService
        )

    return model_registry.get_model("biomedlm")

async def get_tsmixer_service() -> TSMixerService:
    """
    Get the TSMixer service.

    Returns:
        TSMixerService: The TSMixer service
    """
    # Use the model registry to get or create the service
    if not model_registry.is_model_registered("tsmixer"):
        model_registry.register_model_factory(
            "tsmixer",
            lambda: TSMixerService(),
            TSMixerService
        )

    return model_registry.get_model("tsmixer")

async def get_lorentz_embedding_service() -> LorentzEmbeddingService:
    """
    Get the Lorentz embedding service.

    Returns:
        LorentzEmbeddingService: The Lorentz embedding service
    """
    # Use the model registry to get or create the service
    if not model_registry.is_model_registered("lorentz"):
        model_registry.register_model_factory(
            "lorentz",
            lambda: LorentzEmbeddingService(),
            LorentzEmbeddingService
        )

    return model_registry.get_model("lorentz")

async def get_shap_explainer() -> SHAPExplainer:
    """
    Get the SHAP explainer.

    Returns:
        SHAPExplainer: The SHAP explainer
    """
    # Use the model registry to get or create the service
    if not model_registry.is_model_registered("shap_explainer"):
        model_registry.register_model_factory(
            "shap_explainer",
            lambda: SHAPExplainer(),
            SHAPExplainer
        )

    return model_registry.get_model("shap_explainer")

# ML service dependencies
# Removed old contradiction service dependency

async def get_temporal_service_basic() -> TemporalService:
    """
    Get the basic temporal service without dependencies.

    Returns:
        TemporalService: The temporal service
    """
    return TemporalService()

async def get_contradiction_service(
    biomedlm_service: BioMedLMService = Depends(get_biomedlm_service),
    temporal_service: TemporalService = Depends(get_temporal_service_basic),
    shap_explainer: SHAPExplainer = Depends(get_shap_explainer)
) -> EnhancedContradictionService:
    """
    Get the enhanced contradiction service.

    Args:
        biomedlm_service: BioMedLM service for semantic analysis
        temporal_service: Temporal service for temporal analysis
        shap_explainer: SHAP explainer for explainability

    Returns:
        EnhancedContradictionService: The enhanced contradiction service
    """
    return EnhancedContradictionService(
        biomedlm_service=biomedlm_service,
        temporal_service=temporal_service,
        shap_explainer=shap_explainer
    )

async def get_temporal_service(
    tsmixer_service: TSMixerService = Depends(get_tsmixer_service)
) -> TemporalService:
    """
    Get the temporal service.

    Args:
        tsmixer_service: TSMixer service

    Returns:
        TemporalService: The temporal service
    """
    return TemporalService(tsmixer_service=tsmixer_service)

async def get_enhanced_contradiction_service(
    biomedlm_service: BioMedLMService = Depends(get_biomedlm_service),
    temporal_service: TemporalService = Depends(get_temporal_service),
    shap_explainer: SHAPExplainer = Depends(get_shap_explainer)
) -> EnhancedContradictionService:
    """
    Get the enhanced contradiction service.

    Args:
        biomedlm_service: BioMedLM service
        temporal_service: Temporal service
        shap_explainer: SHAP explainer

    Returns:
        EnhancedContradictionService: The enhanced contradiction service
    """
    return EnhancedContradictionService(
        biomedlm_service=biomedlm_service,
        temporal_service=temporal_service,
        shap_explainer=shap_explainer
    )

async def get_prisma_screening_service(
    biomedlm_service: BioMedLMService = Depends(get_biomedlm_service)
) -> PRISMAScreeningService:
    """
    Get the PRISMA screening service.

    Args:
        biomedlm_service: BioMedLM service

    Returns:
        PRISMAScreeningService: The PRISMA screening service
    """
    return PRISMAScreeningService(biomedlm_service=biomedlm_service)

async def get_bias_assessment_service() -> BiasAssessmentService:
    """
    Get the bias assessment service.

    Returns:
        BiasAssessmentService: The bias assessment service
    """
    # Try to load spaCy model, but don't fail if it's not available
    try:
        import spacy
        nlp = spacy.load("en_core_sci_md")
    except Exception as e:
        logger.warning(f"Could not load spaCy model: {str(e)}. Falling back to basic pattern matching.")
        nlp = None

    return BiasAssessmentService(nlp_model=nlp)

# Graph services
async def get_graph_service() -> GraphService:
    """
    Get the graph service.

    Returns:
        GraphService: The graph service
    """
    return GraphService()

async def get_graph_rag(
    graph_service: GraphService = Depends(get_graph_service),
    biomedlm_service: Optional[BioMedLMService] = None
) -> Optional[GraphRAG]:
    """
    Get the GraphRAG service.

    This function attempts to create a GraphRAG service instance with the provided
    dependencies. If any dependency is missing or initialization fails, it returns None
    instead of raising an exception, allowing the application to continue without GraphRAG.

    Args:
        graph_service: The graph service
        biomedlm_service: The BioMedLM service (optional)

    Returns:
        GraphRAG: The GraphRAG service, or None if initialization fails
    """
    # Try to get BioMedLM service if not provided
    if biomedlm_service is None:
        try:
            from asf.medical.ml.models.biomedlm import BioMedLMService
            try:
                biomedlm_service = BioMedLMService()
                logger.info("BioMedLM service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize BioMedLM service: {str(e)}")
                logger.debug(traceback.format_exc())
                return None
        except ImportError as e:
            logger.warning(f"BioMedLM service not available: {str(e)}")
            return None

    # Verify that graph_service is properly initialized
    if graph_service is None:
        logger.warning("Graph service not available")
        return None

    # Initialize GraphRAG
    try:
        graph_rag = GraphRAG(graph_service=graph_service, biomedlm_service=biomedlm_service)
        logger.info("GraphRAG service initialized successfully")
        return graph_rag
    except Exception as e:
        logger.warning(f"Error initializing GraphRAG: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

# Business service dependencies
async def get_search_service(
    ncbi_client: NCBIClient = Depends(get_ncbi_client),
    clinical_trials_client: ClinicalTrialsClient = Depends(get_clinical_trials_client),
    query_repository: QueryRepository = Depends(get_query_repository),
    result_repository: ResultRepository = Depends(get_result_repository),
    graph_rag: Optional[GraphRAG] = Depends(get_graph_rag)
) -> SearchService:
    """
    Get the search service.

    Args:
        ncbi_client: NCBI client
        clinical_trials_client: ClinicalTrials.gov client
        query_repository: Query repository
        result_repository: Result repository
        graph_rag: GraphRAG service (optional)

    Returns:
        SearchService: The search service
    """
    return SearchService(
        ncbi_client=ncbi_client,
        clinical_trials_client=clinical_trials_client,
        query_repository=query_repository,
        result_repository=result_repository,
        graph_rag=graph_rag
    )

async def get_analysis_service(
    enhanced_contradiction_service: EnhancedContradictionService = Depends(get_enhanced_contradiction_service),
    temporal_service: TemporalService = Depends(get_temporal_service),
    search_service: SearchService = Depends(get_search_service),
    result_repository: ResultRepository = Depends(get_result_repository)
) -> AnalysisService:
    """
    Get the analysis service.

    Args:
        enhanced_contradiction_service: Enhanced contradiction service
        temporal_service: Temporal service
        search_service: Search service
        result_repository: Result repository

    Returns:
        AnalysisService: The analysis service
    """
    return AnalysisService(
        contradiction_service=enhanced_contradiction_service,  # Use enhanced service instead
        temporal_service=temporal_service,
        search_service=search_service,
        result_repository=result_repository
    )

async def get_knowledge_base_service(
    search_service: SearchService = Depends(get_search_service),
    kb_repository: KnowledgeBaseRepository = Depends(get_kb_repository)
) -> KnowledgeBaseService:
    """
    Get the knowledge base service.

    Args:
        search_service: Search service
        kb_repository: Knowledge base repository

    Returns:
        KnowledgeBaseService: The knowledge base service
    """
    return KnowledgeBaseService(
        search_service=search_service,
        kb_repository=kb_repository
    )

async def get_export_service() -> ExportService:
    """
    Get the export service.

    Returns:
        ExportService: The export service
    """
    return ExportService()

async def get_synthesizer() -> EnhancedMedicalResearchSynthesizer:
    """
    Get the enhanced medical research synthesizer.

    Returns:
        EnhancedMedicalResearchSynthesizer: The enhanced medical research synthesizer
    """
    # TODO: Implement this when the module is available
    logger.warning("Using placeholder EnhancedMedicalResearchSynthesizer")
    return EnhancedMedicalResearchSynthesizer()
