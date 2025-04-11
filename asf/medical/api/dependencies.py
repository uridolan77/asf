"""
Dependency injection for the Medical Research Synthesizer API.

This module provides dependencies that can be injected into API endpoints
using FastAPI's dependency injection system.
"""

import logging
import traceback
from typing import Optional

from fastapi import Depends, HTTPException, status
from asf.medical.api.auth import get_current_user
from asf.medical.storage.models import User

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
from asf.medical.ml.services.unified_contradiction_service import UnifiedUnifiedUnifiedContradictionService
from asf.medical.ml.services.temporal_service import TemporalService
# TODO: Fix this import when the module is available
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
# from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached

# Set up logging
logger = logging.getLogger(__name__)

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
    Get the user repository.

    Returns:
        UserRepository: The user repository
    Get the query repository.

    Returns:
        QueryRepository: The query repository
    Get the result repository.

    Returns:
        ResultRepository: The result repository
    Get the knowledge base repository.

    Returns:
        KnowledgeBaseRepository: The knowledge base repository
    Get the NCBI client.

    Returns:
        NCBIClient: The NCBI client
    Get the ClinicalTrials.gov client.

    Returns:
        ClinicalTrialsClient: The ClinicalTrials.gov client
    Get the BioMedLM service.

    Returns:
        BioMedLMService: The BioMedLM service
    Get the TSMixer service.

    Returns:
        TSMixerService: The TSMixer service
    Get the Lorentz embedding service.

    Returns:
        LorentzEmbeddingService: The Lorentz embedding service
    Get the SHAP explainer.

    Returns:
        SHAPExplainer: The SHAP explainer
    Get the basic temporal service without dependencies.

    Returns:
        TemporalService: The temporal service
    Get the enhanced contradiction service.

    Args:
        biomedlm_service: BioMedLM service for semantic analysis
        temporal_service: Temporal service for temporal analysis
        shap_explainer: SHAP explainer for explainability

    Returns:
        EnhancedUnifiedUnifiedContradictionService: The enhanced contradiction service
    Get the temporal service.

    Args:
        tsmixer_service: TSMixer service

    Returns:
        TemporalService: The temporal service
    Get the enhanced contradiction service.

    Args:
        biomedlm_service: BioMedLM service
        temporal_service: Temporal service
        shap_explainer: SHAP explainer

    Returns:
        EnhancedUnifiedUnifiedContradictionService: The enhanced contradiction service
    Get the PRISMA screening service.

    Args:
        biomedlm_service: BioMedLM service

    Returns:
        PRISMAScreeningService: The PRISMA screening service
    Get the bias assessment service.

    Returns:
        BiasAssessmentService: The bias assessment service
    Get the graph service.

    Returns:
        GraphService: The graph service
    Get the GraphRAG service.

    This function attempts to create a GraphRAG service instance with the provided
    dependencies. If any dependency is missing or initialization fails, it returns None
    instead of raising an exception, allowing the application to continue without GraphRAG.

    Args:
        graph_service: The graph service
        biomedlm_service: The BioMedLM service (optional)

    Returns:
        GraphRAG: The GraphRAG service, or None if initialization fails
    Get the search service.

    Args:
        ncbi_client: NCBI client
        clinical_trials_client: ClinicalTrials.gov client
        query_repository: Query repository
        result_repository: Result repository
        graph_rag: GraphRAG service (optional)

    Returns:
        SearchService: The search service
    Get the analysis service.

    Args:
        enhanced_contradiction_service: Enhanced contradiction service
        temporal_service: Temporal service
        search_service: Search service
        result_repository: Result repository

    Returns:
        AnalysisService: The analysis service
    Get the knowledge base service.

    Args:
        search_service: Search service
        kb_repository: Knowledge base repository

    Returns:
        KnowledgeBaseService: The knowledge base service
    Get the export service.

    Returns:
        ExportService: The export service
    Get the enhanced medical research synthesizer.

    Returns:
        EnhancedMedicalResearchSynthesizer: The enhanced medical research synthesizer