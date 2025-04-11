"""
ML Inference Tasks

This module defines background tasks for ML model inference operations.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Tuple

import dramatiq
import numpy as np
import torch

from asf.medical.core.observability import (
    trace_ml_operation, update_queue_size, update_model_memory_usage,
    log_ml_event, push_metrics
)
from asf.medical.core.persistent_task_storage import task_storage
from asf.medical.core.resource_limiter import resource_limiter

from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.lorentz_embeddings import LorentzEmbeddingService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.enhanced_contradiction_classifier import ContradictionType, ContradictionConfidence

logger = logging.getLogger(__name__)

task_results = {}

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (ContradictionType, ContradictionConfidence)):
            return obj.value
        return super().default(obj)

@dramatiq.actor(max_retries=2, time_limit=300000)  # 5 minutes
def detect_contradiction(claim1: str, claim2: str, metadata1: Optional[Dict[str, Any]] = None,
                        metadata2: Optional[Dict[str, Any]] = None, use_all_methods: bool = True):
    """
    Detect contradiction between two claims in the background.

    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim
        use_all_methods: Whether to use all available methods

    Returns:
        Contradiction detection result
    Analyze contradictions in a list of articles in the background.

    Args:
        articles: List of articles
        threshold: Threshold for contradiction detection
        use_all_methods: Whether to use all available methods

    Returns:
        List of detected contradictions
    Generate embeddings for a list of texts in the background.

    Args:
        texts: List of texts to embed
        model_name: Name of the model to use (biomedlm, lorentz)

    Returns:
        List of embeddings
    Detect direct contradiction between two claims using BioMedLM.

    Args:
        claim1: First claim
        claim2: Second claim
        biomedlm_service: BioMedLM service

    Returns:
        Direct contradiction detection result
    Detect temporal contradiction between two claims.

    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim
        biomedlm_service: BioMedLM service
        tsmixer_service: TSMixer service

    Returns:
        Temporal contradiction detection result
    Detect statistical contradiction between two claims.

    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim

    Returns:
        Statistical contradiction detection result
    Generate explanation for contradiction using SHAP.

    Args:
        claim1: First claim
        claim2: Second claim
        contradiction_type: Type of contradiction
        shap_explainer: SHAP explainer

    Returns:
        Explanation
    Get the result of a task by ID.

    Args:
        task_id: The ID of the task

    Returns:
        Task result information or None if not found