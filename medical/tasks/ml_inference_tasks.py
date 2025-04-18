"""
ML Inference Tasks for the Medical Research Synthesizer.

This module defines background tasks for ML model inference operations,
including contradiction detection, embedding generation, and explanation
generation. These tasks are designed to run asynchronously in the background
using Dramatiq, allowing the API to respond quickly to requests while the
compute-intensive ML operations continue in separate worker processes.

The module includes tasks for:
- Detecting contradictions between medical claims
- Analyzing contradictions in a collection of articles
- Generating embeddings for text using various models
- Generating explanations for detected contradictions

Each task follows a consistent pattern with proper error handling, logging,
and observability through metrics and tracing.
"""
import logging
import json
from typing import Dict, List, Any, Optional
import dramatiq
import numpy as np
from ..core.observability import (
    trace_ml_operation, update_queue_size, update_model_memory_usage,
    log_ml_event, push_metrics
)
from ..ml.services.contradiction_service import ContradictionType, ContradictionConfidence
logger = logging.getLogger(__name__)
task_results = {}
class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for NumPy types and enum values.

    This encoder handles NumPy arrays, integers, and floating-point numbers,
    as well as ContradictionType and ContradictionConfidence enum values,
    converting them to JSON-serializable Python types.
    """
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

    This task analyzes two medical claims to detect if they contradict each other.
    It uses multiple methods including direct semantic contradiction detection,
    temporal contradiction detection, and statistical contradiction detection.
    The task is traced for observability and metrics are pushed to the monitoring system.

    Args:
        claim1: The text of the first medical claim
        claim2: The text of the second medical claim
        metadata1: Optional metadata for the first claim (publication date, study design, etc.)
        metadata2: Optional metadata for the second claim (publication date, study design, etc.)
        use_all_methods: Whether to use all available contradiction detection methods

    Returns:
        Dictionary containing contradiction detection results, including:
        - is_contradiction: Boolean indicating if a contradiction was detected
        - contradiction_type: Type of contradiction (DIRECT, TEMPORAL, STATISTICAL, etc.)
        - confidence: Confidence level of the contradiction detection
        - explanation: Explanation of why the claims are contradictory
    """