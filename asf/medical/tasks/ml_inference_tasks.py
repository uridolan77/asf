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

# Configure logging
logger = logging.getLogger(__name__)

# Store of task results
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
    """
    # Get task ID
    task_id = dramatiq.middleware.current_message.message_id

    # Log event
    log_ml_event(
        model="biomedlm",
        operation="detect_contradiction",
        event_type="task_start",
        details={
            "task_id": task_id,
            "claim1_length": len(claim1),
            "claim2_length": len(claim2),
            "use_all_methods": use_all_methods
        }
    )

    # Update task status
    task_results[task_id] = {
        "status": "processing",
        "progress": 0
    }

    # Use trace context manager for observability
    with trace_ml_operation(model="biomedlm", operation="detect_contradiction") as trace_id:
        try:
            logger.info(f"Starting contradiction detection between claims")

            # Wait for resources
            resource_acquired = False
            try:
                if not resource_limiter.acquire_task_slot(timeout=600.0):  # 10 minutes timeout
                    raise RuntimeError("Could not acquire resources for contradiction detection")
                resource_acquired = True

                # Initialize services
                if not resource_limiter.acquire_model_lock("biomedlm", timeout=300.0):  # 5 minutes timeout
                    raise RuntimeError("Could not acquire lock for BioMedLM model")

                try:
                    biomedlm_service = BioMedLMService()

                    # Register model usage
                    resource_limiter.register_model_usage("biomedlm", memory_mb=1024)  # 1GB placeholder
                finally:
                    # Release model lock
                    resource_limiter.release_model_lock("biomedlm")

                # Track memory usage
                update_model_memory_usage("biomedlm", 1024 * 1024 * 1024)  # Placeholder: 1GB
            finally:
                # Release resources if acquired
                if resource_acquired:
                    resource_limiter.release_task_slot()

            # Initialize result
            result = {
                "is_contradiction": False,
                "contradiction_score": 0.0,
                "contradiction_type": ContradictionType.NONE.value,
                "confidence": ContradictionConfidence.LOW.value,
                "explanation": "",
                "methods_used": [],
                "details": {},
                "trace_id": trace_id
            }

            # Update progress
            task_results[task_id]["progress"] = 10
            task_storage.update_task_progress(task_id, 10)

            # Detect direct contradiction using BioMedLM
            start_time = time.time()
            direct_result = detect_direct_contradiction(claim1, claim2, biomedlm_service)
            direct_duration = time.time() - start_time

            result["methods_used"].append("biomedlm")
            result["details"]["direct"] = direct_result
            result["details"]["direct"]["duration"] = direct_duration

            # Log direct detection
            log_ml_event(
                model="biomedlm",
                operation="direct_contradiction",
                event_type="inference",
                details={
                    "task_id": task_id,
                    "trace_id": trace_id,
                    "duration": direct_duration,
                    "is_contradiction": direct_result["is_contradiction"],
                    "score": direct_result["score"]
                }
            )

            # Update progress
            task_results[task_id]["progress"] = 50
            task_storage.update_task_progress(task_id, 50)

            # Update result if direct contradiction is detected
            if direct_result["is_contradiction"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = direct_result["score"]
                result["contradiction_type"] = ContradictionType.DIRECT.value
                result["confidence"] = direct_result["confidence"]
                result["explanation"] = direct_result["explanation"]

            # If using all methods and we have metadata, check for temporal and statistical contradictions
            if use_all_methods and metadata1 and metadata2:
                # Initialize additional services if needed
                tsmixer_service = TSMixerService()

                # Track memory usage
                update_model_memory_usage("tsmixer", 512 * 1024 * 1024)  # Placeholder: 512MB

                # Detect temporal contradiction
                start_time = time.time()
                temporal_result = detect_temporal_contradiction(
                    claim1, claim2, metadata1, metadata2, biomedlm_service, tsmixer_service
                )
                temporal_duration = time.time() - start_time

                result["methods_used"].append("temporal")
                result["details"]["temporal"] = temporal_result
                result["details"]["temporal"]["duration"] = temporal_duration

                # Log temporal detection
                log_ml_event(
                    model="tsmixer",
                    operation="temporal_contradiction",
                    event_type="inference",
                    details={
                        "task_id": task_id,
                        "trace_id": trace_id,
                        "duration": temporal_duration,
                        "is_contradiction": temporal_result["is_contradiction"],
                        "score": temporal_result["score"]
                    }
                )

                # Update result if temporal contradiction is detected and has higher score
                if temporal_result["is_contradiction"] and temporal_result["score"] > result["contradiction_score"]:
                    result["is_contradiction"] = True
                    result["contradiction_score"] = temporal_result["score"]
                    result["contradiction_type"] = ContradictionType.TEMPORAL.value
                    result["confidence"] = temporal_result["confidence"]
                    result["explanation"] = temporal_result["explanation"]

                # Detect statistical contradiction
                start_time = time.time()
                statistical_result = detect_statistical_contradiction(
                    claim1, claim2, metadata1, metadata2
                )
                statistical_duration = time.time() - start_time

                result["methods_used"].append("statistical")
                result["details"]["statistical"] = statistical_result
                result["details"]["statistical"]["duration"] = statistical_duration

                # Log statistical detection
                log_ml_event(
                    model="statistical",
                    operation="statistical_contradiction",
                    event_type="inference",
                    details={
                        "task_id": task_id,
                        "trace_id": trace_id,
                        "duration": statistical_duration,
                        "is_contradiction": statistical_result["is_contradiction"],
                        "score": statistical_result["score"]
                    }
                )

                # Update result if statistical contradiction is detected and has higher score
                if statistical_result["is_contradiction"] and statistical_result["score"] > result["contradiction_score"]:
                    result["is_contradiction"] = True
                    result["contradiction_score"] = statistical_result["score"]
                    result["contradiction_type"] = ContradictionType.STATISTICAL.value
                    result["confidence"] = statistical_result["confidence"]
                    result["explanation"] = statistical_result["explanation"]

            # Update progress
            task_results[task_id]["progress"] = 80
            task_storage.update_task_progress(task_id, 80)

            # Generate explanation using SHAP if available and not already set
            if result["is_contradiction"] and not result["explanation"]:
                try:
                    shap_explainer = SHAPExplainer()
                    start_time = time.time()
                    explanation = generate_explanation(
                        claim1, claim2, result["contradiction_type"], shap_explainer
                    )
                    explanation_duration = time.time() - start_time

                    result["explanation"] = explanation
                    result["details"]["explanation_duration"] = explanation_duration

                    # Log explanation generation
                    log_ml_event(
                        model="shap",
                        operation="generate_explanation",
                        event_type="inference",
                        details={
                            "task_id": task_id,
                            "trace_id": trace_id,
                            "duration": explanation_duration,
                            "contradiction_type": result["contradiction_type"]
                        }
                    )
                except Exception as e:
                    logger.error(f"Error generating explanation: {str(e)}")

            # Update task status
            task_results[task_id] = {
                "status": "completed",
                "progress": 100,
                "result": json.dumps(result, cls=NumpyEncoder)
            }

            # Update persistent storage
            task_storage.complete_task(task_id, result)

            # Push metrics
            push_metrics()

            logger.info(f"Contradiction detection completed")
            return result

        except Exception as e:
            # Update task status
            task_results[task_id] = {
                "status": "failed",
                "error": str(e)
            }

            # Update persistent storage
            task_storage.fail_task(task_id, str(e))

            # Log error
            log_ml_event(
                model="biomedlm",
                operation="detect_contradiction",
                event_type="error",
                details={
                    "task_id": task_id,
                    "trace_id": trace_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

            logger.error(f"Error detecting contradiction: {str(e)}")
            raise

@dramatiq.actor(max_retries=2, time_limit=600000)  # 10 minutes
def analyze_contradictions_in_articles(articles: List[Dict[str, Any]], threshold: float = 0.7,
                                      use_all_methods: bool = True):
    """
    Analyze contradictions in a list of articles in the background.

    Args:
        articles: List of articles
        threshold: Threshold for contradiction detection
        use_all_methods: Whether to use all available methods

    Returns:
        List of detected contradictions
    """
    # Get task ID
    task_id = dramatiq.middleware.current_message.message_id

    # Log event
    log_ml_event(
        model="biomedlm",
        operation="analyze_contradictions",
        event_type="task_start",
        details={
            "task_id": task_id,
            "total_articles": len(articles),
            "threshold": threshold,
            "use_all_methods": use_all_methods
        }
    )

    # Update task status
    task_results[task_id] = {
        "status": "processing",
        "progress": 0,
        "total_articles": len(articles),
        "processed_articles": 0
    }

    # Update persistent storage
    task_storage.set_task_status(task_id, {
        "status": "processing",
        "progress": 0,
        "total_articles": len(articles),
        "processed_articles": 0
    })

    # Update queue size
    update_queue_size("biomedlm", 1)  # Placeholder

    # Use trace context manager for observability
    with trace_ml_operation(model="biomedlm", operation="analyze_contradictions") as trace_id:
        try:
            logger.info(f"Starting contradiction analysis for {len(articles)} articles")

            # Wait for resources
            resource_acquired = False
            try:
                if not resource_limiter.acquire_task_slot(timeout=600.0):  # 10 minutes timeout
                    raise RuntimeError("Could not acquire resources for contradiction analysis")
                resource_acquired = True

                # Initialize services
                if not resource_limiter.acquire_model_lock("biomedlm", timeout=300.0):  # 5 minutes timeout
                    raise RuntimeError("Could not acquire lock for BioMedLM model")

                try:
                    biomedlm_service = BioMedLMService()

                    # Register model usage
                    resource_limiter.register_model_usage("biomedlm", memory_mb=1024)  # 1GB placeholder

                    # Track memory usage
                    update_model_memory_usage("biomedlm", 1024 * 1024 * 1024)  # Placeholder: 1GB
                finally:
                    # Release model lock
                    resource_limiter.release_model_lock("biomedlm")

                # Initialize results
                contradictions = []
                total_articles = len(articles)
            finally:
                # Release resources if acquired
                if resource_acquired:
                    resource_limiter.release_task_slot()

            # Compare each pair of articles
            for i in range(total_articles):
                for j in range(i + 1, total_articles):
                    # Get articles
                    article1 = articles[i]
                    article2 = articles[j]

                    # Extract claims
                    claim1 = article1.get("abstract", "")
                    claim2 = article2.get("abstract", "")

                    # Skip if either claim is empty
                    if not claim1 or not claim2:
                        continue

                    # Extract metadata
                    metadata1 = {
                        "publication_date": article1.get("publication_date"),
                        "study_design": article1.get("study_design"),
                        "sample_size": article1.get("sample_size"),
                        "p_value": article1.get("p_value")
                    }

                    metadata2 = {
                        "publication_date": article2.get("publication_date"),
                        "study_design": article2.get("study_design"),
                        "sample_size": article2.get("sample_size"),
                        "p_value": article2.get("p_value")
                    }

                    # Detect contradiction
                    try:
                        # Use direct detection for efficiency
                        start_time = time.time()
                        direct_result = detect_direct_contradiction(claim1, claim2, biomedlm_service)
                        direct_duration = time.time() - start_time

                        # Log detection
                        log_ml_event(
                            model="biomedlm",
                            operation="direct_contradiction",
                            event_type="inference",
                            details={
                                "task_id": task_id,
                                "trace_id": trace_id,
                                "article_pair": f"{i}-{j}",
                                "duration": direct_duration,
                                "is_contradiction": direct_result["is_contradiction"],
                                "score": direct_result["score"]
                            }
                        )

                        # If contradiction is detected and score is above threshold
                        if direct_result["is_contradiction"] and direct_result["score"] > threshold:
                            # Create contradiction result
                            contradiction = {
                                "article1": {
                                    "id": article1.get("id"),
                                    "title": article1.get("title"),
                                    "abstract": claim1,
                                    "metadata": metadata1
                                },
                                "article2": {
                                    "id": article2.get("id"),
                                    "title": article2.get("title"),
                                    "abstract": claim2,
                                    "metadata": metadata2
                                },
                                "contradiction_score": direct_result["score"],
                                "contradiction_type": ContradictionType.DIRECT.value,
                                "confidence": direct_result["confidence"],
                                "explanation": direct_result["explanation"],
                                "detection_duration": direct_duration,
                                "trace_id": trace_id
                            }

                            # Add to results
                            contradictions.append(contradiction)

                            # Log contradiction found
                            log_ml_event(
                                model="biomedlm",
                                operation="contradiction_found",
                                event_type="result",
                                details={
                                    "task_id": task_id,
                                    "trace_id": trace_id,
                                    "article_pair": f"{i}-{j}",
                                    "score": direct_result["score"],
                                    "contradiction_type": ContradictionType.DIRECT.value
                                }
                            )
                    except Exception as e:
                        logger.error(f"Error detecting contradiction between articles {i} and {j}: {str(e)}")

                        # Log error
                        log_ml_event(
                            model="biomedlm",
                            operation="direct_contradiction",
                            event_type="error",
                            details={
                                "task_id": task_id,
                                "trace_id": trace_id,
                                "article_pair": f"{i}-{j}",
                                "error": str(e),
                                "error_type": type(e).__name__
                            }
                        )

                    # Update progress
                    processed_pairs = (i * total_articles) + j - (i * (i + 1)) // 2
                    total_pairs = (total_articles * (total_articles - 1)) // 2
                    progress = min(95, int(100 * processed_pairs / total_pairs))

                    task_results[task_id]["progress"] = progress
                    task_results[task_id]["processed_articles"] = i + 1

                    # Update persistent storage
                    task_storage.update_task_progress(task_id, progress, processed_articles=i + 1)

            # Update task status
            task_results[task_id] = {
                "status": "completed",
                "progress": 100,
                "total_articles": total_articles,
                "total_contradictions": len(contradictions),
                "result": json.dumps(contradictions, cls=NumpyEncoder)
            }

            # Update persistent storage
            task_storage.complete_task(task_id, {
                "total_articles": total_articles,
                "total_contradictions": len(contradictions),
                "contradictions": contradictions
            })

            # Log completion
            log_ml_event(
                model="biomedlm",
                operation="analyze_contradictions",
                event_type="task_complete",
                details={
                    "task_id": task_id,
                    "trace_id": trace_id,
                    "total_articles": total_articles,
                    "total_contradictions": len(contradictions)
                }
            )

            # Push metrics
            push_metrics()

            # Update queue size
            update_queue_size("biomedlm", 0)  # Placeholder

            logger.info(f"Contradiction analysis completed: found {len(contradictions)} contradictions")
            return contradictions

        except Exception as e:
            # Update task status
            task_results[task_id] = {
                "status": "failed",
                "error": str(e)
            }

            # Update persistent storage
            task_storage.fail_task(task_id, str(e))

            # Log error
            log_ml_event(
                model="biomedlm",
                operation="analyze_contradictions",
                event_type="error",
                details={
                    "task_id": task_id,
                    "trace_id": trace_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

            # Update queue size
            update_queue_size("biomedlm", 0)  # Placeholder

            logger.error(f"Error analyzing contradictions: {str(e)}")
            raise

@dramatiq.actor(max_retries=2, time_limit=300000)  # 5 minutes
def generate_embeddings(texts: List[str], model_name: str = "biomedlm"):
    """
    Generate embeddings for a list of texts in the background.

    Args:
        texts: List of texts to embed
        model_name: Name of the model to use (biomedlm, lorentz)

    Returns:
        List of embeddings
    """
    # Get task ID
    task_id = dramatiq.middleware.current_message.message_id

    # Log event
    log_ml_event(
        model=model_name,
        operation="generate_embeddings",
        event_type="task_start",
        details={
            "task_id": task_id,
            "total_texts": len(texts),
            "model_name": model_name
        }
    )

    # Update task status
    task_results[task_id] = {
        "status": "processing",
        "progress": 0,
        "total_texts": len(texts),
        "processed_texts": 0
    }

    # Update persistent storage
    task_storage.set_task_status(task_id, {
        "status": "processing",
        "progress": 0,
        "total_texts": len(texts),
        "processed_texts": 0
    })

    # Update queue size
    update_queue_size(model_name, 1)  # Placeholder

    # Use trace context manager for observability
    with trace_ml_operation(model=model_name, operation="generate_embeddings") as trace_id:
        try:
            logger.info(f"Starting embedding generation for {len(texts)} texts using {model_name}")

            # Wait for resources
            resource_acquired = False
            try:
                if not resource_limiter.acquire_task_slot(timeout=600.0):  # 10 minutes timeout
                    raise RuntimeError(f"Could not acquire resources for embedding generation with {model_name}")
                resource_acquired = True

                # Initialize service based on model name
                if not resource_limiter.acquire_model_lock(model_name, timeout=300.0):  # 5 minutes timeout
                    raise RuntimeError(f"Could not acquire lock for {model_name} model")

                try:
                    if model_name.lower() == "biomedlm":
                        service = BioMedLMService()
                        encode_fn = service.encode
                        memory_mb = 1024  # 1GB placeholder
                        # Track memory usage
                        update_model_memory_usage("biomedlm", memory_mb * 1024 * 1024)  # Convert MB to bytes
                    elif model_name.lower() == "lorentz":
                        service = LorentzEmbeddingService()
                        encode_fn = service.encode
                        memory_mb = 512  # 512MB placeholder
                        # Track memory usage
                        update_model_memory_usage("lorentz", memory_mb * 1024 * 1024)  # Convert MB to bytes
                    else:
                        raise ValueError(f"Unsupported model: {model_name}")

                    # Register model usage
                    resource_limiter.register_model_usage(model_name, memory_mb=memory_mb)
                finally:
                    # Release model lock
                    resource_limiter.release_model_lock(model_name)

                # Generate embeddings
                embeddings = []
                total_embedding_time = 0
            finally:
                # Release resources if acquired
                if resource_acquired:
                    resource_limiter.release_task_slot()

            for i, text in enumerate(texts):
                # Generate embedding
                start_time = time.time()
                embedding = encode_fn(text)
                embedding_time = time.time() - start_time
                total_embedding_time += embedding_time

                embeddings.append(embedding)

                # Log embedding generation
                if i % 10 == 0 or i == len(texts) - 1:  # Log every 10th embedding or the last one
                    log_ml_event(
                        model=model_name,
                        operation="encode_text",
                        event_type="inference",
                        details={
                            "task_id": task_id,
                            "trace_id": trace_id,
                            "text_index": i,
                            "text_length": len(text),
                            "duration": embedding_time
                        }
                    )

                # Update progress
                progress = min(95, int(100 * (i + 1) / len(texts)))
                task_results[task_id]["progress"] = progress
                task_results[task_id]["processed_texts"] = i + 1

                # Update persistent storage (every 10 texts or on the last one)
                if i % 10 == 0 or i == len(texts) - 1:
                    task_storage.update_task_progress(task_id, progress, processed_texts=i + 1)

            # Convert embeddings to list for JSON serialization
            embeddings_list = [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings]

            # Update task status
            task_results[task_id] = {
                "status": "completed",
                "progress": 100,
                "total_texts": len(texts),
                "processed_texts": len(texts),
                "result": json.dumps({
                    "embeddings": embeddings_list,
                    "trace_id": trace_id,
                    "total_embedding_time": total_embedding_time,
                    "average_embedding_time": total_embedding_time / len(texts) if texts else 0
                }, cls=NumpyEncoder)
            }

            # Update persistent storage
            task_storage.complete_task(task_id, {
                "embeddings": embeddings_list,
                "trace_id": trace_id,
                "total_embedding_time": total_embedding_time,
                "average_embedding_time": total_embedding_time / len(texts) if texts else 0
            })

            # Log completion
            log_ml_event(
                model=model_name,
                operation="generate_embeddings",
                event_type="task_complete",
                details={
                    "task_id": task_id,
                    "trace_id": trace_id,
                    "total_texts": len(texts),
                    "total_embedding_time": total_embedding_time,
                    "average_embedding_time": total_embedding_time / len(texts) if texts else 0
                }
            )

            # Push metrics
            push_metrics()

            # Update queue size
            update_queue_size(model_name, 0)  # Placeholder

            logger.info(f"Embedding generation completed")
            return {
                "embeddings": embeddings_list,
                "trace_id": trace_id,
                "total_embedding_time": total_embedding_time,
                "average_embedding_time": total_embedding_time / len(texts) if texts else 0
            }

        except Exception as e:
            # Update task status
            task_results[task_id] = {
                "status": "failed",
                "error": str(e)
            }

            # Update persistent storage
            task_storage.fail_task(task_id, str(e))

            # Log error
            log_ml_event(
                model=model_name,
                operation="generate_embeddings",
                event_type="error",
                details={
                    "task_id": task_id,
                    "trace_id": trace_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )

            # Update queue size
            update_queue_size(model_name, 0)  # Placeholder

            logger.error(f"Error generating embeddings: {str(e)}")
            raise

def detect_direct_contradiction(claim1: str, claim2: str, biomedlm_service: BioMedLMService) -> Dict[str, Any]:
    """
    Detect direct contradiction between two claims using BioMedLM.

    Args:
        claim1: First claim
        claim2: Second claim
        biomedlm_service: BioMedLM service

    Returns:
        Direct contradiction detection result
    """
    # Initialize result
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.LOW.value,
        "explanation": ""
    }

    try:
        # Use BioMedLM to detect contradiction
        # This is a simplified version - in a real implementation, you would use the actual model
        embeddings1 = biomedlm_service.encode(claim1)
        embeddings2 = biomedlm_service.encode(claim2)

        # Calculate cosine similarity
        similarity = np.dot(embeddings1, embeddings2) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

        # Convert similarity to contradiction score (higher similarity = lower contradiction)
        score = 1.0 - max(0.0, similarity)

        # Set result
        result["score"] = float(score)
        result["is_contradiction"] = score > 0.7  # Threshold for contradiction

        # Set confidence based on score
        if score > 0.9:
            result["confidence"] = ContradictionConfidence.HIGH.value
        elif score > 0.8:
            result["confidence"] = ContradictionConfidence.MEDIUM.value
        else:
            result["confidence"] = ContradictionConfidence.LOW.value

        # Generate explanation
        if result["is_contradiction"]:
            result["explanation"] = f"The claims directly contradict each other with a score of {score:.2f}."

        return result
    except Exception as e:
        logger.error(f"Error detecting direct contradiction: {str(e)}")
        return result

def detect_temporal_contradiction(claim1: str, claim2: str, metadata1: Dict[str, Any],
                                 metadata2: Dict[str, Any], biomedlm_service: BioMedLMService,
                                 tsmixer_service: TSMixerService) -> Dict[str, Any]:
    """
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
    """
    # Initialize result
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.LOW.value,
        "explanation": ""
    }

    try:
        # Extract publication dates from metadata
        pub_date1 = metadata1.get("publication_date")
        pub_date2 = metadata2.get("publication_date")

        # Skip if dates are not available
        if not pub_date1 or not pub_date2:
            return result

        # Create sequence for temporal analysis
        sequence = [
            {"claim": claim1, "timestamp": pub_date1},
            {"claim": claim2, "timestamp": pub_date2}
        ]

        # Generate embeddings for claims
        embeddings1 = biomedlm_service.encode(claim1)
        embeddings2 = biomedlm_service.encode(claim2)

        # Calculate temporal score (simplified)
        # In a real implementation, you would use TSMixer for temporal analysis
        temporal_score = 0.5  # Placeholder

        # Set result
        result["score"] = float(temporal_score)
        result["is_contradiction"] = temporal_score > 0.7  # Threshold for contradiction

        # Set confidence based on score
        if temporal_score > 0.9:
            result["confidence"] = ContradictionConfidence.HIGH.value
        elif temporal_score > 0.8:
            result["confidence"] = ContradictionConfidence.MEDIUM.value
        else:
            result["confidence"] = ContradictionConfidence.LOW.value

        # Generate explanation
        if result["is_contradiction"]:
            result["explanation"] = f"The claims show temporal contradiction with a score of {temporal_score:.2f}."

        return result
    except Exception as e:
        logger.error(f"Error detecting temporal contradiction: {str(e)}")
        return result

def detect_statistical_contradiction(claim1: str, claim2: str, metadata1: Dict[str, Any],
                                    metadata2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect statistical contradiction between two claims.

    Args:
        claim1: First claim
        claim2: Second claim
        metadata1: Metadata for first claim
        metadata2: Metadata for second claim

    Returns:
        Statistical contradiction detection result
    """
    # Initialize result
    result = {
        "is_contradiction": False,
        "score": 0.0,
        "confidence": ContradictionConfidence.LOW.value,
        "explanation": ""
    }

    try:
        # Extract statistical information from metadata
        p_value1 = metadata1.get("p_value")
        p_value2 = metadata2.get("p_value")
        sample_size1 = metadata1.get("sample_size")
        sample_size2 = metadata2.get("sample_size")

        # Skip if statistical information is not available
        if not p_value1 or not p_value2 or not sample_size1 or not sample_size2:
            return result

        # Calculate statistical score (simplified)
        # In a real implementation, you would use a more sophisticated method
        statistical_score = 0.5  # Placeholder

        # Set result
        result["score"] = float(statistical_score)
        result["is_contradiction"] = statistical_score > 0.7  # Threshold for contradiction

        # Set confidence based on score
        if statistical_score > 0.9:
            result["confidence"] = ContradictionConfidence.HIGH.value
        elif statistical_score > 0.8:
            result["confidence"] = ContradictionConfidence.MEDIUM.value
        else:
            result["confidence"] = ContradictionConfidence.LOW.value

        # Generate explanation
        if result["is_contradiction"]:
            result["explanation"] = f"The claims show statistical contradiction with a score of {statistical_score:.2f}."

        return result
    except Exception as e:
        logger.error(f"Error detecting statistical contradiction: {str(e)}")
        return result

def generate_explanation(claim1: str, claim2: str, contradiction_type: str,
                        shap_explainer: SHAPExplainer) -> str:
    """
    Generate explanation for contradiction using SHAP.

    Args:
        claim1: First claim
        claim2: Second claim
        contradiction_type: Type of contradiction
        shap_explainer: SHAP explainer

    Returns:
        Explanation
    """
    try:
        # Generate explanation (simplified)
        # In a real implementation, you would use SHAP for explanation
        explanation = f"The claims contradict each other in terms of {contradiction_type}."

        return explanation
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return f"The claims contradict each other."

def get_task_result(task_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the result of a task by ID.

    Args:
        task_id: The ID of the task

    Returns:
        Task result information or None if not found
    """
    return task_results.get(task_id)
