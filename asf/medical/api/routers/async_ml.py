"""
Asynchronous ML Inference Router for the Medical Research Synthesizer API.

This module provides endpoints for asynchronous ML model inference operations.
"""
import logging
import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query

from ..auth import get_current_active_user
from ..models.base import APIResponse
from ...storage.models import User
from ...tasks.ml_inference_tasks import (
    detect_contradiction, analyze_contradictions_in_articles,
    generate_embeddings, get_task_result
)
from ...core.task_storage import task_storage
from ...core.observability import async_timed, log_error

router = APIRouter(prefix="/async-ml", tags=["async-ml"])

logger = logging.getLogger(__name__)

@router.post("/contradiction/detect", response_model=APIResponse[Dict[str, Any]])
@async_timed("async_detect_contradiction_endpoint")
async def async_detect_contradiction(
    claim1: str,
    claim2: str,
    metadata1: Optional[Dict[str, Any]] = None,
    metadata2: Optional[Dict[str, Any]] = None,
    use_all_methods: bool = True,
    current_user: User = Depends(get_current_active_user)
):
    try:
        logger.info(f"Starting asynchronous contradiction detection")

        message = detect_contradiction.send(claim1, claim2, metadata1, metadata2, use_all_methods)
        task_id = message.message_id

        return APIResponse(
            success=True,
            message="Contradiction detection started",
            data={
                "task_id": task_id,
                "status": "processing"
            },
            meta={
                "user_id": current_user.id
            }
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error starting contradiction detection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting contradiction detection: {str(e)}"
        )

@router.post("/contradiction/analyze", response_model=APIResponse[Dict[str, Any]])
@async_timed("async_analyze_contradictions_endpoint")
async def async_analyze_contradictions(
    articles: List[Dict[str, Any]],
    threshold: float = Query(0.7, description="Contradiction detection threshold"),
    use_all_methods: bool = True,
    current_user: User = Depends(get_current_active_user)
):
    try:
        logger.info(f"Starting asynchronous contradiction analysis for {len(articles)} articles")

        message = analyze_contradictions_in_articles.send(articles, threshold, use_all_methods)
        task_id = message.message_id

        return APIResponse(
            success=True,
            message="Contradiction analysis started",
            data={
                "task_id": task_id,
                "status": "processing",
                "total_articles": len(articles)
            },
            meta={
                "user_id": current_user.id
            }
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error starting contradiction analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting contradiction analysis: {str(e)}"
        )

@router.post("/embeddings/generate", response_model=APIResponse[Dict[str, Any]])
@async_timed("async_generate_embeddings_endpoint")
async def async_generate_embeddings(
    texts: List[str],
    model_name: str = Query("biomedlm", description="Model to use for embeddings"),
    current_user: User = Depends(get_current_active_user)
):
    try:
        logger.info(f"Starting asynchronous embedding generation for {len(texts)} texts")

        message = generate_embeddings.send(texts, model_name)
        task_id = message.message_id

        return APIResponse(
            success=True,
            message="Embedding generation started",
            data={
                "task_id": task_id,
                "status": "processing",
                "total_texts": len(texts)
            },
            meta={
                "user_id": current_user.id
            }
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error starting embedding generation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting embedding generation: {str(e)}"
        )

@router.get("/task/{task_id}", response_model=APIResponse[Dict[str, Any]])
@async_timed("get_ml_task_status_endpoint")
async def get_ml_task_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    try:
        logger.info(f"Getting status for ML task {task_id}")

        task_result = get_task_result(task_id)

        if not task_result:
            task_result = task_storage.get_task_status(task_id)

        if task_result:
            if task_result.get("status") == "completed" and "result" in task_result:
                try:
                    result = json.loads(task_result["result"])
                    task_result["result"] = result
                except json.JSONDecodeError:
                    logger.error(f"Error: {str(e)}")
                    pass

            return APIResponse(
                success=True,
                message=f"Task status: {task_result.get('status')}",
                data=task_result,
                meta={
                    "user_id": current_user.id
                }
            )

        return APIResponse(
            success=False,
            message="Task not found",
            data={"status": "not_found"},
            meta={
                "user_id": current_user.id
            }
        )

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting task status: {str(e)}"
        )
