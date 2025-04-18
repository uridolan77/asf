"""Knowledge Base router for the Medical Research Synthesizer API.

This module provides endpoints for creating and managing knowledge bases.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, BackgroundTasks

from ..models.base import APIResponse, ErrorResponse
from ..models.knowledge_base import (
    KnowledgeBaseRequest, 
    KnowledgeBaseResponse
)
from ..dependencies import get_knowledge_base_service
from ..auth import get_current_active_user
from ...core.exceptions import KnowledgeBaseError
from ...core.observability import async_timed, log_error
from ...services.knowledge_base_service import KnowledgeBaseService
from ...storage.models import User

router = APIRouter(prefix="/knowledge-base", tags=["knowledge_base"])

logger = logging.getLogger(__name__)

@router.post("", response_model=APIResponse[KnowledgeBaseResponse])
@async_timed("create_knowledge_base_endpoint")
async def create_knowledge_base(
    request: KnowledgeBaseRequest,
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """Create a new knowledge base.
    
    Args:
        request: Knowledge base creation request containing name and query
        kb_service: Service for managing knowledge bases
        current_user: The authenticated user making the request
        req: The FastAPI request object
        res: The FastAPI response object
        
    Returns:
        APIResponse containing the created knowledge base details
        
    Raises:
        HTTPException: For unexpected server errors
        ValueError: For invalid knowledge base parameters
    """
    try:
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        logger.info(f"Create knowledge base request: name='{request.name}', query='{request.query}', user_id={current_user.id}")
        
        result = await kb_service.create_knowledge_base(
            name=request.name,
            query=request.query,
            update_schedule=request.update_schedule,
            user_id=current_user.id
        )
        
        logger.info(f"Knowledge base created: {result['kb_id']} (name='{result['name']}')")
        
        return APIResponse(
            success=True,
            message="Knowledge base created successfully",
            data=result,
            meta={
                "name": request.name,
                "query": request.query,
                "update_schedule": request.update_schedule,
                "user_id": current_user.id
            }
        )
    except ValueError as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"name": request.name, "query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in knowledge base creation: {str(e)}")
        return ErrorResponse(
            message="Invalid knowledge base parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"name": request.name, "query": request.query, "user_id": current_user.id})
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating knowledge base: {str(e)}"
        )

@router.get("", response_model=APIResponse[List[KnowledgeBaseResponse]])
@async_timed("list_knowledge_bases_endpoint")
async def list_knowledge_bases(
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user)
):
    """List all knowledge bases owned by the current user.
    
    Args:
        kb_service: Service for managing knowledge bases
        current_user: The authenticated user making the request
        
    Returns:
        APIResponse containing a list of knowledge bases
        
    Raises:
        HTTPException: For unexpected server errors
    """
    try:
        logger.info(f"List knowledge bases request: user_id={current_user.id}")
        
        result = await kb_service.list_knowledge_bases(user_id=current_user.id)
        
        logger.info(f"Knowledge bases listed: {len(result)} found")
        
        return APIResponse(
            success=True,
            message="Knowledge bases listed successfully",
            data=result,
            meta={
                "count": len(result),
                "user_id": current_user.id
            }
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"user_id": current_user.id})
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing knowledge bases: {str(e)}"
        )

@router.get("/{kb_id}", response_model=APIResponse[KnowledgeBaseResponse])
@async_timed("get_knowledge_base_endpoint")
async def get_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user)
):
    """Retrieve a knowledge base by its ID.
    
    Args:
        kb_id: ID of the knowledge base to retrieve
        kb_service: Service for managing knowledge bases
        current_user: The authenticated user making the request
        
    Returns:
        APIResponse containing the knowledge base details
        
    Raises:
        HTTPException: For unexpected server errors
        ErrorResponse: If the knowledge base is not found or the user doesn't have access
    """
    try:
        logger.info(f"Get knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        result = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not result:
            logger.warning(f"Knowledge base not found: kb_id={kb_id}")
            return ErrorResponse(
                message="Knowledge base not found",
                errors=[{"detail": f"No knowledge base found with ID {kb_id}"}],
                code="NOT_FOUND"
            )
        
        if result.get("user_id") != current_user.id and current_user.role != "admin":
            logger.warning(f"User {current_user.id} does not have access to knowledge base {kb_id}")
            return ErrorResponse(
                message="Access denied",
                errors=[{"detail": "You do not have access to this knowledge base"}],
                code="ACCESS_DENIED"
            )
        
        logger.info(f"Knowledge base retrieved: kb_id={kb_id}")
        
        return APIResponse(
            success=True,
            message="Knowledge base retrieved successfully",
            data=result,
            meta={
                "kb_id": kb_id,
                "user_id": current_user.id
            }
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"kb_id": kb_id, "user_id": current_user.id})
        logger.error(f"Error retrieving knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving knowledge base: {str(e)}"
        )

@router.post("/{kb_id}/update", response_model=APIResponse[Dict[str, Any]])
@async_timed("update_knowledge_base_endpoint")
async def update_knowledge_base(
    kb_id: str,
    background_tasks: BackgroundTasks,
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user)
):
    """Update a knowledge base with new content asynchronously.
    
    Args:
        kb_id: ID of the knowledge base to update
        background_tasks: FastAPI background tasks manager
        kb_service: Service for managing knowledge bases
        current_user: The authenticated user making the request
        
    Returns:
        APIResponse indicating the update has been started
        
    Raises:
        HTTPException: For unexpected server errors
        ErrorResponse: If the knowledge base is not found or the user doesn't have access
    """
    try:
        logger.info(f"Update knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        kb = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not kb:
            logger.warning(f"Knowledge base not found: kb_id={kb_id}")
            return ErrorResponse(
                message="Knowledge base not found",
                errors=[{"detail": f"No knowledge base found with ID {kb_id}"}],
                code="NOT_FOUND"
            )
        
        if kb.get("user_id") != current_user.id and current_user.role != "admin":
            logger.warning(f"User {current_user.id} does not have access to knowledge base {kb_id}")
            return ErrorResponse(
                message="Access denied",
                errors=[{"detail": "You do not have access to this knowledge base"}],
                code="ACCESS_DENIED"
            )
        
        background_tasks.add_task(kb_service.update_knowledge_base, kb_id)
        
        logger.info(f"Knowledge base update started: kb_id={kb_id}")
        
        return APIResponse(
            success=True,
            message="Knowledge base update started",
            data={
                "kb_id": kb_id,
                "name": kb["name"],
                "status": "updating"
            },
            meta={
                "kb_id": kb_id,
                "user_id": current_user.id
            }
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"kb_id": kb_id, "user_id": current_user.id})
        logger.error(f"Error updating knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating knowledge base: {str(e)}"
        )

@router.delete("/{kb_id}", response_model=APIResponse[Dict[str, Any]])
@async_timed("delete_knowledge_base_endpoint")
async def delete_knowledge_base(
    kb_id: str,
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user)
):
    """Delete a knowledge base.
    
    Args:
        kb_id: ID of the knowledge base to delete
        kb_service: Service for managing knowledge bases
        current_user: The authenticated user making the request
        
    Returns:
        APIResponse confirming the deletion
        
    Raises:
        HTTPException: For unexpected server errors
        ErrorResponse: If the knowledge base is not found, the user doesn't have access, or deletion fails
    """
    try:
        logger.info(f"Delete knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        kb = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not kb:
            logger.warning(f"Knowledge base not found: kb_id={kb_id}")
            return ErrorResponse(
                message="Knowledge base not found",
                errors=[{"detail": f"No knowledge base found with ID {kb_id}"}],
                code="NOT_FOUND"
            )
        
        if kb.get("user_id") != current_user.id and current_user.role != "admin":
            logger.warning(f"User {current_user.id} does not have access to knowledge base {kb_id}")
            return ErrorResponse(
                message="Access denied",
                errors=[{"detail": "You do not have access to this knowledge base"}],
                code="ACCESS_DENIED"
            )
        
        success = await kb_service.delete_knowledge_base(kb_id)
        
        if not success:
            logger.warning(f"Failed to delete knowledge base: kb_id={kb_id}")
            return ErrorResponse(
                message="Failed to delete knowledge base",
                errors=[{"detail": "An error occurred while deleting the knowledge base"}],
                code="DELETE_ERROR"
            )
        
        logger.info(f"Knowledge base deleted: kb_id={kb_id}")
        
        return APIResponse(
            success=True,
            message="Knowledge base deleted successfully",
            data={
                "kb_id": kb_id,
                "name": kb["name"],
                "status": "deleted"
            },
            meta={
                "kb_id": kb_id,
                "user_id": current_user.id
            }
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        log_error(e, {"kb_id": kb_id, "user_id": current_user.id})
        logger.error(f"Error deleting knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting knowledge base: {str(e)}"
        )
