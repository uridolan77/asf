"""
Knowledge Base router for the Medical Research Synthesizer API.

This module provides endpoints for creating and managing knowledge bases.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, BackgroundTasks

from asf.medical.api.models.base import APIResponse, ErrorResponse
from asf.medical.api.models.knowledge_base import (
    KnowledgeBaseRequest, 
    KnowledgeBaseResponse
)
from asf.medical.api.dependencies import get_knowledge_base_service
from asf.medical.api.auth import get_current_active_user, get_admin_user
from asf.medical.services.knowledge_base_service import KnowledgeBaseService
from asf.medical.storage.models import User
from asf.medical.core.monitoring import async_timed, log_error

# Initialize router
router = APIRouter(prefix="/knowledge-base", tags=["knowledge_base"])

# Set up logging
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
    """
    Create a new knowledge base.
    
    This endpoint creates a new knowledge base with the given name and query,
    which can be updated automatically on a schedule.
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(f"Create knowledge base request: name='{request.name}', query='{request.query}', user_id={current_user.id}")
        
        # Create the knowledge base
        result = await kb_service.create_knowledge_base(
            name=request.name,
            query=request.query,
            update_schedule=request.update_schedule,
            user_id=current_user.id
        )
        
        # Log the result
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
        # Handle validation errors
        log_error(e, {"name": request.name, "query": request.query, "user_id": current_user.id})
        logger.warning(f"Validation error in knowledge base creation: {str(e)}")
        return ErrorResponse(
            message="Invalid knowledge base parameters",
            errors=[{"detail": str(e)}],
            code="VALIDATION_ERROR"
        )
    except Exception as e:
        # Handle unexpected errors
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
    """
    List all knowledge bases.
    
    This endpoint returns a list of all knowledge bases accessible to the current user.
    """
    try:
        # Log the request
        logger.info(f"List knowledge bases request: user_id={current_user.id}")
        
        # Get the knowledge bases
        result = await kb_service.list_knowledge_bases(user_id=current_user.id)
        
        # Log the result
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
        # Handle unexpected errors
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
    """
    Get a knowledge base by ID.
    
    This endpoint retrieves a knowledge base by its ID.
    """
    try:
        # Log the request
        logger.info(f"Get knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        # Get the knowledge base
        result = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not result:
            logger.warning(f"Knowledge base not found: kb_id={kb_id}")
            return ErrorResponse(
                message="Knowledge base not found",
                errors=[{"detail": f"No knowledge base found with ID {kb_id}"}],
                code="NOT_FOUND"
            )
        
        # Check if the user has access to this knowledge base
        if result.get("user_id") != current_user.id and current_user.role != "admin":
            logger.warning(f"User {current_user.id} does not have access to knowledge base {kb_id}")
            return ErrorResponse(
                message="Access denied",
                errors=[{"detail": "You do not have access to this knowledge base"}],
                code="ACCESS_DENIED"
            )
        
        # Log the result
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
        # Handle unexpected errors
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
    """
    Update a knowledge base.
    
    This endpoint updates a knowledge base with new articles matching the original query.
    """
    try:
        # Log the request
        logger.info(f"Update knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        # Get the knowledge base
        kb = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not kb:
            logger.warning(f"Knowledge base not found: kb_id={kb_id}")
            return ErrorResponse(
                message="Knowledge base not found",
                errors=[{"detail": f"No knowledge base found with ID {kb_id}"}],
                code="NOT_FOUND"
            )
        
        # Check if the user has access to this knowledge base
        if kb.get("user_id") != current_user.id and current_user.role != "admin":
            logger.warning(f"User {current_user.id} does not have access to knowledge base {kb_id}")
            return ErrorResponse(
                message="Access denied",
                errors=[{"detail": "You do not have access to this knowledge base"}],
                code="ACCESS_DENIED"
            )
        
        # Update the knowledge base in the background
        background_tasks.add_task(kb_service.update_knowledge_base, kb_id)
        
        # Log the result
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
        # Handle unexpected errors
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
    """
    Delete a knowledge base.
    
    This endpoint deletes a knowledge base by its ID.
    """
    try:
        # Log the request
        logger.info(f"Delete knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        # Get the knowledge base
        kb = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not kb:
            logger.warning(f"Knowledge base not found: kb_id={kb_id}")
            return ErrorResponse(
                message="Knowledge base not found",
                errors=[{"detail": f"No knowledge base found with ID {kb_id}"}],
                code="NOT_FOUND"
            )
        
        # Check if the user has access to this knowledge base
        if kb.get("user_id") != current_user.id and current_user.role != "admin":
            logger.warning(f"User {current_user.id} does not have access to knowledge base {kb_id}")
            return ErrorResponse(
                message="Access denied",
                errors=[{"detail": "You do not have access to this knowledge base"}],
                code="ACCESS_DENIED"
            )
        
        # Delete the knowledge base
        success = await kb_service.delete_knowledge_base(kb_id)
        
        if not success:
            logger.warning(f"Failed to delete knowledge base: kb_id={kb_id}")
            return ErrorResponse(
                message="Failed to delete knowledge base",
                errors=[{"detail": "An error occurred while deleting the knowledge base"}],
                code="DELETE_ERROR"
            )
        
        # Log the result
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
        # Handle unexpected errors
        log_error(e, {"kb_id": kb_id, "user_id": current_user.id})
        logger.error(f"Error deleting knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting knowledge base: {str(e)}"
        )
