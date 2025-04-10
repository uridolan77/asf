"""
Knowledge Base router for the Medical Research Synthesizer API.

This module provides endpoints for managing knowledge bases.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response, BackgroundTasks

from asf.medical.api.models.kb import (
    KnowledgeBaseRequest, 
    KnowledgeBaseResponse,
    KnowledgeBaseListResponse
)
from asf.medical.api.dependencies import get_knowledge_base_service
from asf.medical.api.auth import get_current_active_user, get_admin_user
from asf.medical.services.knowledge_base_service import KnowledgeBaseService
from asf.medical.storage.models import User

# Initialize router
router = APIRouter(prefix="/v1/kb", tags=["Knowledge Base"])

# Set up logging
logger = logging.getLogger(__name__)

@router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    request: KnowledgeBaseRequest,
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user),
    req: Request = None,
    res: Response = None
):
    """
    Create a new knowledge base.

    This endpoint creates a new knowledge base with the given name and query.
    """
    try:
        # Add request ID to response headers for tracing
        request_id = req.headers.get("X-Request-ID") if req else None
        if request_id and res:
            res.headers["X-Request-ID"] = request_id
        
        # Log the request
        logger.info(f"Create knowledge base request: name='{request.name}', query='{request.query}', user_id={current_user.id}")
        
        # Validate inputs
        if not request.name:
            raise ValueError("Name is required for knowledge base")
        if not request.query:
            raise ValueError("Query is required for knowledge base")
        
        # Create the knowledge base
        result = await kb_service.create_knowledge_base(
            name=request.name,
            query=request.query,
            update_schedule=request.update_schedule,
            user_id=current_user.id
        )
        
        # Log the result
        logger.info(f"Knowledge base created: {result.get('kb_id')}")
        
        return result
    except ValueError as e:
        # Handle validation errors
        logger.warning(f"Validation error in knowledge base creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating knowledge base: {str(e)}"
        )

@router.get("", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    List all knowledge bases for the current user.
    
    This endpoint retrieves all knowledge bases created by the current user.
    """
    try:
        # Log the request
        logger.info(f"List knowledge bases request: user_id={current_user.id}")
        
        # Get the knowledge bases
        result = await kb_service.list_knowledge_bases(user_id=current_user.id)
        
        # Log the result
        logger.info(f"Knowledge bases listed: {len(result)} found")
        
        return {"knowledge_bases": result}
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing knowledge bases: {str(e)}"
        )

@router.get("/all", response_model=KnowledgeBaseListResponse)
async def list_all_knowledge_bases(
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_admin_user)
):
    """
    List all knowledge bases (admin only).
    
    This endpoint retrieves all knowledge bases in the system.
    """
    try:
        # Log the request
        logger.info(f"List all knowledge bases request: admin_id={current_user.id}")
        
        # Get the knowledge bases
        result = await kb_service.list_knowledge_bases()
        
        # Log the result
        logger.info(f"All knowledge bases listed: {len(result)} found")
        
        return {"knowledge_bases": result}
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error listing all knowledge bases: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing all knowledge bases: {str(e)}"
        )

@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Check if the user has access to this knowledge base
        if result.get('user_id') != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this knowledge base"
            )
        
        # Log the result
        logger.info(f"Knowledge base retrieved: {kb_id}")
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error retrieving knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving knowledge base: {str(e)}"
        )

@router.put("/{kb_id}/update", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    background_tasks: BackgroundTasks,
    kb_service: KnowledgeBaseService = Depends(get_knowledge_base_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a knowledge base.
    
    This endpoint updates a knowledge base with new articles.
    """
    try:
        # Log the request
        logger.info(f"Update knowledge base request: kb_id={kb_id}, user_id={current_user.id}")
        
        # Get the knowledge base
        kb = await kb_service.get_knowledge_base_by_id(kb_id)
        
        if not kb:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Check if the user has access to this knowledge base
        if kb.get('user_id') != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update this knowledge base"
            )
        
        # Update the knowledge base in the background
        background_tasks.add_task(kb_service.update_knowledge_base, kb_id)
        
        # Log the result
        logger.info(f"Knowledge base update started: {kb_id}")
        
        return {
            **kb,
            "message": "Update started in the background"
        }
    except HTTPException:
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error updating knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating knowledge base: {str(e)}"
        )

@router.delete("/{kb_id}")
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
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge base with ID {kb_id} not found"
            )
        
        # Check if the user has access to this knowledge base
        if kb.get('user_id') != current_user.id and current_user.role != "admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this knowledge base"
            )
        
        # Delete the knowledge base
        result = await kb_service.delete_knowledge_base(kb_id)
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete knowledge base with ID {kb_id}"
            )
        
        # Log the result
        logger.info(f"Knowledge base deleted: {kb_id}")
        
        return {"message": f"Knowledge base with ID {kb_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Error deleting knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting knowledge base: {str(e)}"
        )
