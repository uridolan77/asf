"""
Medical Knowledge Base API router for BO backend.
"""
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from sqlalchemy.orm import Session

from api.services.medical_kb_service import MedicalKnowledgeBaseService, get_medical_kb_service
from config.database import get_db
from models.user import User
from api.dependencies import get_current_user

router = APIRouter(
    prefix="/api/medical/kb",
    tags=["medical-knowledge-base"],
    responses={404: {"description": "Not found"}},
)

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_knowledge_base(
    name: str,
    query: str,
    update_schedule: str = "weekly",
    kb_service: MedicalKnowledgeBaseService = Depends(get_medical_kb_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    Create a new knowledge base with the given parameters.
    """
    user_id = current_user.id if current_user else None
    result = await kb_service.create_knowledge_base(
        name=name,
        query=query,
        update_schedule=update_schedule,
        user_id=user_id
    )
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["message"]
        )
        
    return result

@router.get("/{kb_id}")
async def get_knowledge_base(
    kb_id: str = Path(..., description="The ID of the knowledge base to retrieve"),
    kb_service: MedicalKnowledgeBaseService = Depends(get_medical_kb_service)
):
    """
    Get a specific knowledge base by ID.
    """
    result = await kb_service.get_knowledge_base(kb_id=kb_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result["message"]
        )
        
    return result

@router.get("/")
async def list_knowledge_bases(
    user_specific: bool = Query(False, description="Filter knowledge bases by current user"),
    kb_service: MedicalKnowledgeBaseService = Depends(get_medical_kb_service),
    current_user: Optional[User] = Depends(get_current_user)
):
    """
    List all knowledge bases, optionally filtered by user.
    """
    user_id = current_user.id if user_specific and current_user else None
    result = await kb_service.list_knowledge_bases(user_id=user_id)
    
    if not result["success"]:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["message"]
        )
        
    return result

@router.put("/{kb_id}/update")
async def update_knowledge_base(
    kb_id: str = Path(..., description="The ID of the knowledge base to update"),
    kb_service: MedicalKnowledgeBaseService = Depends(get_medical_kb_service)
):
    """
    Trigger an update for a knowledge base.
    """
    result = await kb_service.update_knowledge_base(kb_id=kb_id)
    
    if not result["success"]:
        if "not found" in result["message"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
    return result

@router.delete("/{kb_id}")
async def delete_knowledge_base(
    kb_id: str = Path(..., description="The ID of the knowledge base to delete"),
    kb_service: MedicalKnowledgeBaseService = Depends(get_medical_kb_service)
):
    """
    Delete a knowledge base.
    """
    result = await kb_service.delete_knowledge_base(kb_id=kb_id)
    
    if not result["success"]:
        if "not found" in result["message"].lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["message"]
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["message"]
            )
        
    return result