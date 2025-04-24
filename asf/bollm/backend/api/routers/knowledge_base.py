"""
Router for knowledge base management endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from models.user import User
from config.database import get_db
from api.auth import get_current_user
from api.services.medical_client_integration import get_medical_client, MedicalClientIntegration

router = APIRouter(
    prefix="/api/medical/knowledge-base",
    tags=["knowledge-base"]
)

# Request/Response Models
class KnowledgeBaseCreate(BaseModel):
    name: str = Field(..., description="Knowledge base name")
    query: str = Field(..., description="Search query for populating the knowledge base")
    update_schedule: str = Field("weekly", description="Update frequency (daily, weekly, monthly)")

class KnowledgeBaseResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

@router.post("", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    medical_client: MedicalClientIntegration = Depends(get_medical_client)
):
    """
    Create a new knowledge base for medical research.
    """
    try:
        result = await medical_client.create_knowledge_base(
            db=db,
            user_id=current_user.id,
            name=data.name,
            query=data.query,
            update_schedule=data.update_schedule
        )
        
        return {
            "success": result["success"],
            "message": result["message"],
            "data": result["data"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create knowledge base: {str(e)}"
        )

@router.get("", response_model=KnowledgeBaseResponse)
async def list_knowledge_bases(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    medical_client: MedicalClientIntegration = Depends(get_medical_client)
):
    """
    List all knowledge bases.
    """
    try:
        result = await medical_client.list_knowledge_bases(
            db=db,
            user_id=current_user.id
        )
        
        return {
            "success": result["success"],
            "message": result["message"],
            "data": result["data"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list knowledge bases: {str(e)}"
        )

@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    medical_client: MedicalClientIntegration = Depends(get_medical_client)
):
    """
    Get details of a specific knowledge base.
    """
    try:
        result = await medical_client.get_knowledge_base(
            db=db,
            user_id=current_user.id,
            kb_id=kb_id
        )
        
        return {
            "success": result["success"],
            "message": result["message"],
            "data": result["data"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get knowledge base: {str(e)}"
        )

@router.post("/{kb_id}/update", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    medical_client: MedicalClientIntegration = Depends(get_medical_client)
):
    """
    Trigger an update for a knowledge base.
    """
    try:
        result = await medical_client.update_knowledge_base(
            db=db,
            user_id=current_user.id,
            kb_id=kb_id
        )
        
        return {
            "success": result["success"],
            "message": result["message"],
            "data": result["data"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update knowledge base: {str(e)}"
        )

@router.delete("/{kb_id}", response_model=KnowledgeBaseResponse)
async def delete_knowledge_base(
    kb_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    medical_client: MedicalClientIntegration = Depends(get_medical_client)
):
    """
    Delete a knowledge base.
    """
    try:
        result = await medical_client.delete_knowledge_base(
            db=db,
            user_id=current_user.id,
            kb_id=kb_id
        )
        
        return {
            "success": result["success"],
            "message": result["message"],
            "data": result["data"]
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete knowledge base: {str(e)}"
        )