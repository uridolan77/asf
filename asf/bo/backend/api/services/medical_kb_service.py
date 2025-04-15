"""
Medical Knowledge Base API service for integrating with the knowledge_base_service.py module
from the medical package.
"""
import os
import sys
import logging
import httpx
from typing import Dict, Any, List, Optional
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

# Add the parent directory to sys.path to import the medical module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import from the medical module
from asf.medical.services.knowledge_base_service import KnowledgeBaseService
from asf.medical.services.search_service import SearchService
from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository

from config.database import get_db
from models.user import User

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalKnowledgeBaseService:
    """
    Service for interacting with the medical module's KnowledgeBaseService.
    This provides a bridge between the BO frontend and the Medical Research knowledge base functionality.
    """
    def __init__(self):
        """Initialize with direct access to the medical module's KnowledgeBaseService"""
        # Initialize the required dependencies for KnowledgeBaseService
        self.search_service = SearchService()
        self.kb_repository = KnowledgeBaseRepository()
        self.kb_service = KnowledgeBaseService(
            search_service=self.search_service,
            kb_repository=self.kb_repository
        )
        
    async def create_knowledge_base(
        self, 
        name: str, 
        query: str, 
        update_schedule: str = "weekly",
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new knowledge base.
        
        Args:
            name: Knowledge base name
            query: Search query
            update_schedule: Update frequency (daily, weekly, monthly)
            user_id: BO user ID
            
        Returns:
            Created knowledge base details
        """
        try:
            kb = await self.kb_service.create_knowledge_base(
                name=name,
                query=query,
                update_schedule=update_schedule,
                user_id=user_id
            )
            
            return {
                "success": True,
                "message": f"Knowledge base '{name}' created successfully",
                "data": kb
            }
        except Exception as e:
            logger.error(f"Error creating knowledge base: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to create knowledge base: {str(e)}",
                "data": None
            }
    
    async def get_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """
        Get details of a specific knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            Knowledge base details
        """
        try:
            kb = await self.kb_service.get_knowledge_base_by_id(kb_id=kb_id)
            if not kb:
                return {
                    "success": False,
                    "message": f"Knowledge base with ID '{kb_id}' not found",
                    "data": None
                }
                
            return {
                "success": True,
                "message": "Knowledge base retrieved successfully",
                "data": kb
            }
        except Exception as e:
            logger.error(f"Error retrieving knowledge base: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to retrieve knowledge base: {str(e)}",
                "data": None
            }
    
    async def list_knowledge_bases(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        List all knowledge bases.
        
        Args:
            user_id: Optional user ID to filter by
            
        Returns:
            List of knowledge bases
        """
        try:
            kbs = await self.kb_service.list_knowledge_bases(user_id=user_id)
            
            return {
                "success": True,
                "message": f"Found {len(kbs)} knowledge bases",
                "data": kbs
            }
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to list knowledge bases: {str(e)}",
                "data": []
            }
    
    async def update_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """
        Trigger an update for a knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            Updated knowledge base details
        """
        try:
            kb = await self.kb_service.update_knowledge_base(kb_id=kb_id)
            
            return {
                "success": True,
                "message": f"Knowledge base '{kb['name']}' updated successfully",
                "data": kb
            }
        except Exception as e:
            logger.error(f"Error updating knowledge base: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to update knowledge base: {str(e)}",
                "data": None
            }
    
    async def delete_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """
        Delete a knowledge base.
        
        Args:
            kb_id: Knowledge base ID
            
        Returns:
            Deletion status
        """
        try:
            result = await self.kb_service.delete_knowledge_base(kb_id=kb_id)
            
            return {
                "success": result,
                "message": f"Knowledge base deleted successfully",
                "data": {"kb_id": kb_id}
            }
        except Exception as e:
            logger.error(f"Error deleting knowledge base: {str(e)}")
            return {
                "success": False,
                "message": f"Failed to delete knowledge base: {str(e)}",
                "data": None
            }

# Dependency to get the medical knowledge base service
def get_medical_kb_service() -> MedicalKnowledgeBaseService:
    """Factory function to create and provide a MedicalKnowledgeBaseService instance."""
    return MedicalKnowledgeBaseService()