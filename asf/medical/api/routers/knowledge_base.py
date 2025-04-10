"""
Knowledge Base router for the Medical Research Synthesizer API.

This module provides endpoints for creating and managing knowledge bases.
"""

import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

from asf.medical.api.models import KnowledgeBaseRequest, KnowledgeBaseResponse
from asf.medical.api.dependencies import get_synthesizer, get_current_user
from asf.medical.api.auth import User
from asf.medical.data_ingestion_layer.enhanced_medical_research_synthesizer import EnhancedMedicalResearchSynthesizer

# Initialize router
router = APIRouter(prefix="/v1/knowledge-base", tags=["Knowledge Base"])

# In-memory storage for knowledge bases (will be replaced with database in Phase 2)
kb_storage: Dict[str, Any] = {}

# Set up logging
logger = logging.getLogger(__name__)

@router.post("/", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    request: KnowledgeBaseRequest,
    background_tasks: BackgroundTasks,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Create and schedule updates for a knowledge base.

    This endpoint creates a new knowledge base for tracking publications on a
    specific topic and schedules regular updates to keep it current.
    """
    try:
        logger.info(f"Creating knowledge base: {request.name} (query={request.query})")
        
        # Create knowledge base
        kb_info = synthesizer.create_and_update_knowledge_base(
            name=request.name,
            query=request.query,
            schedule=request.schedule,
            max_results=request.max_results
        )
        
        # Store KB info
        kb_id = str(uuid.uuid4())
        kb_storage[kb_id] = {
            'kb_info': kb_info,
            'created_at': datetime.now().isoformat(),
            'user': current_user.email
        }
        
        logger.info(f"Knowledge base created: {request.name} (kb_id={kb_id})")
        
        return kb_info
    except Exception as e:
        logger.error(f"Error creating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{name}")
async def get_knowledge_base(
    name: str,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Get articles from a knowledge base.

    This endpoint retrieves all articles stored in a specific knowledge base.
    """
    try:
        logger.info(f"Retrieving knowledge base: {name}")
        
        articles = synthesizer.get_knowledge_base(name)
        
        logger.info(f"Knowledge base retrieved: {name} ({len(articles)} articles)")
        
        return {
            "name": name,
            "articles": articles,
            "count": len(articles)
        }
    except Exception as e:
        logger.error(f"Error getting knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def list_knowledge_bases(
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    List all available knowledge bases.

    This endpoint returns a list of all knowledge bases that have been created.
    """
    try:
        logger.info("Listing knowledge bases")
        
        # First check in-memory storage
        kb_list = []
        for kb_id, kb_data in kb_storage.items():
            kb_info = kb_data['kb_info']
            kb_list.append({
                'kb_id': kb_id,
                'name': kb_info['name'],
                'query': kb_info['query'],
                'initial_results': kb_info['initial_results'],
                'update_schedule': kb_info['update_schedule'],
                'created_date': kb_info['created_date']
            })
        
        # Then check file system
        kb_dir = synthesizer.kb_dir
        file_knowledge_bases = []
        
        if os.path.exists(kb_dir):
            for filename in os.listdir(kb_dir):
                if filename.endswith('.json'):
                    kb_name = filename.replace('.json', '')
                    
                    # Skip if already in memory
                    if any(kb['name'] == kb_name for kb in kb_list):
                        continue

                    # Get basic stats
                    kb_path = os.path.join(kb_dir, filename)
                    stat = os.stat(kb_path)

                    kb_info = {
                        "name": kb_name,
                        "file": kb_path,
                        "size": stat.st_size,
                        "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }

                    # Try to get more details
                    try:
                        articles = synthesizer.get_knowledge_base(kb_name)
                        kb_info["article_count"] = len(articles)
                    except:
                        kb_info["article_count"] = "unknown"

                    file_knowledge_bases.append(kb_info)
        
        logger.info(f"Knowledge bases listed: {len(kb_list)} in memory, {len(file_knowledge_bases)} in file system")
        
        return {
            "in_memory_knowledge_bases": kb_list,
            "file_knowledge_bases": file_knowledge_bases,
            "total_count": len(kb_list) + len(file_knowledge_bases)
        }
    except Exception as e:
        logger.error(f"Error listing knowledge bases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{name}/update")
async def update_knowledge_base(
    name: str,
    synthesizer: EnhancedMedicalResearchSynthesizer = Depends(get_synthesizer),
    current_user: User = Depends(get_current_user)
):
    """
    Manually update a knowledge base.
    
    This endpoint triggers an immediate update of the specified knowledge base.
    """
    try:
        logger.info(f"Updating knowledge base: {name}")
        
        # Find the KB in storage
        kb_id = None
        kb_data = None
        kb_query = None
        
        for id, data in kb_storage.items():
            if data['kb_info']['name'] == name:
                kb_id = id
                kb_data = data
                kb_query = data['kb_info']['query']
                break
        
        if not kb_query:
            # Try to find in file system
            kb_path = os.path.join(synthesizer.kb_dir, f"{name}.json")
            if not os.path.exists(kb_path):
                logger.error(f"Knowledge base not found: {name}")
                raise HTTPException(status_code=404, detail=f"Knowledge base '{name}' not found")
            
            # Get query from file
            try:
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                    kb_query = kb_data.get('query', '')
            except Exception as e:
                logger.error(f"Error reading knowledge base file: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error reading knowledge base file")
            
            if not kb_query:
                logger.error(f"Knowledge base query not found: {name}")
                raise HTTPException(status_code=500, detail=f"Knowledge base query not found")
            
            kb_file = kb_path
        else:
            kb_file = kb_data['kb_info']['kb_file']
        
        # Update the KB
        result = synthesizer.incremental_client.search_and_update_knowledge_base(
            kb_query,
            kb_file,
            max_results=100
        )
        
        logger.info(f"Knowledge base updated: {name} (new_count={result['new_count']})")
        
        return {
            'name': name,
            'query': kb_query,
            'total_count': result['total_count'],
            'new_count': result['new_count'],
            'update_time': result['update_time']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
