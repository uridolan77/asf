"""Knowledge base service for the Medical Research Synthesizer.

This module provides a service for creating and managing knowledge bases.
"""
import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from asf.medical.core.enhanced_cache import enhanced_cache_manager, enhanced_cached
from asf.medical.core.exceptions import (
    ResourceNotFoundError, ValidationError,
    ExternalServiceError, DatabaseError, FileError
)
from asf.medical.services.search_service import SearchService
from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository
logger = logging.getLogger(__name__)
class KnowledgeBaseService:
    """Service for creating and managing knowledge bases.

    This service provides methods for creating, updating, and retrieving knowledge bases,
    which are collections of medical literature organized around specific topics.
    """
    def __init__(
        self,
        search_service: SearchService,
        kb_repository: KnowledgeBaseRepository,
        kb_dir: str = "knowledge_bases"
    ):
        self.search_service = search_service
        self.kb_repository = kb_repository
        self.kb_dir = kb_dir
        os.makedirs(self.kb_dir, exist_ok=True)
        self.kb_namespace = "kb:"
    async def create_knowledge_base(
        self,
        name: str,
        query: str,
        update_schedule: str = "weekly",
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        if not name or not name.strip():
            raise ValidationError("Knowledge base name cannot be empty")
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        if update_schedule not in ["daily", "weekly", "monthly"]:
            raise ValidationError("Update schedule must be 'daily', 'weekly', or 'monthly'")
        logger.info(f"Creating knowledge base: {name} (query={query})")
        existing_kb = await self.get_knowledge_base(name)
        if existing_kb:
            raise ValidationError(f"Knowledge base with name '{name}' already exists")
        try:
            search_result = await self.search_service.search(query, max_results=100, user_id=user_id)
            if not search_result or not search_result.get('results'):
                logger.warning(f"No results found for query: {query}")
                raise ResourceNotFoundError("Search Results", query)
        except ValidationError:
            raise
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error searching for articles: {str(e)}")
            raise ExternalServiceError("Search Service", f"Failed to search for articles: {str(e)}")
        articles = search_result['results']
        kb_id = str(uuid.uuid4())
        file_path = os.path.join(self.kb_dir, f"{kb_id}.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(articles, f)
            logger.info(f"Knowledge base file created: {file_path}")
        except Exception as e:
            logger.error(f"Error creating knowledge base file: {str(e)}")
            raise FileError(file_path, f"Failed to create knowledge base file: {str(e)}")
        now = datetime.now()
        try:
            if update_schedule == "daily":
                next_update = now + timedelta(days=1)
            elif update_schedule == "weekly":
                next_update = now + timedelta(weeks=1)
            elif update_schedule == "monthly":
                next_update = now + timedelta(days=30)
            else:
                next_update = now + timedelta(weeks=1)  # Default to weekly
        except Exception as e:
            logger.error(f"Error calculating next update time: {str(e)}")
            next_update = now + timedelta(weeks=1)
        kb = {
            'kb_id': kb_id,
            'name': name,
            'query': query,
            'file_path': file_path,
            'update_schedule': update_schedule,
            'last_updated': now.isoformat(),
            'next_update': next_update.isoformat(),
            'initial_results': len(articles),
            'user_id': user_id,
            'created_at': now.isoformat()
        }
        cache_key = f"{self.kb_namespace}{kb_id}"
        await enhanced_cache_manager.set(cache_key, kb, data_type="knowledge_base")
        name_cache_key = f"{self.kb_namespace}name:{name}"
        await enhanced_cache_manager.set(name_cache_key, kb, data_type="knowledge_base")
        try:
            await self.kb_repository.create_knowledge_base_async(
                db,  # This will be handled by the repository
                name=name,
                query=query,
                file_path=file_path,
                update_schedule=update_schedule,
                initial_results=len(articles),
                user_id=user_id
            )
            logger.info(f"Knowledge base stored in database: {kb_id}")
        except Exception as e:
            logger.error(f"Error storing knowledge base: {str(e)}")
            try:
                os.remove(file_path)
                logger.info(f"Deleted knowledge base file due to database error: {file_path}")
            except Exception as file_e:
                logger.error(f"Error deleting knowledge base file: {str(file_e)}")
            raise DatabaseError(f"Failed to store knowledge base: {str(e)}")
        logger.info(f"Knowledge base created: {kb_id}")
        return kb
    async def get_knowledge_base(self, name: str) -> Optional[Dict[str, Any]]:
        if not name or not name.strip():
            raise ValidationError("Knowledge base name cannot be empty")
        logger.info(f"Getting knowledge base by name: {name}")
        name_cache_key = f"{self.kb_namespace}name:{name}"
        cached_kb = await enhanced_cache_manager.get(name_cache_key, data_type="knowledge_base")
        if cached_kb is not None:
            logger.debug(f"Knowledge base found in cache: {name}")
            return cached_kb
        try:
            kb = await self.kb_repository.get_by_name_async(db, name=name)
            if kb:
                logger.debug(f"Knowledge base found in database: {name}")
                kb_dict = {
                    'kb_id': kb.kb_id,
                    'name': kb.name,
                    'query': kb.query,
                    'file_path': kb.file_path,
                    'update_schedule': kb.update_schedule,
                    'last_updated': kb.last_updated.isoformat() if kb.last_updated else None,
                    'next_update': kb.next_update.isoformat() if kb.next_update else None,
                    'initial_results': kb.initial_results,
                    'user_id': kb.user_id,
                    'created_at': kb.created_at.isoformat()
                }
                cache_key = f"{self.kb_namespace}{kb.kb_id}"
                await enhanced_cache_manager.set(cache_key, kb_dict, data_type="knowledge_base")
                await enhanced_cache_manager.set(name_cache_key, kb_dict, data_type="knowledge_base")
                return kb_dict
            else:
                logger.debug(f"Knowledge base not found: {name}")
                return None
        except Exception as e:
            logger.error(f"Error getting knowledge base by name: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")
    async def get_knowledge_base_by_id(self, kb_id: str) -> Optional[Dict[str, Any]]:
        if not kb_id or not kb_id.strip():
            raise ValidationError("Knowledge base ID cannot be empty")
        logger.info(f"Getting knowledge base by ID: {kb_id}")
        cache_key = f"{self.kb_namespace}{kb_id}"
        cached_kb = await enhanced_cache_manager.get(cache_key, data_type="knowledge_base")
        if cached_kb is not None:
            logger.debug(f"Knowledge base found in cache: {kb_id}")
            return cached_kb
        try:
            kb = await self.kb_repository.get_by_kb_id_async(db, kb_id=kb_id)
            if kb:
                logger.debug(f"Knowledge base found in database: {kb_id}")
                kb_dict = {
                    'kb_id': kb.kb_id,
                    'name': kb.name,
                    'query': kb.query,
                    'file_path': kb.file_path,
                    'update_schedule': kb.update_schedule,
                    'last_updated': kb.last_updated.isoformat() if kb.last_updated else None,
                    'next_update': kb.next_update.isoformat() if kb.next_update else None,
                    'initial_results': kb.initial_results,
                    'user_id': kb.user_id,
                    'created_at': kb.created_at.isoformat()
                }
                await enhanced_cache_manager.set(cache_key, kb_dict, data_type="knowledge_base")
                name_cache_key = f"{self.kb_namespace}name:{kb.name}"
                await enhanced_cache_manager.set(name_cache_key, kb_dict, data_type="knowledge_base")
                return kb_dict
            else:
                logger.warning(f"Knowledge base not found: {kb_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting knowledge base by ID: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")
    async def list_knowledge_bases(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        logger.info(f"Listing knowledge bases: user_id={user_id if user_id else 'all'}")
        cache_key = f"{self.kb_namespace}list:{user_id if user_id else 'all'}"
        cached_list = await enhanced_cache_manager.get(cache_key, data_type="knowledge_base")
        if cached_list is not None:
            logger.debug(f"Knowledge base list found in cache: {len(cached_list)} items")
            return cached_list
        try:
            kbs = await self.kb_repository.list_async(db, user_id=user_id)
            kb_dicts = []
            for kb in kbs:
                kb_dict = {
                    'kb_id': kb.kb_id,
                    'name': kb.name,
                    'query': kb.query,
                    'file_path': kb.file_path,
                    'update_schedule': kb.update_schedule,
                    'last_updated': kb.last_updated.isoformat() if kb.last_updated else None,
                    'next_update': kb.next_update.isoformat() if kb.next_update else None,
                    'initial_results': kb.initial_results,
                    'user_id': kb.user_id,
                    'created_at': kb.created_at.isoformat()
                }
                kb_cache_key = f"{self.kb_namespace}{kb.kb_id}"
                await enhanced_cache_manager.set(kb_cache_key, kb_dict, data_type="knowledge_base")
                name_cache_key = f"{self.kb_namespace}name:{kb.name}"
                await enhanced_cache_manager.set(name_cache_key, kb_dict, data_type="knowledge_base")
                kb_dicts.append(kb_dict)
            await enhanced_cache_manager.set(cache_key, kb_dicts, data_type="knowledge_base")
            logger.info(f"Found {len(kb_dicts)} knowledge bases")
            return kb_dicts
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {str(e)}")
            raise DatabaseError(f"Failed to list knowledge bases: {str(e)}")
    async def update_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        if not kb_id or not kb_id.strip():
            raise ValidationError("Knowledge base ID cannot be empty")
        logger.info(f"Updating knowledge base: {kb_id}")
        try:
            kb = await self.get_knowledge_base_by_id(kb_id)
            if not kb:
                logger.warning(f"Knowledge base not found: {kb_id}")
                raise ResourceNotFoundError("Knowledge Base", kb_id)
        except ValidationError:
            raise
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting knowledge base: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")
        logger.info(f"Updating knowledge base: {kb['name']} (kb_id={kb_id})")
        try:
            search_result = await self.search_service.search(
                kb['query'],
                max_results=100,
                user_id=kb['user_id']
            )
            if not search_result or not search_result.get('results'):
                logger.warning(f"No results found for query: {kb['query']}")
                raise ResourceNotFoundError("Search Results", kb['query'])
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error searching for articles: {str(e)}")
            raise ExternalServiceError("Search Service", f"Failed to search for articles: {str(e)}")
        articles = search_result['results']
        existing_articles = []
        try:
            with open(kb['file_path'], 'r') as f:
                existing_articles = json.load(f)
            logger.debug(f"Loaded {len(existing_articles)} existing articles from {kb['file_path']}")
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {kb['file_path']}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base file: {str(e)}")
            raise FileError(kb['file_path'], f"Invalid JSON in knowledge base file: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load existing articles: {str(e)}")
            raise FileError(kb['file_path'], f"Failed to load existing articles: {str(e)}")
        merged_articles = existing_articles + articles
        logger.info(f"Merged {len(existing_articles)} existing articles with {len(articles)} new articles")
        try:
            with open(kb['file_path'], 'w') as f:
                json.dump(merged_articles, f)
            logger.info(f"Updated knowledge base file: {kb['file_path']} with {len(merged_articles)} articles")
        except Exception as e:
            logger.error(f"Failed to save updated knowledge base: {str(e)}")
            raise FileError(kb['file_path'], f"Failed to save updated knowledge base: {str(e)}")
        now = datetime.now()
        try:
            if kb['update_schedule'] == "daily":
                next_update = now + timedelta(days=1)
            elif kb['update_schedule'] == "weekly":
                next_update = now + timedelta(weeks=1)
            elif kb['update_schedule'] == "monthly":
                next_update = now + timedelta(days=30)
            else:
                next_update = now + timedelta(weeks=1)  # Default to weekly
        except Exception as e:
            logger.error(f"Error calculating next update time: {str(e)}")
            next_update = now + timedelta(weeks=1)
        kb['last_updated'] = now.isoformat()
        kb['next_update'] = next_update.isoformat()
        cache_key = f"{self.kb_namespace}{kb_id}"
        await enhanced_cache_manager.set(cache_key, kb, data_type="knowledge_base")
        name_cache_key = f"{self.kb_namespace}name:{kb['name']}"
        await enhanced_cache_manager.set(name_cache_key, kb, data_type="knowledge_base")
        list_cache_key = f"{self.kb_namespace}list:all"
        await enhanced_cache_manager.delete(list_cache_key)
        if kb['user_id']:
            user_list_cache_key = f"{self.kb_namespace}list:{kb['user_id']}"
            await enhanced_cache_manager.delete(user_list_cache_key)
        try:
            await self.kb_repository.update_async(
                db,  # This will be handled by the repository
                kb_id=kb_id,
                obj_in={
                    'last_updated': now,
                    'next_update': next_update
                }
            )
            logger.info(f"Knowledge base updated in database: {kb_id}")
        except Exception as e:
            logger.error(f"Error updating knowledge base in database: {str(e)}")
            raise DatabaseError(f"Failed to update knowledge base in database: {str(e)}")
        logger.info(f"Knowledge base updated: {kb_id}")
        return kb
    async def delete_knowledge_base(self, kb_id: str) -> bool:
        if not kb_id or not kb_id.strip():
            raise ValidationError("Knowledge base ID cannot be empty")
        logger.info(f"Deleting knowledge base: {kb_id}")
        try:
            kb = await self.get_knowledge_base_by_id(kb_id)
            if not kb:
                logger.warning(f"Knowledge base not found: {kb_id}")
                raise ResourceNotFoundError("Knowledge Base", kb_id)
        except ValidationError:
            raise
        except ResourceNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Error getting knowledge base: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")
        logger.info(f"Deleting knowledge base: {kb['name']} (kb_id={kb_id})")
        try:
            os.remove(kb['file_path'])
            logger.info(f"Deleted knowledge base file: {kb['file_path']}")
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {kb['file_path']}")
        except Exception as e:
            logger.error(f"Failed to delete knowledge base file: {str(e)}")
            raise FileError(kb['file_path'], f"Failed to delete knowledge base file: {str(e)}")
        cache_key = f"{self.kb_namespace}{kb_id}"
        await enhanced_cache_manager.delete(cache_key)
        name_cache_key = f"{self.kb_namespace}name:{kb['name']}"
        await enhanced_cache_manager.delete(name_cache_key)
        list_cache_key = f"{self.kb_namespace}list:all"
        await enhanced_cache_manager.delete(list_cache_key)
        if kb['user_id']:
            user_list_cache_key = f"{self.kb_namespace}list:{kb['user_id']}"
            await enhanced_cache_manager.delete(user_list_cache_key)
        logger.debug(f"Removed knowledge base from cache: {kb_id}")
        try:
            await self.kb_repository.delete_async(db, kb_id=kb_id)
            logger.info(f"Deleted knowledge base from database: {kb_id}")
        except Exception as e:
            logger.error(f"Error deleting knowledge base from database: {str(e)}")
            raise DatabaseError(f"Failed to delete knowledge base from database: {str(e)}")
        logger.info(f"Knowledge base deleted: {kb_id}")
        return True