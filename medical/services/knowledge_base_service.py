"""Knowledge base service for the Medical Research Synthesizer.

This module provides a service for creating and managing knowledge bases.
"""
import os
import json
import logging
import uuid
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from ..core.cache import enhanced_cache_manager, enhanced_cached
from ..core.progress_tracker import ProgressTracker
from ..core.exceptions import (
    ResourceNotFoundError, ValidationError,
    ExternalServiceError, DatabaseError, FileError, KnowledgeBaseError
)
from ..services.search_service import SearchService
from ..storage.repositories.kb_repository import KnowledgeBaseRepository

logger = logging.getLogger(__name__)

class KnowledgeBaseProgressTracker(ProgressTracker):
    def __init__(self, kb_id: str, total_steps: int = 100):
        super().__init__(operation_id=kb_id, total_steps=total_steps)
        self.kb_id = kb_id
        self.kb_name = "unknown"
        self.operation_type = "unknown"
        self.start_time = time.time()
    def set_kb_name(self, kb_name: str):
        self.kb_name = kb_name
    def set_operation_type(self, operation_type: str):
        self.operation_type = operation_type
    def get_progress_details(self) -> Dict[str, Any]:
        details = super().get_progress_details()
        details.update({
            "kb_id": self.kb_id,
            "kb_name": self.kb_name,
            "operation_type": self.operation_type,
            "elapsed_time": time.time() - self.start_time
        })
        return details
    async def save_progress(self):
        progress_key = f"kb_progress:{self.kb_id}"
        await enhanced_cache_manager.set(
            progress_key,
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )

def validate_kb_input(func):
    async def wrapper(self, *args, **kwargs):
        name = kwargs.get('name', '')
        query = kwargs.get('query', '')
        update_schedule = kwargs.get('update_schedule', 'weekly')
        kb_id = kwargs.get('kb_id', '')
        if 'name' in kwargs and (not name or not name.strip()):
            raise ValidationError("Knowledge base name cannot be empty")
        if 'query' in kwargs and (not query or not query.strip()):
            raise ValidationError("Knowledge base query cannot be empty")
        if 'update_schedule' in kwargs:
            valid_schedules = ["daily", "weekly", "monthly"]
            if update_schedule not in valid_schedules:
                raise ValidationError(f"Invalid update schedule: {update_schedule}. Valid values: {', '.join(valid_schedules)}")
        if 'kb_id' in kwargs and (not kb_id or not kb_id.strip()):
            raise ValidationError("Knowledge base ID cannot be empty")
        return await func(self, *args, **kwargs)
    return wrapper

def track_kb_progress(operation_type: str, total_steps: int = 100):
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            kb_id = kwargs.get('kb_id', '')
            kb_name = kwargs.get('name', '')
            if not kb_id:
                param_str = f"{func_name}:{args}:{kwargs}"
                kb_id = hashlib.md5(param_str.encode()).hexdigest()
            tracker = KnowledgeBaseProgressTracker(kb_id, total_steps)
            tracker.set_operation_type(operation_type)
            if kb_name:
                tracker.set_kb_name(kb_name)
            tracker.update(0, f"Starting {operation_type}")
            await tracker.save_progress()
            kwargs['progress_tracker'] = tracker
            try:
                result = await func(self, *args, **kwargs)
                tracker.complete(f"{operation_type.capitalize()} completed successfully")
                await tracker.save_progress()
                return result
            except Exception as e:
                tracker.fail(f"{operation_type.capitalize()} failed: {str(e)}")
                await tracker.save_progress()
                raise
        return wrapper
    return decorator

def enhanced_kb_error_handling(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            raise
        except ExternalServiceError:
            raise
        except DatabaseError:
            raise
        except ResourceNotFoundError:
            raise
        except KnowledgeBaseError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise KnowledgeBaseError(
                component=f"Knowledge Base Service ({func.__name__})",
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper

def cached_kb_query(ttl: int = 3600, prefix: str = "kb_query", data_type: str = "knowledge_base"):
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(self, *args, **kwargs)
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            cache_key = f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"
            cached_result = await enhanced_cache_manager.get(cache_key, data_type=data_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
            logger.debug(f"Cache miss for {func_name}")
            result = await func(self, *args, **kwargs)
            await enhanced_cache_manager.set(cache_key, result, ttl=ttl, data_type=data_type)
            return result
        return wrapper
    return decorator

async def calculate_kb_statistics(kb_file_path: str) -> Dict[str, Any]:
    if not os.path.exists(kb_file_path):
        raise FileNotFoundError(f"Knowledge base file not found: {kb_file_path}")
    try:
        with open(kb_file_path, 'r') as f:
            kb_data = json.load(f)
        articles = kb_data.get('articles', [])
        stats = {
            "total_articles": len(articles),
            "journals": {},
            "publication_years": {},
            "authors": {},
            "keywords": {}
        }
        for article in articles:
            journal = article.get('journal', 'Unknown')
            stats["journals"][journal] = stats["journals"].get(journal, 0) + 1
            pub_date = article.get('publication_date', '')
            year = pub_date[:4] if pub_date and len(pub_date) >= 4 else 'Unknown'
            stats["publication_years"][year] = stats["publication_years"].get(year, 0) + 1
            authors = article.get('authors', [])
            if isinstance(authors, list):
                for author in authors:
                    stats["authors"][author] = stats["authors"].get(author, 0) + 1
            keywords = article.get('keywords', [])
            if isinstance(keywords, list):
                for keyword in keywords:
                    stats["keywords"][keyword] = stats["keywords"].get(keyword, 0) + 1
        stats["journals"] = dict(sorted(stats["journals"].items(), key=lambda x: x[1], reverse=True)[:10])
        stats["publication_years"] = dict(sorted(stats["publication_years"].items(), key=lambda x: x[1], reverse=True))
        stats["authors"] = dict(sorted(stats["authors"].items(), key=lambda x: x[1], reverse=True)[:20])
        stats["keywords"] = dict(sorted(stats["keywords"].items(), key=lambda x: x[1], reverse=True)[:20])
        return stats
    except Exception as e:
        logger.error(f"Error calculating knowledge base statistics: {str(e)}")
        raise KnowledgeBaseError(
            component="Knowledge Base Statistics",
            message=f"Failed to calculate statistics: {str(e)}"
        )

async def detect_kb_updates_needed() -> List[Dict[str, Any]]:
    return [
        {
            "kb_id": "example-kb-id",
            "name": "Example KB",
            "update_schedule": "daily",
            "last_updated": (datetime.now() - timedelta(days=2)).isoformat(),
            "next_update": (datetime.now() - timedelta(days=1)).isoformat()
        }
    ]

async def merge_knowledge_bases(kb_ids: List[str], new_name: str, user_id: Optional[int] = None) -> Dict[str, Any]:
    return {
        "kb_id": str(uuid.uuid4()),
        "name": new_name,
        "merged_from": kb_ids,
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }

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
                None,  # Database session will be handled by the repository
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
            kb = await self.kb_repository.get_by_name_async(None, name=name)
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
            kb = await self.kb_repository.get_by_kb_id_async(None, kb_id=kb_id)
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
            kbs = await self.kb_repository.list_async(None, user_id=user_id)
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
                None,  # Database session will be handled by the repository
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
            await self.kb_repository.delete_async(None, kb_id=kb_id)
            logger.info(f"Deleted knowledge base from database: {kb_id}")
        except Exception as e:
            logger.error(f"Error deleting knowledge base from database: {str(e)}")
            raise DatabaseError(f"Failed to delete knowledge base from database: {str(e)}")
        logger.info(f"Knowledge base deleted: {kb_id}")
        return True
