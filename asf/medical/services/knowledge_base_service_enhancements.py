"""
Enhanced Knowledge Base Service for the Medical Research Synthesizer.
This module provides enhancements to the Knowledge Base Service, including:
- Better error handling for knowledge base operations
- Validation of input data
- Caching of knowledge base queries
- Progress tracking for long-running operations
"""
import os
import json
import logging
import time
import hashlib
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..core.exceptions import (
    ValidationError, ExternalServiceError, DatabaseError,
    ResourceNotFoundError, KnowledgeBaseError
)
from ..core.cache import enhanced_cache_manager, enhanced_cached
from ..core.progress_tracker import ProgressTracker
logger = logging.getLogger(__name__)
class KnowledgeBaseProgressTracker(ProgressTracker):
    """Progress tracker for knowledge base operations.

    This class extends the base ProgressTracker to provide knowledge base-specific
    progress tracking functionality.
    """
    def __init__(self, kb_id: str, total_steps: int = 100):
        """
        Initialize the knowledge base progress tracker.
        Args:
            kb_id: Knowledge base ID
            total_steps: Total number of steps in the operation
        """
        super().__init__(operation_id=kb_id, total_steps=total_steps)
        self.kb_id = kb_id
        self.kb_name = "unknown"
        self.operation_type = "unknown"
        self.start_time = time.time()
    def set_kb_name(self, kb_name: str):
        """
        Set the knowledge base name.
        Args:
            kb_name: Name of the knowledge base
        """
        self.kb_name = kb_name
    def set_operation_type(self, operation_type: str):
        """
        Set the operation type.
        Args:
            operation_type: Type of operation
        """
        self.operation_type = operation_type
    def get_progress_details(self) -> Dict[str, Any]:
        """
        Get detailed progress information.
        Returns:
            Dictionary with progress details
        """
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
    """
    Decorator for validating knowledge base input data.
    This decorator validates input parameters for knowledge base methods.
    """
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
    """
    Decorator for tracking knowledge base operation progress.
    This decorator adds progress tracking to knowledge base methods.
    Args:
        operation_type: Type of operation
        total_steps: Total number of steps in the operation
    """
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
    """
    Decorator for enhanced error handling in knowledge base methods.
    This decorator adds detailed error handling to knowledge base methods.
    """
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
    """
    Decorator for caching knowledge base query results.
    This decorator adds caching to knowledge base query methods.
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "kb_query")
        data_type: Type of data being cached (default: "knowledge_base")
    """
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