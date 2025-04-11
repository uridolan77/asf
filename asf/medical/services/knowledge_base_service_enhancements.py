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
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta

from asf.medical.core.exceptions import (
    ValidationError, ExternalServiceError, DatabaseError, 
    ResourceNotFoundError, KnowledgeBaseError
)
from asf.medical.core.cache import cache_manager
from asf.medical.core.progress_tracker import ProgressTracker

# Set up logging
logger = logging.getLogger(__name__)

class KnowledgeBaseProgressTracker(ProgressTracker):
    """
    Progress tracker for knowledge base operations.
    
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
        """
        Save progress to cache.
        """
        progress_key = f"kb_progress:{self.kb_id}"
        await cache_manager.set(
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
        # Extract common parameters
        name = kwargs.get('name', '')
        query = kwargs.get('query', '')
        update_schedule = kwargs.get('update_schedule', 'weekly')
        kb_id = kwargs.get('kb_id', '')
        
        # Validate name if present
        if 'name' in kwargs and (not name or not name.strip()):
            raise ValidationError("Knowledge base name cannot be empty")
            
        # Validate query if present
        if 'query' in kwargs and (not query or not query.strip()):
            raise ValidationError("Knowledge base query cannot be empty")
            
        # Validate update_schedule if present
        if 'update_schedule' in kwargs:
            valid_schedules = ["daily", "weekly", "monthly"]
            if update_schedule not in valid_schedules:
                raise ValidationError(f"Invalid update schedule: {update_schedule}. Valid values: {', '.join(valid_schedules)}")
                
        # Validate kb_id if present
        if 'kb_id' in kwargs and (not kb_id or not kb_id.strip()):
            raise ValidationError("Knowledge base ID cannot be empty")
                
        # Call the original function
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
            # Generate a deterministic KB ID based on the function and parameters
            func_name = func.__name__
            
            # Use provided kb_id or name if available
            kb_id = kwargs.get('kb_id', '')
            kb_name = kwargs.get('name', '')
            
            if not kb_id:
                # Generate ID from parameters
                param_str = f"{func_name}:{args}:{kwargs}"
                kb_id = hashlib.md5(param_str.encode()).hexdigest()
            
            # Create progress tracker
            tracker = KnowledgeBaseProgressTracker(kb_id, total_steps)
            tracker.set_operation_type(operation_type)
            
            if kb_name:
                tracker.set_kb_name(kb_name)
            
            # Initialize progress
            tracker.update(0, f"Starting {operation_type}")
            await tracker.save_progress()
            
            # Add tracker to kwargs
            kwargs['progress_tracker'] = tracker
            
            try:
                # Call the original function
                result = await func(self, *args, **kwargs)
                
                # Mark as complete
                tracker.complete(f"{operation_type.capitalize()} completed successfully")
                await tracker.save_progress()
                
                return result
            except Exception as e:
                # Mark as failed
                tracker.fail(f"{operation_type.capitalize()} failed: {str(e)}")
                await tracker.save_progress()
                
                # Re-raise the exception
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
            # Call the original function
            return await func(self, *args, **kwargs)
        except ValidationError:
            # Re-raise validation errors
            raise
        except ExternalServiceError:
            # Re-raise external service errors
            raise
        except DatabaseError:
            # Re-raise database errors
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except KnowledgeBaseError:
            # Re-raise knowledge base errors
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Convert to KnowledgeBaseError
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
            # Skip caching if explicitly requested
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(self, *args, **kwargs)
                
            # Generate cache key
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            cache_key = f"{prefix}:{hashlib.md5(param_str.encode()).hexdigest()}"
            
            # Try to get from cache
            cached_result = await cache_manager.get(cache_key, data_type=data_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
                
            # Call the original function
            logger.debug(f"Cache miss for {func_name}")
            result = await func(self, *args, **kwargs)
            
            # Store in cache
            await cache_manager.set(cache_key, result, ttl=ttl, data_type=data_type)
            
            return result
        return wrapper
    return decorator

# Example usage:
"""
class EnhancedKnowledgeBaseService:
    @validate_kb_input
    @track_kb_progress("knowledge base creation", total_steps=5)
    @enhanced_kb_error_handling
    async def create_knowledge_base(
        self,
        name: str,
        query: str,
        update_schedule: str = "weekly",
        user_id: Optional[int] = None,
        progress_tracker: Optional[KnowledgeBaseProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Validating input parameters")
            await progress_tracker.save_progress()
            
        # Generate KB ID
        kb_id = str(uuid.uuid4())
        
        # Update tracker with KB ID and name
        if progress_tracker:
            progress_tracker.kb_id = kb_id
            progress_tracker.set_kb_name(name)
            await progress_tracker.save_progress()
            
        # Search for articles
        if progress_tracker:
            progress_tracker.update(2, "Searching for articles")
            await progress_tracker.save_progress()
            
        search_result = await self.search_service.search(query, max_results=100, user_id=user_id)
        
        # Extract articles
        if progress_tracker:
            progress_tracker.update(3, "Processing search results")
            await progress_tracker.save_progress()
            
        articles = search_result['results']
        
        # Create knowledge base file
        if progress_tracker:
            progress_tracker.update(4, "Creating knowledge base file")
            await progress_tracker.save_progress()
            
        # ... file creation logic ...
        
        # Store in database
        if progress_tracker:
            progress_tracker.update(5, "Storing knowledge base in database")
            await progress_tracker.save_progress()
            
        # ... database storage logic ...
        
        return kb
        
    @validate_kb_input
    @cached_kb_query(ttl=3600, prefix="kb_query", data_type="knowledge_base")
    @enhanced_kb_error_handling
    async def query_knowledge_base(
        self,
        kb_id: str,
        query: str,
        max_results: int = 20
    ) -> Dict[str, Any]:
        # Implementation with caching
        # ... query logic ...
        return results
"""

# Additional knowledge base utility functions:

async def calculate_kb_statistics(kb_file_path: str) -> Dict[str, Any]:
    """
    Calculate statistics for a knowledge base.
    
    Args:
        kb_file_path: Path to the knowledge base file
        
    Returns:
        Dictionary with statistics
    """
    if not os.path.exists(kb_file_path):
        raise FileNotFoundError(f"Knowledge base file not found: {kb_file_path}")
        
    try:
        # Load knowledge base data
        with open(kb_file_path, 'r') as f:
            kb_data = json.load(f)
            
        # Extract articles
        articles = kb_data.get('articles', [])
        
        # Calculate statistics
        stats = {
            "total_articles": len(articles),
            "journals": {},
            "publication_years": {},
            "authors": {},
            "keywords": {}
        }
        
        # Process articles
        for article in articles:
            # Count journals
            journal = article.get('journal', 'Unknown')
            stats["journals"][journal] = stats["journals"].get(journal, 0) + 1
            
            # Count publication years
            pub_date = article.get('publication_date', '')
            year = pub_date[:4] if pub_date and len(pub_date) >= 4 else 'Unknown'
            stats["publication_years"][year] = stats["publication_years"].get(year, 0) + 1
            
            # Count authors
            authors = article.get('authors', [])
            if isinstance(authors, list):
                for author in authors:
                    stats["authors"][author] = stats["authors"].get(author, 0) + 1
            
            # Count keywords
            keywords = article.get('keywords', [])
            if isinstance(keywords, list):
                for keyword in keywords:
                    stats["keywords"][keyword] = stats["keywords"].get(keyword, 0) + 1
                    
        # Sort statistics
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
    """
    Detect knowledge bases that need updates.
    
    Returns:
        List of knowledge bases that need updates
    """
    # This would typically query the database for knowledge bases
    # that are due for updates based on their update_schedule and next_update
    # For this example, we'll just return a placeholder
    
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
    """
    Merge multiple knowledge bases into a new one.
    
    Args:
        kb_ids: List of knowledge base IDs to merge
        new_name: Name for the merged knowledge base
        user_id: User ID
        
    Returns:
        Merged knowledge base
    """
    # This would typically:
    # 1. Load each knowledge base
    # 2. Merge their articles (removing duplicates)
    # 3. Create a new knowledge base with the merged articles
    # For this example, we'll just return a placeholder
    
    return {
        "kb_id": str(uuid.uuid4()),
        "name": new_name,
        "merged_from": kb_ids,
        "user_id": user_id,
        "created_at": datetime.now().isoformat()
    }
