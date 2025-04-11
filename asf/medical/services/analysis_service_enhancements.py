"""
Enhanced Analysis Service for the Medical Research Synthesizer.

This module provides enhancements to the Analysis Service, including:
- Better error handling for analysis errors
- Validation of input data
- Caching of analysis results
- Progress tracking for long-running analyses
"""

import asyncio
import logging
import time
import uuid
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union

from asf.medical.core.exceptions import (
    ValidationError, ExternalServiceError, ModelError, 
    DatabaseError, ResourceNotFoundError, AnalysisError
)
from asf.medical.core.cache import cache_manager
from asf.medical.core.progress_tracker import ProgressTracker

# Set up logging
logger = logging.getLogger(__name__)

class AnalysisProgressTracker(ProgressTracker):
    """
    Progress tracker for analysis operations.
    
    This class extends the base ProgressTracker to provide analysis-specific
    progress tracking functionality.
    """
    
    def __init__(self, analysis_id: str, total_steps: int = 100):
        """
        Initialize the analysis progress tracker.
        
        Args:
            analysis_id: Analysis ID
            total_steps: Total number of steps in the analysis
        """
        super().__init__(operation_id=analysis_id, total_steps=total_steps)
        self.analysis_id = analysis_id
        self.analysis_type = "unknown"
        self.start_time = time.time()
        
    def set_analysis_type(self, analysis_type: str):
        """
        Set the analysis type.
        
        Args:
            analysis_type: Type of analysis
        """
        self.analysis_type = analysis_type
        
    def get_progress_details(self) -> Dict[str, Any]:
        """
        Get detailed progress information.
        
        Returns:
            Dictionary with progress details
        """
        details = super().get_progress_details()
        details.update({
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type,
            "elapsed_time": time.time() - self.start_time
        })
        return details
        
    async def save_progress(self):
        """
        Save progress to cache.
        """
        progress_key = f"analysis_progress:{self.analysis_id}"
        await cache_manager.set(
            progress_key, 
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )

def validate_analysis_input(func):
    """
    Decorator for validating analysis input data.
    
    This decorator validates input parameters for analysis methods.
    """
    async def wrapper(self, *args, **kwargs):
        # Extract common parameters
        query = kwargs.get('query', '')
        max_results = kwargs.get('max_results', 20)
        threshold = kwargs.get('threshold', 0.7)
        
        # Validate query
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
            
        # Validate max_results
        if max_results is not None:
            if not isinstance(max_results, int):
                raise ValidationError("max_results must be an integer")
            if max_results < 1:
                raise ValidationError("max_results must be at least 1")
            if max_results > 500:
                raise ValidationError("max_results cannot exceed 500")
                
        # Validate threshold
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValidationError("threshold must be a number")
            if threshold < 0.0 or threshold > 1.0:
                raise ValidationError("threshold must be between 0.0 and 1.0")
                
        # Call the original function
        return await func(self, *args, **kwargs)
    return wrapper

def track_analysis_progress(analysis_type: str, total_steps: int = 100):
    """
    Decorator for tracking analysis progress.
    
    This decorator adds progress tracking to analysis methods.
    
    Args:
        analysis_type: Type of analysis
        total_steps: Total number of steps in the analysis
    """
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Generate a deterministic analysis ID based on the function and parameters
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            analysis_id = hashlib.md5(param_str.encode()).hexdigest()
            
            # Create progress tracker
            tracker = AnalysisProgressTracker(analysis_id, total_steps)
            tracker.set_analysis_type(analysis_type)
            
            # Initialize progress
            tracker.update(0, "Starting analysis")
            await tracker.save_progress()
            
            # Add tracker to kwargs
            kwargs['progress_tracker'] = tracker
            
            try:
                # Call the original function
                result = await func(self, *args, **kwargs)
                
                # Mark as complete
                tracker.complete("Analysis completed successfully")
                await tracker.save_progress()
                
                return result
            except Exception as e:
                # Mark as failed
                tracker.fail(f"Analysis failed: {str(e)}")
                await tracker.save_progress()
                
                # Re-raise the exception
                raise
        return wrapper
    return decorator

def enhanced_error_handling(func):
    """
    Decorator for enhanced error handling in analysis methods.
    
    This decorator adds detailed error handling to analysis methods.
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
        except ModelError:
            # Re-raise model errors
            raise
        except DatabaseError:
            # Re-raise database errors
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except AnalysisError:
            # Re-raise analysis errors
            raise
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Convert to AnalysisError
            raise AnalysisError(
                component=f"Analysis Service ({func.__name__})",
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper

def cached_analysis(ttl: int = 3600, prefix: str = "analysis", data_type: str = "analysis"):
    """
    Decorator for caching analysis results.
    
    This decorator adds caching to analysis methods.
    
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "analysis")
        data_type: Type of data being cached (default: "analysis")
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
class EnhancedAnalysisService:
    @validate_analysis_input
    @track_analysis_progress("contradiction_analysis", total_steps=5)
    @enhanced_error_handling
    @cached_analysis(ttl=3600, prefix="analyze_contradictions", data_type="analysis")
    async def analyze_contradictions(
        self,
        query: str,
        max_results: int = 20,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        user_id: Optional[int] = None,
        progress_tracker: Optional[AnalysisProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Validating input parameters")
            await progress_tracker.save_progress()
            
        # Search for articles
        if progress_tracker:
            progress_tracker.update(2, "Searching for articles")
            await progress_tracker.save_progress()
            
        search_result = await self.search_service.search(query, max_results, user_id)
        
        # Extract articles
        if progress_tracker:
            progress_tracker.update(3, "Processing search results")
            await progress_tracker.save_progress()
            
        articles = search_result['results']
        
        # Detect contradictions
        if progress_tracker:
            progress_tracker.update(4, "Detecting contradictions")
            await progress_tracker.save_progress()
            
        contradictions = await self.contradiction_service.detect_contradictions_in_articles(
            articles,
            threshold=threshold,
            use_all_methods=use_biomedlm or use_tsmixer or use_lorentz
        )
        
        # Create analysis result
        if progress_tracker:
            progress_tracker.update(5, "Finalizing analysis")
            await progress_tracker.save_progress()
            
        # ... rest of the implementation ...
        
        return analysis_result
"""
