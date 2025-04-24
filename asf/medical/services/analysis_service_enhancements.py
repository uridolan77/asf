"""Enhanced Analysis Service for the Medical Research Synthesizer.

This module provides enhancements to the Analysis Service, including:
- Better error handling for analysis errors
- Validation of input data
- Caching of analysis results
- Progress tracking for long-running analyses
"""
import logging
import time
import hashlib
from ..core.exceptions import (
    ValidationError, ExternalServiceError, ModelError,
    DatabaseError, ResourceNotFoundError, AnalysisError
)
from typing import Dict, Any, Callable
from ..core.enhanced_cache import enhanced_cache_manager, enhanced_cached
from ..core.progress_tracker import ProgressTracker
logger = logging.getLogger(__name__)
class AnalysisProgressTracker(ProgressTracker):
    """Progress tracker for analysis operations.

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
        progress_key = f"analysis_progress:{self.analysis_id}"
        await enhanced_cache_manager.set(
            progress_key,
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )
def validate_analysis_input(func):
    """Decorator for validating analysis input data.

    This decorator validates input parameters for analysis methods.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    async def wrapper(self, *args, **kwargs):
        query = kwargs.get('query', '')
        max_results = kwargs.get('max_results', 20)
        threshold = kwargs.get('threshold', 0.7)
        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")
        if max_results is not None:
            if not isinstance(max_results, int):
                raise ValidationError("max_results must be an integer")
            if max_results < 1:
                raise ValidationError("max_results must be at least 1")
            if max_results > 500:
                raise ValidationError("max_results cannot exceed 500")
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise ValidationError("threshold must be a number")
            if threshold < 0.0 or threshold > 1.0:
                raise ValidationError("threshold must be between 0.0 and 1.0")
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
        """
        decorator function.

        This function provides functionality for...
        Args:
            func: Description of func
        """
        async def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            analysis_id = hashlib.md5(param_str.encode()).hexdigest()
            tracker = AnalysisProgressTracker(analysis_id, total_steps)
            tracker.set_analysis_type(analysis_type)
            tracker.update(0, "Starting analysis")
            await tracker.save_progress()
            kwargs['progress_tracker'] = tracker
            try:
                result = await func(self, *args, **kwargs)
                tracker.complete("Analysis completed successfully")
                await tracker.save_progress()
                return result
            except Exception as e:
                tracker.fail(f"Analysis failed: {str(e)}")
                await tracker.save_progress()
                raise
        return wrapper
    return decorator
def enhanced_error_handling(func):
    """Decorator for enhanced error handling in analysis methods.

    This decorator adds detailed error handling to analysis methods.

    Args:
        func: The function to decorate

    Returns:
        The decorated function
    """

    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            raise
        except ExternalServiceError:
            raise
        except ModelError:
            raise
        except DatabaseError:
            raise
        except ResourceNotFoundError:
            raise
        except AnalysisError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
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
        """
        decorator function.

        This function provides functionality for...
        Args:
            func: Description of func
        """
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