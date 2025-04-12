"""Enhanced ML Services for the Medical Research Synthesizer.

This module provides enhancements to the ML Services, including:
- Better error handling for ML model errors
- Validation of input data
- Caching of ML model predictions
- Progress tracking for long-running ML operations
"""

import logging
import time
import hashlib
from asf.medical.core.exceptions import (
    ValidationError, ModelError, ResourceNotFoundError
)
from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached
from asf.medical.core.progress_tracker import ProgressTracker
logger = logging.getLogger(__name__)
class MLProgressTracker(ProgressTracker):
    """Progress tracker for ML operations.
    
    This class extends the base ProgressTracker to provide ML-specific
    progress tracking functionality.
    """
    def __init__(self, operation_id: str, total_steps: int = 100):
        """
        Initialize the ML progress tracker.
        Args:
            operation_id: Operation ID
            total_steps: Total number of steps in the operation
        """
        super().__init__(operation_id=operation_id, total_steps=total_steps)
        self.model_name = "unknown"
        self.operation_type = "unknown"
        self.start_time = time.time()
        self.input_size = 0
    def set_model_name(self, model_name: str):
        """
        Set the model name.
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
    def set_operation_type(self, operation_type: str):
        """
        Set the operation type.
        Args:
            operation_type: Type of operation
        """
        self.operation_type = operation_type
    def set_input_size(self, input_size: int):
        """
        Set the input size.
        Args:
            input_size: Size of the input data
        """
        self.input_size = input_size
    def get_progress_details(self) -> dict:
        """
        Get detailed progress information.

        Returns:
            Dictionary with progress details
        """
        details = super().get_progress_details()
        details.update({
            "model_name": self.model_name,
            "operation_type": self.operation_type,
            "elapsed_time": time.time() - self.start_time,
            "input_size": self.input_size
        })
        return details
    async def save_progress(self):
        """
        Save progress information to cache.

        This method saves the current progress information to the cache
        for later retrieval.
        """
        progress_key = f"ml_progress:{self.operation_id}"
        await enhanced_cache_manager.set(
            progress_key,
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )
def validate_ml_input(func):
    """
    Decorator for validating ML input data.

    This decorator validates input parameters for ML methods to ensure
    they meet the required criteria before processing.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with input validation
    """
    async def wrapper(self, *args, **kwargs):
        text = kwargs.get('text', '')
        texts = kwargs.get('texts', [])
        claim1 = kwargs.get('claim1', '')
        claim2 = kwargs.get('claim2', '')
        articles = kwargs.get('articles', [])
        claims = kwargs.get('claims', [])
        threshold = kwargs.get('threshold', 0.5)
        if 'text' in kwargs and not text and not isinstance(text, str):
            raise ValidationError("Text cannot be empty and must be a string")
        if 'texts' in kwargs:
            if not texts:
                raise ValidationError("Texts cannot be empty")
            if not all(isinstance(t, str) for t in texts):
                raise ValidationError("All texts must be strings")
        if ('claim1' in kwargs or 'claim2' in kwargs) and (not claim1 or not claim2):
            raise ValidationError("Both claim1 and claim2 must be provided and non-empty")
        if 'articles' in kwargs and not articles:
            raise ValidationError("Articles cannot be empty")
        if 'claims' in kwargs and not claims:
            raise ValidationError("Claims cannot be empty")
        if 'threshold' in kwargs:
            if not isinstance(threshold, (int, float)):
                raise ValidationError("Threshold must be a number")
            if threshold < 0.0 or threshold > 1.0:
                raise ValidationError("Threshold must be between 0.0 and 1.0")
        return await func(self, *args, **kwargs)
    return wrapper
def track_ml_progress(model_name: str, operation_type: str, total_steps: int = 100):
    """
    Decorator for tracking ML operation progress.
    This decorator adds progress tracking to ML methods.
    Args:
        model_name: Name of the model
        operation_type: Type of operation
        total_steps: Total number of steps in the operation
    """
    def decorator(func):
        """
        Inner decorator function.

        Args:
            func: The function to decorate

        Returns:
            Decorated function with progress tracking
        """
        async def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            operation_id = hashlib.md5(param_str.encode()).hexdigest()
            tracker = MLProgressTracker(operation_id, total_steps)
            tracker.set_model_name(model_name)
            tracker.set_operation_type(operation_type)
            if 'text' in kwargs:
                tracker.set_input_size(len(kwargs['text']))
            elif 'texts' in kwargs:
                tracker.set_input_size(len(kwargs['texts']))
            elif 'articles' in kwargs:
                tracker.set_input_size(len(kwargs['articles']))
            elif 'claims' in kwargs:
                tracker.set_input_size(len(kwargs['claims']))
            tracker.update(0, f"Starting {operation_type}")
            await tracker.save_progress()
            kwargs['progress_tracker'] = tracker
            try:
                result = await func(self, *args, **kwargs)
                tracker.complete(f"{operation_type.capitalize()} completed successfully")
                await tracker.save_progress()
                return result
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                logger.error(f"API error: {str(e)}")
                raise APIError(f"API call failed: {str(e)}")
                tracker.fail(f"{operation_type.capitalize()} failed: {str(e)}")
                await tracker.save_progress()
                raise
        return wrapper
    return decorator
def enhanced_ml_error_handling(func):
    """
    Decorator for enhanced error handling in ML methods.

    This decorator adds detailed error handling to ML methods, catching
    and properly handling different types of exceptions.

    Args:
        func: The function to decorate

    Returns:
        Decorated function with enhanced error handling
    """
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            logger.error(f"Error: {str(e)}")
            raise
        except ModelError:
            logger.error(f"Error: {str(e)}")
            raise
        except ResourceNotFoundError:
            logger.error(f"Error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ModelError(
                model=getattr(self, 'model_name', func.__name__),
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper
def cached_ml_prediction(ttl: int = 3600, prefix: str = "ml_prediction", data_type: str = "prediction"):
    """
    Decorator for caching ML model predictions.
    This decorator adds caching to ML prediction methods.
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "ml_prediction")
        data_type: Type of data being cached (default: "prediction")
    """
    def decorator(func):
        """
        Inner decorator function.

        Args:
            func: The function to decorate

        Returns:
            Decorated function with caching
        """
        async def wrapper(self, *args, **kwargs):
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(self, *args, **kwargs)
            func_name = func.__name__
            cache_key_parts = [prefix, func_name]
            model_name = getattr(self, 'model_name', None)
            if model_name:
                cache_key_parts.append(model_name)
            for arg in args[1:]:
                if isinstance(arg, (str, int, float, bool)):
                    cache_key_parts.append(str(arg))
                elif isinstance(arg, (list, tuple)):
                    arg_hash = hashlib.md5(str(arg).encode()).hexdigest()
                    cache_key_parts.append(arg_hash)
                elif isinstance(arg, dict):
                    arg_hash = hashlib.md5(str(sorted(arg.items())).encode()).hexdigest()
                    cache_key_parts.append(arg_hash)
            for key, value in sorted(kwargs.items()):
                if isinstance(value, (str, int, float, bool)):
                    cache_key_parts.append(f"{key}={value}")
                elif isinstance(value, (list, tuple)):
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()
                    cache_key_parts.append(f"{key}={value_hash}")
                elif isinstance(value, dict):
                    value_hash = hashlib.md5(str(sorted(value.items())).encode()).hexdigest()
                    cache_key_parts.append(f"{key}={value_hash}")
            cache_key = hashlib.md5(":".join(cache_key_parts).encode()).hexdigest()
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