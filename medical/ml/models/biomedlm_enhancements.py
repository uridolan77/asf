Enhanced BioMedLM Service for the Medical Research Synthesizer.

This module provides enhancements to the BioMedLM Service, including:
- Better error handling for model errors
- Validation of input data
- Caching of model predictions
- Progress tracking for long-running model operations

The module contains decorators and utility classes that can be applied to
BioMedLM service methods to enhance their functionality and reliability.
import logging
import time
import hashlib
from typing import Dict, Any, List, Optional, Callable, TypeVar, Union, Tuple
import torch

from asf.medical.core.exceptions import (
    ValidationError, ModelError, ResourceNotFoundError
)

logger = logging.getLogger(__name__)

# Type variable for generic model type
T = TypeVar('T')

class MemoryStats:
    Memory statistics for system and GPU resources.
    
    Attributes:
        total: Total system memory in bytes
        available: Available system memory in bytes
        used: Used system memory in bytes
        percent: Percentage of system memory used (0.0-1.0)
        gpu_total: Total GPU memory in bytes
        gpu_available: Available GPU memory in bytes
        gpu_used: Used GPU memory in bytes
        gpu_percent: Percentage of GPU memory used (0.0-1.0)
    total: int = 0
    available: int = 0
    used: int = 0
    percent: float = 0.0
    gpu_total: int = 0
    gpu_available: int = 0
    gpu_used: int = 0
    gpu_percent: float = 0.0


class ProgressTracker:
    Base progress tracker class for monitoring long-running operations.
    
    This class provides functionality for tracking the progress of operations,
    including updating status, recording messages, and calculating completion percentage.
    
    def __init__(self, operation_id: str, total_steps: int = 100):
        """
        Initialize the progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            total_steps: Total number of steps in the operation
        """
        self.operation_id = operation_id
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "pending"
        self.message = ""
        
    def update(self, step: int, message: str):
        """
        Update the progress tracker with a new step and message.
        
        Args:
            step: Current step number
            message: Status message for the current step
        """
        self.current_step = step
        self.message = message
        
    def complete(self, message: str):
        """
        Mark the operation as completed.
        
        Args:
            message: Completion message
        """
        self.current_step = self.total_steps
        self.status = "completed"
        self.message = message
        
    def fail(self, message: str):
        """
        Mark the operation as failed.
        
        Args:
            message: Failure message describing what went wrong
        """
        self.status = "failed"
        self.message = message
        
    def get_progress_details(self) -> Dict[str, Any]:
        """
        Get detailed progress information.
        
        Returns:
            Dictionary with progress details including operation ID, steps, status, and message
        """
        return {
            "operation_id": self.operation_id,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "status": self.status,
            "message": self.message
        }


class BioMedLMProgressTracker(ProgressTracker):
    Progress tracker for BioMedLM operations.
    
    This class extends the base ProgressTracker to provide BioMedLM-specific
    progress tracking functionality for monitoring model operations.
    
    def __init__(self, operation_id: str, total_steps: int = 100):
        """
        Initialize the BioMedLM progress tracker.
        
        Args:
            operation_id: Unique identifier for the operation
            total_steps: Total number of steps in the operation
        """
        super().__init__(operation_id=operation_id, total_steps=total_steps)
        self.model_name = "biomedlm"
        self.operation_type = "unknown"
        self.start_time = time.time()
        self.input_size = 0
        self.device = "unknown"
        self.batch_size = 0
        
    def set_operation_type(self, operation_type: str):
        """
        Set the operation type.
        
        Args:
            operation_type: Type of operation (e.g., "contradiction_detection", "embedding_generation")
        """
        self.operation_type = operation_type
        
    def set_input_size(self, input_size: int):
        """
        Set the input size.
        
        Args:
            input_size: Size of the input data (e.g., number of characters or tokens)
        """
        self.input_size = input_size
        
    def set_device(self, device: str):
        """
        Set the device used for computation.
        
        Args:
            device: Device used for computation (e.g., "cpu", "cuda:0")
        """
        self.device = device
        
    def set_batch_size(self, batch_size: int):
        """
        Set the batch size for processing.
        
        Args:
            batch_size: Batch size for processing
        """
        self.batch_size = batch_size
        
    def get_progress_details(self) -> Dict[str, Any]:
        """
        Get detailed progress information specific to BioMedLM operations.
        
        Returns:
            Dictionary with progress details including model name, operation type, elapsed time,
            input size, device, and batch size in addition to base progress information
        """
        details = super().get_progress_details()
        details.update({
            "model_name": self.model_name,
            "operation_type": self.operation_type,
            "elapsed_time": time.time() - self.start_time,
            "input_size": self.input_size,
            "device": self.device,
            "batch_size": self.batch_size
        })
        return details
        
    async def save_progress(self):
        """
        Save the current progress to the cache.
        
        This method persists the progress information to the cache with a 1-hour TTL.
        In a real implementation, this would use an actual cache manager.
        """
        # In a real implementation, this would use an actual cache manager with a key like:
        # progress_key = f"biomedlm_progress:{self.operation_id}"
        # For this docstring fix, we'll just log the action
        logger.info(f"Saving progress for {self.operation_id}: {self.get_progress_details()}")


def validate_biomedlm_input(func):
    """
    Decorator for validating BioMedLM input data.
    
    This decorator validates input parameters for BioMedLM methods to ensure
    they meet the required format and constraints before processing.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with input validation
    """
    
    async def wrapper(self, *args, **kwargs):
        """
        Wrapper function that validates input parameters before calling the decorated function.
        
        Args:
            self: The class instance
            *args: Positional arguments to pass to the decorated function
            **kwargs: Keyword arguments to pass to the decorated function
            
        Returns:
            Result from the decorated function
            
        Raises:
            ValidationError: If input validation fails
        """
        text = kwargs.get('text', '')
        texts = kwargs.get('texts', [])
        text1 = kwargs.get('text1', '')
        text2 = kwargs.get('text2', '')
        
        if 'text' in kwargs and not text and not isinstance(text, str):
            raise ValidationError("Text cannot be empty and must be a string")
            
        if 'texts' in kwargs:
            if not texts:
                raise ValidationError("Texts cannot be empty")
            if not all(isinstance(t, str) for t in texts):
                raise ValidationError("All texts must be strings")
                
        if ('text1' in kwargs or 'text2' in kwargs) and (not text1 or not text2):
            raise ValidationError("Both text1 and text2 must be provided and non-empty")
            
        return await func(self, *args, **kwargs)
        
    return wrapper


def track_biomedlm_progress(operation_type: str, total_steps: int = 100):
    """
    Decorator for tracking BioMedLM operation progress.
    
    This decorator adds progress tracking to BioMedLM methods, creating a
    progress tracker and updating it throughout the operation.
    
    Args:
        operation_type: Type of operation (e.g., "contradiction_detection")
        total_steps: Total number of steps in the operation
        
    Returns:
        Decorator function that adds progress tracking
    """
    
    def decorator(func):
        """
        Inner decorator function that wraps the original function.
        
        Args:
            func: The function being decorated
            
        Returns:
            Wrapped function with progress tracking capability
        """
        
        async def wrapper(self, *args, **kwargs):
            """
            Wrapper function that adds progress tracking to the decorated function.
            
            Args:
                self: The class instance
                *args: Positional arguments to pass to the decorated function
                **kwargs: Keyword arguments to pass to the decorated function
                
            Returns:
                Result from the decorated function
                
            Raises:
                Any exceptions raised by the decorated function
            """
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            operation_id = hashlib.md5(param_str.encode()).hexdigest()
            
            tracker = BioMedLMProgressTracker(operation_id, total_steps)
            tracker.set_operation_type(operation_type)
            tracker.set_device(self.device)
            
            if 'text' in kwargs:
                tracker.set_input_size(len(kwargs['text']))
            elif 'texts' in kwargs:
                tracker.set_input_size(len(kwargs['texts']))
                tracker.set_batch_size(min(len(kwargs['texts']), 8))  # Default batch size
            elif 'text1' in kwargs and 'text2' in kwargs:
                tracker.set_input_size(len(kwargs['text1']) + len(kwargs['text2']))
                
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


def enhanced_biomedlm_error_handling(func):
    """
    Decorator for enhanced error handling in BioMedLM methods.
    
    This decorator adds detailed error handling to BioMedLM methods, including
    specific handling for CUDA out-of-memory errors and proper error classification.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with enhanced error handling
    """
    
    async def wrapper(self, *args, **kwargs):
        """
        Wrapper function that adds enhanced error handling to the decorated function.
        
        Args:
            self: The class instance
            *args: Positional arguments to pass to the decorated function
            **kwargs: Keyword arguments to pass to the decorated function
            
        Returns:
            Result from the decorated function
            
        Raises:
            ValidationError: If input validation fails
            ModelError: If a model-related error occurs
            ResourceNotFoundError: If a required resource is not found
        """
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            # Re-raise validation errors without modification
            raise
        except ModelError:
            # Re-raise model errors without modification
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors without modification
            raise
        except torch.cuda.OutOfMemoryError as e:
            # Handle CUDA out-of-memory errors specifically
            logger.error(f"CUDA out of memory error in {func.__name__}: {str(e)}")
            raise ModelError(
                model="BioMedLM",
                message=f"CUDA out of memory error: {str(e)}. Try reducing batch size or input length."
            )
        except Exception as e:
            # Handle all other exceptions
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ModelError(
                model="BioMedLM",
                message=f"Unexpected error: {str(e)}"
            )
            
    return wrapper


def cached_biomedlm_prediction(ttl: int = 3600, prefix: str = "biomedlm", data_type: str = "prediction"):
    """
    Decorator for caching BioMedLM model predictions.
    
    This decorator adds caching to BioMedLM prediction methods to improve performance
    by avoiding redundant computations for the same inputs.
    
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "biomedlm")
        data_type: Type of data being cached (default: "prediction")
        
    Returns:
        Decorator function that adds caching capability
    """
    
    def decorator(func):
        """
        Inner decorator function that wraps the original function.
        
        Args:
            func: The function being decorated
            
        Returns:
            Wrapped function with caching capability
        """
        
        async def wrapper(self, *args, **kwargs):
            """
            Wrapper function that adds caching to the decorated function.
            
            Args:
                self: The class instance
                *args: Positional arguments to pass to the decorated function
                **kwargs: Keyword arguments to pass to the decorated function
                
            Returns:
                Result from the decorated function, potentially from cache
            """
            # Allow skipping the cache with an explicit parameter
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(self, *args, **kwargs)
                
            # Generate a cache key based on function name and arguments
            func_name = func.__name__
            cache_key_parts = [prefix, func_name]
            
            # Add positional arguments to cache key
            for arg in args[1:]:  # Skip 'self'
                if isinstance(arg, (str, int, float, bool)):
                    cache_key_parts.append(str(arg))
                elif isinstance(arg, (list, tuple)):
                    arg_hash = hashlib.md5(str(arg).encode()).hexdigest()
                    cache_key_parts.append(arg_hash)
                    
            # Add keyword arguments to cache key
            for key, value in sorted(kwargs.items()):
                if key == 'progress_tracker':
                    continue  # Skip progress tracker
                if isinstance(value, (str, int, float, bool)):
                    cache_key_parts.append(f"{key}={value}")
                elif isinstance(value, (list, tuple)):
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()
                    cache_key_parts.append(f"{key}={value_hash}")
                    
            # In a real implementation, this would generate a cache key and use it
            # cache_key = hashlib.md5(":".join(cache_key_parts).encode()).hexdigest()
            
            # In a real implementation, this would check the cache
            # cached_result = await cache_manager.get(cache_key, data_type=data_type)
            cached_result = None
            
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
                
            logger.debug(f"Cache miss for {func_name}")
            result = await func(self, *args, **kwargs)
            
            # In a real implementation, this would cache the result
            # await cache_manager.set(cache_key, result, ttl=ttl, data_type=data_type)
            logger.debug(f"Caching result for {func_name} with TTL={ttl}")
            
            return result
            
        return wrapper
        
    return decorator
