Enhanced BioMedLM Service for the Medical Research Synthesizer.
This module provides enhancements to the BioMedLM Service, including:
- Better error handling for model errors
- Validation of input data
- Caching of model predictions
- Progress tracking for long-running model operations
import logging
import time
import hashlib
import torch
from asf.medical.core.exceptions import (
    ValidationError, ModelError, ResourceNotFoundError
)
from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached
from asf.medical.core.progress_tracker import ProgressTracker
from asf.medical.ml.services.ml_service_enhancements import (
    validate_ml_input, track_ml_progress, enhanced_ml_error_handling, cached_ml_prediction
)
logger = logging.getLogger(__name__)
class BioMedLMProgressTracker(ProgressTracker):
    Progress tracker for BioMedLM operations.
    This class extends the base ProgressTracker to provide BioMedLM-specific
    progress tracking functionality.
    def __init__(self, operation_id: str, total_steps: int = 100):
        """
        Initialize the BioMedLM progress tracker.
        Args:
            operation_id: Operation ID
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
    def set_device(self, device: str):
        """
        Set the device.
        Args:
            device: Device used for computation
        """
        self.device = device
    def set_batch_size(self, batch_size: int):
        """
        Set the batch size.
        Args:
            batch_size: Batch size for processing
        """
        self.batch_size = batch_size
    def get_progress_details(self) -> Dict[str, Any]:
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
            "input_size": self.input_size,
            "device": self.device,
            "batch_size": self.batch_size
        })
        return details
    async def save_progress(self):
        progress_key = f"biomedlm_progress:{self.operation_id}"
        await enhanced_cache_manager.set(
            progress_key, 
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )
def validate_biomedlm_input(func):
    Decorator for validating BioMedLM input data.
    This decorator validates input parameters for BioMedLM methods.
    
    Args:
        func: Description of func
    
    async def wrapper(self, *args, **kwargs):
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
    This decorator adds progress tracking to BioMedLM methods.
    Args:
        operation_type: Type of operation
        total_steps: Total number of steps in the operation
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
    Decorator for enhanced error handling in BioMedLM methods.
    This decorator adds detailed error handling to BioMedLM methods.
    
    Args:
        func: Description of func
    
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            raise
        except ModelError:
            raise
        except ResourceNotFoundError:
            raise
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory error in {func.__name__}: {str(e)}")
            raise ModelError(
                model="BioMedLM",
                message=f"CUDA out of memory error: {str(e)}. Try reducing batch size or input length."
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ModelError(
                model="BioMedLM",
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper
def cached_biomedlm_prediction(ttl: int = 3600, prefix: str = "biomedlm", data_type: str = "prediction"):
    """
    Decorator for caching BioMedLM model predictions.
    This decorator adds caching to BioMedLM prediction methods.
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "biomedlm")
        data_type: Type of data being cached (default: "prediction")
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
            cache_key_parts = [prefix, func_name]
            for arg in args[1:]:
                if isinstance(arg, (str, int, float, bool)):
                    cache_key_parts.append(str(arg))
                elif isinstance(arg, (list, tuple)):
                    arg_hash = hashlib.md5(str(arg).encode()).hexdigest()
                    cache_key_parts.append(arg_hash)
            for key, value in sorted(kwargs.items()):
                if key == 'progress_tracker':
                    continue  # Skip progress tracker
                if isinstance(value, (str, int, float, bool)):
                    cache_key_parts.append(f"{key}={value}")
                elif isinstance(value, (list, tuple)):
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()
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