"""
Enhanced BioMedLM Service for the Medical Research Synthesizer.

This module provides enhancements to the BioMedLM Service, including:
- Better error handling for model errors
- Validation of input data
- Caching of model predictions
- Progress tracking for long-running model operations
"""

import logging
import time
import hashlib
import asyncio
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime
from functools import lru_cache

from asf.medical.core.exceptions import (
    ValidationError, ModelError, ResourceNotFoundError
)
from asf.medical.core.cache import cache_manager
from asf.medical.core.progress_tracker import ProgressTracker
from asf.medical.ml.services.ml_service_enhancements import (
    validate_ml_input, track_ml_progress, enhanced_ml_error_handling, cached_ml_prediction
)

# Set up logging
logger = logging.getLogger(__name__)

class BioMedLMProgressTracker(ProgressTracker):
    """
    Progress tracker for BioMedLM operations.
    
    This class extends the base ProgressTracker to provide BioMedLM-specific
    progress tracking functionality.
    """
    
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
        """
        Save progress to cache.
        """
        progress_key = f"biomedlm_progress:{self.operation_id}"
        await cache_manager.set(
            progress_key, 
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )

def validate_biomedlm_input(func):
    """
    Decorator for validating BioMedLM input data.
    
    This decorator validates input parameters for BioMedLM methods.
    """
    async def wrapper(self, *args, **kwargs):
        # Extract common parameters
        text = kwargs.get('text', '')
        texts = kwargs.get('texts', [])
        text1 = kwargs.get('text1', '')
        text2 = kwargs.get('text2', '')
        
        # Validate text if present
        if 'text' in kwargs and not text and not isinstance(text, str):
            raise ValidationError("Text cannot be empty and must be a string")
            
        # Validate texts if present
        if 'texts' in kwargs:
            if not texts:
                raise ValidationError("Texts cannot be empty")
            if not all(isinstance(t, str) for t in texts):
                raise ValidationError("All texts must be strings")
                
        # Validate text1 and text2 if present
        if ('text1' in kwargs or 'text2' in kwargs) and (not text1 or not text2):
            raise ValidationError("Both text1 and text2 must be provided and non-empty")
                
        # Call the original function
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
        async def wrapper(self, *args, **kwargs):
            # Generate a deterministic operation ID based on the function and parameters
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            operation_id = hashlib.md5(param_str.encode()).hexdigest()
            
            # Create progress tracker
            tracker = BioMedLMProgressTracker(operation_id, total_steps)
            tracker.set_operation_type(operation_type)
            
            # Set device
            tracker.set_device(self.device)
            
            # Set input size if applicable
            if 'text' in kwargs:
                tracker.set_input_size(len(kwargs['text']))
            elif 'texts' in kwargs:
                tracker.set_input_size(len(kwargs['texts']))
                tracker.set_batch_size(min(len(kwargs['texts']), 8))  # Default batch size
            elif 'text1' in kwargs and 'text2' in kwargs:
                tracker.set_input_size(len(kwargs['text1']) + len(kwargs['text2']))
            
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

def enhanced_biomedlm_error_handling(func):
    """
    Decorator for enhanced error handling in BioMedLM methods.
    
    This decorator adds detailed error handling to BioMedLM methods.
    """
    async def wrapper(self, *args, **kwargs):
        try:
            # Call the original function
            return await func(self, *args, **kwargs)
        except ValidationError:
            # Re-raise validation errors
            raise
        except ModelError:
            # Re-raise model errors
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except torch.cuda.OutOfMemoryError as e:
            # Handle CUDA out of memory errors
            logger.error(f"CUDA out of memory error in {func.__name__}: {str(e)}")
            raise ModelError(
                model="BioMedLM",
                message=f"CUDA out of memory error: {str(e)}. Try reducing batch size or input length."
            )
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Convert to ModelError
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
        async def wrapper(self, *args, **kwargs):
            # Skip caching if explicitly requested
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(self, *args, **kwargs)
                
            # Generate cache key
            func_name = func.__name__
            
            # Create a deterministic cache key based on the function and parameters
            cache_key_parts = [prefix, func_name]
            
            # Add args (excluding self)
            for arg in args[1:]:
                if isinstance(arg, (str, int, float, bool)):
                    cache_key_parts.append(str(arg))
                elif isinstance(arg, (list, tuple)):
                    # For lists/tuples, hash the contents
                    arg_hash = hashlib.md5(str(arg).encode()).hexdigest()
                    cache_key_parts.append(arg_hash)
                    
            # Add kwargs
            for key, value in sorted(kwargs.items()):
                if key == 'progress_tracker':
                    continue  # Skip progress tracker
                    
                if isinstance(value, (str, int, float, bool)):
                    cache_key_parts.append(f"{key}={value}")
                elif isinstance(value, (list, tuple)):
                    # For lists/tuples, hash the contents
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()
                    cache_key_parts.append(f"{key}={value_hash}")
                    
            # Create the final cache key
            cache_key = hashlib.md5(":".join(cache_key_parts).encode()).hexdigest()
            
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
class EnhancedBioMedLMService:
    @validate_biomedlm_input
    @track_biomedlm_progress("encoding", total_steps=3)
    @enhanced_biomedlm_error_handling
    @cached_biomedlm_prediction(ttl=3600, prefix="biomedlm_encode", data_type="embedding")
    async def encode_async(
        self,
        text: str,
        progress_tracker: Optional[BioMedLMProgressTracker] = None
    ) -> np.ndarray:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Tokenizing text")
            await progress_tracker.save_progress()
            
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        if progress_tracker:
            progress_tracker.update(2, "Moving inputs to device")
            await progress_tracker.save_progress()
            
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if progress_tracker:
            progress_tracker.update(3, "Computing embeddings")
            await progress_tracker.save_progress()
            
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use mean pooling to get a single vector
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return embeddings
        
    @validate_biomedlm_input
    @track_biomedlm_progress("batch_encoding", total_steps=4)
    @enhanced_biomedlm_error_handling
    @cached_biomedlm_prediction(ttl=3600, prefix="biomedlm_batch_encode", data_type="embedding")
    async def batch_encode_async(
        self,
        texts: List[str],
        batch_size: int = 8,
        progress_tracker: Optional[BioMedLMProgressTracker] = None
    ) -> List[np.ndarray]:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Preparing batches")
            progress_tracker.set_batch_size(batch_size)
            await progress_tracker.save_progress()
            
        # Prepare batches
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        total_batches = len(batches)
        
        # Process batches
        all_embeddings = []
        
        for i, batch in enumerate(batches):
            if progress_tracker:
                progress_tracker.update(
                    2 + (i / total_batches),
                    f"Processing batch {i+1}/{total_batches}"
                )
                await progress_tracker.save_progress()
                
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Use mean pooling to get vectors
            batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            all_embeddings.extend(batch_embeddings)
            
        if progress_tracker:
            progress_tracker.update(4, "Finalizing embeddings")
            await progress_tracker.save_progress()
            
        return all_embeddings
        
    @validate_biomedlm_input
    @track_biomedlm_progress("contradiction_detection", total_steps=4)
    @enhanced_biomedlm_error_handling
    @cached_biomedlm_prediction(ttl=3600, prefix="biomedlm_contradiction", data_type="prediction")
    async def detect_contradiction(
        self,
        text1: str,
        text2: str,
        threshold: float = 0.7,
        progress_tracker: Optional[BioMedLMProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Encoding text1")
            await progress_tracker.save_progress()
            
        # Encode text1
        embedding1 = await self.encode_async(text1)
        
        if progress_tracker:
            progress_tracker.update(2, "Encoding text2")
            await progress_tracker.save_progress()
            
        # Encode text2
        embedding2 = await self.encode_async(text2)
        
        if progress_tracker:
            progress_tracker.update(3, "Calculating contradiction score")
            await progress_tracker.save_progress()
            
        # Calculate contradiction score
        similarity = self.calculate_similarity(embedding1, embedding2)
        contradiction_score = 1.0 - similarity
        
        if progress_tracker:
            progress_tracker.update(4, "Finalizing results")
            await progress_tracker.save_progress()
            
        # Create result
        result = {
            "text1": text1,
            "text2": text2,
            "contradiction_score": contradiction_score,
            "is_contradiction": contradiction_score >= threshold,
            "similarity": similarity
        }
        
        return result
"""
