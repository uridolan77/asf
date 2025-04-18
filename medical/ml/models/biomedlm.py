"""BioMedLM model wrapper for the Medical Research Synthesizer.

This module provides a wrapper for the BioMedLM model for contradiction detection.
"""
import logging
import torch
import numpy as np
from typing import Tuple, TypeVar, Any, Dict, List, Optional, Callable, Union
import time
import hashlib
from transformers import AutoModel
from asf.medical.core.config import settings
from asf.medical.core.exceptions import (
    ValidationError, ModelError, ResourceNotFoundError
)

logger = logging.getLogger(__name__)
T = TypeVar('T')

class ProgressTracker:
    def __init__(self, operation_id: str, total_steps: int = 100):
        self.operation_id = operation_id
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "pending"
        self.message = ""
    def update(self, step: int, message: str):
        self.current_step = step
        self.message = message
    def complete(self, message: str):
        self.current_step = self.total_steps
        self.status = "completed"
        self.message = message
    def fail(self, message: str):
        self.status = "failed"
        self.message = message
    def get_progress_details(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "status": self.status,
            "message": self.message
        }

class BioMedLMProgressTracker(ProgressTracker):
    def __init__(self, operation_id: str, total_steps: int = 100):
        super().__init__(operation_id=operation_id, total_steps=total_steps)
        self.model_name = "biomedlm"
        self.operation_type = "unknown"
        self.start_time = time.time()
        self.input_size = 0
        self.device = "unknown"
        self.batch_size = 0
    def set_operation_type(self, operation_type: str):
        self.operation_type = operation_type
    def set_input_size(self, input_size: int):
        self.input_size = input_size
    def set_device(self, device: str):
        self.device = device
    def set_batch_size(self, batch_size: int):
        self.batch_size = batch_size
    def get_progress_details(self) -> Dict[str, Any]:
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
        logger.info(f"Saving progress for {self.operation_id}: {self.get_progress_details()}")

def validate_biomedlm_input(func):
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
    def decorator(func):
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
                tracker.set_batch_size(min(len(kwargs['texts']), 8))
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
    def decorator(func):
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
                    continue
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

class BioMedLMService:
    """Service for the BioMedLM model.

    This service provides methods for using the BioMedLM model for contradiction detection.
    """
    _instance = None
    def __new__(cls):
        """
        Create a singleton instance of the BioMedLM service.
        Returns:
            BioMedLMService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(BioMedLMService, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        """
        Initialize the BioMedLM service.

        This method sets up the service with the model name and device configuration
        based on application settings.
        """

        self.model_name = settings.BIOMEDLM_MODEL
        self.use_gpu = settings.USE_GPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        logger.info(f"BioMedLM service initialized with device: {self.device}")
    @property
    def model(self):
        """Get the BioMedLM model, loading it if necessary.
        
        This property loads the BioMedLM model from the specified model name
        and moves it to the appropriate device (CPU/GPU).
        
        Returns:
            torch.nn.Module: The loaded BioMedLM model instance
        """
        logger.info(f"Loading BioMedLM model: {self.model_name}")
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        logger.info("BioMedLM model loaded")
        return model
    @property
    def tokenizer(self):
        """Get the BioMedLM tokenizer for text processing.
        
        This property loads the tokenizer associated with the BioMedLM model
        for processing input text into tokens.
        
        Returns:
            transformers.PreTrainedTokenizer: The BioMedLM tokenizer instance
        """
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_name)

    async def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using the BioMedLM model.

        Args:
            text: Text to encode

        Returns:
            Text embedding as a numpy array
        """
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the [CLS] token embedding as the text embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        embedding1 = await self.encode_text(text1)
        embedding2 = await self.encode_text(text2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float((similarity + 1) / 2)  # Scale from [-1, 1] to [0, 1]

    async def detect_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Tuple of (is_contradiction, confidence)
        """
        similarity = await self.calculate_similarity(claim1, claim2)
        contradiction_score = 1.0 - similarity
        is_contradiction = contradiction_score > 0.7
        return (is_contradiction, contradiction_score)

    async def get_token_importance(self, claim1: str, claim2: str) -> dict:
        """
        Get the importance of each token in the claims for contradiction detection.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Dictionary mapping tokens to importance scores
        """
        tokenizer = self.tokenizer
        tokens1 = tokenizer.tokenize(claim1)
        tokens2 = tokenizer.tokenize(claim2)

        # Calculate baseline contradiction score
        baseline_result = await self.detect_contradiction(claim1, claim2)
        baseline_score = baseline_result[1]

        # Calculate importance for each token in claim1
        importance = {}
        for i, token in enumerate(tokens1):
            # Create a version of claim1 without this token
            modified_tokens = tokens1.copy()
            modified_tokens.pop(i)
            modified_claim = tokenizer.convert_tokens_to_string(modified_tokens)

            # Calculate contradiction score without this token
            modified_result = await self.detect_contradiction(modified_claim, claim2)
            modified_score = modified_result[1]

            # Importance is the difference in scores
            importance[token] = abs(baseline_score - modified_score)

        # Calculate importance for each token in claim2
        for i, token in enumerate(tokens2):
            # Create a version of claim2 without this token
            modified_tokens = tokens2.copy()
            modified_tokens.pop(i)
            modified_claim = tokenizer.convert_tokens_to_string(modified_tokens)

            # Calculate contradiction score without this token
            modified_result = await self.detect_contradiction(claim1, modified_claim)
            modified_score = modified_result[1]

            # Importance is the difference in scores
            importance[token] = abs(baseline_score - modified_score)

        return importance