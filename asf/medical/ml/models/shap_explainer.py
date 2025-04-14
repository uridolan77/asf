"""SHAP explainer for the Medical Research Synthesizer.

This module provides a SHAP-based explainer for model predictions.
"""
import logging
import shap
import io
import base64
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable
import time
import hashlib
import numpy as np
from asf.medical.core.progress_tracker import ProgressTracker
from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached
from asf.medical.ml.services.ml_service_enhancements import (
    validate_ml_input, track_ml_progress, enhanced_ml_error_handling, cached_ml_prediction
)

logger = logging.getLogger(__name__)

class SHAPProgressTracker(ProgressTracker):
    def __init__(self, operation_id: str, total_steps: int = 100):
        super().__init__(operation_id=operation_id, total_steps=total_steps)
        self.model_name = "shap"
        self.operation_type = "unknown"
        self.start_time = time.time()
        self.input_size = 0
        self.num_samples = 0
        self.visualization_type = "none"
    def set_operation_type(self, operation_type: str):
        self.operation_type = operation_type
    def set_input_size(self, input_size: int):
        self.input_size = input_size
    def set_num_samples(self, num_samples: int):
        self.num_samples = num_samples
    def set_visualization_type(self, visualization_type: str):
        self.visualization_type = visualization_type
    def get_progress_details(self) -> dict:
        details = super().get_progress_details()
        details.update({
            "model_name": self.model_name,
            "operation_type": self.operation_type,
            "elapsed_time": time.time() - self.start_time,
            "input_size": self.input_size,
            "num_samples": self.num_samples,
            "visualization_type": self.visualization_type
        })
        return details
    async def save_progress(self):
        progress_key = f"shap_progress:{self.operation_id}"
        await cache_manager.set(
            progress_key,
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )

def validate_shap_input(func):
    async def wrapper(self, *args, **kwargs):
        text = kwargs.get('text', '')
        claim1 = kwargs.get('claim1', '')
        claim2 = kwargs.get('claim2', '')
        background_data = kwargs.get('background_data', None)
        num_samples = kwargs.get('num_samples', 100)
        if 'text' in kwargs and not text and not isinstance(text, str):
            raise ValidationError("Text cannot be empty and must be a string")
        if ('claim1' in kwargs or 'claim2' in kwargs) and (not claim1 or not claim2):
            raise ValidationError("Both claim1 and claim2 must be provided and non-empty")
        if 'num_samples' in kwargs:
            if not isinstance(num_samples, int):
                raise ValidationError("num_samples must be an integer")
            if num_samples < 10:
                raise ValidationError("num_samples must be at least 10")
            if num_samples > 1000:
                raise ValidationError("num_samples cannot exceed 1000")
        return await func(self, *args, **kwargs)
    return wrapper

def track_shap_progress(operation_type: str, total_steps: int = 100):
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            operation_id = hashlib.md5(param_str.encode()).hexdigest()
            tracker = SHAPProgressTracker(operation_id, total_steps)
            tracker.set_operation_type(operation_type)
            if 'text' in kwargs:
                tracker.set_input_size(len(kwargs['text']))
            elif 'claim1' in kwargs and 'claim2' in kwargs:
                tracker.set_input_size(len(kwargs['claim1']) + len(kwargs['claim2']))
            if 'num_samples' in kwargs:
                tracker.set_num_samples(kwargs['num_samples'])
            if 'visualization_type' in kwargs:
                tracker.set_visualization_type(kwargs['visualization_type'])
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

def enhanced_shap_error_handling(func):
    async def wrapper(self, *args, **kwargs):
        try:
            return await func(self, *args, **kwargs)
        except ValidationError:
            raise
        except ModelError:
            raise
        except ResourceNotFoundError:
            raise
        except ImportError as e:
            logger.error(f"Missing dependency in {func.__name__}: {str(e)}")
            raise ModelError(
                model="SHAP",
                message=f"Missing dependency: {str(e)}. Please install SHAP and its dependencies."
            )
        except MemoryError as e:
            logger.error(f"Memory error in {func.__name__}: {str(e)}")
            raise ModelError(
                model="SHAP",
                message=f"Memory error: {str(e)}. Try reducing the number of samples or input size."
            )
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            raise ModelError(
                model="SHAP",
                message=f"Unexpected error: {str(e)}"
            )
    return wrapper

def cached_shap_explanation(ttl: int = 3600, prefix: str = "shap", data_type: str = "explanation"):
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
                    continue  # Skip progress tracker
                if isinstance(value, (str, int, float, bool)):
                    cache_key_parts.append(f"{key}={value}")
                elif isinstance(value, (list, tuple)):
                    value_hash = hashlib.md5(str(value).encode()).hexdigest()
                    cache_key_parts.append(f"{key}={value_hash}")
            cache_key = hashlib.md5(":".join(cache_key_parts).encode()).hexdigest()
            cached_result = await cache_manager.get(cache_key, data_type=data_type)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
            logger.debug(f"Cache miss for {func_name}")
            result = await func(self, *args, **kwargs)
            await cache_manager.set(cache_key, result, ttl=ttl, data_type=data_type)
            return result
        return wrapper
    return decorator

async def generate_shap_report(explanation: Dict[str, Any], output_path: str) -> str:
    # ...existing code for HTML report generation...
    # ...existing code...
    return output_path

async def generate_batch_explanations(
    contradictions: List[Dict[str, Any]],
    explainer: Any,
    output_dir: str,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    # ...existing code for batch explanation generation...
    # ...existing code...
    return []

class SHAPExplainer:
    """SHAP-based explainer for model predictions.
    
    This class provides methods for explaining model predictions using SHAP.
    """
    def __init__(self, model_fn: Optional[Callable] = None, tokenizer: Optional[Any] = None):
        """
        Initialize the SHAP explainer.
        Args:
            model_fn: Model prediction function (optional)
            tokenizer: Tokenizer for text data (optional)
        """
        self.model_fn = model_fn
        self.tokenizer = tokenizer
        self.explainer = None
        logger.info("SHAP explainer initialized")
    def initialize(self, model_fn: callable, tokenizer: Any = None):
        """
        Initialize the SHAP explainer with a model function and tokenizer.
        Args:
            model_fn: Model prediction function
            tokenizer: Tokenizer for text data
        """
        self.model_fn = model_fn
        self.tokenizer = tokenizer
        self.explainer = None
        logger.info("SHAP explainer initialized with model function and tokenizer")
    def is_initialized(self) -> bool:
        """
        Check if the SHAP explainer is initialized.
        Returns:
            True if the SHAP explainer is initialized, False otherwise
        """
        return self.model_fn is not None
    def _initialize_explainer(self, background_data: List[str]):
        """
        Initialize the SHAP explainer with background data.
        Args:
            background_data: Background data for SHAP
        """
        if self.tokenizer:
            self.explainer = shap.Explainer(self.model_fn, self.tokenizer)
        else:
            self.explainer = shap.KernelExplainer(self.model_fn, background_data)
        logger.info("SHAP explainer initialized with background data")
    def explain_text(
        self,
        text: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Explain a text prediction using SHAP values.

        Args:
            text: The text to explain
            background_data: Background data for SHAP (optional)
            num_samples: Number of samples for SHAP (default: 100)

        Returns:
            Dictionary containing explanation details including token importance,
            top influential words, summary, and visualization
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is required for text explanation")
        if self.explainer is None and background_data is not None:
            self._initialize_explainer(background_data)
        if self.explainer is None:
            raise ValueError("Explainer has not been initialized")
        shap_values = self.explainer([text])
        tokens = shap_values.data[0]
        values = shap_values.values[0]
        token_importance = {}
        for token, value in zip(tokens, values):
            if token.strip():
                token_importance[token] = float(value)
        sorted_tokens = sorted(
            token_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        top_words = [token for token, _ in sorted_tokens[:10]]
        top_values = [value for _, value in sorted_tokens[:10]]
        positive_words = [token for token, value in sorted_tokens if value > 0][:5]
        negative_words = [token for token, value in sorted_tokens if value < 0][:5]
        summary = f"The prediction is influenced positively by {', '.join(positive_words)}"
        if negative_words:
            summary += f" and negatively by {', '.join(negative_words)}"
        plt.figure(figsize=(10, 6))
        plt.barh([token for token, _ in sorted_tokens[:10]], [value for _, value in sorted_tokens[:10]])
        plt.xlabel("SHAP Value")
        plt.title("Top Influential Words")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        explanation = {
            "token_importance": token_importance,
            "top_words": top_words,
            "top_values": top_values,
            "summary": summary,
            "visualization": f"data:image/png;base64,{img_str}"
        }
        return explanation
    def explain_contradiction(
        self,
        claim1: str,
        claim2: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Explain a contradiction prediction between two claims using SHAP values.

        Args:
            claim1: The first claim
            claim2: The second claim
            background_data: Background data for SHAP (optional)
            num_samples: Number of samples for SHAP (default: 100)

        Returns:
            Dictionary containing explanation details including token importance,
            top influential words, summary, and visualization
        """
        combined_text = f"{claim1} [SEP] {claim2}"
        explanation = self.explain_text(
            combined_text,
            background_data,
            num_samples
        )
        explanation["claim1"] = claim1
        explanation["claim2"] = claim2
        return explanation