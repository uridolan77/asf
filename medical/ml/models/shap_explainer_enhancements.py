"""Enhanced SHAP Explainer for the Medical Research Synthesizer.

This module provides enhancements to the SHAP Explainer, including:
- Better error handling for explanation errors
- Validation of input data
- Caching of explanations 
- Progress tracking for long-running explanations
"""
import logging
import time
import hashlib
import numpy as np
from typing import Dict, List, Any
from asf.medical.core.exceptions import (
    ValidationError, ModelError, ResourceNotFoundError
)
from asf.medical.core.enhanced_cache import enhanced_cache_manager as cache_manager, enhanced_cached as cached
from asf.medical.core.progress_tracker import ProgressTracker
from asf.medical.ml.services.ml_service_enhancements import (
    validate_ml_input, track_ml_progress, enhanced_ml_error_handling, cached_ml_prediction
)
logger = logging.getLogger(__name__)
class SHAPProgressTracker(ProgressTracker):
    """Progress tracker for SHAP operations.
    
    This class extends the base ProgressTracker to provide SHAP-specific
    progress tracking functionality.
    """
    def __init__(self, operation_id: str, total_steps: int = 100):
        """
        Initialize the SHAP progress tracker.
        Args:
            operation_id: Operation ID
            total_steps: Total number of steps in the operation
        """
        super().__init__(operation_id=operation_id, total_steps=total_steps)
        self.model_name = "shap"
        self.operation_type = "unknown"
        self.start_time = time.time()
        self.input_size = 0
        self.num_samples = 0
        self.visualization_type = "none"
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
    def set_num_samples(self, num_samples: int):
        """
        Set the number of samples.
        Args:
            num_samples: Number of samples for SHAP
        """
        self.num_samples = num_samples
    def set_visualization_type(self, visualization_type: str):
        """
        Set the visualization type.
        Args:
            visualization_type: Type of visualization
        """
        self.visualization_type = visualization_type
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
            "num_samples": self.num_samples,
            "visualization_type": self.visualization_type
        })
        return details
    async def save_progress(self):
        progress_key = f"shap_progress:{self.operation_id}"
        await enhanced_cache_manager.set(
            progress_key,
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )
def validate_shap_input(func):
    """Decorator for validating SHAP input data.
    
    This decorator validates input parameters for SHAP methods to ensure 
    they meet the required format and constraints before processing.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with input validation
    """
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
    """
    Decorator for tracking SHAP operation progress.
    This decorator adds progress tracking to SHAP methods.
    Args:
        operation_type: Type of operation
        total_steps: Total number of steps in the operation
    """
    def decorator(func):
        """Inner decorator function that wraps the original function.
        
        Args:
            func: The function being decorated
            
        Returns:
            Wrapped function with progress tracking capability
        """
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
    """Decorator for enhanced error handling in SHAP methods.
    
    This decorator adds detailed error handling to SHAP methods, including
    specific handling for import errors and memory errors.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with enhanced error handling
    """
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
    """
    Decorator for caching SHAP explanations.
    This decorator adds caching to SHAP explanation methods.
    Args:
        ttl: Time-to-live in seconds (default: 3600 = 1 hour)
        prefix: Cache key prefix (default: "shap")
        data_type: Type of data being cached (default: "explanation")
    """
    def decorator(func):
        """Inner decorator function that wraps the original function.
        
        Args:
            func: The function being decorated
            
        Returns:
            Wrapped function with caching capability
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
async def generate_shap_report(explanation: Dict[str, Any], output_path: str) -> str:
    """
    Generate a HTML report from a SHAP explanation.

    Args:
        explanation: Dictionary containing SHAP explanation data
        output_path: Path to save the HTML report

    Returns:
        Path to the generated HTML report
    """
    # Build HTML tokens for top words
    top_words_html = ""
    for word, value in explanation.get('top_words', {}).items():
        css_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
        top_words_html += f"<div class=\"token {css_class}\">\n"
        top_words_html += f"    {word} ({value:.4f})\n"
        top_words_html += f"</div>\n"

    # Build HTML for contradictory terms
    contradictory_terms_html = ""
    for term in explanation.get('contradictory_terms', []):
        css_class = "positive" if term.get('importance', 0) > 0 else "negative" if term.get('importance', 0) < 0 else "neutral"
        location_str = "Claim 1" if term.get('location') == 'claim1' else "Claim 2" if term.get('location') == 'claim2' else "Both claims"
        contradictory_terms_html += f"<div class=\"token {css_class}\">\n"
        contradictory_terms_html += f"    {term.get('token', '')} ({term.get('importance', 0):.4f}) - {location_str}\n"
        contradictory_terms_html += f"</div>\n"

    # Generate the HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Explanation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .section {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; }}
            .token {{ display: inline-block; margin: 5px; padding: 5px; border-radius: 3px; }}
            .positive {{ background-color: #d4efdf; }}
            .negative {{ background-color: #f5b7b1; }}
            .neutral {{ background-color: #eaeded; }}
            .visualization {{ margin: 20px 0; }}
            .summary {{ font-weight: bold; margin: 10px 0; }}
            .claims {{ margin: 10px 0; }}
            .claim {{ margin: 5px 0; padding: 10px; background-color: #f8f9f9; }}
        </style>
    </head>
    <body>
        <h1>SHAP Explanation Report</h1>
        <div class="section">
            <h2>Summary</h2>
            <p class="summary">{explanation.get('summary', 'No summary available')}</p>
        </div>
        <div class="section">
            <h2>Claims</h2>
            <div class="claims">
                <div class="claim">
                    <strong>Claim 1:</strong> {explanation.get('claim1', 'N/A')}
                </div>
                <div class="claim">
                    <strong>Claim 2:</strong> {explanation.get('claim2', 'N/A')}
                </div>
            </div>
        </div>
        <div class="section">
            <h2>Visualization</h2>
            <div class="visualization">
                <img src="{explanation.get('visualization', '')}" alt="SHAP Visualization" style="max-width: 100%;">
            </div>
        </div>
        <div class="section">
            <h2>Top Influential Words</h2>
            <div class="tokens">
                {top_words_html}
            </div>
        </div>
        <div class="section">
            <h2>Contradictory Terms</h2>
            <div class="tokens">
                {contradictory_terms_html}
            </div>
        </div>
    </body>
    </html>
    """

    # Write the HTML to the output file
    with open(output_path, 'w') as f:
        f.write(html)

    return output_path

async def generate_batch_explanations(
    contradictions: List[Dict[str, Any]],
    explainer: Any,
    output_dir: str,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate SHAP explanations for a batch of contradictions.

    Args:
        contradictions: List of contradictions to explain
        explainer: SHAP explainer instance
        output_dir: Directory to save reports
        max_concurrent: Maximum number of concurrent explanations

    Returns:
        List of contradictions with explanations
    """