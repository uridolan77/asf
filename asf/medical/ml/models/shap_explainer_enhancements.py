"""
Enhanced SHAP Explainer for the Medical Research Synthesizer.

This module provides enhancements to the SHAP Explainer, including:
- Better error handling for explanation errors
- Validation of input data
- Caching of explanations
- Progress tracking for long-running explanations
"""

import logging
import time
import hashlib
import asyncio
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime

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

class SHAPProgressTracker(ProgressTracker):
    """
    Progress tracker for SHAP operations.
    
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
        """
        Save progress to cache.
        """
        progress_key = f"shap_progress:{self.operation_id}"
        await cache_manager.set(
            progress_key, 
            self.get_progress_details(),
            ttl=3600,  # 1 hour TTL
            data_type="progress"
        )

def validate_shap_input(func):
    """
    Decorator for validating SHAP input data.
    
    This decorator validates input parameters for SHAP methods.
    """
    async def wrapper(self, *args, **kwargs):
        # Extract common parameters
        text = kwargs.get('text', '')
        claim1 = kwargs.get('claim1', '')
        claim2 = kwargs.get('claim2', '')
        background_data = kwargs.get('background_data', None)
        num_samples = kwargs.get('num_samples', 100)
        
        # Validate text if present
        if 'text' in kwargs and not text and not isinstance(text, str):
            raise ValidationError("Text cannot be empty and must be a string")
            
        # Validate claim1 and claim2 if present
        if ('claim1' in kwargs or 'claim2' in kwargs) and (not claim1 or not claim2):
            raise ValidationError("Both claim1 and claim2 must be provided and non-empty")
            
        # Validate num_samples if present
        if 'num_samples' in kwargs:
            if not isinstance(num_samples, int):
                raise ValidationError("num_samples must be an integer")
            if num_samples < 10:
                raise ValidationError("num_samples must be at least 10")
            if num_samples > 1000:
                raise ValidationError("num_samples cannot exceed 1000")
                
        # Call the original function
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
        async def wrapper(self, *args, **kwargs):
            # Generate a deterministic operation ID based on the function and parameters
            func_name = func.__name__
            param_str = f"{func_name}:{args}:{kwargs}"
            operation_id = hashlib.md5(param_str.encode()).hexdigest()
            
            # Create progress tracker
            tracker = SHAPProgressTracker(operation_id, total_steps)
            tracker.set_operation_type(operation_type)
            
            # Set input size if applicable
            if 'text' in kwargs:
                tracker.set_input_size(len(kwargs['text']))
            elif 'claim1' in kwargs and 'claim2' in kwargs:
                tracker.set_input_size(len(kwargs['claim1']) + len(kwargs['claim2']))
                
            # Set num_samples if applicable
            if 'num_samples' in kwargs:
                tracker.set_num_samples(kwargs['num_samples'])
                
            # Set visualization type if applicable
            if 'visualization_type' in kwargs:
                tracker.set_visualization_type(kwargs['visualization_type'])
            
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

def enhanced_shap_error_handling(func):
    """
    Decorator for enhanced error handling in SHAP methods.
    
    This decorator adds detailed error handling to SHAP methods.
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
        except ImportError as e:
            # Handle missing dependencies
            logger.error(f"Missing dependency in {func.__name__}: {str(e)}")
            raise ModelError(
                model="SHAP",
                message=f"Missing dependency: {str(e)}. Please install SHAP and its dependencies."
            )
        except MemoryError as e:
            # Handle memory errors
            logger.error(f"Memory error in {func.__name__}: {str(e)}")
            raise ModelError(
                model="SHAP",
                message=f"Memory error: {str(e)}. Try reducing the number of samples or input size."
            )
        except Exception as e:
            # Log unexpected errors
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}", exc_info=True)
            
            # Convert to ModelError
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
class EnhancedSHAPExplainer:
    @validate_shap_input
    @track_shap_progress("text_explanation", total_steps=5)
    @enhanced_shap_error_handling
    @cached_shap_explanation(ttl=3600, prefix="shap_text", data_type="explanation")
    async def explain_text(
        self,
        text: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100,
        visualization_type: str = "bar",
        progress_tracker: Optional[SHAPProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Checking SHAP availability")
            await progress_tracker.save_progress()
            
        # Check if SHAP is available
        try:
            import shap
        except ImportError:
            raise ModelError(
                model="SHAP",
                message="SHAP is not installed. Please install it with: pip install shap"
            )
            
        if progress_tracker:
            progress_tracker.update(2, "Initializing explainer")
            await progress_tracker.save_progress()
            
        # Initialize explainer if needed
        if not self.is_initialized():
            raise ModelError(
                model="SHAP",
                message="SHAP explainer is not initialized. Call initialize() first."
            )
            
        if self.explainer is None:
            self._initialize_explainer(background_data or [text])
            
        if progress_tracker:
            progress_tracker.update(3, "Computing SHAP values")
            await progress_tracker.save_progress()
            
        # Get SHAP values
        shap_values = self.explainer([text])
        
        if progress_tracker:
            progress_tracker.update(4, "Processing SHAP values")
            await progress_tracker.save_progress()
            
        # Extract token-level SHAP values
        tokens = shap_values.data[0]
        values = shap_values.values[0]
        
        # Create explanation
        token_importance = {}
        for token, value in zip(tokens, values):
            if token.strip():
                token_importance[token] = float(value)
                
        # Sort tokens by absolute importance
        sorted_tokens = sorted(
            token_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Get top words
        top_words = [token for token, _ in sorted_tokens[:10]]
        top_values = [float(value) for _, value in sorted_tokens[:10]]
        
        # Create summary
        summary = f"The most influential words in the text are: {', '.join(top_words[:5])}"
        
        if progress_tracker:
            progress_tracker.update(5, "Generating visualization")
            await progress_tracker.save_progress()
            
        # Generate visualization
        plt.figure(figsize=(10, 6))
        plt.barh([token for token, _ in sorted_tokens[:10]], [value for _, value in sorted_tokens[:10]])
        plt.xlabel("SHAP Value")
        plt.title("Top Influential Words")
        plt.tight_layout()
        
        # Save visualization to base64
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        
        # Create explanation
        explanation = {
            "token_importance": token_importance,
            "top_words": top_words,
            "top_values": top_values,
            "summary": summary,
            "visualization": f"data:image/png;base64,{img_str}"
        }
        
        return explanation
        
    @validate_shap_input
    @track_shap_progress("contradiction_explanation", total_steps=6)
    @enhanced_shap_error_handling
    @cached_shap_explanation(ttl=3600, prefix="shap_contradiction", data_type="explanation")
    async def explain_contradiction(
        self,
        claim1: str,
        claim2: str,
        background_data: Optional[List[str]] = None,
        num_samples: int = 100,
        progress_tracker: Optional[SHAPProgressTracker] = None
    ) -> Dict[str, Any]:
        # Implementation with progress tracking
        if progress_tracker:
            progress_tracker.update(1, "Combining claims")
            await progress_tracker.save_progress()
            
        # Combine claims
        combined_text = f"{claim1} [SEP] {claim2}"
        
        if progress_tracker:
            progress_tracker.update(2, "Generating text explanation")
            await progress_tracker.save_progress()
            
        # Get explanation
        explanation = await self.explain_text(
            combined_text,
            background_data,
            num_samples,
            progress_tracker=None  # Don't pass the tracker to avoid nested tracking
        )
        
        if progress_tracker:
            progress_tracker.update(3, "Analyzing claim1")
            await progress_tracker.save_progress()
            
        # Analyze claim1
        claim1_tokens = {}
        for token, value in explanation["token_importance"].items():
            if token in claim1:
                claim1_tokens[token] = value
                
        if progress_tracker:
            progress_tracker.update(4, "Analyzing claim2")
            await progress_tracker.save_progress()
            
        # Analyze claim2
        claim2_tokens = {}
        for token, value in explanation["token_importance"].items():
            if token in claim2:
                claim2_tokens[token] = value
                
        if progress_tracker:
            progress_tracker.update(5, "Identifying contradictory terms")
            await progress_tracker.save_progress()
            
        # Identify contradictory terms
        contradictory_terms = []
        for token, value in explanation["token_importance"].items():
            if abs(value) > 0.1:  # Threshold for significance
                contradictory_terms.append({
                    "token": token,
                    "importance": float(value),
                    "in_claim1": token in claim1,
                    "in_claim2": token in claim2
                })
                
        # Sort by absolute importance
        contradictory_terms = sorted(
            contradictory_terms,
            key=lambda x: abs(x["importance"]),
            reverse=True
        )
        
        if progress_tracker:
            progress_tracker.update(6, "Finalizing explanation")
            await progress_tracker.save_progress()
            
        # Add claims to explanation
        explanation["claim1"] = claim1
        explanation["claim2"] = claim2
        explanation["claim1_tokens"] = claim1_tokens
        explanation["claim2_tokens"] = claim2_tokens
        explanation["contradictory_terms"] = contradictory_terms
        
        # Generate a more detailed summary
        top_contradictory = [term["token"] for term in contradictory_terms[:5]]
        explanation["summary"] = f"The contradiction is primarily due to: {', '.join(top_contradictory)}"
        
        return explanation
"""

# Additional SHAP utility functions:

async def generate_shap_report(explanation: Dict[str, Any], output_path: str) -> str:
    """
    Generate a detailed HTML report from a SHAP explanation.
    
    Args:
        explanation: SHAP explanation
        output_path: Path to save the HTML report
        
    Returns:
        Path to the HTML report
    """
    # Create HTML content
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
    """
    
    # Add claims if available
    if 'claim1' in explanation and 'claim2' in explanation:
        html += f"""
        <div class="section">
            <h2>Claims</h2>
            <div class="claims">
                <div class="claim">
                    <strong>Claim 1:</strong> {explanation['claim1']}
                </div>
                <div class="claim">
                    <strong>Claim 2:</strong> {explanation['claim2']}
                </div>
            </div>
        </div>
        """
    
    # Add visualization if available
    if 'visualization' in explanation:
        html += f"""
        <div class="section">
            <h2>Visualization</h2>
            <div class="visualization">
                <img src="{explanation['visualization']}" alt="SHAP Visualization" style="max-width: 100%;">
            </div>
        </div>
        """
    
    # Add top influential words
    if 'top_words' in explanation and 'top_values' in explanation:
        html += """
        <div class="section">
            <h2>Top Influential Words</h2>
            <div class="tokens">
        """
        
        for word, value in zip(explanation['top_words'], explanation['top_values']):
            css_class = "positive" if value > 0 else "negative" if value < 0 else "neutral"
            html += f"""
                <div class="token {css_class}">
                    {word} ({value:.4f})
                </div>
            """
            
        html += """
            </div>
        </div>
        """
    
    # Add contradictory terms if available
    if 'contradictory_terms' in explanation:
        html += """
        <div class="section">
            <h2>Contradictory Terms</h2>
            <div class="tokens">
        """
        
        for term in explanation['contradictory_terms'][:20]:  # Limit to top 20
            css_class = "positive" if term['importance'] > 0 else "negative" if term['importance'] < 0 else "neutral"
            location = []
            if term.get('in_claim1'):
                location.append("Claim 1")
            if term.get('in_claim2'):
                location.append("Claim 2")
            location_str = " & ".join(location) if location else "Unknown"
            
            html += f"""
                <div class="token {css_class}">
                    {term['token']} ({term['importance']:.4f}) - {location_str}
                </div>
            """
            
        html += """
            </div>
        </div>
        """
    
    # Close HTML
    html += """
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    return output_path

async def batch_explain_contradictions(
    contradictions: List[Dict[str, Any]],
    explainer: Any,
    output_dir: str,
    max_concurrent: int = 3
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def explain_contradiction(contradiction: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            try:
                # Extract claims
                claim1 = contradiction.get('claim1', '')
                claim2 = contradiction.get('claim2', '')
                
                if not claim1 or not claim2:
                    logger.warning(f"Missing claims in contradiction: {contradiction}")
                    return contradiction
                
                # Generate explanation
                explanation = await explainer.explain_contradiction(claim1, claim2)
                
                # Generate report
                report_path = os.path.join(output_dir, f"contradiction_{hash(claim1 + claim2) % 10000}.html")
                await generate_shap_report(explanation, report_path)
                
                # Add explanation to contradiction
                contradiction['explanation'] = explanation
                contradiction['explanation_report'] = report_path
                
                return contradiction
            except Exception as e:
                logger.error(f"Error explaining contradiction: {str(e)}")
                # Return the original contradiction without explanation
                return contradiction
    
    # Process all contradictions concurrently
    tasks = [explain_contradiction(c) for c in contradictions]
    explained_contradictions = await asyncio.gather(*tasks)
    
    return explained_contradictions
