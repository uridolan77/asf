"""
Utility functions for replay strategies.

This module provides utility functions for replay strategies, including:
- Example generation
- Batch processing
- Caching mechanisms
"""

import torch
import random
import hashlib
import json
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class GenerationCache:
    """
    Cache for generated examples.
    
    This class provides a cache for generated examples to avoid
    regenerating the same examples multiple times.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_cache_size: int = 10000,
        use_disk_cache: bool = True
    ):
        """
        Initialize the generation cache.
        
        Args:
            cache_dir: Directory for disk cache
            max_cache_size: Maximum number of examples to cache in memory
            use_disk_cache: Whether to use disk cache
        """
        self.memory_cache = {}
        self.max_cache_size = max_cache_size
        self.use_disk_cache = use_disk_cache
        
        # Set up disk cache if enabled
        self.cache_dir = None
        if use_disk_cache:
            if cache_dir:
                self.cache_dir = cache_dir
            else:
                # Use default cache directory
                self.cache_dir = os.path.join(
                    os.path.expanduser("~"),
                    ".cache",
                    "cl_peft",
                    "generation_cache"
                )
            
            # Create cache directory if it doesn't exist
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Using disk cache at {self.cache_dir}")
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get an example from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached example or None if not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]
        
        # Check disk cache if enabled
        if self.use_disk_cache and self.cache_dir:
            cache_file = self._get_cache_file(key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        example = json.load(f)
                    
                    # Add to memory cache
                    self._add_to_memory_cache(key, example)
                    
                    return example
                except Exception as e:
                    logger.warning(f"Error loading from disk cache: {str(e)}")
        
        return None
    
    def put(self, key: str, example: Dict[str, Any]):
        """
        Put an example in the cache.
        
        Args:
            key: Cache key
            example: Example to cache
        """
        # Add to memory cache
        self._add_to_memory_cache(key, example)
        
        # Add to disk cache if enabled
        if self.use_disk_cache and self.cache_dir:
            cache_file = self._get_cache_file(key)
            try:
                with open(cache_file, 'w') as f:
                    json.dump(example, f)
            except Exception as e:
                logger.warning(f"Error saving to disk cache: {str(e)}")
    
    def _add_to_memory_cache(self, key: str, example: Dict[str, Any]):
        """
        Add an example to the memory cache.
        
        Args:
            key: Cache key
            example: Example to cache
        """
        # If cache is full, remove a random entry
        if len(self.memory_cache) >= self.max_cache_size:
            random_key = random.choice(list(self.memory_cache.keys()))
            del self.memory_cache[random_key]
        
        # Add to memory cache
        self.memory_cache[key] = example
    
    def _get_cache_file(self, key: str) -> str:
        """
        Get the cache file path for a key.
        
        Args:
            key: Cache key
            
        Returns:
            Cache file path
        """
        # Hash the key to get a filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")
    
    def clear(self):
        """Clear the cache."""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache if enabled
        if self.use_disk_cache and self.cache_dir and os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        logger.warning(f"Error removing cache file: {str(e)}")

def get_generation_parameters(
    task_id: str,
    task_params: Dict[str, Dict[str, Any]],
    default_params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get generation parameters for a task.
    
    Args:
        task_id: Task ID
        task_params: Dictionary mapping task IDs to generation parameters
        default_params: Default generation parameters
        
    Returns:
        Generation parameters for the task
    """
    # Get task-specific parameters if available
    if task_id in task_params:
        # Start with default parameters
        params = default_params.copy()
        # Update with task-specific parameters
        params.update(task_params[task_id])
        return params
    
    # Return default parameters
    return default_params.copy()

def create_generation_key(prompt: str, params: Dict[str, Any]) -> str:
    """
    Create a cache key for generation.
    
    Args:
        prompt: Generation prompt
        params: Generation parameters
        
    Returns:
        Cache key
    """
    # Create a string representation of the parameters
    params_str = json.dumps(params, sort_keys=True)
    
    # Combine prompt and parameters
    key_str = f"{prompt}|{params_str}"
    
    # Hash the key string
    return hashlib.md5(key_str.encode()).hexdigest()

def batch_generate_examples(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int = 4,
    **generation_kwargs
) -> List[str]:
    """
    Generate examples in batches.
    
    Args:
        model: Model for generation
        tokenizer: Tokenizer for generation
        prompts: List of prompts
        batch_size: Batch size for generation
        **generation_kwargs: Additional arguments for generation
        
    Returns:
        List of generated texts
    """
    generated_texts = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        try:
            # Tokenize prompts
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                **generation_kwargs
            )
            
            # Decode outputs
            batch_texts = tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            generated_texts.extend(batch_texts)
            
        except Exception as e:
            logger.warning(f"Error in batch generation: {str(e)}")
            
            # Fall back to individual generation
            for prompt in batch_prompts:
                try:
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    
                    outputs = model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    
                    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_texts.append(text)
                    
                except Exception as e2:
                    logger.warning(f"Error in individual generation: {str(e2)}")
                    # Add empty string as fallback
                    generated_texts.append("")
    
    return generated_texts

def adaptive_temperature_sampling(
    base_temperature: float,
    diversity_factor: float,
    iteration: int,
    max_iterations: int
) -> float:
    """
    Adaptive temperature sampling for generation.
    
    This function adjusts the temperature based on the iteration
    to balance quality and diversity.
    
    Args:
        base_temperature: Base temperature
        diversity_factor: Factor to increase diversity
        iteration: Current iteration
        max_iterations: Maximum number of iterations
        
    Returns:
        Adjusted temperature
    """
    # Start with lower temperature for quality
    if iteration < max_iterations * 0.3:
        return base_temperature * 0.8
    
    # Increase temperature for diversity
    if iteration > max_iterations * 0.7:
        return base_temperature * (1.0 + diversity_factor)
    
    # Use base temperature for middle iterations
    return base_temperature
