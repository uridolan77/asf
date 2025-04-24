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
        max_cache_size: int = 10000,
        use_disk_cache: bool = True,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the generation cache.
        
        Args:
            max_cache_size: Maximum number of examples to store in memory
            use_disk_cache: Whether to use disk cache
            cache_dir: Directory for disk cache
        """
        self.max_cache_size = max_cache_size
        self.use_disk_cache = use_disk_cache
        self.cache_dir = cache_dir
        
        # Memory cache
        self.memory_cache = {}
        
        # Create cache directory if needed
        if self.use_disk_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
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
        """
        Clear the cache.
        """
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache if enabled
        if self.use_disk_cache and self.cache_dir and os.path.exists(self.cache_dir):
            for file in os.listdir(self.cache_dir):
                try:
                    os.remove(os.path.join(self.cache_dir, file))
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
    if task_params and task_id in task_params:
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
    generation_params: Dict[str, Any],
    batch_size: int = 4
) -> List[str]:
    """
    Generate examples in batches.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for generation
        prompts: List of prompts
        generation_params: Generation parameters
        batch_size: Batch size for generation
        
    Returns:
        List of generated texts
    """
    generated_texts = []
    
    # Process prompts in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize prompts
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        try:
            # Generate text
            outputs = model.generate(
                **inputs,
                **generation_params,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode the generated text
            for j, output in enumerate(outputs):
                text = tokenizer.decode(output, skip_special_tokens=True)
                
                # Remove the prompt from the generated text
                prompt_text = batch_prompts[j]
                if text.startswith(prompt_text):
                    text = text[len(prompt_text):].strip()
                
                generated_texts.append(text)
        
        except Exception as e:
            logger.warning(f"Error generating batch: {str(e)}")
            # Add empty strings as fallback
            generated_texts.extend([""] * len(batch_prompts))
    
    return generated_texts


def adaptive_temperature_sampling(
    base_temperature: float,
    diversity_factor: float,
    current_index: int,
    total_samples: int
) -> float:
    """
    Adaptive temperature sampling for diverse generation.
    
    This function increases the temperature as generation progresses
    to encourage diversity in the generated examples.
    
    Args:
        base_temperature: Base temperature value
        diversity_factor: Factor to control diversity (0 to 1)
        current_index: Current sample index
        total_samples: Total number of samples
        
    Returns:
        Adapted temperature value
    """
    # Compute progress (0 to 1)
    progress = current_index / max(1, total_samples - 1)
    
    # Increase temperature based on progress and diversity factor
    temperature = base_temperature * (1.0 + diversity_factor * progress)
    
    return temperature
