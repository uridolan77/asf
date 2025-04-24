"""
Generative Replay strategy for CL-PEFT.

This module provides the GenerativeReplay class for generating and replaying
synthetic examples from previous tasks.
"""

import torch
import random
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import time
import os

from peft import PeftModel
from torch.utils.data import Dataset

from asf.medical.core.logging_config import get_logger
from .replay_base import ReplayStrategy, MixedDataset
from .replay_quality import QualityController
from .replay_utils import (
    GenerationCache,
    get_generation_parameters,
    create_generation_key,
    batch_generate_examples,
    adaptive_temperature_sampling
)

logger = get_logger(__name__)

class GenerativeReplay(ReplayStrategy):
    """
    Generative Replay strategy for CL-PEFT.
    
    This strategy uses the model itself to generate synthetic examples from previous tasks
    and mixes them with current task examples during training.
    """
    
    def __init__(
        self,
        model: PeftModel,
        task_prompts: Dict[str, str],
        examples_per_task: int = 100,
        replay_ratio: float = 0.3,
        replay_frequency: int = 10,
        quality_threshold: float = 0.7,
        diversity_threshold: float = 0.8,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        generation_batch_size: int = 4,
        task_generation_params: Optional[Dict[str, Dict[str, Any]]] = None,
        embedding_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Generative Replay strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            task_prompts: Dictionary mapping task IDs to prompts for generating examples
            examples_per_task: Number of examples to generate per previous task
            replay_ratio: Ratio of replay examples to current task examples
            replay_frequency: How often to perform replay (every N batches)
            quality_threshold: Threshold for quality filtering
            diversity_threshold: Threshold for diversity filtering
            use_cache: Whether to cache generated examples
            cache_dir: Directory for caching generated examples
            generation_batch_size: Batch size for generation
            task_generation_params: Task-specific generation parameters
            embedding_model_name: Name of the model to use for embeddings
            **kwargs: Additional parameters
        """
        super().__init__(model, replay_ratio, replay_frequency, **kwargs)
        self.task_prompts = task_prompts
        self.examples_per_task = examples_per_task
        self.generation_batch_size = generation_batch_size
        self.task_generation_params = task_generation_params or {}
        
        # Default generation parameters
        self.default_generation_params = {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": 1
        }
        
        # Generated examples: task_id -> list of examples
        self.generated_examples = {}
        
        # Tokenizer for generating examples
        self.tokenizer = kwargs.get('tokenizer', None)
        
        # Quality controller
        self.quality_controller = QualityController(
            quality_threshold=quality_threshold,
            diversity_threshold=diversity_threshold,
            embedding_model_name=embedding_model_name
        )
        
        # Generation cache
        self.use_cache = use_cache
        self.cache = None
        
        if self.use_cache:
            if cache_dir is None:
                cache_dir = os.path.join(os.getcwd(), "generative_replay_cache")
            
            self.cache = GenerationCache(
                max_cache_size=10000,
                use_disk_cache=True,
                cache_dir=cache_dir
            )
    
    def before_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Prepare for training on a new task.
        
        This method generates synthetic examples from previous tasks and
        creates a mixed dataset with examples from the current task.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        super().before_training(task_id, train_dataset, **kwargs)
        
        # If there are no previous tasks, no replay needed
        if not self.task_history:
            logger.info(f"No previous tasks for generative replay for task {task_id}")
            return
        
        # Check if tokenizer is available
        if self.tokenizer is None and 'tokenizer' in kwargs:
            self.tokenizer = kwargs['tokenizer']
        
        if self.tokenizer is None:
            logger.warning("No tokenizer provided for Generative Replay, cannot generate examples")
            return
        
        logger.info(f"Generating synthetic examples for previous tasks for task {task_id}")
        
        # Generate synthetic examples for previous tasks
        self._generate_examples_with_quality_control()
        
        # Create a mixed dataset with synthetic examples
        replay_examples = self._get_all_generated_examples()
        mixed_dataset = self._create_mixed_dataset(train_dataset, replay_examples)
        
        # Update the train_dataset in kwargs
        kwargs['train_dataset'] = mixed_dataset
    
    def _generate_examples_with_quality_control(self):
        """
        Generate examples with quality control.
        
        This method generates synthetic examples for previous tasks,
        applies quality control, and stores the examples.
        """
        # Skip the current task
        previous_tasks = [task for task in self.task_history 
                         if task != self.current_task_id]
        
        for task_id in previous_tasks:
            # Skip if no prompt is available for this task
            if task_id not in self.task_prompts:
                logger.warning(f"No prompt available for task {task_id}, skipping generation")
                continue
            
            # Get the prompt for this task
            prompt = self.task_prompts[task_id]
            
            # Get generation parameters for this task
            gen_params = get_generation_parameters(
                task_id,
                self.task_generation_params,
                self.default_generation_params
            )
            
            # Generate more examples than needed to allow for filtering
            num_to_generate = self.examples_per_task * 2
            
            logger.info(f"Generating {num_to_generate} examples for task {task_id} (before filtering)")
            
            # Generate examples
            generated_texts = self._generate_texts(prompt, num_to_generate, gen_params)
            
            # Create examples from generated texts
            examples = []
            for text in generated_texts:
                if not text:  # Skip empty texts
                    continue
                
                example = {
                    "text": text,
                    "task_id": task_id,
                    "is_generated": True,
                    "generation_params": gen_params
                }
                
                examples.append(example)
            
            # Apply quality control
            filtered_examples = self.quality_controller.filter_examples(examples)
            
            # Limit to the desired number of examples
            filtered_examples = filtered_examples[:self.examples_per_task]
            
            logger.info(f"Generated {len(filtered_examples)} examples for task {task_id} after filtering")
            
            # Store the generated examples
            self.generated_examples[task_id] = filtered_examples
    
    def _generate_texts(self, prompt: str, num_examples: int, gen_params: Dict[str, Any]) -> List[str]:
        """
        Generate texts for a prompt.
        
        Args:
            prompt: Generation prompt
            num_examples: Number of examples to generate
            gen_params: Generation parameters
            
        Returns:
            List of generated texts
        """
        generated_texts = []
        
        # Check if batch generation is enabled
        if self.generation_batch_size > 1:
            # Create prompts for batch generation
            prompts = [prompt] * num_examples
            
            # Generate in batches
            batch_texts = batch_generate_examples(
                self.model,
                self.tokenizer,
                prompts,
                gen_params,
                self.generation_batch_size
            )
            
            generated_texts.extend(batch_texts)
        else:
            # Generate one by one
            for i in range(num_examples):
                # Use adaptive temperature sampling
                temp_params = gen_params.copy()
                temp_params['temperature'] = adaptive_temperature_sampling(
                    gen_params.get('temperature', 0.7),
                    0.3,  # diversity factor
                    i,
                    num_examples
                )
                
                # Create cache key if cache is enabled
                cache_key = None
                if self.cache:
                    cache_key = create_generation_key(prompt, temp_params)
                    cached_example = self.cache.get(cache_key)
                    if cached_example:
                        generated_texts.append(cached_example.get('text', ''))
                        continue
                
                try:
                    # Generate text
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    outputs = self.model.generate(
                        **inputs,
                        **temp_params,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    # Decode the generated text
                    text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Remove the prompt from the generated text
                    if text.startswith(prompt):
                        text = text[len(prompt):].strip()
                    
                    generated_texts.append(text)
                    
                    # Cache the generated example
                    if self.cache and cache_key:
                        self.cache.put(cache_key, {'text': text, 'task_id': self.current_task_id})
                    
                except Exception as e:
                    logger.warning(f"Error generating text: {str(e)}")
                    # Add empty string as fallback
                    generated_texts.append("")
        
        return generated_texts
    
    def _get_all_generated_examples(self):
        """
        Get all generated examples.
        
        Returns:
            List of all generated examples
        """
        all_examples = []
        for task_examples in self.generated_examples.values():
            all_examples.extend(task_examples)
        
        return all_examples
    
    def get_replay_batch(self, batch_size):
        """
        Get a batch of examples for replay.
        
        Args:
            batch_size: Size of the current batch
            
        Returns:
            List of examples for replay
        """
        # Get all generated examples
        all_examples = self._get_all_generated_examples()
        
        if not all_examples:
            return []
        
        # Determine replay batch size
        replay_size = int(batch_size * self.replay_ratio)
        replay_size = min(replay_size, len(all_examples))
        
        if replay_size == 0:
            return []
        
        # Sample examples for replay
        return random.sample(all_examples, replay_size)
    
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.
        
        For Generative Replay, this is mainly a bookkeeping operation.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        logger.info(f"Completed training on task {task_id} with Generative Replay")
    
    def get_generation_statistics(self):
        """
        Get statistics about generated examples.
        
        Returns:
            Dictionary with generation statistics
        """
        stats = {
            "total_examples": sum(len(examples) for examples in self.generated_examples.values()),
            "examples_per_task": {task_id: len(examples) for task_id, examples in self.generated_examples.items()},
            "quality_scores": {}
        }
        
        # Collect quality scores
        for task_id, examples in self.generated_examples.items():
            quality_scores = [example.get("quality_score", 0.0) for example in examples]
            if quality_scores:
                stats["quality_scores"][task_id] = {
                    "mean": np.mean(quality_scores),
                    "min": np.min(quality_scores),
                    "max": np.max(quality_scores)
                }
        
        return stats
