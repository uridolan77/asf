"""
Experience Replay strategy for CL-PEFT.

This module provides the ExperienceReplay class for storing and replaying
examples from previous tasks.
"""

import torch
import random
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import copy

from peft import PeftModel
from torch.utils.data import Dataset, Subset

from asf.medical.core.logging_config import get_logger
from .replay_base import ReplayStrategy, MixedDataset

logger = get_logger(__name__)

class ExperienceReplay(ReplayStrategy):
    """
    Experience Replay strategy for CL-PEFT.
    
    This strategy maintains a buffer of examples from previous tasks and
    mixes them with current task examples during training.
    """
    
    def __init__(
        self,
        model: PeftModel,
        buffer_size: int = 1000,
        replay_ratio: float = 0.3,
        replay_frequency: int = 10,
        sampling_strategy: str = "uniform",
        **kwargs
    ):
        """
        Initialize the Experience Replay strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            buffer_size: Maximum number of examples to store in the replay buffer
            replay_ratio: Ratio of replay examples to current task examples
            replay_frequency: How often to perform replay (every N batches)
            sampling_strategy: Strategy for sampling examples from the buffer
                              ("uniform", "importance", "class_balanced")
            **kwargs: Additional parameters
        """
        super().__init__(model, replay_ratio, replay_frequency, **kwargs)
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        
        # Replay buffer: task_id -> list of examples
        self.replay_buffer = {}
        
        # For importance sampling
        self.importance_scores = {}  # task_id -> {example_idx -> score}
        
        # For class-balanced sampling
        self.class_distribution = {}  # task_id -> {class -> count}
    
    def before_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Prepare for training on a new task.
        
        This method creates a mixed dataset with examples from the current task
        and the replay buffer.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        super().before_training(task_id, train_dataset, **kwargs)
        
        # If there are no previous tasks, no replay needed
        if not self.replay_buffer:
            logger.info(f"No previous tasks in replay buffer for task {task_id}")
            return
        
        logger.info(f"Preparing mixed dataset with replay for task {task_id}")
        
        # Create a mixed dataset with examples from the replay buffer
        replay_examples = self._sample_from_buffer()
        mixed_dataset = self._create_mixed_dataset(train_dataset, replay_examples)
        
        # Update the train_dataset in kwargs
        kwargs['train_dataset'] = mixed_dataset
    
    def _sample_from_buffer(self):
        """
        Sample examples from the replay buffer.
        
        Returns:
            List of examples from the replay buffer
        """
        # Flatten the replay buffer into a single list of examples
        all_examples = []
        for task_id, task_examples in self.replay_buffer.items():
            all_examples.extend(task_examples)
        
        # If replay buffer is empty, return empty list
        if not all_examples:
            return []
        
        # Determine how many examples to sample
        num_examples = min(len(all_examples), self.buffer_size)
        
        # Sample examples based on the sampling strategy
        if self.sampling_strategy == "importance" and self.importance_scores:
            return self._importance_sampling(num_examples)
        elif self.sampling_strategy == "class_balanced" and self.class_distribution:
            return self._class_balanced_sampling(num_examples)
        else:
            # Default to uniform sampling
            return random.sample(all_examples, num_examples)
    
    def _importance_sampling(self, num_examples):
        """
        Sample examples based on their importance scores.
        
        Args:
            num_examples: Number of examples to sample
            
        Returns:
            List of sampled examples
        """
        # Flatten importance scores and examples
        flat_scores = []
        flat_examples = []
        
        for task_id, task_examples in self.replay_buffer.items():
            for idx, example in enumerate(task_examples):
                if task_id in self.importance_scores and idx in self.importance_scores[task_id]:
                    score = self.importance_scores[task_id][idx]
                    flat_scores.append(score)
                    flat_examples.append(example)
                else:
                    # If no importance score, use a default value
                    flat_scores.append(1.0)
                    flat_examples.append(example)
        
        # If no examples with scores, return empty list
        if not flat_examples:
            return []
        
        # Convert scores to probabilities
        total_score = sum(flat_scores)
        if total_score > 0:
            probs = [score / total_score for score in flat_scores]
        else:
            # If all scores are 0, use uniform probabilities
            probs = [1.0 / len(flat_scores)] * len(flat_scores)
        
        # Sample examples based on probabilities
        indices = np.random.choice(
            len(flat_examples),
            size=min(num_examples, len(flat_examples)),
            replace=False,
            p=probs
        )
        
        return [flat_examples[i] for i in indices]
    
    def _class_balanced_sampling(self, num_examples):
        """
        Sample examples to maintain class balance.
        
        Args:
            num_examples: Number of examples to sample
            
        Returns:
            List of sampled examples
        """
        # Get all classes
        all_classes = set()
        for task_classes in self.class_distribution.values():
            all_classes.update(task_classes.keys())
        
        # If no classes, return empty list
        if not all_classes:
            return []
        
        # Determine examples per class
        examples_per_class = num_examples // len(all_classes)
        if examples_per_class == 0:
            examples_per_class = 1
        
        # Sample examples for each class
        sampled_examples = []
        
        for cls in all_classes:
            # Get examples for this class
            class_examples = []
            
            for task_id, task_examples in self.replay_buffer.items():
                for example in task_examples:
                    # Get class from example
                    example_class = example.get('label', example.get('class', None))
                    
                    if example_class == cls:
                        class_examples.append(example)
            
            # Sample examples for this class
            if class_examples:
                sampled_examples.extend(
                    random.sample(
                        class_examples,
                        min(examples_per_class, len(class_examples))
                    )
                )
        
        # If we need more examples, sample randomly
        if len(sampled_examples) < num_examples:
            # Get all examples not already sampled
            remaining_examples = []
            
            for task_id, task_examples in self.replay_buffer.items():
                for example in task_examples:
                    if example not in sampled_examples:
                        remaining_examples.append(example)
            
            # Sample additional examples
            if remaining_examples:
                additional_examples = random.sample(
                    remaining_examples,
                    min(num_examples - len(sampled_examples), len(remaining_examples))
                )
                
                sampled_examples.extend(additional_examples)
        
        return sampled_examples
    
    def get_replay_batch(self, batch_size):
        """
        Get a batch of examples for replay.
        
        Args:
            batch_size: Size of the current batch
            
        Returns:
            List of examples for replay
        """
        # Sample from the replay buffer
        replay_examples = self._sample_from_buffer()
        
        if not replay_examples:
            return []
        
        # Determine replay batch size
        replay_size = int(batch_size * self.replay_ratio)
        replay_size = min(replay_size, len(replay_examples))
        
        if replay_size == 0:
            return []
        
        # Sample examples for replay
        return random.sample(replay_examples, replay_size)
    
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.
        
        This method updates the replay buffer with examples from the current task.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        if self.current_dataset is None:
            logger.warning(f"No dataset available to update replay buffer for task {task_id}")
            return
        
        logger.info(f"Updating replay buffer with examples from task {task_id}")
        
        # Sample examples from the current dataset for the replay buffer
        examples_per_task = self.buffer_size // max(1, len(self.replay_buffer) + 1)
        
        # Sample examples from the current dataset
        indices = random.sample(
            range(len(self.current_dataset)),
            min(examples_per_task, len(self.current_dataset))
        )
        
        # Initialize replay buffer for this task
        self.replay_buffer[task_id] = []
        
        # Initialize importance scores for this task
        self.importance_scores[task_id] = {}
        
        # Store examples in the replay buffer
        for i, idx in enumerate(indices):
            # Get example from dataset
            example = self.current_dataset[idx]
            
            # Convert to dict if not already
            if not isinstance(example, dict):
                if hasattr(example, "__dict__"):
                    example = example.__dict__
                else:
                    # Try to convert to dict
                    try:
                        example = dict(example)
                    except:
                        # If conversion fails, wrap in a dict
                        example = {"data": example}
            
            # Add task ID to the example
            example['task_id'] = task_id
            
            # Add to replay buffer
            self.replay_buffer[task_id].append(example)
            
            # Update class distribution if label is available
            cls = example.get('label', example.get('class', None))
            if cls is not None:
                if task_id not in self.class_distribution:
                    self.class_distribution[task_id] = {}
                self.class_distribution[task_id][cls] = self.class_distribution[task_id].get(cls, 0) + 1
        
        # If the buffer is too large, reduce the number of examples per task
        if sum(len(examples) for examples in self.replay_buffer.values()) > self.buffer_size:
            self._resize_buffer()
        
        logger.info(f"Replay buffer updated: {sum(len(examples) for examples in self.replay_buffer.values())} examples total")
    
    def _resize_buffer(self):
        """
        Resize the replay buffer to stay within the buffer size limit.
        """
        # Calculate new examples per task
        examples_per_task = self.buffer_size // len(self.replay_buffer)
        
        # Resize each task's examples
        for task_id in self.replay_buffer:
            if len(self.replay_buffer[task_id]) > examples_per_task:
                # Sample examples to keep
                indices = random.sample(
                    range(len(self.replay_buffer[task_id])),
                    examples_per_task
                )
                
                # Keep only the sampled examples
                self.replay_buffer[task_id] = [self.replay_buffer[task_id][i] for i in indices]
                
                # Update importance scores
                if task_id in self.importance_scores:
                    new_scores = {}
                    for i, idx in enumerate(indices):
                        if idx in self.importance_scores[task_id]:
                            new_scores[i] = self.importance_scores[task_id][idx]
                    self.importance_scores[task_id] = new_scores
    
    def update_importance_scores(self, task_id, example_indices, scores):
        """
        Update importance scores for examples.
        
        Args:
            task_id: Task identifier
            example_indices: Indices of examples to update
            scores: New importance scores
        """
        if task_id not in self.importance_scores:
            self.importance_scores[task_id] = {}
        
        for idx, score in zip(example_indices, scores):
            self.importance_scores[task_id][idx] = score
    
    def get_buffer_statistics(self):
        """
        Get statistics about the replay buffer.
        
        Returns:
            Dictionary with buffer statistics
        """
        stats = {
            "total_examples": sum(len(examples) for examples in self.replay_buffer.values()),
            "examples_per_task": {task_id: len(examples) for task_id, examples in self.replay_buffer.items()},
            "class_distribution": self.class_distribution
        }
        
        return stats
