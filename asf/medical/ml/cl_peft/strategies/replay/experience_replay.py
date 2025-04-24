"""
Experience Replay strategy for continual learning.

This module provides the ExperienceReplay class for storing and replaying
examples from previous tasks.
"""

import random
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from peft import PeftModel
from torch.utils.data import Dataset, Subset

from asf.medical.core.logging_config import get_logger
from .replay_base import ReplayStrategy

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
        
        # Example importance scores: task_id -> {example_id -> importance}
        self.importance_scores = {}
        
        # Class distribution: task_id -> {class_label -> count}
        self.class_distribution = {}
    
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
        Sample examples based on importance scores.
        
        Args:
            num_examples: Number of examples to sample
            
        Returns:
            List of sampled examples
        """
        # Flatten the replay buffer and importance scores
        examples = []
        scores = []
        
        for task_id, task_examples in self.replay_buffer.items():
            task_scores = self.importance_scores.get(task_id, {})
            for i, example in enumerate(task_examples):
                examples.append(example)
                # Use example ID if available, otherwise use index
                example_id = example.get('id', str(i))
                scores.append(task_scores.get(example_id, 1.0))
        
        # Normalize scores to probabilities
        if sum(scores) > 0:
            probs = [s / sum(scores) for s in scores]
        else:
            # If all scores are 0, use uniform sampling
            probs = [1.0 / len(scores)] * len(scores)
        
        # Sample examples based on importance scores
        indices = np.random.choice(
            len(examples),
            size=min(num_examples, len(examples)),
            replace=False,
            p=probs
        )
        
        return [examples[i] for i in indices]
    
    def _class_balanced_sampling(self, num_examples):
        """
        Sample examples to maintain class balance.
        
        Args:
            num_examples: Number of examples to sample
            
        Returns:
            List of sampled examples
        """
        # Get all classes across all tasks
        all_classes = set()
        for class_dist in self.class_distribution.values():
            all_classes.update(class_dist.keys())
        
        if not all_classes:
            # Fall back to uniform sampling if no class information
            return self._sample_from_buffer()
        
        # Group examples by class
        class_examples = {cls: [] for cls in all_classes}
        
        for task_id, task_examples in self.replay_buffer.items():
            for example in task_examples:
                cls = example.get('label', example.get('class', None))
                if cls is not None and cls in class_examples:
                    class_examples[cls].append(example)
        
        # Sample examples from each class
        sampled_examples = []
        examples_per_class = num_examples // len(all_classes)
        
        for cls, examples in class_examples.items():
            if examples:
                # Sample examples for this class
                n_samples = min(examples_per_class, len(examples))
                sampled_examples.extend(random.sample(examples, n_samples))
        
        # If we need more examples, sample randomly from the remaining
        if len(sampled_examples) < num_examples:
            # Get all examples not already sampled
            remaining = []
            for task_examples in self.replay_buffer.values():
                for example in task_examples:
                    if example not in sampled_examples:
                        remaining.append(example)
            
            # Sample additional examples
            additional = min(num_examples - len(sampled_examples), len(remaining))
            if additional > 0:
                sampled_examples.extend(random.sample(remaining, additional))
        
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
        
        # Store examples in the replay buffer
        self.replay_buffer[task_id] = []
        for i in indices:
            example = self.current_dataset[i]
            
            # Convert to dict if not already
            if not isinstance(example, dict):
                if hasattr(example, '__dict__'):
                    example = example.__dict__
                else:
                    # Try to convert to dict based on common formats
                    try:
                        if hasattr(example, 'input_ids'):
                            example = {
                                'input_ids': example.input_ids,
                                'attention_mask': getattr(example, 'attention_mask', None),
                                'labels': getattr(example, 'labels', None)
                            }
                        elif isinstance(example, tuple) and len(example) == 2:
                            example = {'input': example[0], 'label': example[1]}
                        else:
                            logger.warning(f"Could not convert example to dict: {type(example)}")
                            continue
                    except Exception as e:
                        logger.warning(f"Error converting example to dict: {str(e)}")
                        continue
            
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
                self.replay_buffer[task_id] = random.sample(
                    self.replay_buffer[task_id],
                    examples_per_task
                )
                
                # Update class distribution
                if task_id in self.class_distribution:
                    self.class_distribution[task_id] = {}
                    for example in self.replay_buffer[task_id]:
                        cls = example.get('label', example.get('class', None))
                        if cls is not None:
                            self.class_distribution[task_id][cls] = self.class_distribution[task_id].get(cls, 0) + 1
    
    def update_importance_scores(self, task_id: str, scores: Dict[str, float]):
        """
        Update importance scores for examples.
        
        Args:
            task_id: Task ID
            scores: Dictionary mapping example IDs to importance scores
        """
        if task_id not in self.importance_scores:
            self.importance_scores[task_id] = {}
        
        self.importance_scores[task_id].update(scores)
