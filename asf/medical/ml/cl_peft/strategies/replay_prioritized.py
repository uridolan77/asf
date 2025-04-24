"""
Prioritized Experience Replay strategy for CL-PEFT.

This module provides the PrioritizedExperienceReplay class for storing and replaying
examples from previous tasks based on their importance.
"""

import torch
import random
import heapq
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import copy

from peft import PeftModel
from torch.utils.data import Dataset, Subset

from asf.medical.core.logging_config import get_logger
from .replay_experience import ExperienceReplay

logger = get_logger(__name__)

class PrioritizedExperienceReplay(ExperienceReplay):
    """
    Prioritized Experience Replay strategy for CL-PEFT.
    
    This strategy extends Experience Replay by prioritizing examples based on
    their importance, which can be measured by loss, gradient magnitude, etc.
    """
    
    def __init__(
        self,
        model: PeftModel,
        buffer_size: int = 1000,
        replay_ratio: float = 0.3,
        replay_frequency: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: bool = True,
        beta_annealing_steps: int = 1000,
        **kwargs
    ):
        """
        Initialize the Prioritized Experience Replay strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            buffer_size: Maximum number of examples to store in the replay buffer
            replay_ratio: Ratio of replay examples to current task examples
            replay_frequency: How often to perform replay (every N batches)
            alpha: Exponent for prioritization (0 = uniform, 1 = fully prioritized)
            beta: Exponent for importance sampling correction (0 = no correction, 1 = full correction)
            beta_annealing: Whether to anneal beta from its initial value to 1
            beta_annealing_steps: Number of steps for beta annealing
            **kwargs: Additional parameters
        """
        super().__init__(
            model=model,
            buffer_size=buffer_size,
            replay_ratio=replay_ratio,
            replay_frequency=replay_frequency,
            sampling_strategy="importance",
            **kwargs
        )
        self.alpha = alpha
        self.beta = beta
        self.initial_beta = beta
        self.beta_annealing = beta_annealing
        self.beta_annealing_steps = beta_annealing_steps
        
        # Step counter for beta annealing
        self.steps = 0
        
        # Priority sum tree for efficient sampling
        self.priority_trees = {}  # task_id -> SumTree
        
        # Maximum priority for new examples
        self.max_priority = 1.0
    
    def before_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Prepare for training on a new task.
        
        This method creates a mixed dataset with examples from the current task
        and the replay buffer, sampled based on priorities.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        # Reset step counter for beta annealing
        self.steps = 0
        
        # Call parent method
        super().before_training(task_id, train_dataset, **kwargs)
    
    def _importance_sampling(self, num_examples):
        """
        Sample examples based on their priorities.
        
        Args:
            num_examples: Number of examples to sample
            
        Returns:
            List of sampled examples
        """
        # Update beta if annealing is enabled
        if self.beta_annealing:
            self.beta = min(1.0, self.initial_beta + (1.0 - self.initial_beta) * (self.steps / self.beta_annealing_steps))
        
        # Flatten priorities and examples
        flat_priorities = []
        flat_examples = []
        flat_indices = []  # (task_id, idx) pairs
        
        for task_id, task_examples in self.replay_buffer.items():
            for idx, example in enumerate(task_examples):
                priority = 1.0  # Default priority
                
                # Get priority from importance scores
                if task_id in self.importance_scores and idx in self.importance_scores[task_id]:
                    priority = self.importance_scores[task_id][idx]
                
                # Apply alpha exponent for prioritization
                priority = priority ** self.alpha
                
                flat_priorities.append(priority)
                flat_examples.append(example)
                flat_indices.append((task_id, idx))
        
        # If no examples, return empty list
        if not flat_examples:
            return []
        
        # Convert priorities to probabilities
        total_priority = sum(flat_priorities)
        if total_priority > 0:
            probs = [priority / total_priority for priority in flat_priorities]
        else:
            # If all priorities are 0, use uniform probabilities
            probs = [1.0 / len(flat_priorities)] * len(flat_priorities)
        
        # Sample examples based on probabilities
        indices = np.random.choice(
            len(flat_examples),
            size=min(num_examples, len(flat_examples)),
            replace=False,
            p=probs
        )
        
        # Get sampled examples
        sampled_examples = []
        
        for i in indices:
            example = copy.deepcopy(flat_examples[i])
            
            # Compute importance sampling weight
            weight = (1.0 / (len(flat_examples) * probs[i])) ** self.beta
            
            # Add weight to example
            example['is_weight'] = weight
            
            # Add index to example for priority updates
            example['priority_idx'] = flat_indices[i]
            
            sampled_examples.append(example)
        
        # Normalize weights
        max_weight = max(example['is_weight'] for example in sampled_examples)
        for example in sampled_examples:
            example['is_weight'] /= max_weight
        
        return sampled_examples
    
    def on_batch_end(self, trainer, outputs, batch, **kwargs):
        """
        Called at the end of each batch during training.
        
        This method updates priorities for examples in the batch.
        
        Args:
            trainer: Trainer instance
            outputs: Batch outputs
            batch: Current batch
            **kwargs: Additional arguments
        """
        # Increment step counter for beta annealing
        self.steps += 1
        
        # Check if batch contains replay examples
        if not isinstance(batch, dict) or 'priority_idx' not in batch:
            return
        
        # Get losses for priority updates
        if hasattr(outputs, "loss"):
            losses = outputs.loss.detach()
            
            # If loss is a scalar, convert to a tensor with one element
            if losses.dim() == 0:
                losses = losses.unsqueeze(0)
        else:
            # If no loss in outputs, skip priority updates
            return
        
        # Get priority indices
        priority_indices = batch['priority_idx']
        
        # Update priorities
        for i, (task_id, idx) in enumerate(priority_indices):
            if i < len(losses):
                loss = losses[i].item()
                
                # Update importance score
                if task_id not in self.importance_scores:
                    self.importance_scores[task_id] = {}
                
                self.importance_scores[task_id][idx] = loss
                
                # Update max priority
                self.max_priority = max(self.max_priority, loss)
    
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.
        
        This method updates the replay buffer with examples from the current task
        and initializes their priorities.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        # Call parent method to update replay buffer
        super().after_training(task_id, **kwargs)
        
        # Initialize priorities for new examples
        if task_id in self.replay_buffer:
            if task_id not in self.importance_scores:
                self.importance_scores[task_id] = {}
            
            for idx in range(len(self.replay_buffer[task_id])):
                if idx not in self.importance_scores[task_id]:
                    # Initialize with max priority to encourage exploration
                    self.importance_scores[task_id][idx] = self.max_priority
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for examples.
        
        Args:
            indices: List of (task_id, idx) pairs
            priorities: New priorities
        """
        for (task_id, idx), priority in zip(indices, priorities):
            if task_id in self.importance_scores:
                self.importance_scores[task_id][idx] = priority
                
                # Update max priority
                self.max_priority = max(self.max_priority, priority)


class SumTree:
    """
    Sum Tree data structure for efficient sampling from prioritized replay buffer.
    
    A sum tree is a binary tree where each node is the sum of its children.
    Leaf nodes contain priorities, and internal nodes contain the sum of priorities
    in their subtree.
    """
    
    def __init__(self, capacity):
        """
        Initialize the sum tree.
        
        Args:
            capacity: Maximum number of elements in the tree
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.size = 0
        self.next_idx = 0
    
    def add(self, priority, data):
        """
        Add an element with priority to the tree.
        
        Args:
            priority: Priority of the element
            data: Data to store
        """
        # Get index in the tree
        idx = self.next_idx + self.capacity - 1
        
        # Store data
        self.data[self.next_idx] = data
        
        # Update tree
        self.update(idx, priority)
        
        # Update next index
        self.next_idx = (self.next_idx + 1) % self.capacity
        
        # Update size
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, idx, priority):
        """
        Update the priority of an element.
        
        Args:
            idx: Index in the tree
            priority: New priority
        """
        # Compute change in priority
        change = priority - self.tree[idx]
        
        # Update tree
        self.tree[idx] = priority
        
        # Propagate change up the tree
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change
    
    def get(self, s):
        """
        Get element with cumulative priority s.
        
        Args:
            s: Cumulative priority
            
        Returns:
            (idx, priority, data) tuple
        """
        # Start from the root
        idx = 0
        
        # Traverse the tree
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            
            # If s is less than the left child's priority, go left
            if s <= self.tree[left]:
                idx = left
            else:
                # Otherwise, go right and subtract left child's priority
                s -= self.tree[left]
                idx = right
        
        # Get data index
        data_idx = idx - (self.capacity - 1)
        
        return (idx, self.tree[idx], self.data[data_idx])
    
    def total(self):
        """
        Get the total priority in the tree.
        
        Returns:
            Total priority
        """
        return self.tree[0]
    
    def __len__(self):
        """
        Get the number of elements in the tree.
        
        Returns:
            Number of elements
        """
        return self.size
