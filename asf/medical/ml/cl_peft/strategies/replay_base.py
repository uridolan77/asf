"""
Base Replay Strategy for CL-PEFT.

This module provides the base class for replay-based strategies for mitigating
catastrophic forgetting in sequential fine-tuning of LLMs with PEFT.
"""

import torch
import random
from typing import Dict, List, Optional, Any, Union, Tuple
import copy
import numpy as np

from peft import PeftModel
from transformers import Trainer
from torch.utils.data import Dataset, ConcatDataset, Subset

from asf.medical.core.logging_config import get_logger
from .base import CLStrategy

logger = get_logger(__name__)

class ReplayStrategy(CLStrategy):
    """
    Base class for replay-based strategies for CL-PEFT.
    
    Replay strategies mitigate catastrophic forgetting by replaying examples
    from previous tasks during training on new tasks.
    """
    
    def __init__(
        self,
        model: PeftModel,
        replay_ratio: float = 0.3,
        replay_frequency: int = 10,
        **kwargs
    ):
        """
        Initialize the replay strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            replay_ratio: Ratio of replay examples to current task examples
            replay_frequency: How often to perform replay (every N batches)
            **kwargs: Additional parameters
        """
        super().__init__(model, **kwargs)
        self.replay_ratio = replay_ratio
        self.replay_frequency = replay_frequency
        
        # Current dataset being used for training
        self.current_dataset = None
        
        # Batch counter for replay frequency
        self.batch_counter = 0
    
    def before_training(self, task_id: str, train_dataset=None, **kwargs):
        """
        Prepare for training on a new task.
        
        This method should be implemented by subclasses to prepare for training
        on a new task, such as creating a mixed dataset with replay examples.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        
        if train_dataset is not None:
            self.current_dataset = train_dataset
    
    def _create_mixed_dataset(self, current_dataset, replay_examples):
        """
        Create a mixed dataset with examples from the current task and replay examples.
        
        Args:
            current_dataset: Dataset for the current task
            replay_examples: List of examples from replay buffer or generation
            
        Returns:
            Mixed dataset
        """
        # If no replay examples, return the current dataset
        if not replay_examples:
            return current_dataset
        
        # Create a custom mixed dataset
        return MixedDataset(current_dataset, replay_examples)
    
    def on_batch_start(self, trainer, batch, **kwargs):
        """
        Called at the start of each batch during training.
        
        This method implements the replay mechanism by injecting replay
        examples into the current batch.
        
        Args:
            trainer: Trainer instance
            batch: Current batch
            **kwargs: Additional arguments
            
        Returns:
            Modified batch
        """
        # Increment batch counter
        self.batch_counter += 1
        
        # Check if replay should be performed
        if self.batch_counter % self.replay_frequency != 0:
            return batch
        
        # Get replay batch (to be implemented by subclasses)
        replay_batch = self.get_replay_batch(len(batch))
        
        # If no replay batch, return original batch
        if not replay_batch:
            return batch
        
        # Combine current batch with replay batch
        return self._combine_batches(batch, replay_batch)
    
    def _combine_batches(self, current_batch, replay_batch):
        """
        Combine current batch with replay batch.
        
        Args:
            current_batch: Current batch
            replay_batch: Replay batch
            
        Returns:
            Combined batch
        """
        # Handle different batch formats
        if isinstance(current_batch, dict):
            # Dictionary batch (most common in HF)
            combined_batch = {}
            for key in current_batch.keys():
                if key in replay_batch[0]:
                    current_tensors = current_batch[key]
                    
                    # For text inputs, we need to tokenize them first
                    if key == "text" or key == "input_ids":
                        # If already tokenized or no tokenizer
                        replay_tensors = [torch.tensor(example[key]) for example in replay_batch]
                        replay_inputs = torch.stack(replay_tensors).to(current_tensors.device)
                        
                        # Combine with current batch
                        combined_batch[key] = torch.cat([current_tensors, replay_inputs], dim=0)
                    else:
                        # Handle other tensor types
                        if isinstance(current_tensors, torch.Tensor):
                            # Try to convert to tensors and stack
                            try:
                                replay_tensors = [torch.tensor(example[key]).to(current_tensors.device)
                                                 for example in replay_batch]
                                replay_tensors = torch.stack(replay_tensors)
                                combined_batch[key] = torch.cat([current_tensors, replay_tensors], dim=0)
                            except:
                                # Fallback if tensor conversion fails
                                logger.warning(f"Could not combine tensors for key {key}")
                                combined_batch[key] = current_tensors
                        else:
                            # For non-tensor data
                            combined_batch[key] = current_tensors + [example[key] for example in replay_batch]
                else:
                    # Key not in replay batch, keep original
                    combined_batch[key] = current_batch[key]
            
            return combined_batch
        else:
            # For non-dictionary batches, this needs to be customized based on the expected format
            logger.warning(f"Unsupported batch format for replay: {type(current_batch)}")
            return current_batch
    
    def get_replay_batch(self, batch_size):
        """
        Get a batch of examples for replay.
        
        This method should be implemented by subclasses to get a batch
        of examples for replay.
        
        Args:
            batch_size: Size of the current batch
            
        Returns:
            List of examples for replay
        """
        raise NotImplementedError("Subclasses must implement get_replay_batch")
    
    def modify_loss(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Modify the loss function.
        
        For replay strategies, the loss is already modified by using a mixed dataset,
        so no additional modification is needed here.
        
        Args:
            loss: Original loss value
            **kwargs: Additional arguments
            
        Returns:
            Modified loss value (same as original for replay strategies)
        """
        # No loss modification needed for replay strategies
        return loss
    
    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.
        
        For replay strategies, this is a no-op as the replay is handled
        through batch modification.
        
        Args:
            **kwargs: Additional arguments
        """
        # No gradient modification needed for replay strategies
        pass
    
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.
        
        This method should be implemented by subclasses to perform post-training
        operations, such as updating the replay buffer.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        pass


class MixedDataset(Dataset):
    """
    Custom dataset that combines original examples with replay/generated examples.
    """
    
    def __init__(self, original_dataset, replay_examples):
        """
        Initialize the mixed dataset.
        
        Args:
            original_dataset: Dataset for the current task
            replay_examples: List of examples from replay buffer or generation
        """
        self.original_dataset = original_dataset
        self.replay_examples = replay_examples
        
        # Total length is the sum of original and replay examples
        self.length = len(original_dataset) + len(replay_examples)
    
    def __len__(self):
        """Get the total length of the dataset."""
        return self.length
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dataset item
        """
        # If index is within original dataset range, return from original
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        
        # Otherwise, return from replay examples
        replay_idx = idx - len(self.original_dataset)
        return self.replay_examples[replay_idx]
