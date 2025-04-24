"""
Base classes for replay-based continual learning strategies.

This module provides base classes for replay-based strategies, including:
- ReplayStrategy: Base class for all replay strategies
"""

import torch
from typing import Dict, List, Optional, Any, Union, Tuple
import random

from peft import PeftModel
from torch.utils.data import Dataset

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.cl_peft.strategies.base import CLStrategy

logger = get_logger(__name__)

class ReplayStrategy(CLStrategy):
    """
    Base class for replay-based continual learning strategies.
    
    This class provides common functionality for replay-based strategies,
    including dataset mixing and replay batch creation.
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
        
        This method should be implemented by subclasses to prepare
        for training on a new task.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset for the current task
            **kwargs: Additional arguments
        """
        self.current_task_id = task_id
        
        if train_dataset is None:
            logger.warning(f"No training dataset provided for {self.__class__.__name__}")
            return
        
        # Store the original dataset
        self.current_dataset = train_dataset
    
    def _create_mixed_dataset(self, current_dataset, replay_examples):
        """
        Create a mixed dataset with examples from the current task and replay examples.
        
        Args:
            current_dataset: Dataset for the current task
            replay_examples: List of examples for replay
            
        Returns:
            Mixed dataset
        """
        # If no replay examples, return the current dataset
        if not replay_examples:
            return current_dataset
        
        # Import here to avoid circular imports
        from .mixed_dataset import MixedDataset
        
        # Create a mixed dataset
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
        # If current batch is a dictionary (most common in HF)
        if isinstance(current_batch, dict):
            combined_batch = {}
            for key in current_batch.keys():
                if key in replay_batch[0]:
                    current_tensors = current_batch[key]
                    
                    # For text inputs, we need to tokenize them first
                    if key == "text" or key == "input_ids":
                        combined_batch[key] = self._combine_text_tensors(
                            current_tensors, replay_batch, key
                        )
                    else:
                        # Handle other tensor types
                        combined_batch[key] = self._combine_other_tensors(
                            current_tensors, replay_batch, key
                        )
                else:
                    # Key not in replay batch, keep original
                    combined_batch[key] = current_batch[key]
            
            return combined_batch
        else:
            # For non-dictionary batches, this needs to be customized
            logger.warning(f"Unsupported batch format for replay: {type(current_batch)}")
            return current_batch
    
    def _combine_text_tensors(self, current_tensors, replay_batch, key):
        """
        Combine text tensors from current batch and replay batch.
        
        Args:
            current_tensors: Tensors from current batch
            replay_batch: Replay batch
            key: Key for the tensors
            
        Returns:
            Combined tensors
        """
        # Tokenize text inputs if needed
        if key == "text" and hasattr(self, 'tokenizer') and self.tokenizer:
            replay_inputs = self.tokenizer(
                [example[key] for example in replay_batch],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).input_ids.to(current_tensors.device)
        else:
            # If already tokenized or no tokenizer
            try:
                replay_tensors = [torch.tensor(example[key]) for example in replay_batch]
                replay_inputs = torch.stack(replay_tensors).to(current_tensors.device)
            except Exception as e:
                logger.warning(f"Error combining text tensors: {str(e)}")
                return current_tensors
        
        # Combine with current batch
        try:
            return torch.cat([current_tensors, replay_inputs], dim=0)
        except Exception as e:
            logger.warning(f"Error concatenating text tensors: {str(e)}")
            return current_tensors
    
    def _combine_other_tensors(self, current_tensors, replay_batch, key):
        """
        Combine non-text tensors from current batch and replay batch.
        
        Args:
            current_tensors: Tensors from current batch
            replay_batch: Replay batch
            key: Key for the tensors
            
        Returns:
            Combined tensors
        """
        if isinstance(current_tensors, torch.Tensor):
            # Try to convert to tensors and stack
            try:
                replay_tensors = [torch.tensor(example[key]).to(current_tensors.device) 
                                 for example in replay_batch]
                replay_tensors = torch.stack(replay_tensors)
                return torch.cat([current_tensors, replay_tensors], dim=0)
            except Exception as e:
                # Fallback if tensor conversion fails
                logger.warning(f"Could not combine tensors for key {key}: {str(e)}")
                return current_tensors
        else:
            # For non-tensor data
            try:
                return current_tensors + [example[key] for example in replay_batch]
            except Exception as e:
                logger.warning(f"Could not combine non-tensor data for key {key}: {str(e)}")
                return current_tensors
    
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
        # No explicit gradient modification needed
        pass
