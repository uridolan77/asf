"""
Mixed dataset implementation for replay strategies.

This module provides the MixedDataset class for combining original examples
with replay/generated examples.
"""

from torch.utils.data import Dataset
from typing import List, Dict, Any

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
        
        # Store dataset metadata if available
        self.metadata = {}
        if hasattr(original_dataset, 'metadata'):
            self.metadata.update(original_dataset.metadata)
        
        # Add replay metadata
        self.metadata['num_replay_examples'] = len(replay_examples)
        self.metadata['num_original_examples'] = len(original_dataset)
    
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
    
    def get_original_item(self, idx):
        """
        Get an item from the original dataset.
        
        Args:
            idx: Index of the item
            
        Returns:
            Original dataset item
        """
        if idx >= len(self.original_dataset):
            raise IndexError(f"Index {idx} out of range for original dataset")
        return self.original_dataset[idx]
    
    def get_replay_item(self, idx):
        """
        Get an item from the replay examples.
        
        Args:
            idx: Index of the item
            
        Returns:
            Replay example
        """
        if idx >= len(self.replay_examples):
            raise IndexError(f"Index {idx} out of range for replay examples")
        return self.replay_examples[idx]
    
    def get_metadata(self):
        """
        Get dataset metadata.
        
        Returns:
            Dataset metadata
        """
        return self.metadata
