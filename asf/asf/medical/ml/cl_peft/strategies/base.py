"""
Base class for Continual Learning strategies.
"""

import torch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple

from transformers import Trainer, TrainingArguments
from peft import PeftModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class CLStrategy(ABC):
    """
    Base class for Continual Learning strategies.
    
    This abstract class defines the interface for all CL strategies.
    """
    
    def __init__(self, model: PeftModel, **kwargs):
        """
        Initialize the CL strategy.
        
        Args:
            model: The PEFT model to apply the strategy to
            **kwargs: Additional strategy-specific parameters
        """
        self.model = model
        self.task_history = []
        self.current_task_id = None
    
    @abstractmethod
    def before_training(self, task_id: str, **kwargs):
        """
        Prepare for training on a new task.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def modify_loss(self, loss: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Modify the loss function to incorporate CL constraints.
        
        Args:
            loss: Original loss value
            **kwargs: Additional arguments
            
        Returns:
            Modified loss value
        """
        pass
    
    @abstractmethod
    def after_training(self, task_id: str, **kwargs):
        """
        Perform post-training operations.
        
        Args:
            task_id: Unique identifier for the task
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def modify_gradients(self, **kwargs):
        """
        Modify gradients during training.
        
        Args:
            **kwargs: Additional arguments
        """
        pass
    
    def train(
        self,
        task_id: str,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        **kwargs
    ):
        """
        Train the model on a task using this CL strategy.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments (optional)
            **kwargs: Additional arguments for the trainer
            
        Returns:
            Training results
        """
        # Set current task
        self.current_task_id = task_id
        
        # Prepare for training
        self.before_training(task_id, **kwargs)
        
        # Create custom trainer with CL-specific loss
        trainer = CLTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            cl_strategy=self,
            **kwargs
        )
        
        # Train the model
        train_results = trainer.train()
        
        # Perform post-training operations
        self.after_training(task_id, **kwargs)
        
        # Update task history
        self.task_history.append({
            "task_id": task_id,
            "metrics": train_results
        })
        
        return train_results

class CLTrainer(Trainer):
    """
    Custom trainer for Continual Learning.
    
    This trainer extends the Hugging Face Trainer to incorporate
    CL-specific loss modifications and gradient manipulations.
    """
    
    def __init__(self, cl_strategy: CLStrategy, **kwargs):
        """
        Initialize the CL trainer.
        
        Args:
            cl_strategy: The CL strategy to apply
            **kwargs: Additional arguments for the Trainer
        """
        super().__init__(**kwargs)
        self.cl_strategy = cl_strategy
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute the loss with CL-specific modifications.
        
        Args:
            model: The model
            inputs: The inputs
            return_outputs: Whether to return outputs along with the loss
            
        Returns:
            Loss value or (loss, outputs) tuple
        """
        # Compute original loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply CL-specific loss modification
        modified_loss = self.cl_strategy.modify_loss(loss, model=model, inputs=inputs, outputs=outputs)
        
        return (modified_loss, outputs) if return_outputs else modified_loss
    
    def training_step(self, model, inputs):
        """
        Perform a training step with CL-specific gradient modifications.
        
        Args:
            model: The model
            inputs: The inputs
            
        Returns:
            Loss value
        """
        # Regular training step
        loss = super().training_step(model, inputs)
        
        # Apply CL-specific gradient modifications
        self.cl_strategy.modify_gradients(model=model, inputs=inputs)
        
        return loss
