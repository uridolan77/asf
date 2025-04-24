"""
CL-PEFT Trainer Module

This module provides a custom trainer for Continual Learning with Parameter-Efficient Fine-Tuning.
It extends the Hugging Face Trainer to incorporate CL-specific loss modifications and gradient manipulations.
"""

import os
import torch
from typing import Dict, List, Optional, Any, Union, Callable

from transformers import Trainer, TrainingArguments
from peft import PeftModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class CLTrainer(Trainer):
    """
    Custom trainer for Continual Learning with PEFT.
    
    This trainer extends the Hugging Face Trainer to incorporate
    CL-specific loss modifications and gradient manipulations.
    """
    
    def __init__(
        self, 
        cl_strategy,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        **kwargs
    ):
        """
        Initialize the CL trainer.
        
        Args:
            cl_strategy: The CL strategy to apply
            model: The model to train
            args: Training arguments
            data_collator: Function to collate batches
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            tokenizer: Tokenizer
            model_init: Model initialization function
            compute_metrics: Function to compute metrics
            callbacks: List of callbacks
            optimizers: Tuple of optimizer and scheduler
            preprocess_logits_for_metrics: Function to preprocess logits for metrics
            **kwargs: Additional arguments for the Trainer
        """
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        self.cl_strategy = cl_strategy
        
        # Register batch start callback for strategies that need it
        if hasattr(self.cl_strategy, 'on_batch_start'):
            logger.info("Strategy has on_batch_start method, will call it during training")
            self._has_batch_start_callback = True
        else:
            self._has_batch_start_callback = False
    
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
        modified_loss = self.cl_strategy.modify_loss(
            loss=loss, 
            model=model, 
            inputs=inputs, 
            outputs=outputs
        )
        
        if return_outputs:
            return modified_loss, outputs
        return modified_loss
    
    def training_step(self, model, inputs):
        """
        Perform a training step with CL-specific gradient modifications.
        
        Args:
            model: The model
            inputs: The inputs
            
        Returns:
            Loss value
        """
        model.train()
        
        # Apply batch start callback if available
        if self._has_batch_start_callback:
            inputs = self.cl_strategy.on_batch_start(self, inputs)
        
        # Move inputs to appropriate device
        inputs = self._prepare_inputs(inputs)
        
        # Compute loss and perform backward pass
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Apply CL-specific gradient modifications
        self.cl_strategy.modify_gradients(
            model=model, 
            inputs=inputs, 
            outputs=outputs
        )
        
        return loss.detach()
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """
        Log metrics, save model and evaluate if needed.
        
        This overrides the parent method to add CL-specific metrics.
        
        Args:
            tr_loss: Training loss
            model: The model
            trial: Trial object for hyperparameter search
            epoch: Current epoch
            ignore_keys_for_eval: Keys to ignore during evaluation
            
        Returns:
            Evaluation metrics
        """
        # Call parent method
        metrics = super()._maybe_log_save_evaluate(
            tr_loss, model, trial, epoch, ignore_keys_for_eval
        )
        
        # Add CL-specific metrics if available
        if hasattr(self.cl_strategy, 'compute_metrics') and metrics is not None:
            cl_metrics = self.cl_strategy.compute_metrics()
            if cl_metrics:
                # Add CL metrics with cl_ prefix
                for key, value in cl_metrics.items():
                    metrics[f"cl_{key}"] = value
                
                # Log CL metrics
                self.log(metrics)
        
        return metrics
