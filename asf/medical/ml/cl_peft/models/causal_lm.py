"""
Causal language model implementation for CL-PEFT.

This module provides a causal language model implementation for CL-PEFT.
"""

import torch
from typing import Dict, List, Optional, Any, Union, Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.cl_peft.adapter import CLPEFTAdapter
from asf.medical.ml.cl_peft.config import CLPEFTAdapterConfig

logger = get_logger(__name__)

class CLPEFTCausalLM:
    """
    Causal language model implementation for CL-PEFT.
    
    This class provides a causal language model implementation for CL-PEFT,
    with methods for training, evaluation, and generation.
    """
    
    def __init__(
        self,
        adapter: CLPEFTAdapter,
        max_length: int = 512,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the CL-PEFT causal language model.
        
        Args:
            adapter: CL-PEFT adapter
            max_length: Maximum sequence length
            generation_config: Configuration for text generation
        """
        self.adapter = adapter
        self.max_length = max_length
        self.generation_config = generation_config or {
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
        
        logger.info(f"Initialized CL-PEFT causal language model with adapter {adapter.config.adapter_id}")
    
    def train(
        self,
        task_id: str,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        **kwargs
    ):
        """
        Train the model on a task.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments (optional)
            **kwargs: Additional arguments for the trainer
            
        Returns:
            Training results
        """
        return self.adapter.train_on_task(
            task_id,
            train_dataset,
            eval_dataset,
            training_args,
            **kwargs
        )
    
    def evaluate(
        self,
        task_id: str,
        eval_dataset,
        metric_fn=None
    ):
        """
        Evaluate the model on a task.
        
        Args:
            task_id: Unique identifier for the task
            eval_dataset: Evaluation dataset
            metric_fn: Function to compute metrics (optional)
            
        Returns:
            Evaluation results
        """
        return self.adapter.evaluate_on_task(
            task_id,
            eval_dataset,
            metric_fn
        )
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional arguments for the generate method
            
        Returns:
            Generated text
        """
        # Use provided values or defaults from generation_config
        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.generation_config["max_new_tokens"],
            "temperature": temperature or self.generation_config["temperature"],
            "top_p": top_p or self.generation_config["top_p"],
            "do_sample": do_sample if do_sample is not None else self.generation_config["do_sample"],
            **kwargs
        }
        
        return self.adapter.generate(prompt, **generation_kwargs)
    
    def compute_perplexity(
        self,
        text: str,
        stride: int = 512
    ) -> float:
        """
        Compute perplexity for a text.
        
        Args:
            text: Input text
            stride: Stride for sliding window
            
        Returns:
            Perplexity
        """
        if self.adapter.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Tokenize input
        encodings = self.adapter.tokenizer(text, return_tensors="pt")
        
        # Get input IDs and create a tensor on the same device as the model
        input_ids = encodings.input_ids.to(self.adapter.adapter_model.device)
        
        # Initialize variables for perplexity calculation
        nlls = []
        max_length = self.max_length
        
        # Compute perplexity using sliding window approach
        for i in range(0, input_ids.size(1), stride):
            # Extract a window of input IDs
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.size(1))
            target_len = end_loc - i
            
            # Extract input IDs for this window
            input_ids_window = input_ids[:, begin_loc:end_loc]
            
            # Compute loss for this window
            with torch.no_grad():
                outputs = self.adapter.adapter_model(input_ids_window, labels=input_ids_window)
                neg_log_likelihood = outputs.loss
            
            # Add to list of negative log likelihoods
            nlls.append(neg_log_likelihood.item() * target_len)
        
        # Compute perplexity
        ppl = torch.exp(torch.tensor(sum(nlls) / input_ids.size(1)))
        
        return ppl.item()
    
    @classmethod
    def from_pretrained(
        cls,
        adapter_id: str,
        registry=None,
        device_map: str = "auto",
        **kwargs
    ) -> "CLPEFTCausalLM":
        """
        Load a CL-PEFT causal language model from a pretrained adapter.
        
        Args:
            adapter_id: Adapter ID
            registry: Optional registry instance
            device_map: Device mapping strategy
            **kwargs: Additional arguments for the CLPEFTCausalLM constructor
            
        Returns:
            CLPEFTCausalLM instance
        """
        # Load adapter
        adapter = CLPEFTAdapter.load_adapter(
            adapter_id,
            registry=registry,
            device_map=device_map
        )
        
        # Create CLPEFTCausalLM instance
        return cls(adapter, **kwargs)
