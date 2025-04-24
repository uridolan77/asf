"""
Model utility functions for CL-PEFT.

This module provides utility functions for working with models in CL-PEFT.
"""

from typing import List, Dict, Optional, Any, Union

def get_target_modules_for_model(model_name: str) -> List[str]:
    """
    Get the appropriate target modules for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of target module names
    """
    # Default target modules
    default_targets = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    # Model-specific target modules
    model_targets = {
        # Llama models
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama3": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Mistral models
        "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Gemma models
        "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        
        # Falcon models
        "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # GPT-NeoX models
        "gpt-neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        
        # MPT models
        "mpt": ["Wqkv", "out_proj", "fc1", "fc2"],
        
        # BERT models
        "bert": ["query", "key", "value", "dense"],
        
        # RoBERTa models
        "roberta": ["query", "key", "value", "dense"],
        
        # T5 models
        "t5": ["q", "k", "v", "o", "wi", "wo"],
        
        # Flan-T5 models
        "flan-t5": ["q", "k", "v", "o", "wi", "wo"],
    }
    
    # Check if the model name contains any of the keys in model_targets
    for key, targets in model_targets.items():
        if key.lower() in model_name.lower():
            return targets
    
    # Return default targets if no match is found
    return default_targets

def get_model_hidden_size(model_name: str) -> int:
    """
    Get the hidden size for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Hidden size
    """
    # Default hidden size
    default_hidden_size = 768
    
    # Model-specific hidden sizes
    model_hidden_sizes = {
        # Llama models
        "llama-7b": 4096,
        "llama-13b": 5120,
        "llama-70b": 8192,
        "llama2-7b": 4096,
        "llama2-13b": 5120,
        "llama2-70b": 8192,
        "llama3-8b": 4096,
        "llama3-70b": 8192,
        
        # Mistral models
        "mistral-7b": 4096,
        
        # Gemma models
        "gemma-2b": 2048,
        "gemma-7b": 3072,
        
        # Falcon models
        "falcon-7b": 4544,
        "falcon-40b": 8192,
        
        # BERT models
        "bert-base": 768,
        "bert-large": 1024,
        
        # RoBERTa models
        "roberta-base": 768,
        "roberta-large": 1024,
        
        # T5 models
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        
        # Flan-T5 models
        "flan-t5-small": 512,
        "flan-t5-base": 768,
        "flan-t5-large": 1024,
        "flan-t5-xl": 2048,
        "flan-t5-xxl": 4096,
    }
    
    # Check if the model name contains any of the keys in model_hidden_sizes
    for key, hidden_size in model_hidden_sizes.items():
        if key.lower() in model_name.lower():
            return hidden_size
    
    # Return default hidden size if no match is found
    return default_hidden_size
