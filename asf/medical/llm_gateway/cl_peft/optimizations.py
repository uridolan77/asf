"""
Memory and computation optimizations for CL-PEFT.

This module provides optimizations for memory usage and computation speed
in continual learning with parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Union, Tuple
import gc
import os
import numpy as np
from functools import partial

from peft import PeftModel
from transformers import Trainer, TrainingArguments

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

def enable_gradient_checkpointing(model):
    """
    Enable gradient checkpointing to reduce memory usage.
    
    Args:
        model: The model to enable gradient checkpointing for
        
    Returns:
        Model with gradient checkpointing enabled
    """
    # Get the base model
    if hasattr(model, "base_model"):
        base_model = model.base_model
    elif hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model
    
    # Enable gradient checkpointing
    if hasattr(base_model, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing")
        base_model.gradient_checkpointing_enable()
    elif hasattr(base_model, "enable_gradient_checkpointing"):
        logger.info("Enabling gradient checkpointing")
        base_model.enable_gradient_checkpointing()
    else:
        logger.warning("Gradient checkpointing not supported by this model")
    
    return model

def get_8bit_optimizer(model, lr=5e-5, weight_decay=0.0):
    """
    Get an 8-bit Adam optimizer for memory-efficient training.
    
    Args:
        model: The model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        8-bit Adam optimizer
    """
    try:
        import bitsandbytes as bnb
        
        # Get trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        
        # Create 8-bit Adam optimizer
        optimizer = bnb.optim.Adam8bit(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        logger.info("Using 8-bit Adam optimizer")
        return optimizer
    
    except ImportError:
        logger.warning("bitsandbytes not installed, falling back to regular Adam")
        return torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay
        )

def get_memory_efficient_trainer(
    model,
    args,
    train_dataset=None,
    eval_dataset=None,
    data_collator=None,
    compute_metrics=None,
    optimizers=(None, None),
    callbacks=None,
    use_8bit_optimizer=True,
    use_gradient_checkpointing=True,
    **kwargs
):
    """
    Get a memory-efficient trainer.
    
    Args:
        model: The model to train
        args: Training arguments
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        data_collator: Data collator
        compute_metrics: Function to compute metrics
        optimizers: Tuple of (optimizer, scheduler)
        callbacks: List of callbacks
        use_8bit_optimizer: Whether to use 8-bit optimizer
        use_gradient_checkpointing: Whether to use gradient checkpointing
        **kwargs: Additional arguments for the Trainer
        
    Returns:
        Memory-efficient trainer
    """
    # Enable gradient checkpointing if requested
    if use_gradient_checkpointing:
        model = enable_gradient_checkpointing(model)
    
    # Create optimizer if not provided
    optimizer, scheduler = optimizers
    if optimizer is None and use_8bit_optimizer:
        optimizer = get_8bit_optimizer(
            model,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler),
        callbacks=callbacks,
        **kwargs
    )
    
    return trainer

def clear_memory():
    """
    Clear unused memory.
    
    This function clears CUDA cache and runs garbage collection.
    """
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()

def get_memory_stats():
    """
    Get memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    stats = {
        "cpu": {
            "available": 0,
            "used": 0,
            "percent": 0
        },
        "gpu": {
            "available": 0,
            "used": 0,
            "percent": 0
        }
    }
    
    # Get CPU memory usage
    try:
        import psutil
        
        # Get CPU memory usage
        mem = psutil.virtual_memory()
        stats["cpu"]["available"] = mem.available / (1024 ** 3)  # GB
        stats["cpu"]["used"] = mem.used / (1024 ** 3)  # GB
        stats["cpu"]["percent"] = mem.percent
    except ImportError:
        logger.warning("psutil not installed, CPU memory stats not available")
    
    # Get GPU memory usage
    if torch.cuda.is_available():
        try:
            # Get GPU memory usage
            for i in range(torch.cuda.device_count()):
                gpu_stats = {
                    "available": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),  # GB
                    "used": torch.cuda.memory_allocated(i) / (1024 ** 3),  # GB
                    "percent": torch.cuda.memory_allocated(i) / torch.cuda.get_device_properties(i).total_memory * 100
                }
                
                stats[f"gpu_{i}"] = gpu_stats
        except Exception as e:
            logger.warning(f"Error getting GPU memory stats: {str(e)}")
    
    return stats

def log_memory_stats():
    """
    Log memory usage statistics.
    """
    stats = get_memory_stats()
    
    # Log CPU memory usage
    logger.info(f"CPU Memory: {stats['cpu']['used']:.2f} GB used ({stats['cpu']['percent']:.1f}%)")
    
    # Log GPU memory usage
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_stats = stats.get(f"gpu_{i}")
            if gpu_stats:
                logger.info(f"GPU {i} Memory: {gpu_stats['used']:.2f} GB used ({gpu_stats['percent']:.1f}%)")

def optimize_inference(model, use_bettertransformer=True, use_flash_attention=True):
    """
    Optimize model for inference.
    
    Args:
        model: The model to optimize
        use_bettertransformer: Whether to use BetterTransformer
        use_flash_attention: Whether to use Flash Attention
        
    Returns:
        Optimized model
    """
    # Get the base model
    if hasattr(model, "base_model"):
        base_model = model.base_model
    elif hasattr(model, "model"):
        base_model = model.model
    else:
        base_model = model
    
    # Use BetterTransformer if available
    if use_bettertransformer:
        try:
            from optimum.bettertransformer import BetterTransformer
            
            logger.info("Optimizing model with BetterTransformer")
            base_model = BetterTransformer.transform(base_model)
        except ImportError:
            logger.warning("optimum not installed, BetterTransformer not available")
    
    # Use Flash Attention if available
    if use_flash_attention:
        try:
            # Check if Flash Attention is already enabled
            if hasattr(base_model.config, "attn_implementation") and base_model.config.attn_implementation == "flash_attention_2":
                logger.info("Flash Attention already enabled")
            else:
                # Try to enable Flash Attention
                try:
                    logger.info("Enabling Flash Attention")
                    base_model.config.attn_implementation = "flash_attention_2"
                except Exception as e:
                    logger.warning(f"Error enabling Flash Attention: {str(e)}")
        except Exception as e:
            logger.warning(f"Error checking Flash Attention: {str(e)}")
    
    return model

def quantize_model(model, quantization_method="4bit", **kwargs):
    """
    Quantize model for memory-efficient inference.
    
    Args:
        model: The model to quantize
        quantization_method: Quantization method ("4bit", "8bit", "gptq", "awq")
        **kwargs: Additional arguments for quantization
        
    Returns:
        Quantized model
    """
    if quantization_method == "4bit":
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            # Get the base model
            if hasattr(model, "base_model"):
                base_model = model.base_model
            elif hasattr(model, "model"):
                base_model = model.model
            else:
                base_model = model
            
            # Create 4-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            # Apply quantization
            logger.info("Quantizing model to 4-bit")
            
            # This is a simplified approach - in practice, you would need to reload the model
            # with the quantization config
            logger.warning("4-bit quantization requires reloading the model, returning original model")
            return model
        
        except ImportError:
            logger.warning("bitsandbytes not installed, 4-bit quantization not available")
            return model
    
    elif quantization_method == "8bit":
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            
            # Get the base model
            if hasattr(model, "base_model"):
                base_model = model.base_model
            elif hasattr(model, "model"):
                base_model = model.model
            else:
                base_model = model
            
            # Create 8-bit quantization config
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            
            # Apply quantization
            logger.info("Quantizing model to 8-bit")
            
            # This is a simplified approach - in practice, you would need to reload the model
            # with the quantization config
            logger.warning("8-bit quantization requires reloading the model, returning original model")
            return model
        
        except ImportError:
            logger.warning("bitsandbytes not installed, 8-bit quantization not available")
            return model
    
    elif quantization_method == "gptq":
        try:
            # GPTQ quantization requires a more complex setup
            logger.warning("GPTQ quantization requires a specialized setup, returning original model")
            return model
        
        except Exception as e:
            logger.warning(f"Error applying GPTQ quantization: {str(e)}")
            return model
    
    elif quantization_method == "awq":
        try:
            # AWQ quantization requires a more complex setup
            logger.warning("AWQ quantization requires a specialized setup, returning original model")
            return model
        
        except Exception as e:
            logger.warning(f"Error applying AWQ quantization: {str(e)}")
            return model
    
    else:
        logger.warning(f"Unknown quantization method: {quantization_method}, returning original model")
        return model

def get_distributed_training_args(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.0,
    warmup_ratio=0.1,
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    bf16=False,
    **kwargs
):
    """
    Get training arguments for distributed training.
    
    Args:
        output_dir: Output directory
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device for training
        per_device_eval_batch_size: Batch size per device for evaluation
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_ratio: Warmup ratio
        logging_steps: Logging steps
        evaluation_strategy: Evaluation strategy
        save_strategy: Save strategy
        fp16: Whether to use FP16
        bf16: Whether to use BF16
        **kwargs: Additional arguments for TrainingArguments
        
    Returns:
        Training arguments for distributed training
    """
    # Determine if distributed training is available
    is_distributed = torch.cuda.device_count() > 1
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        fp16=fp16,
        bf16=bf16,
        # Distributed training arguments
        local_rank=-1 if not is_distributed else int(os.environ.get("LOCAL_RANK", -1)),
        ddp_find_unused_parameters=False,
        **kwargs
    )
    
    return training_args

def get_fsdp_config():
    """
    Get configuration for Fully Sharded Data Parallel (FSDP) training.
    
    Returns:
        FSDP configuration
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            BackwardPrefetch,
            ShardingStrategy,
            CPUOffload,
        )
        from torch.distributed.fsdp.wrap import (
            transformer_auto_wrap_policy,
            enable_wrap,
            wrap,
        )
        
        # Define FSDP configuration
        fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": ["LlamaDecoderLayer", "LlamaLayer", "GPTJBlock", "GPT2Block", "OPTDecoderLayer", "BertLayer"],
            "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_offload_params": False,
            "fsdp_sharding_strategy": 1,  # SHARD_GRAD_OP
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        }
        
        logger.info("FSDP configuration created")
        return fsdp_config
    
    except ImportError:
        logger.warning("FSDP not available, returning empty config")
        return {}

def get_deepspeed_config(
    stage=2,
    offload_optimizer=False,
    offload_param=False,
    zero3_init_flag=False,
    zero3_save_16bit_model=False,
    fp16=True,
    bf16=False,
    gradient_accumulation_steps=1,
    gradient_clipping=1.0,
    **kwargs
):
    """
    Get configuration for DeepSpeed.
    
    Args:
        stage: ZeRO stage (0, 1, 2, or 3)
        offload_optimizer: Whether to offload optimizer states to CPU
        offload_param: Whether to offload parameters to CPU
        zero3_init_flag: Whether to use ZeRO-3 initialization
        zero3_save_16bit_model: Whether to save 16-bit model with ZeRO-3
        fp16: Whether to use FP16
        bf16: Whether to use BF16
        gradient_accumulation_steps: Gradient accumulation steps
        gradient_clipping: Gradient clipping
        **kwargs: Additional arguments for DeepSpeed config
        
    Returns:
        DeepSpeed configuration
    """
    # Base config
    config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            }
        },
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        }
    }
    
    # Add precision settings
    if fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        }
    elif bf16:
        config["bf16"] = {
            "enabled": True
        }
    
    # Add ZeRO-2 specific settings
    if stage == 2:
        config["zero_optimization"]["cpu_offload"] = offload_optimizer
    
    # Add ZeRO-3 specific settings
    if stage == 3:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu" if offload_optimizer else "none",
            "pin_memory": True
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu" if offload_param else "none",
            "pin_memory": True
        }
        config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"] = zero3_save_16bit_model
        config["zero_optimization"]["zero_init_flag"] = zero3_init_flag
    
    # Add additional settings
    for key, value in kwargs.items():
        config[key] = value
    
    logger.info(f"DeepSpeed configuration created for stage {stage}")
    return config
