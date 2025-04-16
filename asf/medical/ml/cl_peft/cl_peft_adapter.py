"""
CL-PEFT Adapter Module

This module provides the core CLPEFTAdapter class for combining Continual Learning (CL)
with Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA and QLoRA.
"""

import os
import json
import torch
import hashlib
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments
)

# Import PEFT for LoRA
try:
    from peft import (
        LoraConfig,
        TaskType,
        get_peft_model,
        PeftModel,
        PeftConfig,
        prepare_model_for_kbit_training,
        PeftModelForCausalLM
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.models.model_registry import ModelRegistry, ModelStatus, get_model_registry
from .registry import get_cl_peft_registry, CLPEFTAdapterStatus

logger = get_logger(__name__)

class CLStrategy(str, Enum):
    """Continual Learning strategy."""
    NAIVE = "naive"  # Sequential fine-tuning without CL mechanisms
    EWC = "ewc"  # Elastic Weight Consolidation
    REPLAY = "replay"  # Experience Replay
    GENERATIVE_REPLAY = "generative_replay"  # Generative Replay
    ORTHOGONAL_LORA = "orthogonal_lora"  # Orthogonal LoRA (O-LoRA)
    ADAPTIVE_SVD = "adaptive_svd"  # Adaptive SVD
    MASK_BASED = "mask_based"  # Mask-based methods

class QuantizationMode(str, Enum):
    """Quantization mode for the base model."""
    NONE = "none"  # No quantization
    INT8 = "int8"  # 8-bit quantization
    INT4 = "int4"  # 4-bit quantization (QLoRA)

class CLPEFTAdapterConfig(BaseModel):
    """Configuration for CL-PEFT adapter."""
    adapter_id: str = Field(..., description="Unique identifier for the adapter.")
    adapter_name: str = Field(..., description="Human-readable name for the adapter.")
    base_model_name: str = Field(..., description="Base model identifier.")
    description: Optional[str] = Field(None, description="Description of the adapter.")

    # CL configuration
    cl_strategy: CLStrategy = Field(CLStrategy.NAIVE, description="Continual Learning strategy.")
    replay_buffer_size: int = Field(1000, description="Size of the replay buffer for replay-based strategies.")
    ewc_lambda: float = Field(5000.0, description="EWC regularization strength.")

    # PEFT configuration
    peft_method: str = Field("lora", description="PEFT method (lora, prefix_tuning, etc.).")
    lora_r: int = Field(16, description="LoRA attention dimension.")
    lora_alpha: int = Field(32, description="Alpha parameter for LoRA scaling.")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA layers.")

    # Target modules to apply LoRA to
    target_modules: List[str] = Field(
        ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="List of modules to apply LoRA to."
    )

    # Quantization configuration
    quantization_mode: QuantizationMode = Field(
        QuantizationMode.NONE,
        description="Quantization mode for the base model."
    )

    # Task configuration
    task_type: str = Field("causal_lm", description="Task type (causal_lm, seq_cls, etc.).")

    # Additional configuration
    bias: str = Field("none", description="Bias type for LoRA.")
    fan_in_fan_out: bool = Field(False, description="Fan in/out for LoRA.")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Tags for the adapter.")

    class Config:
        """Pydantic config."""
        use_enum_values = True

class CLPEFTAdapter:
    """
    CL-PEFT Adapter for combining Continual Learning with Parameter-Efficient Fine-Tuning.

    This class provides a framework for creating and managing adapters that combine
    Continual Learning strategies with PEFT methods like LoRA and QLoRA.
    """

    def __init__(
        self,
        config: CLPEFTAdapterConfig,
        device_map: str = "auto",
        registry = None
    ):
        """
        Initialize the CL-PEFT adapter.

        Args:
            config: Configuration for the adapter
            device_map: Device mapping strategy
            registry: Optional registry instance
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required for CL-PEFT. Please install it with `pip install peft`.")

        self.config = config
        self.device_map = device_map
        self.registry = registry or get_cl_peft_registry()

        # Initialize components
        self.base_model = None
        self.adapter_model = None
        self.tokenizer = None
        self.task_type = self._get_task_type()

        # CL-specific components
        self.replay_buffer = []
        self.task_params_importance = {}  # For EWC
        self.current_task_id = None
        self.task_history = []

        # Register the adapter
        self._register_adapter()

        logger.info(f"Initialized CL-PEFT adapter {config.adapter_id} with strategy {config.cl_strategy}")

    def _register_adapter(self):
        """Register the adapter with the registry."""
        metadata = {
            "adapter_id": self.config.adapter_id,
            "adapter_name": self.config.adapter_name,
            "base_model_name": self.config.base_model_name,
            "description": self.config.description,
            "cl_strategy": self.config.cl_strategy,
            "peft_method": self.config.peft_method,
            "quantization_mode": self.config.quantization_mode,
            "task_type": self.config.task_type,
            "status": CLPEFTAdapterStatus.INITIALIZING.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tags": self.config.tags,
            "task_history": self.task_history
        }

        self.registry.register_adapter(self.config.adapter_id, metadata)

    def _get_task_type(self) -> TaskType:
        """
        Get the PEFT task type from the config.

        Returns:
            TaskType enum value
        """
        task_map = {
            "causal_lm": TaskType.CAUSAL_LM,
            "seq_cls": TaskType.SEQ_CLS,
            "seq2seq": TaskType.SEQ_2_SEQ_LM,
            "token_cls": TaskType.TOKEN_CLS,
            "question_answering": TaskType.QUESTION_ANS
        }

        return task_map.get(self.config.task_type.lower(), TaskType.CAUSAL_LM)

    def load_base_model(self):
        """
        Load the base model.

        This method loads the base model with the specified quantization mode.
        """
        logger.info(f"Loading base model {self.config.base_model_name}")

        # Update status
        self.registry.update_adapter_status(
            self.config.adapter_id,
            CLPEFTAdapterStatus.INITIALIZING
        )

        try:
            # Configure quantization if needed
            quantization_config = None
            if self.config.quantization_mode == QuantizationMode.INT4:
                logger.info("Using 4-bit quantization (QLoRA)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            elif self.config.quantization_mode == QuantizationMode.INT8:
                logger.info("Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16
                )

            # Load the appropriate model class based on task type
            if self.task_type == TaskType.CAUSAL_LM:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )
            elif self.task_type == TaskType.SEQ_CLS:
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.base_model_name,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )
            else:
                # Default to causal LM
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    self.config.base_model_name,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )

            # Prepare model for k-bit training if quantized
            if self.config.quantization_mode != QuantizationMode.NONE:
                self.base_model = prepare_model_for_kbit_training(self.base_model)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model_name)

            # Update status
            self.registry.update_adapter_status(
                self.config.adapter_id,
                CLPEFTAdapterStatus.READY
            )

            logger.info(f"Successfully loaded base model {self.config.base_model_name}")
            return True

        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            self.registry.update_adapter_status(
                self.config.adapter_id,
                CLPEFTAdapterStatus.ERROR
            )
            raise

    def create_lora_config(self) -> LoraConfig:
        """
        Create LoRA configuration.

        Returns:
            LoraConfig for the model.
        """
        return LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.bias,
            task_type=self.task_type,
            fan_in_fan_out=self.config.fan_in_fan_out
        )

    def create_adapter_model(self) -> None:
        """
        Create a LoRA adapter model from the base model.

        This prepares the model for fine-tuning with LoRA.
        """
        if self.base_model is None:
            self.load_base_model()

        # Create LoRA configuration
        lora_config = self.create_lora_config()

        logger.info(f"Creating LoRA adapter model with config: {lora_config}")

        # Apply LoRA adapters to the model
        self.adapter_model = get_peft_model(self.base_model, lora_config)

        # Print trainable parameters
        self._print_trainable_parameters()

        # Update status
        self.registry.update_adapter_status(
            self.config.adapter_id,
            CLPEFTAdapterStatus.READY
        )

    def _print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        if self.adapter_model is None:
            logger.warning("Adapter model not created yet")
            return

        trainable_params = 0
        all_params = 0

        for _, param in self.adapter_model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        logger.info(
            f"Trainable params: {trainable_params} ({100 * trainable_params / all_params:.2f}% of all params)"
        )

    def train_on_task(
        self,
        task_id: str,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        strategy_config=None,
        **kwargs
    ):
        """
        Train the adapter on a specific task.

        This method implements the core continual learning logic, applying the
        selected CL strategy during training.

        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments (optional)
            strategy_config: Configuration for the CL strategy (optional)
            **kwargs: Additional arguments for the trainer

        Returns:
            Training results
        """
        if self.adapter_model is None:
            self.create_adapter_model()

        # Update status
        self.registry.update_adapter_status(
            self.config.adapter_id,
            CLPEFTAdapterStatus.TRAINING
        )

        # Set current task
        self.current_task_id = task_id

        # Set up default training arguments if not provided
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=os.path.join(self.registry.get_adapter_path(self.config.adapter_id), "checkpoints"),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.registry.get_adapter_path(self.config.adapter_id), "logs"),
                logging_steps=10,
                evaluation_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if eval_dataset else False,
            )

        # For naive strategy, use standard training
        if self.config.cl_strategy == CLStrategy.NAIVE:
            return self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)

        # For all other strategies, use the strategy manager and CLTrainer
        try:
            # Import here to avoid circular imports
            from .strategy_manager import strategy_manager
            from .trainer import CLTrainer

            # Prepare strategy configuration
            config = strategy_config or {}

            # Add strategy-specific configurations from adapter config
            if self.config.cl_strategy == CLStrategy.EWC:
                config.setdefault('ewc_lambda', self.config.ewc_lambda)
            elif self.config.cl_strategy in [CLStrategy.REPLAY, CLStrategy.GENERATIVE_REPLAY]:
                config.setdefault('replay_buffer_size', self.config.replay_buffer_size)
                if self.tokenizer:
                    config.setdefault('tokenizer', self.tokenizer)

            # Get the appropriate strategy
            strategy = strategy_manager.get_strategy(
                strategy_name=self.config.cl_strategy,
                model=self.adapter_model,
                strategy_config=config,
                **kwargs.get('strategy_kwargs', {})
            )

            if strategy is None:
                logger.warning(f"Failed to initialize strategy {self.config.cl_strategy}, falling back to naive training")
                return self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)

            # Update strategy with task history
            strategy.task_history = [task['task_id'] for task in self.task_history]

            # Prepare for training on this task
            strategy.before_training(task_id, train_dataset=train_dataset, **kwargs)

            # Create CL trainer
            trainer = CLTrainer(
                cl_strategy=strategy,
                model=self.adapter_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                **{k: v for k, v in kwargs.items() if k != 'strategy_kwargs'}
            )

            # Train the model
            train_results = trainer.train()

            # Perform post-training operations
            strategy.after_training(task_id, **kwargs)

            # Save the adapter
            self.save_adapter()

            # Update task history
            self.task_history.append({
                "task_id": task_id,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": train_results
            })

            # Update adapter metadata
            self.registry.update_adapter_metadata(
                self.config.adapter_id,
                {
                    "task_history": self.task_history,
                    "current_task_id": self.current_task_id
                }
            )

            # Update status
            self.registry.update_adapter_status(
                self.config.adapter_id,
                CLPEFTAdapterStatus.READY
            )

            return train_results

        except Exception as e:
            logger.error(f"Error during CL training: {str(e)}")
            logger.warning(f"Falling back to naive training due to error")
            results = self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)

            # Update task history
            self.task_history.append({
                "task_id": task_id,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "metrics": results
            })

            # Update adapter metadata
            self.registry.update_adapter_metadata(
                self.config.adapter_id,
                {
                    "task_history": self.task_history,
                    "current_task_id": self.current_task_id
                }
            )

            # Update status
            self.registry.update_adapter_status(
                self.config.adapter_id,
                CLPEFTAdapterStatus.READY
            )

            return results

    def _train_naive(self, train_dataset, eval_dataset, training_args, **kwargs):
        """
        Train with naive sequential fine-tuning (no CL mechanisms).

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            **kwargs: Additional arguments for the trainer

        Returns:
            Training results
        """
        logger.info(f"Training on task {self.current_task_id} with naive sequential fine-tuning")

        # Set up default training arguments if not provided
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=os.path.join(self.registry.get_adapter_path(self.config.adapter_id), "checkpoints"),
                num_train_epochs=3,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(self.registry.get_adapter_path(self.config.adapter_id), "logs"),
                logging_steps=10,
                evaluation_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                load_best_model_at_end=True if eval_dataset else False,
            )

        # Create trainer
        trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )

        # Train the model
        train_results = trainer.train()

        # Save the adapter
        self.save_adapter()

        return train_results

    def _train_ewc(self, train_dataset, eval_dataset, training_args, **kwargs):
        """
        Train with Elastic Weight Consolidation (EWC).

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            **kwargs: Additional arguments for the trainer

        Returns:
            Training results
        """
        logger.info(f"Training on task {self.current_task_id} with EWC")

        # TODO: Implement EWC training logic
        # This requires computing Fisher information matrix and
        # adding EWC regularization to the loss function

        # For now, fall back to naive training
        logger.warning("EWC not fully implemented, falling back to naive training")
        return self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)

    def _train_replay(self, train_dataset, eval_dataset, training_args, **kwargs):
        """
        Train with Experience Replay.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            **kwargs: Additional arguments for the trainer

        Returns:
            Training results
        """
        logger.info(f"Training on task {self.current_task_id} with Experience Replay")

        # TODO: Implement Experience Replay logic
        # This requires maintaining a replay buffer and
        # mixing samples from previous tasks with current task

        # For now, fall back to naive training
        logger.warning("Experience Replay not fully implemented, falling back to naive training")
        return self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)

    def _train_orthogonal_lora(self, train_dataset, eval_dataset, training_args, **kwargs):
        """
        Train with Orthogonal LoRA (O-LoRA).

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            **kwargs: Additional arguments for the trainer

        Returns:
            Training results
        """
        logger.info(f"Training on task {self.current_task_id} with Orthogonal LoRA")

        # TODO: Implement Orthogonal LoRA training logic
        # This requires enforcing orthogonality constraints between
        # LoRA updates for different tasks

        # For now, fall back to naive training
        logger.warning("Orthogonal LoRA not fully implemented, falling back to naive training")
        return self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)

    def save_adapter(self, path: Optional[str] = None) -> str:
        """
        Save the adapter to disk.

        Args:
            path: Path to save the adapter (optional)

        Returns:
            Path where the adapter was saved
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")

        # Use default path if not provided
        if path is None:
            path = os.path.join(
                self.registry.get_adapter_path(self.config.adapter_id),
                "adapter"
            )

        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save adapter
        self.adapter_model.save_pretrained(path)

        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

        # Save config
        with open(os.path.join(path, "cl_peft_config.json"), "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)

        logger.info(f"Saved adapter to {path}")

        return path

    @classmethod
    def load_adapter(
        cls,
        adapter_id: str,
        registry = None,
        device_map: str = "auto"
    ) -> "CLPEFTAdapter":
        """
        Load a CL-PEFT adapter from the registry.

        Args:
            adapter_id: Adapter ID
            registry: Optional registry instance
            device_map: Device mapping strategy

        Returns:
            CLPEFTAdapter instance
        """
        registry = registry or get_cl_peft_registry()

        # Get adapter metadata
        metadata = registry.get_adapter(adapter_id)
        if metadata is None:
            raise ValueError(f"Adapter {adapter_id} not found in registry")

        # Get adapter path
        adapter_path = os.path.join(registry.get_adapter_path(adapter_id), "adapter")
        if not os.path.exists(adapter_path):
            raise ValueError(f"Adapter path {adapter_path} does not exist")

        # Load config
        config_path = os.path.join(adapter_path, "cl_peft_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            config = CLPEFTAdapterConfig(**config_dict)
        else:
            # Create config from metadata
            config = CLPEFTAdapterConfig(
                adapter_id=metadata["adapter_id"],
                adapter_name=metadata["adapter_name"],
                base_model_name=metadata["base_model_name"],
                description=metadata.get("description"),
                cl_strategy=metadata.get("cl_strategy", CLStrategy.NAIVE),
                peft_method=metadata.get("peft_method", "lora"),
                quantization_mode=metadata.get("quantization_mode", QuantizationMode.NONE),
                task_type=metadata.get("task_type", "causal_lm"),
                tags=metadata.get("tags", [])
            )

        # Create adapter instance
        adapter = cls(config, device_map, registry)

        # Load base model
        adapter.load_base_model()

        # Load adapter weights
        adapter.adapter_model = PeftModel.from_pretrained(
            adapter.base_model,
            adapter_path,
            device_map=device_map
        )

        # Load task history
        adapter.task_history = metadata.get("task_history", [])
        adapter.current_task_id = metadata.get("current_task_id")

        logger.info(f"Loaded adapter {adapter_id}")

        return adapter

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the adapter model.

        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for the generate method

        Returns:
            Generated text
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.adapter_model.device) for k, v in inputs.items()}

        # Generate
        outputs = self.adapter_model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 100),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("do_sample", True),
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def evaluate_on_task(self, task_id: str, eval_dataset, metric_fn=None):
        """
        Evaluate the adapter on a specific task.

        Args:
            task_id: Task ID
            eval_dataset: Evaluation dataset
            metric_fn: Function to compute metrics (optional)

        Returns:
            Evaluation results
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")

        logger.info(f"Evaluating adapter on task {task_id}")

        # Set up trainer
        training_args = TrainingArguments(
            output_dir=os.path.join(self.registry.get_adapter_path(self.config.adapter_id), "eval"),
            per_device_eval_batch_size=8,
            remove_unused_columns=False
        )

        trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            eval_dataset=eval_dataset
        )

        # Evaluate
        results = trainer.evaluate()

        # Apply custom metric function if provided
        if metric_fn is not None:
            custom_metrics = metric_fn(self.adapter_model, eval_dataset)
            results.update(custom_metrics)

        # Update task history
        for task in self.task_history:
            if task["task_id"] == task_id:
                task["eval_metrics"] = results
                break

        # Update adapter metadata
        self.registry.update_adapter_metadata(
            self.config.adapter_id,
            {"task_history": self.task_history}
        )

        return results

    def compute_forgetting(self, task_id: str, eval_dataset, metric_key: str = "eval_loss"):
        """
        Compute forgetting for a specific task.

        Args:
            task_id: Task ID
            eval_dataset: Evaluation dataset
            metric_key: Key for the metric to use for forgetting calculation

        Returns:
            Forgetting metric
        """
        # Find original performance
        original_perf = None
        for task in self.task_history:
            if task["task_id"] == task_id and "eval_metrics" in task:
                original_perf = task["eval_metrics"].get(metric_key)
                break

        if original_perf is None:
            logger.warning(f"No original performance found for task {task_id}")
            return None

        # Evaluate current performance
        current_results = self.evaluate_on_task(task_id, eval_dataset)
        current_perf = current_results.get(metric_key)

        if current_perf is None:
            logger.warning(f"Metric {metric_key} not found in evaluation results")
            return None

        # Compute forgetting (higher is worse for loss, lower is worse for accuracy)
        if "loss" in metric_key.lower():
            forgetting = current_perf - original_perf
        else:
            forgetting = original_perf - current_perf

        logger.info(f"Forgetting for task {task_id}: {forgetting}")

        return forgetting
