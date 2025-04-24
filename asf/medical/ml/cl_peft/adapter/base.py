"""
Base adapter class for CL-PEFT.

This module provides the base adapter class for combining Continual Learning (CL)
with Parameter-Efficient Fine-Tuning (PEFT) techniques.
"""

import os
import json
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from abc import ABC, abstractmethod

from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    AutoTokenizer
)

# Import PEFT for LoRA
try:
    from peft import (
        TaskType, 
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.cl_peft.config import CLPEFTAdapterConfig, CLStrategy
from asf.medical.ml.cl_peft.registry import get_cl_peft_registry, CLPEFTAdapterStatus

logger = get_logger(__name__)

class CLPEFTAdapter(ABC):
    """
    Base class for CL-PEFT adapters.
    
    This abstract class provides the common interface and functionality for
    all CL-PEFT adapters, regardless of the specific PEFT method used.
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
            "created_at": datetime.utcnow().isoformat(),
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
    
    @abstractmethod
    def load_base_model(self):
        """
        Load the base model.
        
        This method should be implemented by subclasses to load the base model
        with the appropriate configuration.
        """
        pass
    
    @abstractmethod
    def create_adapter_model(self) -> None:
        """
        Create an adapter model from the base model.
        
        This method should be implemented by subclasses to create the adapter model
        with the appropriate PEFT method.
        """
        pass
    
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
        
        # Apply CL strategy
        if self.config.cl_strategy == CLStrategy.NAIVE:
            # Naive sequential fine-tuning (no CL mechanisms)
            results = self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)
        
        elif self.config.cl_strategy == CLStrategy.EWC:
            # Elastic Weight Consolidation
            results = self._train_ewc(train_dataset, eval_dataset, training_args, **kwargs)
        
        elif self.config.cl_strategy == CLStrategy.REPLAY:
            # Experience Replay
            results = self._train_replay(train_dataset, eval_dataset, training_args, **kwargs)
        
        elif self.config.cl_strategy == CLStrategy.ORTHOGONAL_LORA:
            # Orthogonal LoRA (O-LoRA)
            results = self._train_orthogonal_lora(train_dataset, eval_dataset, training_args, **kwargs)
        
        else:
            # Default to naive
            logger.warning(f"CL strategy {self.config.cl_strategy} not fully implemented, using naive")
            results = self._train_naive(train_dataset, eval_dataset, training_args, **kwargs)
        
        # Update task history
        self.task_history.append({
            "task_id": task_id,
            "trained_at": datetime.utcnow().isoformat(),
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
        
        # Import the EWC strategy
        from asf.medical.ml.cl_peft.strategies.ewc import ElasticWeightConsolidation
        
        # Create EWC strategy
        ewc_strategy = ElasticWeightConsolidation(
            model=self.adapter_model,
            ewc_lambda=self.config.ewc_lambda
        )
        
        # Train with EWC
        results = ewc_strategy.train(
            task_id=self.current_task_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            **kwargs
        )
        
        # Save the adapter
        self.save_adapter()
        
        return results
    
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
        
        # Import the Replay strategy
        from asf.medical.ml.cl_peft.strategies.replay import ExperienceReplay
        
        # Create Replay strategy
        replay_strategy = ExperienceReplay(
            model=self.adapter_model,
            buffer_size=self.config.replay_buffer_size
        )
        
        # Train with Replay
        results = replay_strategy.train(
            task_id=self.current_task_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            **kwargs
        )
        
        # Save the adapter
        self.save_adapter()
        
        return results
    
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
        
        # Import the Orthogonal LoRA strategy
        from asf.medical.ml.cl_peft.strategies.orthogonal import OrthogonalLoRA
        
        # Create Orthogonal LoRA strategy
        olora_strategy = OrthogonalLoRA(
            model=self.adapter_model
        )
        
        # Train with Orthogonal LoRA
        results = olora_strategy.train(
            task_id=self.current_task_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            **kwargs
        )
        
        # Save the adapter
        self.save_adapter()
        
        return results
    
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
            json.dump(self.config.dict(), f, indent=2)
        
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
                quantization_mode=metadata.get("quantization_mode", "none"),
                task_type=metadata.get("task_type", "causal_lm"),
                tags=metadata.get("tags", [])
            )
        
        # Create appropriate adapter instance based on PEFT method
        if config.peft_method.lower() == "lora":
            if config.quantization_mode == "int4":
                from asf.medical.ml.cl_peft.adapter.qlora import QLoRAAdapter
                adapter = QLoRAAdapter(config, device_map, registry)
            else:
                from asf.medical.ml.cl_peft.adapter.lora import LoRAAdapter
                adapter = LoRAAdapter(config, device_map, registry)
        else:
            # Default to LoRA
            from asf.medical.ml.cl_peft.adapter.lora import LoRAAdapter
            adapter = LoRAAdapter(config, device_map, registry)
        
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
