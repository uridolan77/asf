"""
Service layer for CL-PEFT operations.

This module provides a service for managing CL-PEFT adapters, including:
- Creating adapters
- Training adapters on sequential tasks
- Evaluating adapter performance
- Managing adapter lifecycle
"""

import os
import json
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

import torch
from pydantic import BaseModel, Field

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.cl_peft import (
    CLPEFTAdapter,
    CLPEFTAdapterConfig,
    get_cl_peft_registry,
    CLPEFTAdapterRegistry,
    CLPEFTAdapterStatus,
    LoRAAdapter,
    QLoRAAdapter,
    CLPEFTCausalLM
)

logger = get_logger(__name__)

class CLPEFTService:
    """
    Service for managing CL-PEFT adapters.
    """

    def __init__(self, registry: Optional[CLPEFTAdapterRegistry] = None):
        """
        Initialize the CL-PEFT service.

        Args:
            registry: Optional registry instance
        """
        self.registry = registry or get_cl_peft_registry()
        logger.info("Initialized CL-PEFT service")

    def create_adapter(self, config: CLPEFTAdapterConfig) -> str:
        """
        Create a new CL-PEFT adapter.

        Args:
            config: Configuration for the adapter

        Returns:
            Adapter ID
        """
        logger.info(f"Creating adapter {config.adapter_id} for base model {config.base_model_name}")

        try:
            # Create appropriate adapter based on configuration
            if config.peft_method.lower() == "lora":
                if config.quantization_mode == "int4":
                    # Create QLoRA adapter
                    adapter = QLoRAAdapter(config, registry=self.registry)
                else:
                    # Create LoRA adapter
                    adapter = LoRAAdapter(config, registry=self.registry)
            else:
                # Default to LoRA adapter
                adapter = LoRAAdapter(config, registry=self.registry)

            # Load base model
            adapter.load_base_model()

            # Create adapter model
            adapter.create_adapter_model()

            # Save adapter
            adapter.save_adapter()

            logger.info(f"Successfully created adapter {config.adapter_id}")

            return config.adapter_id

        except Exception as e:
            logger.error(f"Error creating adapter: {str(e)}")

            # Update status to error
            self.registry.update_adapter_status(
                config.adapter_id,
                CLPEFTAdapterStatus.ERROR
            )

            raise

    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get adapter metadata.

        Args:
            adapter_id: Adapter ID

        Returns:
            Adapter metadata or None if not found
        """
        return self.registry.get_adapter(adapter_id)

    def list_adapters(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List all adapters, optionally filtered.

        Args:
            filter_by: Filter criteria

        Returns:
            List of adapter metadata
        """
        return self.registry.list_adapters(filter_by)

    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter.

        Args:
            adapter_id: Adapter ID

        Returns:
            Success flag
        """
        return self.registry.delete_adapter(adapter_id)

    def train_adapter(
        self,
        adapter_id: str,
        task_id: str,
        train_dataset,
        eval_dataset=None,
        training_args=None,
        strategy_config=None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train an adapter on a task.

        Args:
            adapter_id: Adapter ID
            task_id: Task ID
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            training_args: Training arguments (optional)
            strategy_config: Configuration for the CL strategy (optional)
            **kwargs: Additional arguments for the trainer

        Returns:
            Training results
        """
        logger.info(f"Training adapter {adapter_id} on task {task_id}")

        try:
            # Load adapter
            adapter = CLPEFTAdapter.load_adapter(adapter_id, registry=self.registry)

            # Get adapter metadata
            adapter_metadata = self.registry.get_adapter(adapter_id)
            cl_strategy = adapter_metadata.get('cl_strategy', 'naive')

            # Prepare strategy configuration
            if strategy_config is None:
                strategy_config = {}

            # Add task-specific prompts for generative replay if provided
            if cl_strategy == 'generative_replay' and 'task_prompts' in kwargs:
                strategy_config['task_prompts'] = kwargs.pop('task_prompts')

            # Train adapter
            results = adapter.train_on_task(
                task_id,
                train_dataset,
                eval_dataset,
                training_args,
                strategy_config=strategy_config,
                **kwargs
            )

            # Save adapter
            adapter.save_adapter()

            logger.info(f"Successfully trained adapter {adapter_id} on task {task_id}")

            return {
                "adapter_id": adapter_id,
                "task_id": task_id,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error training adapter: {str(e)}")

            # Update status to error
            self.registry.update_adapter_status(
                adapter_id,
                CLPEFTAdapterStatus.ERROR
            )

            raise

    def evaluate_adapter(
        self,
        adapter_id: str,
        task_id: str,
        eval_dataset,
        metric_fn=None
    ) -> Dict[str, Any]:
        """
        Evaluate an adapter on a task.

        Args:
            adapter_id: Adapter ID
            task_id: Task ID
            eval_dataset: Evaluation dataset
            metric_fn: Function to compute metrics (optional)

        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating adapter {adapter_id} on task {task_id}")

        try:
            # Load adapter
            adapter = CLPEFTAdapter.load_adapter(adapter_id, registry=self.registry)

            # Evaluate adapter
            results = adapter.evaluate_on_task(task_id, eval_dataset, metric_fn)

            logger.info(f"Successfully evaluated adapter {adapter_id} on task {task_id}")

            return {
                "adapter_id": adapter_id,
                "task_id": task_id,
                "results": results
            }

        except Exception as e:
            logger.error(f"Error evaluating adapter: {str(e)}")
            raise

    def compute_forgetting(
        self,
        adapter_id: str,
        task_id: str,
        eval_dataset,
        metric_key: str = "eval_loss"
    ) -> Dict[str, Any]:
        """
        Compute forgetting for a task.

        Args:
            adapter_id: Adapter ID
            task_id: Task ID
            eval_dataset: Evaluation dataset
            metric_key: Key for the metric to use for forgetting calculation

        Returns:
            Forgetting metric
        """
        logger.info(f"Computing forgetting for adapter {adapter_id} on task {task_id}")

        try:
            # Load adapter
            adapter = CLPEFTAdapter.load_adapter(adapter_id, registry=self.registry)

            # Compute forgetting
            forgetting = adapter.compute_forgetting(task_id, eval_dataset, metric_key)

            logger.info(f"Successfully computed forgetting for adapter {adapter_id} on task {task_id}")

            return {
                "adapter_id": adapter_id,
                "task_id": task_id,
                "forgetting": forgetting,
                "metric_key": metric_key
            }

        except Exception as e:
            logger.error(f"Error computing forgetting: {str(e)}")
            raise

    def generate_text(
        self,
        adapter_id: str,
        prompt: str,
        **kwargs
    ) -> str:
        """
        Generate text using an adapter.

        Args:
            adapter_id: Adapter ID
            prompt: Input prompt
            **kwargs: Additional arguments for the generate method

        Returns:
            Generated text
        """
        logger.info(f"Generating text with adapter {adapter_id}")

        try:
            # Load adapter
            adapter = CLPEFTAdapter.load_adapter(adapter_id, registry=self.registry)

            # Generate text
            generated_text = adapter.generate(prompt, **kwargs)

            return generated_text

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

# Singleton instance
_cl_peft_service = None

def get_cl_peft_service() -> CLPEFTService:
    """
    Get the singleton instance of the CL-PEFT service.

    Returns:
        CLPEFTService instance
    """
    global _cl_peft_service
    if _cl_peft_service is None:
        _cl_peft_service = CLPEFTService()
    return _cl_peft_service
