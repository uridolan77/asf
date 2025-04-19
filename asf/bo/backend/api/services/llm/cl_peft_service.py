"""
CL-PEFT service for BO backend.

This module provides a service for interacting with the CL-PEFT functionality,
including adapter management, training, evaluation, and text generation.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import from medical.ml.cl_peft instead of asf.medical.ml.cl_peft
try:
    import sys
    import os
    # Add the project root to sys.path if not already there
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to Python path in cl_peft_service.py")

    from medical.ml.cl_peft import (
        CLPEFTAdapterConfig,
        CLStrategy,
        QuantizationMode,
        CLPEFTAdapterStatus,
        get_target_modules_for_model
    )
    from medical.ml.services.cl_peft_service import CLPEFTService as MedicalCLPEFTService
    print("Successfully imported from medical.ml.cl_peft")
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import from medical.ml.cl_peft: {str(e)}")

    # Define mock classes for development/testing
    class CLStrategy:
        NAIVE = "naive"
        EWC = "ewc"
        REPLAY = "replay"
        GENERATIVE_REPLAY = "generative_replay"
        ORTHOGONAL_LORA = "orthogonal_lora"
        ADAPTIVE_SVD = "adaptive_svd"
        MASK_BASED = "mask_based"

    class QuantizationMode:
        NONE = "none"
        INT8 = "int8"
        INT4 = "int4"

    class CLPEFTAdapterStatus:
        CREATED = "created"
        TRAINING = "training"
        READY = "ready"
        ERROR = "error"

    class CLPEFTAdapterConfig:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    def get_target_modules_for_model(model_name):
        return ["q_proj", "v_proj", "k_proj", "o_proj"]

    class MedicalCLPEFTService:
        def __init__(self):
            pass

        def create_adapter(self, config):
            return "mock-adapter-id"

        def get_adapter(self, adapter_id):
            return {"id": adapter_id, "status": CLPEFTAdapterStatus.READY}

        def list_adapters(self, filter_by=None):
            return [{"id": "mock-adapter-id", "status": CLPEFTAdapterStatus.READY}]

        def delete_adapter(self, adapter_id):
            return True

        def train_adapter(self, *args, **kwargs):
            return {"success": True, "metrics": {"loss": 0.1}}

        def evaluate_adapter(self, *args, **kwargs):
            return {"success": True, "metrics": {"eval_loss": 0.2}}

        def compute_forgetting(self, *args, **kwargs):
            return {"forgetting": 0.05}

        def generate_text(self, adapter_id, prompt, **kwargs):
            return f"Generated text for prompt: {prompt}"

logger = logging.getLogger(__name__)

class CLPEFTService:
    """
    Service for interacting with CL-PEFT functionality.
    """

    def __init__(self, medical_service: Optional[MedicalCLPEFTService] = None):
        """
        Initialize the CL-PEFT service.

        Args:
            medical_service: Optional medical CL-PEFT service instance
        """
        try:
            # Try to import from medical.ml.services instead of asf.medical.ml.services
            from medical.ml.services.cl_peft_service import get_cl_peft_service
            self.medical_service = medical_service or get_cl_peft_service()
        except ImportError:
            # Use mock service if import fails
            self.medical_service = medical_service or MedicalCLPEFTService()

        logger.info("Initialized BO CL-PEFT service")

    def create_adapter(self, config: CLPEFTAdapterConfig) -> str:
        """
        Create a new CL-PEFT adapter.

        Args:
            config: Configuration for the adapter

        Returns:
            Adapter ID
        """
        return self.medical_service.create_adapter(config)

    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get adapter metadata.

        Args:
            adapter_id: Adapter ID

        Returns:
            Adapter metadata or None if not found
        """
        return self.medical_service.get_adapter(adapter_id)

    def list_adapters(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List all adapters, optionally filtered.

        Args:
            filter_by: Filter criteria

        Returns:
            List of adapter metadata
        """
        return self.medical_service.list_adapters(filter_by)

    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter.

        Args:
            adapter_id: Adapter ID

        Returns:
            Success flag
        """
        return self.medical_service.delete_adapter(adapter_id)

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
        return self.medical_service.train_adapter(
            adapter_id,
            task_id,
            train_dataset,
            eval_dataset,
            training_args,
            strategy_config,
            **kwargs
        )

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
        return self.medical_service.evaluate_adapter(
            adapter_id,
            task_id,
            eval_dataset,
            metric_fn
        )

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
            metric_key: Metric key for forgetting calculation

        Returns:
            Forgetting metric
        """
        return self.medical_service.compute_forgetting(
            adapter_id,
            task_id,
            eval_dataset,
            metric_key
        )

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
        return self.medical_service.generate_text(
            adapter_id,
            prompt,
            **kwargs
        )

    def get_available_cl_strategies(self) -> List[Dict[str, Any]]:
        """
        Get available CL strategies.

        Returns:
            List of available CL strategies with metadata
        """
        strategies = [
            {
                "id": CLStrategy.NAIVE,
                "name": "Naive Sequential Fine-Tuning",
                "description": "Simple sequential fine-tuning without any CL mechanisms."
            },
            {
                "id": CLStrategy.EWC,
                "name": "Elastic Weight Consolidation (EWC)",
                "description": "Prevents forgetting by adding a penalty for changing parameters important for previous tasks."
            },
            {
                "id": CLStrategy.REPLAY,
                "name": "Experience Replay",
                "description": "Prevents forgetting by replaying examples from previous tasks during training."
            },
            {
                "id": CLStrategy.GENERATIVE_REPLAY,
                "name": "Generative Replay",
                "description": "Uses a generative model to create synthetic examples from previous tasks."
            },
            {
                "id": CLStrategy.ORTHOGONAL_LORA,
                "name": "Orthogonal LoRA (O-LoRA)",
                "description": "Enforces orthogonality between LoRA updates for different tasks."
            },
            {
                "id": CLStrategy.ADAPTIVE_SVD,
                "name": "Adaptive SVD",
                "description": "Projects gradient updates onto orthogonal subspaces using Singular Value Decomposition."
            },
            {
                "id": CLStrategy.MASK_BASED,
                "name": "Mask-Based CL",
                "description": "Uses binary masks to protect parameters important for previous tasks."
            }
        ]

        return strategies

    def get_available_peft_methods(self) -> List[Dict[str, Any]]:
        """
        Get available PEFT methods.

        Returns:
            List of available PEFT methods with metadata
        """
        methods = [
            {
                "id": "lora",
                "name": "LoRA",
                "description": "Low-Rank Adaptation for efficient fine-tuning."
            },
            {
                "id": "qlora",
                "name": "QLoRA",
                "description": "Quantized Low-Rank Adaptation for memory-efficient fine-tuning."
            }
        ]

        return methods

    def get_available_base_models(self) -> List[Dict[str, Any]]:
        """
        Get available base models.

        Returns:
            List of available base models with metadata
        """
        models = [
            {
                "id": "meta-llama/Llama-2-7b-hf",
                "name": "Llama 2 (7B)",
                "description": "Meta's Llama 2 model with 7 billion parameters."
            },
            {
                "id": "meta-llama/Llama-2-13b-hf",
                "name": "Llama 2 (13B)",
                "description": "Meta's Llama 2 model with 13 billion parameters."
            },
            {
                "id": "mistralai/Mistral-7B-v0.1",
                "name": "Mistral (7B)",
                "description": "Mistral AI's 7 billion parameter model."
            },
            {
                "id": "google/gemma-2b",
                "name": "Gemma (2B)",
                "description": "Google's Gemma model with 2 billion parameters."
            },
            {
                "id": "google/gemma-7b",
                "name": "Gemma (7B)",
                "description": "Google's Gemma model with 7 billion parameters."
            },
            {
                "id": "tiiuae/falcon-7b",
                "name": "Falcon (7B)",
                "description": "TII's Falcon model with 7 billion parameters."
            }
        ]

        return models

def get_cl_peft_service() -> CLPEFTService:
    """
    Get the CL-PEFT service instance.

    Returns:
        CL-PEFT service instance
    """
    return CLPEFTService()
