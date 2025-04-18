"""
LoRA Training Script for Medical Contradiction Detection Models.

This script trains LoRA adapters for GPT-4o-Mini on medical contradiction detection,
creating lightweight fine-tuned models that can be integrated with the Medical Research
Synthesizer for improved contradiction detection capabilities.

Key features:
- Training data preparation from CSV/JSON sources
- LoRA adapter training with optimized quantization
- Evaluation on contradiction detection benchmarks
- Integration with model registry for versioning

Requires PEFT, transformers, and accelerate libraries.
"""

import os
import json
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Import PEFT for LoRA
try:
    from peft import (
        LoraConfig, 
        TaskType, 
        get_peft_model, 
        PeftModel,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.models.model_registry import (
    ModelRegistry, ModelStatus, ModelMetrics, ModelFramework, get_model_registry
)
from asf.medical.ml.models.lora_adapter import (
    LoraAdapter,
    LoraAdapterConfig,
    QuantizationMode
)

# Set up logger
logger = get_logger(__name__)

# Define contradiction types
CONTRADICTION_TYPES = [
    "no_contradiction",
    "direct",
    "methodological",
    "population",
    "temporal",
    "partial",
    "contextual",
    "terminological"
]

class ContradictionDataProcessor:
    """
    Processor for preparing contradiction detection datasets.
    
    This class handles loading, preprocessing, and tokenizing
    medical contradiction datasets for training.
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        task_type: str = "classification",  # or "generation"
        label_names: Optional[List[str]] = None,
    ):
        """
        Initialize the contradiction data processor.
        
        Args:
            tokenizer: Tokenizer to use for tokenizing the data.
            max_seq_length: Maximum sequence length for tokenization.
            task_type: Type of task, either "classification" or "generation".
            label_names: Names of labels for classification tasks.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.task_type = task_type
        self.label_names = label_names or CONTRADICTION_TYPES
        
        # Create label mapping
        self.label2id = {label: i for i, label in enumerate(self.label_names)}
        self.id2label = {i: label for i, label in enumerate(self.label_names)}
    
    def prepare_dataset(
        self,
        data_path: str,
        split_ratio: float = 0.1,
        seed: int = 42
    ) -> DatasetDict:
        """
        Prepare dataset from file.
        
        Args:
            data_path: Path to the dataset file (CSV/JSON).
            split_ratio: Ratio of data to use for validation.
            seed: Random seed for splitting.
            
        Returns:
            DatasetDict with train and validation splits.
        """
        # Load dataset based on file extension
        extension = Path(data_path).suffix.lower()
        
        if extension == ".csv":
            df = pd.read_csv(data_path)
            dataset = Dataset.from_pandas(df)
        elif extension in (".json", ".jsonl"):
            dataset = load_dataset("json", data_files=data_path)["train"]
        else:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Validate dataset columns
        required_columns = ["claim1", "claim2"]
        if self.task_type == "classification":
            required_columns.append("label")
        
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}")
        
        # Split dataset
        train_test = dataset.train_test_split(test_size=split_ratio, seed=seed)
        
        # Rename test to validation
        return DatasetDict({
            "train": train_test["train"],
            "validation": train_test["test"]
        })
    
    def preprocess_classification(self, example):
        """
        Preprocess an example for classification.
        
        Args:
            example: Dataset example with claim1, claim2, and label.
            
        Returns:
            Processed example with input_ids, attention_mask, and labels.
        """
        # Format input text
        text = f"Claim 1: {example['claim1']}\nClaim 2: {example['claim2']}\nContradiction type:"
        
        # Tokenize
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Convert label to id
        if isinstance(example["label"], str):
            label_id = self.label2id.get(example["label"], 0)
        else:
            label_id = int(example["label"])
        
        # Add label to tokenized output
        tokenized["labels"] = label_id
        
        # Remove batch dimension
        return {k: v[0] if isinstance(v, torch.Tensor) else v for k, v in tokenized.items()}
    
    def preprocess_generation(self, example):
        """
        Preprocess an example for generative training.
        
        Args:
            example: Dataset example with claim1, claim2, and label.
            
        Returns:
            Processed example with input_ids, attention_mask, and labels.
        """
        # Format input text for instruction tuning
        if isinstance(example["label"], str):
            label = example["label"]
        else:
            label = self.id2label.get(int(example["label"]), "unknown")
            
        # Create instruction format
        instruction = (
            f"Analyze the following medical claims for contradictions:\n\n"
            f"Claim 1: {example['claim1']}\n\n"
            f"Claim 2: {example['claim2']}\n\n"
            f"Determine if these claims contradict each other and classify the type of contradiction."
        )
        
        # Create completion with the label
        completion = f"Contradiction type: {label}"
        
        if label != "no_contradiction" and "explanation" in example:
            completion += f"\n\nExplanation: {example['explanation']}"
        
        # Combine instruction and completion
        full_text = f"{instruction}\n\n{completion}"
        
        # Tokenize
        tokenized = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt"
        )
        
        # Add input and output for generative training
        input_ids = tokenized["input_ids"][0]
        labels = input_ids.clone()
        
        # Find the start of the completion
        completion_tokens = self.tokenizer(completion, add_special_tokens=False)["input_ids"]
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        
        # Set labels to -100 for instruction part (we don't want to train on predicting the instruction)
        labels[:len(instruction_tokens)] = -100
        
        result = {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"][0],
            "labels": labels
        }
        
        return result
    
    def preprocess_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Preprocess the dataset for training.
        
        Args:
            dataset: Dataset to preprocess.
            
        Returns:
            Preprocessed dataset.
        """
        # Choose preprocessing function based on task type
        preprocess_fn = (
            self.preprocess_generation if self.task_type == "generation" 
            else self.preprocess_classification
        )
        
        # Apply preprocessing
        processed = dataset.map(
            preprocess_fn,
            remove_columns=dataset["train"].column_names,
            desc="Preprocessing dataset"
        )
        
        return processed

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics for contradiction classification.
    
    Args:
        eval_pred: EvalPrediction object with predictions and labels.
        
    Returns:
        Dictionary of metrics.
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_weighted": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }

def train_lora_adapter(
    base_model_name: str = "gpt-4o-mini",
    adapter_name: str = "contradiction-detection",
    adapter_version: str = None,
    data_path: str = None,
    output_dir: str = None,
    task_type: str = "classification",  # or "generation"
    quantization: str = "int4",
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    num_epochs: int = 3,
    max_seq_length: int = 512,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    register_model: bool = True,
    push_to_hub: bool = False,
    hub_repo: str = None,
    hub_token: str = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train a LoRA adapter for contradiction detection.
    
    Args:
        base_model_name: Base model to fine-tune.
        adapter_name: Name for the adapter.
        adapter_version: Version for the adapter (default: timestamp).
        data_path: Path to training data.
        output_dir: Directory to save adapter.
        task_type: Type of task ("classification" or "generation").
        quantization: Quantization method ("none", "int8", "int4", "gptq").
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        num_epochs: Number of training epochs.
        max_seq_length: Maximum sequence length.
        lora_r: LoRA attention dimension.
        lora_alpha: Alpha parameter for LoRA scaling.
        lora_dropout: LoRA dropout rate.
        register_model: Whether to register model in model registry.
        push_to_hub: Whether to push model to Hugging Face Hub.
        hub_repo: Hub repository name.
        hub_token: Hub token for authentication.
        random_seed: Random seed.
        
    Returns:
        Dictionary with training results.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required for LoRA training. "
                         "Install it with: pip install peft")
    
    # Set default adapter version if not provided
    if adapter_version is None:
        adapter_version = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(
            "models", "adapters", 
            adapter_name, adapter_version
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Start timing
    start_time = time.time()
    training_metrics = {}
    
    try:
        # Create LoRA adapter config
        adapter_config = LoraAdapterConfig(
            base_model_name=base_model_name,
            adapter_name=adapter_name,
            adapter_version=adapter_version,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantization_mode=quantization,
            task_type="SEQ_CLS" if task_type == "classification" else "CAUSAL_LM"
        )
        
        # Initialize adapter
        lora_adapter = LoraAdapter(adapter_config)
        
        # Load base model
        lora_adapter.load_base_model()
        
        # Get tokenizer
        tokenizer = lora_adapter.tokenizer
        
        # Set label names for tokenizer
        if task_type == "classification":
            tokenizer.label2id = {label: i for i, label in enumerate(CONTRADICTION_TYPES)}
            tokenizer.id2label = {i: label for i, label in enumerate(CONTRADICTION_TYPES)}
        
        # Load and process dataset
        processor = ContradictionDataProcessor(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            task_type=task_type,
            label_names=CONTRADICTION_TYPES
        )
        
        # Load dataset from file
        dataset = processor.prepare_dataset(data_path)
        
        # Preprocess dataset
        processed_dataset = processor.preprocess_dataset(dataset)
        
        # Create LoRA model
        lora_adapter.create_adapter_model()
        model = lora_adapter.adapter_model
        
        # Calculate training steps
        train_size = len(processed_dataset["train"])
        steps_per_epoch = train_size // batch_size
        total_steps = steps_per_epoch * num_epochs
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_total_limit=3,
            remove_unused_columns=False,
            push_to_hub=push_to_hub,
            hub_model_id=hub_repo,
            hub_token=hub_token,
            load_best_model_at_end=True,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_strategy="steps",
            logging_steps=steps_per_epoch // 5,
            report_to="tensorboard",
            seed=random_seed,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,  # Helpful for colab environments
        )
        
        # Create trainer
        if task_type == "classification":
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset["train"],
                eval_dataset=processed_dataset["validation"],
                compute_metrics=compute_metrics
            )
        else:
            # For generative models
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset["train"],
                eval_dataset=processed_dataset["validation"],
            )
        
        # Train model
        logger.info(f"Starting LoRA training with {train_size} examples for {num_epochs} epochs")
        train_result = trainer.train()
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Evaluate model
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Also save as adapter only
        model.save_pretrained(os.path.join(output_dir, "adapter"))
        
        # Save adapter configuration
        with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
            json.dump(adapter_config.model_dump(), f, indent=2)
        
        # Save training metrics
        training_metrics = {
            "train_loss": float(train_result.metrics.get("train_loss", 0.0)),
            "eval_loss": float(eval_result.get("eval_loss", 0.0)),
            "accuracy": float(eval_result.get("eval_accuracy", eval_result.get("accuracy", 0.0))),
            "f1_score": float(eval_result.get("eval_f1_weighted", eval_result.get("f1_weighted", 0.0))),
            "precision": float(eval_result.get("eval_precision", eval_result.get("precision", 0.0))),
            "recall": float(eval_result.get("eval_recall", eval_result.get("recall", 0.0))),
            "training_time": training_time,
            "epochs": num_epochs,
            "train_samples": train_size,
            "eval_samples": len(processed_dataset["validation"])
        }
        
        # Register model in model registry if requested
        if register_model:
            registry = get_model_registry()
            
            # Register the adapter as a model in the registry
            registry_metrics = ModelMetrics(
                accuracy=training_metrics["accuracy"],
                precision=training_metrics["precision"],
                recall=training_metrics["recall"],
                f1_score=training_metrics["f1_score"]
            )
            
            # Register model
            model_metadata = registry.register_model(
                name=adapter_name,
                version=adapter_version,
                framework=ModelFramework.CUSTOM,
                description=f"LoRA adapter for {base_model_name} for contradiction detection",
                status=ModelStatus.STAGING,
                metrics=registry_metrics,
                path=output_dir,
                parameters={
                    "base_model": base_model_name,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "quantization": quantization,
                    "task_type": task_type,
                    "training_args": training_args.to_dict()
                }
            )
            
            training_metrics["model_registry_id"] = model_metadata.id
        
        # Save metrics to file
        with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
            json.dump(training_metrics, f, indent=2)
        
        logger.info(f"Adapter training completed in {training_time:.2f} seconds")
        logger.info(f"Adapter saved to {output_dir}")
        
        return {
            "status": "success",
            "metrics": training_metrics,
            "output_dir": output_dir
        }
        
    except Exception as e:
        logger.error(f"Error during LoRA training: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "metrics": training_metrics,
            "output_dir": output_dir
        }

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA adapter for contradiction detection")
    
    # Model configuration
    parser.add_argument("--base_model", type=str, default="gpt-4o-mini",
                        help="Base model to fine-tune")
    parser.add_argument("--adapter_name", type=str, default="contradiction-detection",
                        help="Name for the adapter")
    parser.add_argument("--adapter_version", type=str, default=None,
                        help="Version for the adapter")
    
    # Data and output paths
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to training data (CSV/JSON)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save adapter")
    
    # Training parameters
    parser.add_argument("--task_type", type=str, default="classification",
                        choices=["classification", "generation"],
                        help="Task type (classification or generation)")
    parser.add_argument("--quantization", type=str, default="int4",
                        choices=["none", "int8", "int4", "gptq"],
                        help="Quantization method")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    
    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout rate")
    
    # Model registry and HuggingFace Hub
    parser.add_argument("--register_model", action="store_true",
                        help="Register model in model registry")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_repo", type=str, default=None,
                        help="Hub repository name")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="Hub token for authentication")
    
    # Other
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    result = train_lora_adapter(
        base_model_name=args.base_model,
        adapter_name=args.adapter_name,
        adapter_version=args.adapter_version,
        data_path=args.data_path,
        output_dir=args.output_dir,
        task_type=args.task_type,
        quantization=args.quantization,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        register_model=args.register_model,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
        hub_token=args.hub_token,
        random_seed=args.seed
    )
    
    if result["status"] == "success":
        print(f"Training completed successfully. Model saved to: {result['output_dir']}")
    else:
        print(f"Training failed: {result.get('error', 'Unknown error')}")