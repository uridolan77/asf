"""
PEFT Integration Module

This module provides integration between the ASF framework and Parameter-Efficient Fine-Tuning methods.
"""

import time
import os
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable


class PEFTIntegration:
    """
    Integrates ASF with Parameter-Efficient Fine-Tuning methods.
    
    This class provides a bridge between the ASF framework and PEFT methods like LoRA,
    enabling efficient adaptation of large language models for specific domains or tasks.
    """
    
    def __init__(
        self, 
        base_model: Any = None, 
        confidence_ecosystem: Any = None
    ):
        """
        Initialize PEFT integration.
        
        Args:
            base_model: Base LLM model (optional)
            confidence_ecosystem: Reference to the confidence ecosystem (optional)
        """
        self.base_model = base_model
        self.confidence_ecosystem = confidence_ecosystem
        self.adapters: Dict[str, Dict[str, Any]] = {}  # Map of adapter_id -> adapter_info
        self.is_initialized = False
        self.adapter_usage_stats: Dict[str, Dict[str, Any]] = {}  # Stats for each adapter
        self.adapter_save_dir = "adapters"  # Directory to save adapters
    
    async def initialize(self, base_model: Any = None) -> bool:
        """
        Initialize PEFT integration.
        
        Args:
            base_model: Base LLM model (optional)
            
        Returns:
            Success flag
        """
        if base_model:
            self.base_model = base_model
        
        if not self.base_model:
            return False
        
        # Create adapter save directory if it doesn't exist
        os.makedirs(self.adapter_save_dir, exist_ok=True)
        
        self.is_initialized = True
        return True
    
    async def create_adapter(
        self, 
        adapter_id: str, 
        adapter_config: Dict[str, Any]
    ) -> bool:
        """
        Create a new PEFT adapter.
        
        Args:
            adapter_id: Adapter ID
            adapter_config: Configuration for the adapter
            
        Returns:
            Success flag
        """
        if not self.is_initialized:
            return False
        
        try:
            # Import here to avoid dependency if not used
            from peft import LoraConfig, get_peft_model
            import torch
            
            # Create LoRA configuration
            lora_config = LoraConfig(
                r=adapter_config.get("lora_r", 8),
                lora_alpha=adapter_config.get("lora_alpha", 16),
                target_modules=adapter_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=adapter_config.get("lora_dropout", 0.05),
                bias=adapter_config.get("bias", "none"),
                task_type=adapter_config.get("task_type", "CAUSAL_LM")
            )
            
            # Create PEFT model
            peft_model = get_peft_model(self.base_model, lora_config)
            
            # Store adapter information
            self.adapters[adapter_id] = {
                "model": peft_model,
                "config": adapter_config,
                "created_at": time.time(),
                "last_used": time.time()
            }
            
            # Initialize usage stats
            self.adapter_usage_stats[adapter_id] = {
                "usage_count": 0,
                "total_tokens_processed": 0,
                "total_generations": 0,
                "generation_times": []
            }
            
            return True
        except Exception as e:
            print(f"Error creating adapter: {e}")
            return False
    
    async def get_adapter(self, adapter_id: str) -> Optional[Any]:
        """
        Get a specific adapter.
        
        Args:
            adapter_id: Adapter ID
            
        Returns:
            Adapter model or None if not found
        """
        return self.adapters.get(adapter_id, {}).get("model")
    
    async def list_adapters(self) -> List[str]:
        """
        List all available adapters.
        
        Returns:
            List of adapter IDs
        """
        return list(self.adapters.keys())
    
    async def save_adapter(self, adapter_id: str, path: Optional[str] = None) -> Optional[str]:
        """
        Save an adapter to disk.
        
        Args:
            adapter_id: Adapter ID
            path: Path to save the adapter (optional)
            
        Returns:
            Path where the adapter was saved, or None if failed
        """
        if adapter_id not in self.adapters:
            return None
            
        adapter = self.adapters[adapter_id]
        
        try:
            # Determine save path
            if path is None:
                path = os.path.join(self.adapter_save_dir, adapter_id)
                
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save adapter model
            adapter["model"].save_pretrained(path)
            
            # Save adapter config
            config_path = os.path.join(path, "adapter_config.json")
            with open(config_path, "w") as f:
                json.dump(adapter["config"], f, indent=2)
                
            # Save metadata
            metadata_path = os.path.join(path, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "adapter_id": adapter_id,
                    "created_at": adapter["created_at"],
                    "last_used": adapter["last_used"],
                    "usage_stats": self.adapter_usage_stats.get(adapter_id, {})
                }, f, indent=2)
                
            return path
        except Exception as e:
            print(f"Error saving adapter: {e}")
            return None
    
    async def load_adapter(self, adapter_id: str, path: str) -> bool:
        """
        Load an adapter from disk.
        
        Args:
            adapter_id: Adapter ID
            path: Path to load the adapter from
            
        Returns:
            Success flag
        """
        if not self.is_initialized:
            return False
            
        try:
            # Import here to avoid dependency if not used
            from peft import PeftModel
            
            # Load adapter model
            peft_model = PeftModel.from_pretrained(self.base_model, path)
            
            # Load adapter config
            config_path = os.path.join(path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    adapter_config = json.load(f)
            else:
                adapter_config = {}
                
            # Load metadata
            metadata_path = os.path.join(path, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                    created_at = metadata.get("created_at", time.time())
                    last_used = metadata.get("last_used", time.time())
                    usage_stats = metadata.get("usage_stats", {})
            else:
                created_at = time.time()
                last_used = time.time()
                usage_stats = {}
                
            # Store adapter information
            self.adapters[adapter_id] = {
                "model": peft_model,
                "config": adapter_config,
                "created_at": created_at,
                "last_used": last_used
            }
            
            # Initialize usage stats
            self.adapter_usage_stats[adapter_id] = usage_stats or {
                "usage_count": 0,
                "total_tokens_processed": 0,
                "total_generations": 0,
                "generation_times": []
            }
            
            return True
        except Exception as e:
            print(f"Error loading adapter: {e}")
            return False
    
    async def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter.
        
        Args:
            adapter_id: Adapter ID
            
        Returns:
            Success flag
        """
        if adapter_id not in self.adapters:
            return False
            
        # Remove from adapters dict
        del self.adapters[adapter_id]
        
        # Remove from usage stats
        if adapter_id in self.adapter_usage_stats:
            del self.adapter_usage_stats[adapter_id]
            
        return True
    
    async def generate_with_adapter(
        self, 
        adapter_id: str, 
        prompt: str, 
        tokenizer: Any,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **generation_kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text using a specific adapter.
        
        Args:
            adapter_id: Adapter ID
            prompt: Input prompt
            tokenizer: Tokenizer to use
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text or list of generated texts
        """
        adapter = await self.get_adapter(adapter_id)
        if adapter is None:
            return f"Adapter {adapter_id} not found."
        
        try:
            # Import here to avoid dependency if not used
            import torch
            
            # Record start time
            start_time = time.time()
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt").to(adapter.device)
            input_tokens = inputs.input_ids.shape[1]
            
            # Set up generation parameters
            gen_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_return_sequences": num_return_sequences,
                "pad_token_id": tokenizer.eos_token_id
            }
            
            # Add any additional kwargs
            gen_kwargs.update(generation_kwargs)
            
            # Generate output
            with torch.no_grad():
                outputs = adapter.generate(
                    inputs.input_ids,
                    **gen_kwargs
                )
            
            # Decode output
            if num_return_sequences == 1:
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
                result = decoded_output
            else:
                decoded_outputs = [
                    tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]
                result = decoded_outputs
            
            # Record end time
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Update adapter stats
            self.adapters[adapter_id]["last_used"] = end_time
            
            stats = self.adapter_usage_stats[adapter_id]
            stats["usage_count"] += 1
            stats["total_tokens_processed"] += input_tokens + (outputs.shape[1] - input_tokens) * num_return_sequences
            stats["total_generations"] += 1
            stats["generation_times"].append(generation_time)
            
            # Trim generation times list if it gets too long
            if len(stats["generation_times"]) > 100:
                stats["generation_times"] = stats["generation_times"][-100:]
            
            return result
        except Exception as e:
            return f"Error generating with adapter: {e}"
    
    async def confidence_weighted_generate(
        self, 
        prompt: str, 
        tokenizer: Any,
        adapter_ids: Optional[List[str]] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text using multiple adapters weighted by confidence.
        
        Args:
            prompt: Input prompt
            tokenizer: Tokenizer to use
            adapter_ids: List of adapter IDs to use (optional)
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        if adapter_ids is None:
            adapter_ids = await self.list_adapters()
        
        if not adapter_ids:
            return "No adapters available."
        
        # Get confidence scores for each adapter
        adapter_confidences = {}
        for adapter_id in adapter_ids:
            if adapter_id in self.adapters:
                if self.confidence_ecosystem:
                    confidence = self.confidence_ecosystem.get_confidence(adapter_id)
                else:
                    confidence = 0.5  # Default confidence
                adapter_confidences[adapter_id] = confidence
        
        # Normalize confidences
        total_confidence = sum(adapter_confidences.values())
        if total_confidence == 0:
            # Equal weighting if all confidences are 0
            normalized_confidences = {
                adapter_id: 1.0 / len(adapter_confidences)
                for adapter_id in adapter_confidences
            }
        else:
            normalized_confidences = {
                adapter_id: confidence / total_confidence
                for adapter_id, confidence in adapter_confidences.items()
            }
        
        # Generate from each adapter
        adapter_outputs = {}
        for adapter_id in normalized_confidences:
            adapter_outputs[adapter_id] = await self.generate_with_adapter(
                adapter_id, 
                prompt, 
                tokenizer,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                **generation_kwargs
            )
        
        # For now, just return the output from the highest confidence adapter
        # In practice, we would need a more sophisticated method to combine outputs
        best_adapter_id = max(normalized_confidences, key=normalized_confidences.get)
        return adapter_outputs[best_adapter_id]
    
    async def fine_tune_adapter(
        self, 
        adapter_id: str, 
        train_data: List[Dict[str, str]],
        eval_data: Optional[List[Dict[str, str]]] = None,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        tokenizer: Any = None,
        **training_kwargs
    ) -> Dict[str, Any]:
        """
        Fine-tune an adapter on training data.
        
        Args:
            adapter_id: Adapter ID
            train_data: Training data (list of dicts with 'input' and 'output' keys)
            eval_data: Evaluation data (optional)
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            tokenizer: Tokenizer to use
            **training_kwargs: Additional training parameters
            
        Returns:
            Training results
        """
        if adapter_id not in self.adapters:
            return {"status": "error", "message": f"Adapter {adapter_id} not found."}
            
        adapter = self.adapters[adapter_id]
        
        try:
            # Import here to avoid dependency if not used
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import Trainer, TrainingArguments
            
            # Define dataset class
            class TextDataset(Dataset):
                def __init__(self, data, tokenizer, max_length=512):
                    self.data = data
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                    
                def __len__(self):
                    return len(self.data)
                    
                def __getitem__(self, idx):
                    item = self.data[idx]
                    input_text = item["input"]
                    output_text = item["output"]
                    
                    # Combine input and output for causal language modeling
                    full_text = f"{input_text} {output_text}"
                    
                    # Tokenize
                    encodings = self.tokenizer(
                        full_text,
                        max_length=self.max_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    )
                    
                    # Create labels (same as input_ids for causal LM)
                    encodings["labels"] = encodings["input_ids"].clone()
                    
                    # Convert to tensors
                    return {
                        key: val.squeeze(0) for key, val in encodings.items()
                    }
            
            # Create datasets
            train_dataset = TextDataset(train_data, tokenizer)
            eval_dataset = TextDataset(eval_data, tokenizer) if eval_data else None
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=f"./results/{adapter_id}",
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=f"./logs/{adapter_id}",
                logging_steps=10,
                save_steps=100,
                eval_steps=100 if eval_dataset else None,
                **training_kwargs
            )
            
            # Create trainer
            trainer = Trainer(
                model=adapter["model"],
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset
            )
            
            # Train the model
            train_result = trainer.train()
            
            # Save the adapter
            await self.save_adapter(adapter_id)
            
            # Return training results
            return {
                "status": "success",
                "adapter_id": adapter_id,
                "train_loss": train_result.training_loss,
                "train_steps": train_result.global_step,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0)
            }
        except Exception as e:
            return {"status": "error", "message": f"Error fine-tuning adapter: {e}"}
    
    def get_adapter_info(self, adapter_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about adapters.
        
        Args:
            adapter_id: Adapter ID (optional)
            
        Returns:
            Dictionary of adapter information
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        if adapter_id:
            if adapter_id not in self.adapters:
                return {"status": "adapter_not_found"}
            
            adapter = self.adapters[adapter_id]
            stats = self.adapter_usage_stats.get(adapter_id, {})
            
            # Calculate average generation time
            generation_times = stats.get("generation_times", [])
            avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
            
            return {
                "adapter_id": adapter_id,
                "config": adapter["config"],
                "created_at": adapter["created_at"],
                "last_used": adapter["last_used"],
                "usage_count": stats.get("usage_count", 0),
                "total_tokens_processed": stats.get("total_tokens_processed", 0),
                "total_generations": stats.get("total_generations", 0),
                "average_generation_time": avg_gen_time
            }
        else:
            adapter_info = {}
            for adapter_id, adapter in self.adapters.items():
                stats = self.adapter_usage_stats.get(adapter_id, {})
                
                # Calculate average generation time
                generation_times = stats.get("generation_times", [])
                avg_gen_time = sum(generation_times) / len(generation_times) if generation_times else 0
                
                adapter_info[adapter_id] = {
                    "config": adapter["config"],
                    "created_at": adapter["created_at"],
                    "last_used": adapter["last_used"],
                    "usage_count": stats.get("usage_count", 0),
                    "total_tokens_processed": stats.get("total_tokens_processed", 0),
                    "total_generations": stats.get("total_generations", 0),
                    "average_generation_time": avg_gen_time
                }
            
            return {
                "adapter_count": len(self.adapters),
                "adapters": adapter_info
            }
