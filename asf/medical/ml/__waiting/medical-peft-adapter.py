import os
import torch
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig,
    prepare_model_for_kbit_training
)

logger = logging.getLogger(__name__)

class BiomedicNERAdapter:
    """
    A Parameter-Efficient Fine-Tuning (PEFT) adapter for biomedical NER tasks
    using continual learning to preserve knowledge across different entity types.
    """
    
    def __init__(
        self,
        base_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        adapter_name: str = "biomedical-ner-adapter",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_4bit: bool = False,
        device_map: str = "auto",
        labels_list: Optional[List[str]] = None
    ):
        """
        Initialize a biomedical NER adapter with LoRA.
        
        Args:
            base_model_name: Base model to use (PubMedBERT recommended)
            adapter_name: Name for this adapter
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            use_4bit: Whether to use 4-bit quantization (QLoRA)
            device_map: Device mapping strategy
            labels_list: List of NER labels
        """
        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.device_map = device_map
        
        # Store labels if provided
        self.labels_list = labels_list
        self.id2label = {i: label for i, label in enumerate(labels_list)} if labels_list else None
        self.label2id = {label: i for i, label in enumerate(labels_list)} if labels_list else None
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        
        # Task history for continual learning
        self.task_history = []
        self.current_task_id = None
        
        # Default target modules for PubMedBERT
        self.target_modules = ["query", "key", "value", "dense"]
        
        # Create adapter directory
        os.makedirs(f"adapters/{self.adapter_name}", exist_ok=True)
    
    def load_base_model(self):
        """Load the base model for NER tasks."""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Define quantization config for 4-bit if enabled
            quantization_config = None
            if self.use_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model with proper NER configuration
            if self.labels_list:
                self.base_model = AutoModelForTokenClassification.from_pretrained(
                    self.base_model_name,
                    num_labels=len(self.labels_list),
                    id2label=self.id2label,
                    label2id=self.label2id,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )
            else:
                # Load model and then add token classification head based on the dataset
                self.base_model = AutoModelForTokenClassification.from_pretrained(
                    self.base_model_name,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Prepare for k-bit training if using quantization
            if self.use_4bit:
                self.base_model = prepare_model_for_kbit_training(self.base_model)
            
            logger.info(f"Successfully loaded base model: {self.base_model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def create_adapter(self):
        """Create a LoRA adapter for the base model."""
        if self.base_model is None:
            self.load_base_model()
        
        try:
            logger.info("Creating LoRA adapter")
            
            # Create LoRA configuration for token classification
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.TOKEN_CLS,
                inference_mode=False,
            )
            
            # Apply LoRA adapter to model
            self.adapter_model = get_peft_model(self.base_model, lora_config)
            
            # Log trainable parameters
            self._print_trainable_parameters()
            
            logger.info("Successfully created LoRA adapter")
            return True
            
        except Exception as e:
            logger.error(f"Error creating adapter: {str(e)}")
            raise
    
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
        output_dir: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 5e-4,
        **kwargs
    ):
        """
        Train the adapter on a specific NER task.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            learning_rate: Learning rate
            **kwargs: Additional arguments for TrainingArguments
        
        Returns:
            Training results
        """
        # Set current task
        self.current_task_id = task_id
        
        # Create adapter if not already created
        if self.adapter_model is None:
            self.create_adapter()
        
        # Set output directory
        if output_dir is None:
            output_dir = f"adapters/{self.adapter_name}/{task_id}"
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            **kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logger.info(f"Training adapter on task {task_id}")
        train_results = trainer.train()
        
        # Save the adapter
        self.save_adapter(f"{output_dir}/final_adapter")
        
        # Update task history
        self.task_history.append({
            "task_id": task_id,
            "trained_at": datetime.now().isoformat(),
            "metrics": train_results.metrics
        })
        
        logger.info(f"Completed training on task {task_id}")
        return train_results
    
    def save_adapter(self, path: str):
        """
        Save the adapter model.
        
        Args:
            path: Path to save the adapter
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save adapter
        self.adapter_model.save_pretrained(path)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        
        # Save task history and configuration
        config = {
            "base_model_name": self.base_model_name,
            "adapter_name": self.adapter_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_4bit": self.use_4bit,
            "task_history": self.task_history,
            "current_task_id": self.current_task_id,
            "labels_list": self.labels_list
        }
        
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved adapter to {path}")
    
    @classmethod
    def load_adapter(cls, adapter_path: str, device_map: str = "auto"):
        """
        Load a saved adapter.
        
        Args:
            adapter_path: Path to the saved adapter
            device_map: Device mapping strategy
            
        Returns:
            BiomedicNERAdapter instance
        """
        # Load adapter configuration
        with open(f"{adapter_path}/adapter_config.json", "r") as f:
            config = json.load(f)
        
        # Create adapter instance
        adapter = cls(
            base_model_name=config["base_model_name"],
            adapter_name=config["adapter_name"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            use_4bit=config["use_4bit"],
            device_map=device_map,
            labels_list=config["labels_list"]
        )
        
        # Load tokenizer
        adapter.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        adapter.load_base_model()
        
        # Load adapter
        adapter.adapter_model = PeftModel.from_pretrained(
            adapter.base_model,
            adapter_path,
            device_map=device_map
        )
        
        # Load task history
        adapter.task_history = config["task_history"]
        adapter.current_task_id = config["current_task_id"]
        
        logger.info(f"Loaded adapter from {adapter_path}")
        return adapter
    
    def predict(self, texts: List[str], **kwargs):
        """
        Predict NER tags for a list of texts.
        
        Args:
            texts: List of texts to predict
            
        Returns:
            List of dictionaries with predictions
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            return_offsets_mapping=True,
            **kwargs
        )
        
        # Get input tensors
        input_ids = encodings.input_ids.to(self.adapter_model.device)
        attention_mask = encodings.attention_mask.to(self.adapter_model.device)
        offset_mapping = encodings.offset_mapping
        
        # Run prediction
        with torch.no_grad():
            outputs = self.adapter_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Process predictions
        predictions = []
        
        for i, text in enumerate(texts):
            # Get predictions for this text
            text_predictions = []
            logits = outputs.logits[i]
            offset_map = offset_mapping[i]
            
            # Get predicted labels
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Filter out predictions for special tokens
            valid_predictions = []
            current_entity = None
            
            for j, (offset, pred_id) in enumerate(zip(offset_map, preds)):
                # Skip special tokens (CLS, SEP, etc.)
                if offset[0] == offset[1] == 0:
                    continue
                
                # Get predicted label
                pred_label = self.id2label[pred_id] if self.id2label else str(pred_id)
                
                # Skip padding tokens
                if pred_label == "O" or pred_label.startswith("O-"):
                    if current_entity:
                        valid_predictions.append(current_entity)
                        current_entity = None
                    continue
                
                # Extract entity type (B-GENE, I-GENE, etc.)
                if pred_label.startswith("B-"):
                    if current_entity:
                        valid_predictions.append(current_entity)
                    
                    entity_type = pred_label[2:]  # Remove "B-"
                    start, end = offset
                    
                    current_entity = {
                        "entity": text[start:end],
                        "type": entity_type,
                        "start": int(start),
                        "end": int(end)
                    }
                
                elif pred_label.startswith("I-") and current_entity:
                    # Continuing an entity
                    entity_type = pred_label[2:]  # Remove "I-"
                    
                    # Only extend if it's the same entity type
                    if entity_type == current_entity["type"]:
                        start, end = offset
                        current_entity["entity"] = text[current_entity["start"]:end]
                        current_entity["end"] = int(end)
            
            # Add final entity if any
            if current_entity:
                valid_predictions.append(current_entity)
            
            # Add predictions for this text
            predictions.append({
                "text": text,
                "entities": valid_predictions
            })
        
        return predictions

class BiomedRelationExtractor:
    """
    A Parameter-Efficient Fine-Tuning (PEFT) adapter for biomedical relation extraction
    using continual learning to preserve knowledge across different relation types.
    """
    
    def __init__(
        self,
        base_model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        adapter_name: str = "biomedical-re-adapter",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_4bit: bool = False,
        device_map: str = "auto",
        relations_list: Optional[List[str]] = None
    ):
        """
        Initialize a biomedical relation extraction adapter with LoRA.
        
        Args:
            base_model_name: Base model to use (PubMedBERT recommended)
            adapter_name: Name for this adapter
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            use_4bit: Whether to use 4-bit quantization (QLoRA)
            device_map: Device mapping strategy
            relations_list: List of relation labels
        """
        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.device_map = device_map
        
        # Store relations if provided
        self.relations_list = relations_list
        self.id2relation = {i: rel for i, rel in enumerate(relations_list)} if relations_list else None
        self.relation2id = {rel: i for i, rel in enumerate(relations_list)} if relations_list else None
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        
        # Task history for continual learning
        self.task_history = []
        self.current_task_id = None
        
        # Default target modules for PubMedBERT
        self.target_modules = ["query", "key", "value", "dense"]
        
        # Create adapter directory
        os.makedirs(f"adapters/{self.adapter_name}", exist_ok=True)
    
    def load_base_model(self):
        """Load the base model for relation extraction tasks."""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Define quantization config for 4-bit if enabled
            quantization_config = None
            if self.use_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model for sequence classification (relations)
            if self.relations_list:
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model_name,
                    num_labels=len(self.relations_list),
                    id2label=self.id2relation,
                    label2id=self.relation2id,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )
            else:
                # Load model with default configuration
                self.base_model = AutoModelForSequenceClassification.from_pretrained(
                    self.base_model_name,
                    quantization_config=quantization_config,
                    device_map=self.device_map
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Prepare for k-bit training if using quantization
            if self.use_4bit:
                self.base_model = prepare_model_for_kbit_training(self.base_model)
            
            logger.info(f"Successfully loaded base model: {self.base_model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def create_adapter(self):
        """Create a LoRA adapter for the base model."""
        if self.base_model is None:
            self.load_base_model()
        
        try:
            logger.info("Creating LoRA adapter")
            
            # Create LoRA configuration for sequence classification
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_CLS,
                inference_mode=False,
            )
            
            # Apply LoRA adapter to model
            self.adapter_model = get_peft_model(self.base_model, lora_config)
            
            # Log trainable parameters
            self._print_trainable_parameters()
            
            logger.info("Successfully created LoRA adapter")
            return True
            
        except Exception as e:
            logger.error(f"Error creating adapter: {str(e)}")
            raise
    
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
        output_dir: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        learning_rate: float = 5e-4,
        **kwargs
    ):
        """
        Train the adapter on a specific relation extraction task.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            learning_rate: Learning rate
            **kwargs: Additional arguments for TrainingArguments
        
        Returns:
            Training results
        """
        # Set current task
        self.current_task_id = task_id
        
        # Create adapter if not already created
        if self.adapter_model is None:
            self.create_adapter()
        
        # Set output directory
        if output_dir is None:
            output_dir = f"adapters/{self.adapter_name}/{task_id}"
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            **kwargs
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train the model
        logger.info(f"Training adapter on task {task_id}")
        train_results = trainer.train()
        
        # Save the adapter
        self.save_adapter(f"{output_dir}/final_adapter")
        
        # Update task history
        self.task_history.append({
            "task_id": task_id,
            "trained_at": datetime.now().isoformat(),
            "metrics": train_results.metrics
        })
        
        logger.info(f"Completed training on task {task_id}")
        return train_results
    
    def save_adapter(self, path: str):
        """
        Save the adapter model.
        
        Args:
            path: Path to save the adapter
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save adapter
        self.adapter_model.save_pretrained(path)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        
        # Save task history and configuration
        config = {
            "base_model_name": self.base_model_name,
            "adapter_name": self.adapter_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_4bit": self.use_4bit,
            "task_history": self.task_history,
            "current_task_id": self.current_task_id,
            "relations_list": self.relations_list
        }
        
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved adapter to {path}")
    
    @classmethod
    def load_adapter(cls, adapter_path: str, device_map: str = "auto"):
        """
        Load a saved adapter.
        
        Args:
            adapter_path: Path to the saved adapter
            device_map: Device mapping strategy
            
        Returns:
            BiomedRelationExtractor instance
        """
        # Load adapter configuration
        with open(f"{adapter_path}/adapter_config.json", "r") as f:
            config = json.load(f)
        
        # Create adapter instance
        adapter = cls(
            base_model_name=config["base_model_name"],
            adapter_name=config["adapter_name"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            use_4bit=config["use_4bit"],
            device_map=device_map,
            relations_list=config["relations_list"]
        )
        
        # Load tokenizer
        adapter.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        adapter.load_base_model()
        
        # Load adapter
        adapter.adapter_model = PeftModel.from_pretrained(
            adapter.base_model,
            adapter_path,
            device_map=device_map
        )
        
        # Load task history
        adapter.task_history = config["task_history"]
        adapter.current_task_id = config["current_task_id"]
        
        logger.info(f"Loaded adapter from {adapter_path}")
        return adapter
    
    def predict_relation(self, entity1: str, entity2: str, context: str, **kwargs):
        """
        Predict the relation between two entities in a context.
        
        Args:
            entity1: First entity
            entity2: Second entity
            context: Context containing both entities
            
        Returns:
            Dictionary with predicted relation and confidence
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Mark the entities in the context with special tokens
        # Find entity positions
        entity1_start = context.find(entity1)
        entity2_start = context.find(entity2)
        
        if entity1_start == -1 or entity2_start == -1:
            raise ValueError(f"One or both entities not found in context: {entity1}, {entity2}")
        
        # Ensure entity1 appears before entity2 in the text
        if entity1_start > entity2_start:
            entity1, entity2 = entity2, entity1
            entity1_start, entity2_start = entity2_start, entity1_start
        
        entity1_end = entity1_start + len(entity1)
        entity2_end = entity2_start + len(entity2)
        
        # Mark entities in the context
        marked_context = (
            context[:entity1_start] + 
            f"[E1] {entity1} [/E1]" + 
            context[entity1_end:entity2_start] + 
            f"[E2] {entity2} [/E2]" + 
            context[entity2_end:]
        )
        
        # Tokenize text
        inputs = self.tokenizer(
            marked_context,
            padding=True,
            truncation=True,
            return_tensors="pt",
            **kwargs
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.adapter_model.device) for k, v in inputs.items()}
        
        # Run prediction
        with torch.no_grad():
            outputs = self.adapter_model(**inputs)
        
        # Get prediction
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=0)
        pred_id = torch.argmax(logits).item()
        confidence = probs[pred_id].item()
        
        # Get relation label
        relation = self.id2relation[pred_id] if self.id2relation else str(pred_id)
        
        return {
            "entity1": entity1,
            "entity2": entity2,
            "relation": relation,
            "confidence": confidence,
            "probabilities": {self.id2relation[i] if self.id2relation else str(i): prob.item() 
                            for i, prob in enumerate(probs)}
        }

    def predict_relations_from_entities(self, text: str, entities: List[Dict[str, Any]], **kwargs):
        """
        Predict relations between entities in a text.
        
        Args:
            text: Text containing entities
            entities: List of entity dictionaries (from NER model)
            
        Returns:
            List of dictionaries with predicted relations
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Find all possible entity pairs
        relations = []
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities):
                # Skip self-relations
                if i == j:
                    continue
                
                try:
                    # Get relation prediction
                    relation = self.predict_relation(
                        entity1["entity"],
                        entity2["entity"],
                        text,
                        **kwargs
                    )
                    
                    # Add entity information
                    relation["entity1_type"] = entity1["type"]
                    relation["entity2_type"] = entity2["type"]
                    relation["entity1_start"] = entity1["start"]
                    relation["entity1_end"] = entity1["end"]
                    relation["entity2_start"] = entity2["start"]
                    relation["entity2_end"] = entity2["end"]
                    
                    # Add to relations list
                    relations.append(relation)
                except Exception as e:
                    logger.warning(f"Error predicting relation between {entity1['entity']} and {entity2['entity']}: {str(e)}")
        
        return relations


class BiomedicalSummarizer:
    """
    A Parameter-Efficient Fine-Tuning (PEFT) adapter for biomedical text summarization
    using continual learning to preserve knowledge across different summarization tasks.
    """
    
    def __init__(
        self,
        base_model_name: str = "facebook/bart-large-cnn",
        adapter_name: str = "biomedical-summarizer",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        use_4bit: bool = False,
        device_map: str = "auto",
    ):
        """
        Initialize a biomedical summarization adapter with LoRA.
        
        Args:
            base_model_name: Base model to use (BART or similar)
            adapter_name: Name for this adapter
            lora_r: LoRA attention dimension
            lora_alpha: LoRA alpha parameter
            lora_dropout: Dropout probability for LoRA layers
            use_4bit: Whether to use 4-bit quantization (QLoRA)
            device_map: Device mapping strategy
        """
        self.base_model_name = base_model_name
        self.adapter_name = adapter_name
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_4bit = use_4bit
        self.device_map = device_map
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        
        # Task history for continual learning
        self.task_history = []
        self.current_task_id = None
        
        # Target modules for sequence-to-sequence models like BART
        self.target_modules = ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"]
        
        # Create adapter directory
        os.makedirs(f"adapters/{self.adapter_name}", exist_ok=True)
    
    def load_base_model(self):
        """Load the base model for summarization tasks."""
        try:
            logger.info(f"Loading base model: {self.base_model_name}")
            
            # Define quantization config for 4-bit if enabled
            quantization_config = None
            if self.use_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load model for sequence-to-sequence tasks
            from transformers import AutoModelForSeq2SeqLM
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                quantization_config=quantization_config,
                device_map=self.device_map
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Prepare for k-bit training if using quantization
            if self.use_4bit:
                self.base_model = prepare_model_for_kbit_training(self.base_model)
            
            logger.info(f"Successfully loaded base model: {self.base_model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading base model: {str(e)}")
            raise
    
    def create_adapter(self):
        """Create a LoRA adapter for the base model."""
        if self.base_model is None:
            self.load_base_model()
        
        try:
            logger.info("Creating LoRA adapter")
            
            # Create LoRA configuration for seq2seq models
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
            )
            
            # Apply LoRA adapter to model
            self.adapter_model = get_peft_model(self.base_model, lora_config)
            
            # Log trainable parameters
            self._print_trainable_parameters()
            
            logger.info("Successfully created LoRA adapter")
            return True
            
        except Exception as e:
            logger.error(f"Error creating adapter: {str(e)}")
            raise
    
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
        output_dir: Optional[str] = None,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_source_length: int = 1024,
        max_target_length: int = 256,
        **kwargs
    ):
        """
        Train the adapter on a specific summarization task.
        
        Args:
            task_id: Unique identifier for the task
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            output_dir: Directory to save checkpoints
            num_train_epochs: Number of training epochs
            per_device_train_batch_size: Batch size for training
            learning_rate: Learning rate
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            **kwargs: Additional arguments for TrainingArguments
        
        Returns:
            Training results
        """
        # Set current task
        self.current_task_id = task_id
        
        # Create adapter if not already created
        if self.adapter_model is None:
            self.create_adapter()
        
        # Set output directory
        if output_dir is None:
            output_dir = f"adapters/{self.adapter_name}/{task_id}"
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=max_target_length,
            load_best_model_at_end=True if eval_dataset else False,
            **kwargs
        )
        
        # Data collator for seq2seq tasks
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.adapter_model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if self.use_4bit else None,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.adapter_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train the model
        logger.info(f"Training adapter on task {task_id}")
        train_results = trainer.train()
        
        # Save the adapter
        self.save_adapter(f"{output_dir}/final_adapter")
        
        # Update task history
        self.task_history.append({
            "task_id": task_id,
            "trained_at": datetime.now().isoformat(),
            "metrics": train_results.metrics
        })
        
        logger.info(f"Completed training on task {task_id}")
        return train_results
    
    def save_adapter(self, path: str):
        """
        Save the adapter model.
        
        Args:
            path: Path to save the adapter
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save adapter
        self.adapter_model.save_pretrained(path)
        
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        
        # Save task history and configuration
        config = {
            "base_model_name": self.base_model_name,
            "adapter_name": self.adapter_name,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_4bit": self.use_4bit,
            "task_history": self.task_history,
            "current_task_id": self.current_task_id
        }
        
        with open(f"{path}/adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved adapter to {path}")
    
    @classmethod
    def load_adapter(cls, adapter_path: str, device_map: str = "auto"):
        """
        Load a saved adapter.
        
        Args:
            adapter_path: Path to the saved adapter
            device_map: Device mapping strategy
            
        Returns:
            BiomedicalSummarizer instance
        """
        # Load adapter configuration
        with open(f"{adapter_path}/adapter_config.json", "r") as f:
            config = json.load(f)
        
        # Create adapter instance
        adapter = cls(
            base_model_name=config["base_model_name"],
            adapter_name=config["adapter_name"],
            lora_r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            use_4bit=config["use_4bit"],
            device_map=device_map
        )
        
        # Load tokenizer
        adapter.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        
        # Load base model
        adapter.load_base_model()
        
        # Load adapter
        adapter.adapter_model = PeftModel.from_pretrained(
            adapter.base_model,
            adapter_path,
            device_map=device_map
        )
        
        # Load task history
        adapter.task_history = config["task_history"]
        adapter.current_task_id = config["current_task_id"]
        
        logger.info(f"Loaded adapter from {adapter_path}")
        return adapter
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 256,
        min_length: int = 50,
        num_beams: int = 4,
        **kwargs
    ):
        """
        Generate a summary for a biomedical text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of the summary
            min_length: Minimum length of the summary
            num_beams: Number of beams for beam search
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary with generated summary and metadata
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        # Prepare inputs
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.adapter_model.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            output_ids = self.adapter_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                **kwargs
            )
        
        # Decode summary
        summary = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return {
            "text": text[:1000] + "..." if len(text) > 1000 else text,  # Truncate original text for display
            "summary": summary,
            "summary_length": len(summary.split()),
            "original_length": len(text.split()),
            "compression_ratio": len(summary.split()) / len(text.split())
        }
    
    def batch_generate_summaries(self, texts: List[str], batch_size: int = 2, **kwargs):
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts: List of texts to summarize
            batch_size: Batch size for generation
            **kwargs: Additional generation parameters
            
        Returns:
            List of dictionaries with generated summaries
        """
        if self.adapter_model is None:
            raise ValueError("Adapter model not created yet")
        
        results = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Prepare inputs
            inputs = self.tokenizer(
                batch_texts,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.adapter_model.device) for k, v in inputs.items()}
            
            # Generate summaries
            with torch.no_grad():
                output_ids = self.adapter_model.generate(
                    **inputs,
                    **kwargs
                )
            
            # Decode summaries
            summaries = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            
            # Create result dictionaries
            for text, summary in zip(batch_texts, summaries):
                results.append({
                    "text": text[:1000] + "..." if len(text) > 1000 else text,
                    "summary": summary,
                    "summary_length": len(summary.split()),
                    "original_length": len(text.split()),
                    "compression_ratio": len(summary.split()) / len(text.split())
                })
        
        return results
