"""
Scientific Document Section Classifier

This module provides functionality for classifying sections of scientific documents
into standard IMRAD (Introduction, Methods, Results, and Discussion) categories
using a fine-tuned SciBERT model.
"""

import os
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IMRADSectionClassifier:
    """
    Scientific document section classifier using SciBERT.
    
    This classifier identifies standard IMRAD sections (Introduction, Methods,
    Results, and Discussion) in scientific documents using a fine-tuned SciBERT model.
    """
    
    # Standard IMRAD section types
    SECTION_TYPES = [
        "title",
        "abstract", 
        "introduction",
        "background",
        "methods", 
        "materials_and_methods",
        "results", 
        "discussion", 
        "conclusion",
        "references",
        "acknowledgments",
        "other"
    ]
    
    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the IMRAD section classifier.
        
        Args:
            model_name: SciBERT model name or path
            device: Device for PyTorch models
            cache_dir: Directory to cache models
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Initialize tokenizer and model
        try:
            logger.info(f"Loading section classifier model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir
            )
            
            # For a fine-tuned model, load directly
            if "imrad" in model_name.lower() or "section" in model_name.lower():
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                ).to(self.device)
                logger.info(f"Loaded fine-tuned section classifier: {model_name}")
            else:
                # For base SciBERT, initialize with the right number of labels
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.SECTION_TYPES),
                    cache_dir=cache_dir
                ).to(self.device)
                logger.info(f"Initialized base model for section classification: {model_name}")
                logger.warning("Using base model without fine-tuning. For better results, fine-tune on IMRAD data.")
            
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to initialize section classifier: {str(e)}")
            self.tokenizer = None
            self.model = None
    
    def classify_section(
        self, 
        heading: str, 
        text: Optional[str] = None,
        return_scores: bool = False
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Classify a section based on its heading and optional text.
        
        Args:
            heading: Section heading
            text: Optional section text (improves classification)
            return_scores: Whether to return confidence scores
            
        Returns:
            Section type or tuple of (section_type, scores) if return_scores=True
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Section classifier not initialized")
            return "other" if not return_scores else ("other", {})
        
        # First try rule-based classification for common headings
        rule_based_result = self._rule_based_classification(heading)
        if rule_based_result and not return_scores:
            return rule_based_result
        
        # Prepare input for model
        if text:
            # Use heading and beginning of text
            input_text = heading + ". " + text[:200]
        else:
            # Use only heading
            input_text = heading
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                prediction = torch.argmax(probs).item()
            
            # Get section type and scores
            section_type = self.SECTION_TYPES[prediction]
            
            # If rule-based classification gave a result, use it for common sections
            if rule_based_result and rule_based_result in ["abstract", "introduction", "methods", "results", "discussion", "conclusion", "references"]:
                section_type = rule_based_result
            
            if return_scores:
                scores = {self.SECTION_TYPES[i]: probs[i].item() for i in range(len(self.SECTION_TYPES))}
                return section_type, scores
            else:
                return section_type
        except Exception as e:
            logger.error(f"Error in model-based section classification: {str(e)}")
            return rule_based_result or "other" if not return_scores else (rule_based_result or "other", {})
    
    def _rule_based_classification(self, heading: str) -> Optional[str]:
        """
        Classify section type based on heading using rules.
        
        Args:
            heading: Section heading
            
        Returns:
            Section type or None if no match
        """
        heading_lower = heading.lower()
        
        # Check for common section types
        if any(x in heading_lower for x in ["abstract", "summary"]):
            return "abstract"
        elif any(x in heading_lower for x in ["introduction", "background"]):
            return "introduction"
        elif any(x in heading_lower for x in ["method", "materials", "procedure", "experimental"]):
            return "methods"
        elif "result" in heading_lower:
            return "results"
        elif "discussion" in heading_lower:
            return "discussion"
        elif any(x in heading_lower for x in ["conclusion", "concluding", "findings"]):
            return "conclusion"
        elif any(x in heading_lower for x in ["reference", "bibliography", "literature"]):
            return "references"
        elif any(x in heading_lower for x in ["acknowledgment", "acknowledgement"]):
            return "acknowledgments"
        
        return None
    
    def classify_sections(
        self, 
        sections: List[Dict[str, str]],
        text_key: str = "text",
        heading_key: str = "heading"
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple sections.
        
        Args:
            sections: List of section dictionaries
            text_key: Key for section text in dictionaries
            heading_key: Key for section heading in dictionaries
            
        Returns:
            List of section dictionaries with added 'section_type' key
        """
        result = []
        for section in sections:
            heading = section.get(heading_key, "")
            text = section.get(text_key, "")
            
            section_type, scores = self.classify_section(heading, text, return_scores=True)
            
            # Add classification to section dictionary
            section_with_type = section.copy()
            section_with_type["section_type"] = section_type
            section_with_type["classification_scores"] = scores
            
            result.append(section_with_type)
        
        return result
    
    def fine_tune(
        self,
        training_data: List[Dict[str, str]],
        validation_data: Optional[List[Dict[str, str]]] = None,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        output_dir: str = "imrad_classifier",
        text_key: str = "text",
        heading_key: str = "heading",
        label_key: str = "section_type"
    ) -> None:
        """
        Fine-tune the section classifier on custom data.
        
        Args:
            training_data: List of dictionaries with text, heading, and section_type
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            output_dir: Directory to save the fine-tuned model
            text_key: Key for text in data dictionaries
            heading_key: Key for heading in data dictionaries
            label_key: Key for section type in data dictionaries
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Section classifier not initialized")
            return
        
        try:
            from transformers import Trainer, TrainingArguments
            from datasets import Dataset
            
            # Prepare datasets
            def prepare_dataset(data):
                processed_data = []
                for item in data:
                    heading = item.get(heading_key, "")
                    text = item.get(text_key, "")
                    label = item.get(label_key, "other")
                    
                    # Convert label to index
                    if label in self.SECTION_TYPES:
                        label_idx = self.SECTION_TYPES.index(label)
                    else:
                        label_idx = self.SECTION_TYPES.index("other")
                    
                    # Combine heading and text
                    input_text = heading
                    if text:
                        input_text += ". " + text[:200]
                    
                    processed_data.append({
                        "text": input_text,
                        "label": label_idx
                    })
                
                return Dataset.from_list(processed_data)
            
            train_dataset = prepare_dataset(training_data)
            if validation_data:
                eval_dataset = prepare_dataset(validation_data)
            else:
                # Use 10% of training data for validation if no validation data provided
                train_size = int(0.9 * len(train_dataset))
                eval_size = len(train_dataset) - train_size
                train_dataset, eval_dataset = torch.utils.data.random_split(
                    train_dataset, [train_size, eval_size]
                )
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                learning_rate=learning_rate
            )
            
            # Define compute_metrics function
            def compute_metrics(eval_pred):
                logits, labels = eval_pred
                predictions = np.argmax(logits, axis=1)
                accuracy = np.mean(predictions == labels)
                return {"accuracy": accuracy}
            
            # Define tokenization function
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )
            
            # Tokenize datasets
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics
            )
            
            # Train model
            logger.info("Starting fine-tuning of section classifier")
            trainer.train()
            
            # Save model
            logger.info(f"Saving fine-tuned model to {output_dir}")
            trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Reload model
            self.model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(self.device)
            self.model.eval()
            
            logger.info("Fine-tuning complete")
        except Exception as e:
            logger.error(f"Error in fine-tuning section classifier: {str(e)}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> "IMRADSectionClassifier":
        """
        Load a fine-tuned section classifier.
        
        Args:
            model_path: Path to fine-tuned model
            device: Device for PyTorch models
            cache_dir: Directory to cache models
            
        Returns:
            IMRADSectionClassifier instance
        """
        return cls(model_name=model_path, device=device, cache_dir=cache_dir)
