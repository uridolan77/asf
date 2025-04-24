"""
Quality control for generative replay.

This module provides quality control mechanisms for filtering and selecting
high-quality examples generated for replay.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class QualityController:
    """
    Quality controller for generative replay.
    
    This class provides methods for filtering and selecting high-quality
    examples generated for replay based on various criteria.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        diversity_threshold: float = 0.8,
        min_length: int = 10,
        max_length: int = 1000,
        embedding_model_name: Optional[str] = None,
        embedding_batch_size: int = 8,
        **kwargs
    ):
        """
        Initialize the quality controller.
        
        Args:
            quality_threshold: Threshold for quality filtering (0 to 1)
            diversity_threshold: Threshold for diversity filtering (0 to 1)
            min_length: Minimum length for generated examples
            max_length: Maximum length for generated examples
            embedding_model_name: Name of the model to use for embeddings
            embedding_batch_size: Batch size for computing embeddings
            **kwargs: Additional parameters
        """
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.min_length = min_length
        self.max_length = max_length
        self.embedding_model_name = embedding_model_name
        self.embedding_batch_size = embedding_batch_size
        
        # Initialize embedding model if provided
        self.embedding_model = None
        self.embedding_tokenizer = None
        
        if self.embedding_model_name:
            try:
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
                self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
                
                # Move model to GPU if available
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.cuda()
                
                logger.info(f"Initialized embedding model: {self.embedding_model_name}")
            except Exception as e:
                logger.warning(f"Error loading embedding model: {str(e)}")
                self.embedding_model = None
                self.embedding_tokenizer = None
    
    def filter_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples based on quality and diversity.
        
        Args:
            examples: List of examples to filter
            
        Returns:
            Filtered list of examples
        """
        # Apply basic filtering
        filtered_examples = self._basic_filtering(examples)
        
        # Apply quality filtering if embedding model is available
        if self.embedding_model and self.embedding_tokenizer:
            filtered_examples = self._quality_filtering(filtered_examples)
            filtered_examples = self._diversity_filtering(filtered_examples)
        
        return filtered_examples
    
    def _basic_filtering(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply basic filtering based on length and content.
        
        Args:
            examples: List of examples to filter
            
        Returns:
            Filtered list of examples
        """
        filtered_examples = []
        
        for example in examples:
            # Get text from example
            text = example.get("text", "")
            
            # Skip empty texts
            if not text:
                continue
            
            # Check length
            if len(text) < self.min_length or len(text) > self.max_length:
                continue
            
            # Check for repetitive content
            if self._is_repetitive(text):
                continue
            
            # Add to filtered examples
            filtered_examples.append(example)
        
        return filtered_examples
    
    def _is_repetitive(self, text: str, threshold: float = 0.5) -> bool:
        """
        Check if text is repetitive.
        
        Args:
            text: Text to check
            threshold: Threshold for repetition detection
            
        Returns:
            True if text is repetitive, False otherwise
        """
        # Split text into words
        words = text.split()
        
        # If text is too short, it's not repetitive
        if len(words) < 10:
            return False
        
        # Count unique words
        unique_words = set(words)
        
        # Compute ratio of unique words to total words
        unique_ratio = len(unique_words) / len(words)
        
        # Check if ratio is below threshold
        return unique_ratio < threshold
    
    def _quality_filtering(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples based on quality.
        
        Args:
            examples: List of examples to filter
            
        Returns:
            Filtered list of examples
        """
        # If no examples or no embedding model, return as is
        if not examples or not self.embedding_model or not self.embedding_tokenizer:
            return examples
        
        # Compute quality scores
        quality_scores = self._compute_quality_scores(examples)
        
        # Filter examples based on quality scores
        filtered_examples = []
        
        for example, score in zip(examples, quality_scores):
            if score >= self.quality_threshold:
                # Add quality score to example
                example["quality_score"] = float(score)
                filtered_examples.append(example)
        
        return filtered_examples
    
    def _compute_quality_scores(self, examples: List[Dict[str, Any]]) -> List[float]:
        """
        Compute quality scores for examples.
        
        Args:
            examples: List of examples
            
        Returns:
            List of quality scores
        """
        # Get texts from examples
        texts = [example.get("text", "") for example in examples]
        
        # Compute embeddings
        embeddings = self._compute_embeddings(texts)
        
        # Compute quality scores based on embedding norms
        # Higher norm indicates more confident/coherent text
        quality_scores = []
        
        for embedding in embeddings:
            # Compute L2 norm
            norm = np.linalg.norm(embedding)
            
            # Normalize to [0, 1] range (assuming typical norms are in [0, 10])
            score = min(1.0, norm / 10.0)
            
            quality_scores.append(score)
        
        return quality_scores
    
    def _diversity_filtering(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples based on diversity.
        
        Args:
            examples: List of examples to filter
            
        Returns:
            Filtered list of examples
        """
        # If no examples or too few examples, return as is
        if len(examples) <= 1 or not self.embedding_model or not self.embedding_tokenizer:
            return examples
        
        # Get texts from examples
        texts = [example.get("text", "") for example in examples]
        
        # Compute embeddings
        embeddings = self._compute_embeddings(texts)
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # Apply greedy diversity filtering
        selected_indices = self._greedy_diversity_selection(similarities)
        
        # Get selected examples
        selected_examples = [examples[i] for i in selected_indices]
        
        return selected_examples
    
    def _greedy_diversity_selection(self, similarities: np.ndarray) -> List[int]:
        """
        Greedy diversity selection algorithm.
        
        Args:
            similarities: Pairwise similarity matrix
            
        Returns:
            Indices of selected examples
        """
        n = similarities.shape[0]
        
        # Start with the example that has the highest quality score (if available)
        # or the first example
        selected_indices = [0]
        
        # Greedily select examples that are most diverse from already selected ones
        while len(selected_indices) < n:
            # Compute maximum similarity to already selected examples
            max_similarities = np.max(similarities[selected_indices, :], axis=0)
            
            # Find the example with the lowest maximum similarity
            # (i.e., the most diverse from already selected ones)
            candidate_indices = list(set(range(n)) - set(selected_indices))
            
            if not candidate_indices:
                break
            
            # Get similarities for candidate indices
            candidate_similarities = max_similarities[candidate_indices]
            
            # Find the candidate with the lowest similarity
            min_idx = np.argmin(candidate_similarities)
            next_idx = candidate_indices[min_idx]
            
            # If similarity is above threshold, stop
            if max_similarities[next_idx] > self.diversity_threshold:
                break
            
            # Add to selected indices
            selected_indices.append(next_idx)
        
        return selected_indices
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Array of embeddings
        """
        # If no embedding model, return empty embeddings
        if not self.embedding_model or not self.embedding_tokenizer:
            return np.zeros((len(texts), 1))
        
        embeddings = []
        
        # Process texts in batches
        for i in range(0, len(texts), self.embedding_batch_size):
            batch_texts = texts[i:i+self.embedding_batch_size]
            
            try:
                # Tokenize texts
                inputs = self.embedding_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Move inputs to the same device as the model
                inputs = {k: v.to(self.embedding_model.device) for k, v in inputs.items()}
                
                # Compute embeddings
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                
                # Use CLS token embedding as text embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                embeddings.append(batch_embeddings)
            
            except Exception as e:
                logger.warning(f"Error computing embeddings: {str(e)}")
                # Use zero embeddings as fallback
                batch_embeddings = np.zeros((len(batch_texts), self.embedding_model.config.hidden_size))
                embeddings.append(batch_embeddings)
        
        # Concatenate batch embeddings
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.zeros((len(texts), 1))
