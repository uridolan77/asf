"""
Quality control mechanisms for generative replay.

This module provides quality control mechanisms for generated examples,
including filtering, scoring, and diversity sampling.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

class QualityController:
    """
    Quality controller for generated examples.
    
    This class provides methods for filtering, scoring, and sampling
    generated examples based on quality and diversity.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.7,
        diversity_threshold: float = 0.8,
        max_similarity: float = 0.9,
        embedding_model_name: Optional[str] = None,
        custom_quality_fn: Optional[Callable] = None
    ):
        """
        Initialize the quality controller.
        
        Args:
            quality_threshold: Threshold for quality filtering
            diversity_threshold: Threshold for diversity filtering
            max_similarity: Maximum similarity between examples
            embedding_model_name: Name of the model to use for embeddings
            custom_quality_fn: Custom function for quality assessment
        """
        self.quality_threshold = quality_threshold
        self.diversity_threshold = diversity_threshold
        self.max_similarity = max_similarity
        self.custom_quality_fn = custom_quality_fn
        
        # Initialize embedding model if specified
        self.embedding_model = None
        self.embedding_tokenizer = None
        if embedding_model_name:
            try:
                self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
                self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
                logger.info(f"Initialized embedding model: {embedding_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {str(e)}")
    
    def filter_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples based on quality and diversity.
        
        Args:
            examples: List of generated examples
            
        Returns:
            Filtered list of examples
        """
        if not examples:
            return []
        
        # Score examples for quality
        scored_examples = self.score_examples(examples)
        
        # Filter by quality threshold
        quality_filtered = [
            ex for ex in scored_examples 
            if ex.get('quality_score', 0) >= self.quality_threshold
        ]
        
        if not quality_filtered:
            logger.warning("No examples passed quality filtering")
            # Return top examples if none pass the threshold
            scored_examples.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            return scored_examples[:max(1, len(examples) // 10)]
        
        # Filter for diversity
        return self.filter_for_diversity(quality_filtered)
    
    def score_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score examples for quality.
        
        Args:
            examples: List of generated examples
            
        Returns:
            List of examples with quality scores
        """
        scored_examples = []
        
        for example in examples:
            # Make a copy to avoid modifying the original
            scored_example = example.copy()
            
            # Use custom quality function if provided
            if self.custom_quality_fn:
                quality_score = self.custom_quality_fn(example)
            else:
                # Default quality assessment
                quality_score = self._assess_quality(example)
            
            scored_example['quality_score'] = quality_score
            scored_examples.append(scored_example)
        
        return scored_examples
    
    def _assess_quality(self, example: Dict[str, Any]) -> float:
        """
        Assess the quality of a generated example.
        
        Args:
            example: Generated example
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        # Get the text from the example
        text = example.get('text', '')
        if not text:
            return 0.0
        
        # Basic quality checks
        quality_score = 1.0
        
        # Length check (penalize very short or very long texts)
        text_length = len(text.split())
        if text_length < 5:
            quality_score *= 0.5
        elif text_length > 500:
            quality_score *= 0.8
        
        # Repetition check
        repetition_score = self._check_repetition(text)
        quality_score *= repetition_score
        
        # Coherence check using embeddings if available
        if self.embedding_model and self.embedding_tokenizer:
            coherence_score = self._check_coherence(text)
            quality_score *= coherence_score
        
        return quality_score
    
    def _check_repetition(self, text: str) -> float:
        """
        Check for repetitions in the text.
        
        Args:
            text: Generated text
            
        Returns:
            Repetition score (0.0 to 1.0)
        """
        words = text.split()
        if len(words) <= 1:
            return 1.0
        
        # Check for repeated words
        repeated_words = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                repeated_words += 1
        
        repetition_rate = repeated_words / (len(words) - 1)
        repetition_score = 1.0 - min(1.0, repetition_rate * 5)  # Penalize repetitions
        
        # Check for repeated phrases
        if len(words) >= 4:
            phrases = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
            unique_phrases = set(phrases)
            phrase_repetition_rate = 1.0 - (len(unique_phrases) / len(phrases))
            repetition_score *= (1.0 - min(1.0, phrase_repetition_rate * 2))
        
        return max(0.1, repetition_score)  # Ensure minimum score
    
    def _check_coherence(self, text: str) -> float:
        """
        Check the coherence of the text using embeddings.
        
        Args:
            text: Generated text
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return 1.0
        
        try:
            # Get embeddings for each sentence
            inputs = self.embedding_tokenizer(
                sentences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=128
            )
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use CLS token
            
            # Calculate coherence based on cosine similarity between adjacent sentences
            coherence_scores = []
            for i in range(1, len(embeddings)):
                sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
                coherence_scores.append(sim)
            
            # Average coherence score
            avg_coherence = sum(coherence_scores) / len(coherence_scores)
            
            # Scale to 0.0-1.0 range (higher similarity = higher coherence)
            return min(1.0, max(0.0, avg_coherence))
            
        except Exception as e:
            logger.warning(f"Error calculating coherence: {str(e)}")
            return 0.8  # Default value
    
    def filter_for_diversity(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter examples for diversity.
        
        Args:
            examples: List of examples with quality scores
            
        Returns:
            Diverse subset of examples
        """
        if not examples:
            return []
        
        if len(examples) <= 1:
            return examples
        
        # Sort by quality score
        examples.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Always keep the highest quality example
        diverse_examples = [examples[0]]
        
        # Get embeddings if embedding model is available
        if self.embedding_model and self.embedding_tokenizer:
            try:
                # Get text from examples
                texts = [ex.get('text', '') for ex in examples]
                
                # Get embeddings
                inputs = self.embedding_tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Use CLS token
                
                # Add diverse examples
                for i in range(1, len(examples)):
                    # Check similarity with already selected examples
                    similarities = [
                        cosine_similarity([embeddings[i]], [embeddings[examples.index(ex)]])[0][0]
                        for ex in diverse_examples
                    ]
                    
                    # Add if not too similar to any existing example
                    if max(similarities) < self.max_similarity:
                        diverse_examples.append(examples[i])
                
                return diverse_examples
                
            except Exception as e:
                logger.warning(f"Error in diversity filtering with embeddings: {str(e)}")
                # Fall back to simpler diversity filtering
        
        # Simple diversity filtering based on text overlap
        for example in examples[1:]:
            text = example.get('text', '')
            
            # Check overlap with already selected examples
            is_diverse = True
            for selected in diverse_examples:
                selected_text = selected.get('text', '')
                
                # Skip if either text is empty
                if not text or not selected_text:
                    continue
                
                # Calculate text overlap
                overlap = self._calculate_text_overlap(text, selected_text)
                
                if overlap > (1.0 - self.diversity_threshold):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_examples.append(example)
        
        return diverse_examples
    
    def _calculate_text_overlap(self, text1: str, text2: str) -> float:
        """
        Calculate text overlap between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Overlap score (0.0 to 1.0)
        """
        # Convert to sets of words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
