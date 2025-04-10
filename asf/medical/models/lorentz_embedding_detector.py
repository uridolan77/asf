"""
Lorentz Embedding-based Contradiction Detector for Medical Claims

This module provides a Lorentz manifold embedding-based implementation for detecting
contradictions in medical claims. It integrates with the BioMedLM wrapper to provide
enhanced contradiction detection capabilities.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

from asf.layer1_knowledge_substrate.embeddings.lorentz_embeddings import (
    LorentzEmbedding, LorentzLinear, LorentzDistance, LorentzFusion,
    HybridLorentzEuclideanEmbedding, HybridLorentzEuclideanDistance
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lorentz-embedding-detector")

class LorentzEmbeddingContradictionDetector:
    """
    Lorentz embedding-based contradiction detector for medical claims.
    
    This class uses Lorentz manifold embeddings to detect contradictions in medical claims,
    focusing on hierarchical relationships and semantic similarity.
    """
    
    def __init__(
        self, 
        biomedlm_scorer=None, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Lorentz embedding-based contradiction detector.
        
        Args:
            biomedlm_scorer: BioMedLMScorer instance for semantic contradiction detection
            device: Device to run the model on
            config: Configuration dictionary
        """
        self.biomedlm_scorer = biomedlm_scorer
        self.device = device
        self.config = config or {}
        
        # Initialize Lorentz embedding model
        self.embedding_model = None
        self.tokenizer = None
        self.distance_calculator = None
        
        # Initialize Lorentz embedding model if requested
        if self.config.get("use_lorentz", True):
            self._initialize_lorentz_model()
    
    def _initialize_lorentz_model(self):
        """Initialize Lorentz embedding model for contradiction detection."""
        try:
            # Get Lorentz configuration
            vocab_size = self.config.get("vocab_size", 30000)
            embedding_dim = self.config.get("embedding_dim", 128)
            lorentz_k = self.config.get("lorentz_k", -1.0)
            
            # Create Lorentz embedding model
            self.embedding_model = HybridLorentzEuclideanEmbedding(
                num_embeddings=vocab_size,
                lorentz_dim=embedding_dim,
                euclidean_dim=embedding_dim,
                k=lorentz_k
            ).to(self.device)
            
            # Create distance calculator
            self.distance_calculator = HybridLorentzEuclideanDistance(
                lorentz_weight=0.7,
                euclidean_weight=0.3,
                k=lorentz_k
            ).to(self.device)
            
            # Use BioMedLM tokenizer if available
            if self.biomedlm_scorer is not None and hasattr(self.biomedlm_scorer, 'tokenizer'):
                self.tokenizer = self.biomedlm_scorer.tokenizer
                logger.info("Using BioMedLM tokenizer for Lorentz embeddings")
            else:
                # Use a simple tokenizer
                self.tokenizer = SimpleTokenizer()
                logger.info("Using simple tokenizer for Lorentz embeddings")
            
            logger.info("Lorentz embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Lorentz embedding model: {e}")
            self.embedding_model = None
            self.distance_calculator = None
    
    def detect_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect contradiction between medical claims.
        
        This method combines Lorentz embedding-based analysis with BioMedLM-based
        semantic analysis to detect contradictions in medical claims.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            
        Returns:
            Dictionary with contradiction detection results
        """
        result = {
            "text1": claim1,
            "text2": claim2,
            "has_contradiction": False,
            "contradiction_score": 0.0,
            "method": "lorentz_biomedlm",
            "embedding_analysis": {}
        }
        
        # Get semantic contradiction score from BioMedLM if available
        semantic_score = 0.0
        if self.biomedlm_scorer is not None:
            try:
                biomedlm_result = self.biomedlm_scorer.detect_contradiction(claim1, claim2)
                semantic_score = biomedlm_result.get("contradiction_score", 0.0)
                
                # Add BioMedLM results to our result
                result["biomedlm_result"] = {
                    "contradiction_score": semantic_score,
                    "agreement_score": biomedlm_result.get("agreement_score", 0.0),
                    "confidence": biomedlm_result.get("confidence", 0.0)
                }
            except Exception as e:
                logger.error(f"Error getting BioMedLM contradiction score: {e}")
        
        # If we have Lorentz embedding model, analyze embeddings
        embedding_score = 0.0
        if self.embedding_model is not None and self.tokenizer is not None and self.distance_calculator is not None:
            try:
                # Analyze embeddings
                embedding_result = self._analyze_embeddings(claim1, claim2)
                embedding_score = embedding_result.get("contradiction_score", 0.0)
                
                # Add embedding analysis results to our result
                result["embedding_analysis"] = embedding_result
            except Exception as e:
                logger.error(f"Error in Lorentz embedding analysis: {e}")
        
        # Combine semantic and embedding scores
        if self.embedding_model is not None:
            # If we have embedding model, use a weighted combination
            combined_score = 0.6 * semantic_score + 0.4 * embedding_score
        else:
            # If we don't have embedding model, use only semantic score
            combined_score = semantic_score
        
        # Update result with combined score
        result["contradiction_score"] = combined_score
        result["has_contradiction"] = combined_score > 0.7
        
        return result
    
    def _analyze_embeddings(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Analyze embeddings for contradiction detection.
        
        This method uses Lorentz manifold embeddings to analyze the hierarchical
        relationships and semantic similarity between medical claims.
        
        Args:
            text1: First medical claim
            text2: Second medical claim
            
        Returns:
            Dictionary with embedding analysis results
        """
        result = {
            "contradiction_score": 0.0,
            "contradiction_type": None,
            "details": {}
        }
        
        # Tokenize texts
        if hasattr(self.tokenizer, 'encode_plus'):
            # BioMedLM tokenizer
            tokens1 = self.tokenizer.encode_plus(
                text1, 
                add_special_tokens=True, 
                return_tensors="pt"
            )["input_ids"].to(self.device)
            
            tokens2 = self.tokenizer.encode_plus(
                text2, 
                add_special_tokens=True, 
                return_tensors="pt"
            )["input_ids"].to(self.device)
        else:
            # Simple tokenizer
            tokens1 = self.tokenizer.tokenize(text1).to(self.device)
            tokens2 = self.tokenizer.tokenize(text2).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings1 = self.embedding_model(tokens1)
            embeddings2 = self.embedding_model(tokens2)
        
        # Calculate distance between embeddings
        distance = self.distance_calculator(embeddings1, embeddings2)
        
        # Calculate contradiction score based on distance
        # Higher distance means higher contradiction
        contradiction_score = torch.mean(distance).item()
        
        # Normalize contradiction score to [0, 1]
        normalized_score = min(contradiction_score / 2.0, 1.0)
        
        # Update result
        result["contradiction_score"] = normalized_score
        result["details"]["distance"] = contradiction_score
        
        # Analyze hierarchical relationships
        if 'lorentz' in embeddings1 and 'lorentz' in embeddings2:
            lorentz_embeddings1 = embeddings1['lorentz']
            lorentz_embeddings2 = embeddings2['lorentz']
            
            # Check if one claim is a generalization of the other
            # In Lorentz space, points closer to the origin are higher in the hierarchy
            norm1 = torch.norm(lorentz_embeddings1[:, 1:], dim=-1).mean().item()
            norm2 = torch.norm(lorentz_embeddings2[:, 1:], dim=-1).mean().item()
            
            if abs(norm1 - norm2) > 0.5:
                # Significant difference in hierarchy
                result["contradiction_type"] = "hierarchical"
                result["details"]["hierarchical"] = {
                    "norm1": norm1,
                    "norm2": norm2,
                    "difference": abs(norm1 - norm2)
                }
        
        return result


class SimpleTokenizer:
    """
    Simple tokenizer for Lorentz embeddings.
    
    This class provides a simple tokenizer for Lorentz embeddings when
    BioMedLM tokenizer is not available.
    """
    
    def __init__(self, vocab_size: int = 30000):
        """
        Initialize the simple tokenizer.
        
        Args:
            vocab_size: Size of the vocabulary
        """
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx = 0
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tensor of token indices
        """
        # Split text into words
        words = text.lower().split()
        
        # Convert words to indices
        indices = []
        for word in words:
            if word not in self.word_to_idx:
                if self.idx < self.vocab_size:
                    self.word_to_idx[word] = self.idx
                    self.idx += 1
                else:
                    # Use UNK token
                    self.word_to_idx[word] = self.vocab_size - 1
            
            indices.append(self.word_to_idx[word])
        
        # Convert to tensor
        return torch.tensor([indices], dtype=torch.long)
