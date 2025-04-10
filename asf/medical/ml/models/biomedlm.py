"""
BioMedLM model wrapper for the Medical Research Synthesizer.

This module provides a wrapper for the BioMedLM model for contradiction detection.
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from transformers import AutoModel, AutoTokenizer
from functools import lru_cache

from asf.medical.core.config import settings
from asf.medical.ml.model_cache import model_cache

# Set up logging
logger = logging.getLogger(__name__)

class BioMedLMService:
    """
    Service for the BioMedLM model.

    This service provides methods for using the BioMedLM model for contradiction detection.
    """

    _instance = None

    def __new__(cls):
        """
        Create a singleton instance of the BioMedLM service.

        Returns:
            BioMedLMService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(BioMedLMService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the BioMedLM service."""
        self.model_name = settings.BIOMEDLM_MODEL
        self.use_gpu = settings.USE_GPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"

        logger.info(f"BioMedLM service initialized with device: {self.device}")

    @property
    def model(self):
        """
        Get the BioMedLM model.

        Returns:
            The BioMedLM model
        """
        # Use model cache to get or create the model
        return model_cache.get_or_create(
            model_id=f"biomedlm:{self.model_name}",
            factory=self._create_model,
            metadata={
                "model_name": self.model_name,
                "device": self.device,
                "memory_mb": 1024  # 1GB placeholder
            }
        )

    def _create_model(self):
        """Create the BioMedLM model."""
        logger.info(f"Loading BioMedLM model: {self.model_name}")
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        logger.info("BioMedLM model loaded")
        return model

    @property
    def tokenizer(self):
        """
        Get the BioMedLM tokenizer.

        Returns:
            The BioMedLM tokenizer
        """
        # Use model cache to get or create the tokenizer
        return model_cache.get_or_create(
            model_id=f"biomedlm_tokenizer:{self.model_name}",
            factory=self._create_tokenizer,
            metadata={
                "model_name": self.model_name,
                "memory_mb": 100  # 100MB placeholder
            }
        )

    def _create_tokenizer(self):
        """Create the BioMedLM tokenizer."""
        logger.info(f"Loading BioMedLM tokenizer: {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("BioMedLM tokenizer loaded")
        return tokenizer

    def unload_model(self):
        """Unload the model from memory."""
        # Remove model from cache
        model_cache.remove(f"biomedlm:{self.model_name}")
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("BioMedLM model unloaded")

    @lru_cache(maxsize=1000)
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text using the BioMedLM model.

        Args:
            text: Text to encode

        Returns:
            Text embedding
        """
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling to get a single vector
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

        return embeddings

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        embedding1 = self.encode(text1)
        embedding2 = self.encode(text2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        return float(similarity)

    def detect_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Tuple of (is_contradiction, confidence)
        """
        # Calculate similarity
        similarity = self.calculate_similarity(claim1, claim2)

        # Invert similarity to get contradiction score
        contradiction_score = 1.0 - similarity

        # Determine if it's a contradiction
        is_contradiction = contradiction_score > 0.5

        return is_contradiction, contradiction_score

    def get_token_importance(self, claim1: str, claim2: str) -> Dict[str, float]:
        """
        Get the importance of each token in the claims for contradiction detection.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Dictionary mapping tokens to importance scores
        """
        # Tokenize claims
        tokens1 = self.tokenizer.tokenize(claim1)
        tokens2 = self.tokenizer.tokenize(claim2)

        # Calculate baseline contradiction score
        _, baseline_score = self.detect_contradiction(claim1, claim2)

        # Calculate importance for each token in claim1
        token_importance = {}

        for i, token in enumerate(tokens1):
            # Create a version of claim1 without this token
            modified_tokens = tokens1.copy()
            modified_tokens.pop(i)
            modified_claim = self.tokenizer.convert_tokens_to_string(modified_tokens)

            # Calculate contradiction score without this token
            _, modified_score = self.detect_contradiction(modified_claim, claim2)

            # Calculate importance
            importance = baseline_score - modified_score

            # Add to dictionary
            token_importance[token] = importance

        # Calculate importance for each token in claim2
        for i, token in enumerate(tokens2):
            # Create a version of claim2 without this token
            modified_tokens = tokens2.copy()
            modified_tokens.pop(i)
            modified_claim = self.tokenizer.convert_tokens_to_string(modified_tokens)

            # Calculate contradiction score without this token
            _, modified_score = self.detect_contradiction(claim1, modified_claim)

            # Calculate importance
            importance = baseline_score - modified_score

            # Add to dictionary
            token_importance[token] = importance

        return token_importance
