"""BioMedLM model wrapper for the Medical Research Synthesizer.

This module provides a wrapper for the BioMedLM model for contradiction detection.
"""
import logging
import torch
import numpy as np
from typing import Tuple
from transformers import AutoModel
from asf.medical.core.config import settings
logger = logging.getLogger(__name__)
class BioMedLMService:
    """Service for the BioMedLM model.

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
        """
        Initialize the BioMedLM service.

        This method sets up the service with the model name and device configuration
        based on application settings.
        """

        self.model_name = settings.BIOMEDLM_MODEL
        self.use_gpu = settings.USE_GPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        logger.info(f"BioMedLM service initialized with device: {self.device}")
    @property
    def model(self):
        """Get the BioMedLM model, loading it if necessary.
        
        This property loads the BioMedLM model from the specified model name
        and moves it to the appropriate device (CPU/GPU).
        
        Returns:
            torch.nn.Module: The loaded BioMedLM model instance
        """
        logger.info(f"Loading BioMedLM model: {self.model_name}")
        model = AutoModel.from_pretrained(self.model_name)
        model.to(self.device)
        logger.info("BioMedLM model loaded")
        return model
    @property
    def tokenizer(self):
        """Get the BioMedLM tokenizer for text processing.
        
        This property loads the tokenizer associated with the BioMedLM model
        for processing input text into tokens.
        
        Returns:
            transformers.PreTrainedTokenizer: The BioMedLM tokenizer instance
        """
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_name)

    async def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text using the BioMedLM model.

        Args:
            text: Text to encode

        Returns:
            Text embedding as a numpy array
        """
        tokenizer = self.tokenizer
        model = self.model

        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = model(**inputs)

        # Use the [CLS] token embedding as the text embedding
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate the similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        embedding1 = await self.encode_text(text1)
        embedding2 = await self.encode_text(text2)

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float((similarity + 1) / 2)  # Scale from [-1, 1] to [0, 1]

    async def detect_contradiction(self, claim1: str, claim2: str) -> Tuple[bool, float]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Tuple of (is_contradiction, confidence)
        """
        similarity = await self.calculate_similarity(claim1, claim2)
        contradiction_score = 1.0 - similarity
        is_contradiction = contradiction_score > 0.7
        return (is_contradiction, contradiction_score)

    async def get_token_importance(self, claim1: str, claim2: str) -> dict:
        """
        Get the importance of each token in the claims for contradiction detection.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Dictionary mapping tokens to importance scores
        """
        tokenizer = self.tokenizer
        tokens1 = tokenizer.tokenize(claim1)
        tokens2 = tokenizer.tokenize(claim2)

        # Calculate baseline contradiction score
        baseline_result = await self.detect_contradiction(claim1, claim2)
        baseline_score = baseline_result[1]

        # Calculate importance for each token in claim1
        importance = {}
        for i, token in enumerate(tokens1):
            # Create a version of claim1 without this token
            modified_tokens = tokens1.copy()
            modified_tokens.pop(i)
            modified_claim = tokenizer.convert_tokens_to_string(modified_tokens)

            # Calculate contradiction score without this token
            modified_result = await self.detect_contradiction(modified_claim, claim2)
            modified_score = modified_result[1]

            # Importance is the difference in scores
            importance[token] = abs(baseline_score - modified_score)

        # Calculate importance for each token in claim2
        for i, token in enumerate(tokens2):
            # Create a version of claim2 without this token
            modified_tokens = tokens2.copy()
            modified_tokens.pop(i)
            modified_claim = tokenizer.convert_tokens_to_string(modified_tokens)

            # Calculate contradiction score without this token
            modified_result = await self.detect_contradiction(claim1, modified_claim)
            modified_score = modified_result[1]

            # Importance is the difference in scores
            importance[token] = abs(baseline_score - modified_score)

        return importance