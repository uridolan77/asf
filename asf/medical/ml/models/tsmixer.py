"""TSMixer model wrapper for the Medical Research Synthesizer.

This module provides a wrapper for the TSMixer model for temporal analysis.
"""
import logging
import torch
import numpy as np
from typing import Dict, List, Any
import torch.nn as nn
import torch.nn.functional as F
from asf.medical.core.config import settings
logger = logging.getLogger(__name__)
class TSMixerLayer(nn.Module):
    """TSMixer layer for temporal analysis.
    
    This layer implements the TSMixer architecture for time series analysis.
    """
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize the TSMixer layer.
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.temporal_fc1 = nn.Linear(input_dim, hidden_dim)
        self.temporal_fc2 = nn.Linear(hidden_dim, input_dim)
        self.feature_fc1 = nn.Linear(input_dim, hidden_dim)
        self.feature_fc2 = nn.Linear(hidden_dim, input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TSMixer layer.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        residual = x
        x = self.layer_norm1(x)
        x = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
        x = self.temporal_fc1(x)  # (batch_size, input_dim, hidden_dim)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.temporal_fc2(x)  # (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, input_dim)
        x = x + residual
        residual = x
        x = self.layer_norm2(x)
        x = self.feature_fc1(x)  # (batch_size, seq_len, hidden_dim)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.feature_fc2(x)  # (batch_size, seq_len, input_dim)
        x = x + residual
        return x
class TSMixer(nn.Module):
    """TSMixer model for temporal analysis.
    
    This model implements the TSMixer architecture for time series analysis.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize the TSMixer model.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_layers: Number of TSMixer layers
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            TSMixerLayer(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TSMixer model.
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        x = self.output_projection(x)  # (batch_size, seq_len, output_dim)
        return x
class TSMixerService:
    """Service for the TSMixer model.
    
    This service provides methods for using the TSMixer model for temporal analysis.
    """
    _instance = None
    _model = None
    def __new__(cls):
        """
        Create a singleton instance of the TSMixer service.
        Returns:
            TSMixerService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(TSMixerService, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        """Initialize the TSMixer service.
        
        This method sets up the service with default parameters and device configuration.
        """
        self.input_dim = 768  # Embedding dimension
        self.hidden_dim = 512
        self.output_dim = 1  # Contradiction score
        self.num_layers = 4
        self.dropout = 0.1
        self.use_gpu = settings.USE_GPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        logger.info(f"TSMixer service initialized with device: {self.device}")
    @property
    def model(self):
        """
        Get the TSMixer model, creating it if it doesn't exist.

        Returns:
            The TSMixer model instance
        """
        if self._model is None:
            self._model = TSMixer(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
            logger.info("TSMixer model created")
        return self._model

    def unload_model(self):
        """Unload the model from memory to free resources.
        
        This method removes the TSMixer model from memory and clears CUDA cache.
        """
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("TSMixer model unloaded")
    def analyze_temporal_sequence(
        self,
        sequence: List[Dict[str, Any]],
        embedding_fn: callable
    ) -> Dict[str, Any]:
        """
        Analyze a temporal sequence of claims for contradictions.

        Args:
            sequence: List of dictionaries containing claims and timestamps
                     Each dictionary should have at least 'claim' and 'timestamp' keys
            embedding_fn: Function to convert a claim text to an embedding vector

        Returns:
            Dictionary containing contradiction scores and temporal analysis results
            including trend, volatility, and maximum contradiction information
        """
        sequence = sorted(sequence, key=lambda x: x["timestamp"])
        embeddings = []
        for item in sequence:
            embedding = embedding_fn(item["claim"])
            embeddings.append(embedding)
        embeddings = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        embeddings = embeddings.unsqueeze(0)  # (1, seq_len, input_dim)
        with torch.no_grad():
            outputs = self.model(embeddings)
        contradiction_scores = outputs.squeeze().cpu().numpy()
        result = {
            "contradiction_scores": contradiction_scores.tolist(),
            "temporal_analysis": {
                "trend": "increasing" if np.mean(np.diff(contradiction_scores)) > 0 else "decreasing",
                "volatility": float(np.std(contradiction_scores)),
                "max_contradiction": float(np.max(contradiction_scores)),
                "max_contradiction_index": int(np.argmax(contradiction_scores))
            }
        }
        return result