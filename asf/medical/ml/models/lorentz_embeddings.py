"""
Lorentz embeddings model for the Medical Research Synthesizer.
This module provides a model for Lorentz embeddings for hierarchical data.
"""
import logging
import torch
import numpy as np
from typing import List
import geoopt
from geoopt.manifolds.lorentz import Lorentz
from asf.medical.core.config import settings
from asf.medical.ml.model_cache import model_cache
logger = logging.getLogger(__name__)
class LorentzEmbedding(torch.nn.Module):
    """
    Lorentz embedding model for hierarchical data.
    This model embeds data in Lorentz space, which is well-suited for hierarchical data.
    """
    def __init__(self, num_entities: int, dim: int = 128, c: float = 1.0):
        """
        Initialize the Lorentz embedding model.
        Args:
            num_entities: Number of entities to embed
            dim: Embedding dimension
            c: Curvature parameter
        """
        super().__init__()
        self.manifold = Lorentz(c=c)
        self.embeddings = geoopt.ManifoldParameter(
            torch.randn(num_entities, dim + 1),
            manifold=self.manifold
        )
        with torch.no_grad():
            self.embeddings.set_(self.manifold.random(num_entities, dim + 1))
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for the given indices.
        Args:
            indices: Entity indices
        Returns:
            Entity embeddings
        """
        return self.embeddings[indices]
    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Calculate the distance between embeddings.
        Args:
            x: First embedding
            y: Second embedding
        Returns:
            Distance
        """
        return self.manifold.dist(x, y)
class LorentzEmbeddingService:
    """
    Service for the Lorentz embedding model.
    This service provides methods for using the Lorentz embedding model.
    """
    _instance = None
    _model = None
    _entity_to_idx = None
    _idx_to_entity = None
    def __new__(cls):
        """
        Create a singleton instance of the Lorentz embedding service.
        Returns:
            LorentzEmbeddingService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(LorentzEmbeddingService, cls).__new__(cls)
        return cls._instance
    def __init__(self):
        """Initialize the Lorentz embedding service.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        self.dim = 128
        self.c = 1.0
        self.use_gpu = settings.USE_GPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        self._entity_to_idx = {}
        self._idx_to_entity = {}
        self._model_id = None
        logger.info(f"Lorentz embedding service initialized with device: {self.device}")
    def initialize_model(self, entities: List[str]):
        """
        Initialize the model with the given entities.
        Args:
            entities: List of entity names
        Get the Lorentz embedding model.
        Returns:
            The Lorentz embedding model
        Raises:
            ValueError: If the model has not been initialized
        if self._model_id is not None:
            model_cache.remove(self._model_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Lorentz embedding model unloaded: {self._model_id}")
    def get_embedding(self, entity: str) -> np.ndarray:
        """
        Get the embedding for an entity.
        Args:
            entity: Entity name
        Returns:
            Entity embedding
        Raises:
            ValueError: If the entity is not in the model
        Get the distance between two entities.
        Args:
            entity1: First entity name
            entity2: Second entity name
        Returns:
            Distance between entities
        Raises:
            ValueError: If either entity is not in the model
        Get the k nearest neighbors of an entity.
        Args:
            entity: Entity name
            k: Number of neighbors to return
        Returns:
            List of (entity, distance) tuples
        Raises:
            ValueError: If the entity is not in the model