"""Lorentz embeddings model for the Medical Research Synthesizer.

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
    """Lorentz embedding model for hierarchical data.
    
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
    """Service for the Lorentz embedding model.
    
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
        
        This method sets up the service with default parameters and device configuration.
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
            entities: List of entity names to embed in the Lorentz space
        """
        # Unload existing model if any
        if self._model_id is not None:
            self.unload_model()
        # Create entity mappings
        self._entity_to_idx = {entity: i for i, entity in enumerate(entities)}
        self._idx_to_entity = {i: entity for i, entity in enumerate(entities)}

        # Create model
        model = LorentzEmbedding(
            num_entities=len(entities),
            dim=self.dim,
            c=self.c
        ).to(self.device)

        # Generate model ID
        self._model_id = f"lorentz_embedding_{len(entities)}_{self.dim}_{self.c}"

        # Store in cache
        model_cache.put(self._model_id, model)

        logger.info(f"Lorentz embedding model initialized with {len(entities)} entities")

    def get_model(self):
        """
        Get the Lorentz embedding model.

        Returns:
            The Lorentz embedding model

        Raises:
            ValueError: If the model has not been initialized
        """
        if self._model_id is None:
            raise ValueError("Model has not been initialized. Call initialize_model first.")

        model = model_cache.get(self._model_id)
        if model is None:
            raise ValueError(f"Model not found in cache: {self._model_id}")

        return model

    def unload_model(self):
        """Unload the model from memory.
        
        This method removes the model from cache and cleans up GPU memory.
        """
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
            Entity embedding as a numpy array

        Raises:
            ValueError: If the entity is not in the model
        """
        if entity not in self._entity_to_idx:
            raise ValueError(f"Entity not found: {entity}")

        model = self.get_model()
        idx = self._entity_to_idx[entity]
        with torch.no_grad():
            embedding = model(torch.tensor([idx]).to(self.device))[0].cpu().numpy()
        return embedding

    def get_distance(self, entity1: str, entity2: str) -> float:
        """
        Get the distance between two entities in Lorentz space.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            Distance between entities

        Raises:
            ValueError: If either entity is not in the model
        """
        if entity1 not in self._entity_to_idx:
            raise ValueError(f"Entity not found: {entity1}")
        if entity2 not in self._entity_to_idx:
            raise ValueError(f"Entity not found: {entity2}")

        model = self.get_model()
        idx1 = self._entity_to_idx[entity1]
        idx2 = self._entity_to_idx[entity2]
        with torch.no_grad():
            emb1 = model(torch.tensor([idx1]).to(self.device))
            emb2 = model(torch.tensor([idx2]).to(self.device))
            distance = model.distance(emb1, emb2).item()
        return distance

    def get_nearest_neighbors(self, entity: str, k: int = 5) -> List[tuple]:
        """
        Get the k nearest neighbors of an entity in Lorentz space.

        Args:
            entity: Entity name
            k: Number of neighbors to return

        Returns:
            List of (entity, distance) tuples sorted by distance

        Raises:
            ValueError: If the entity is not in the model
        """
        if entity not in self._entity_to_idx:
            raise ValueError(f"Entity not found: {entity}")

        model = self.get_model()
        idx = self._entity_to_idx[entity]
        with torch.no_grad():
            emb = model(torch.tensor([idx]).to(self.device))

            # Calculate distances to all entities
            distances = []
            for other_idx, other_entity in self._idx_to_entity.items():
                if other_idx == idx:  # Skip self
                    continue

                other_emb = model(torch.tensor([other_idx]).to(self.device))
                distance = model.distance(emb, other_emb).item()
                distances.append((other_entity, distance))

            # Sort by distance and return top k
            distances.sort(key=lambda x: x[1])
            return distances[:k]