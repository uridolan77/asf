"""
Lorentz embeddings model for the Medical Research Synthesizer.

This module provides a model for Lorentz embeddings for hierarchical data.
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import geoopt
from geoopt.manifolds.lorentz import Lorentz

from asf.medical.core.config import settings

# Set up logging
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
        
        # Manifold
        self.manifold = Lorentz(c=c)
        
        # Embeddings
        self.embeddings = geoopt.ManifoldParameter(
            torch.randn(num_entities, dim + 1),
            manifold=self.manifold
        )
        
        # Initialize embeddings on the manifold
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
        """Initialize the Lorentz embedding service."""
        self.dim = 128
        self.c = 1.0
        
        self.use_gpu = settings.USE_GPU
        self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        
        self._entity_to_idx = {}
        self._idx_to_entity = {}
        
        logger.info(f"Lorentz embedding service initialized with device: {self.device}")
    
    def initialize_model(self, entities: List[str]):
        """
        Initialize the model with the given entities.
        
        Args:
            entities: List of entity names
        """
        # Create entity mappings
        self._entity_to_idx = {entity: i for i, entity in enumerate(entities)}
        self._idx_to_entity = {i: entity for i, entity in enumerate(entities)}
        
        # Create model
        logger.info(f"Creating Lorentz embedding model with {len(entities)} entities")
        self._model = LorentzEmbedding(
            num_entities=len(entities),
            dim=self.dim,
            c=self.c
        )
        self._model.to(self.device)
        logger.info("Lorentz embedding model created")
    
    @property
    def model(self):
        """
        Get the Lorentz embedding model.
        
        Returns:
            The Lorentz embedding model
            
        Raises:
            ValueError: If the model has not been initialized
        """
        if self._model is None:
            raise ValueError("Lorentz embedding model has not been initialized")
        return self._model
    
    def unload_model(self):
        """Unload the model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("Lorentz embedding model unloaded")
    
    def get_embedding(self, entity: str) -> np.ndarray:
        """
        Get the embedding for an entity.
        
        Args:
            entity: Entity name
            
        Returns:
            Entity embedding
            
        Raises:
            ValueError: If the entity is not in the model
        """
        if entity not in self._entity_to_idx:
            raise ValueError(f"Entity '{entity}' not in model")
        
        idx = self._entity_to_idx[entity]
        indices = torch.tensor([idx], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(indices)
        
        return embedding.cpu().numpy()[0]
    
    def get_distance(self, entity1: str, entity2: str) -> float:
        """
        Get the distance between two entities.
        
        Args:
            entity1: First entity name
            entity2: Second entity name
            
        Returns:
            Distance between entities
            
        Raises:
            ValueError: If either entity is not in the model
        """
        if entity1 not in self._entity_to_idx:
            raise ValueError(f"Entity '{entity1}' not in model")
        if entity2 not in self._entity_to_idx:
            raise ValueError(f"Entity '{entity2}' not in model")
        
        idx1 = self._entity_to_idx[entity1]
        idx2 = self._entity_to_idx[entity2]
        
        indices = torch.tensor([idx1, idx2], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            embeddings = self.model(indices)
            distance = self.model.distance(embeddings[0], embeddings[1])
        
        return distance.item()
    
    def get_nearest_neighbors(self, entity: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the k nearest neighbors of an entity.
        
        Args:
            entity: Entity name
            k: Number of neighbors to return
            
        Returns:
            List of (entity, distance) tuples
            
        Raises:
            ValueError: If the entity is not in the model
        """
        if entity not in self._entity_to_idx:
            raise ValueError(f"Entity '{entity}' not in model")
        
        idx = self._entity_to_idx[entity]
        indices = torch.arange(len(self._entity_to_idx), dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model(torch.tensor([idx], dtype=torch.long).to(self.device))
            all_embeddings = self.model(indices)
            
            distances = self.model.distance(
                query_embedding.expand(all_embeddings.size(0), -1),
                all_embeddings
            )
        
        # Get top k
        distances = distances.cpu().numpy()
        indices = np.argsort(distances)[:k+1]  # +1 to include the query itself
        
        # Filter out the query
        neighbors = []
        for i in indices:
            if i != idx:
                neighbors.append((self._idx_to_entity[i.item()], distances[i]))
        
        return neighbors[:k]
