"""
Lorentz Embeddings
This module provides utilities for working with Lorentz manifold embeddings,
which offer improved numerical stability and performance compared to Poincaré
embeddings for hierarchical data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math
import logging
from geoopt import Lorentz
from geoopt.manifolds.lorentz import math as lorentz_math
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("lorentz-embeddings")
class LorentzEmbedding(nn.Module):
    """
    Lorentz embedding layer.
    This layer maps input indices to embeddings in Lorentz space,
    which is particularly well-suited for hierarchical data.
    """
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        k: float = -1.0,
        sparse: bool = False,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        initialize: str = "random"
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.k = k
        self.manifold = Lorentz(k=k)
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self._initialize_weights(initialize)
        self.sparse = sparse
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
    def _initialize_weights(self, method: str):
        """
        Initialize embedding weights.
        Args:
            method: Initialization method ("random", "uniform", or "normal")
        """
        if method == "random":
            with torch.no_grad():
                tangent_vectors = torch.randn(self.num_embeddings, self.embedding_dim)
                tangent_vectors = F.normalize(tangent_vectors, p=2, dim=1) * 0.1
                tangent_vectors[:, 0] = 0.0  # Zero out the time component
                origin = torch.zeros(1, self.embedding_dim, device=tangent_vectors.device)
                origin[:, 0] = 1.0  # Set the time component to 1
                self.weight.data = self.manifold.expmap(origin, tangent_vectors)
        elif method == "uniform":
            nn.init.uniform_(self.weight, -0.1, 0.1)
            self._project_to_lorentz()
        elif method == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=0.1)
            self._project_to_lorentz()
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    def _project_to_lorentz(self):
        """Project embeddings to the Lorentz manifold.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        with torch.no_grad():
            spatial_norm_squared = torch.sum(self.weight[:, 1:] ** 2, dim=1, keepdim=True)
            self.weight[:, 0] = torch.sqrt(spatial_norm_squared - 1.0 / self.k)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Lorentz embedding layer.
        Args:
            input: Input tensor of indices
        Returns:
            Tensor of Lorentz embeddings
    Linear layer in Lorentz space.
    This layer performs a linear transformation in the tangent space
    of the Lorentz manifold, followed by an exponential map to project
    back to the manifold.
        Initialize Lorentz linear layer.
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            k: Curvature of the Lorentz manifold (default: -1.0)
            bias: If True, adds a learnable bias to the output
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Lorentz linear layer.
        Args:
            input: Input tensor of Lorentz embeddings
        Returns:
            Tensor of transformed Lorentz embeddings
    Lorentz distance module.
    This module computes the distance between points in Lorentz space,
    which corresponds to the hyperbolic distance in the Poincaré model.
        Initialize Lorentz distance module.
        Args:
            k: Curvature of the Lorentz manifold (default: -1.0)
        Compute Lorentz distance between points.
        Args:
            x: First tensor of Lorentz embeddings
            y: Second tensor of Lorentz embeddings
        Returns:
            Tensor of distances
    Multinomial Logistic Regression in Lorentz space.
    This module implements a classifier based on Lorentz embeddings,
    using the hyperbolic distance to compute class probabilities.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
        Forward pass of the Lorentz MLR.
        Args:
            input: Input tensor of Lorentz embeddings
        Returns:
            Tensor of class logits
    Fusion module for Lorentz embeddings.
    This module combines multiple Lorentz embeddings into a single embedding,
    preserving the manifold structure.
        Initialize Lorentz fusion module.
        Args:
            in_features: Size of each input embedding
            out_features: Size of the output embedding
            k: Curvature of the Lorentz manifold (default: -1.0)
        Fuse multiple Lorentz embeddings.
        Args:
            embeddings: List of Lorentz embedding tensors
        Returns:
            Fused Lorentz embedding
    Hybrid embedding combining Lorentz and Euclidean spaces.
    This module maintains separate embeddings in Lorentz and Euclidean spaces,
    allowing for both hierarchical and similarity-based relationships.
        Initialize hybrid Lorentz-Euclidean embedding.
        Args:
            num_embeddings: Number of embeddings
            lorentz_dim: Dimension of Lorentz embeddings
            euclidean_dim: Dimension of Euclidean embeddings
            k: Curvature of the Lorentz manifold (default: -1.0)
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
            initialize: Initialization method for embeddings
        Forward pass of the hybrid embedding.
        Args:
            input: Input tensor of indices
        Returns:
            Dictionary with 'lorentz' and 'euclidean' embeddings
    Distance module for hybrid Lorentz-Euclidean embeddings.
    This module computes a weighted combination of distances in
    Lorentz and Euclidean spaces.
        Initialize hybrid distance module.
        Args:
            lorentz_weight: Weight for Lorentz distance
            euclidean_weight: Weight for Euclidean distance
            k: Curvature of the Lorentz manifold (default: -1.0)
            learnable_weights: If True, weights are learnable parameters
        Compute hybrid distance between embeddings.
        Args:
            x: First dictionary with 'lorentz' and 'euclidean' embeddings
            y: Second dictionary with 'lorentz' and 'euclidean' embeddings
        Returns:
            Tensor of hybrid distances