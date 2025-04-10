"""
Lorentz Embeddings

This module provides utilities for working with Lorentz manifold embeddings,
which offer improved numerical stability and performance compared to Poincaré
embeddings for hierarchical data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import logging

# Import geoopt for Lorentz manifold operations
from geoopt import Lorentz
from geoopt.manifolds.lorentz import math as lorentz_math

# Configure logging
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
        """
        Initialize Lorentz embedding layer.
        
        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Dimension of each embedding
            k: Curvature of the Lorentz manifold (default: -1.0)
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized
            norm_type: The p of the p-norm to compute for the max_norm option
            scale_grad_by_freq: If True, scale gradients by the inverse of frequency of the words in the mini-batch
            initialize: Initialization method ("random", "uniform", or "normal")
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.k = k
        self.manifold = Lorentz(k=k)
        
        # Create standard Euclidean embedding
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        
        # Initialize embedding
        self._initialize_weights(initialize)
        
        # Store other parameters
        self.sparse = sparse
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        
        # Handle padding_idx
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
            # Initialize with random points on the Lorentz manifold
            with torch.no_grad():
                # Initialize in the tangent space at the origin
                tangent_vectors = torch.randn(self.num_embeddings, self.embedding_dim)
                # Normalize to ensure valid Lorentz embeddings
                tangent_vectors = F.normalize(tangent_vectors, p=2, dim=1) * 0.1
                # Set the time component to ensure points lie on the manifold
                tangent_vectors[:, 0] = 0.0  # Zero out the time component
                
                # Map to the Lorentz manifold using the exponential map
                origin = torch.zeros(1, self.embedding_dim, device=tangent_vectors.device)
                origin[:, 0] = 1.0  # Set the time component to 1
                
                # Use the exponential map to get points on the manifold
                self.weight.data = self.manifold.expmap(origin, tangent_vectors)
        
        elif method == "uniform":
            # Initialize with uniform distribution
            nn.init.uniform_(self.weight, -0.1, 0.1)
            # Project to the Lorentz manifold
            self._project_to_lorentz()
        
        elif method == "normal":
            # Initialize with normal distribution
            nn.init.normal_(self.weight, mean=0.0, std=0.1)
            # Project to the Lorentz manifold
            self._project_to_lorentz()
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
    
    def _project_to_lorentz(self):
        """Project embeddings to the Lorentz manifold."""
        with torch.no_grad():
            # Compute the squared norm of the spatial components
            spatial_norm_squared = torch.sum(self.weight[:, 1:] ** 2, dim=1, keepdim=True)
            
            # Set the time component to ensure points lie on the manifold
            # For Lorentz manifold with curvature k, we need x_0^2 - sum(x_i^2) = 1/k
            self.weight[:, 0] = torch.sqrt(spatial_norm_squared - 1.0 / self.k)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Lorentz embedding layer.
        
        Args:
            input: Input tensor of indices
            
        Returns:
            Tensor of Lorentz embeddings
        """
        # Use standard embedding function with the stored parameters
        embeddings = F.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )
        
        # Project back to the Lorentz manifold if needed
        # This ensures that the embeddings always lie on the manifold
        if self.training and self.max_norm is not None:
            embeddings = self.manifold.projx(embeddings)
        
        return embeddings


class LorentzLinear(nn.Module):
    """
    Linear layer in Lorentz space.
    
    This layer performs a linear transformation in the tangent space
    of the Lorentz manifold, followed by an exponential map to project
    back to the manifold.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        k: float = -1.0,
        bias: bool = True
    ):
        """
        Initialize Lorentz linear layer.
        
        Args:
            in_features: Size of each input sample
            out_features: Size of each output sample
            k: Curvature of the Lorentz manifold (default: -1.0)
            bias: If True, adds a learnable bias to the output
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.manifold = Lorentz(k=k)
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weight and bias parameters."""
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
        """
        # Map input to the tangent space at the origin
        origin = torch.zeros(1, self.in_features, device=input.device)
        origin[:, 0] = 1.0  # Set the time component to 1
        
        # Compute the logarithmic map to get tangent vectors
        tangent_vectors = self.manifold.logmap(origin, input)
        
        # Apply linear transformation in the tangent space
        transformed = F.linear(tangent_vectors, self.weight, self.bias)
        
        # Map back to the Lorentz manifold using the exponential map
        output = self.manifold.expmap(origin, transformed)
        
        return output


class LorentzDistance(nn.Module):
    """
    Lorentz distance module.
    
    This module computes the distance between points in Lorentz space,
    which corresponds to the hyperbolic distance in the Poincaré model.
    """
    
    def __init__(self, k: float = -1.0):
        """
        Initialize Lorentz distance module.
        
        Args:
            k: Curvature of the Lorentz manifold (default: -1.0)
        """
        super().__init__()
        self.k = k
        self.manifold = Lorentz(k=k)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute Lorentz distance between points.
        
        Args:
            x: First tensor of Lorentz embeddings
            y: Second tensor of Lorentz embeddings
            
        Returns:
            Tensor of distances
        """
        return self.manifold.dist(x, y)


class LorentzMLR(nn.Module):
    """
    Multinomial Logistic Regression in Lorentz space.
    
    This module implements a classifier based on Lorentz embeddings,
    using the hyperbolic distance to compute class probabilities.
    """
    
    def __init__(
        self, 
        in_features: int, 
        num_classes: int, 
        k: float = -1.0
    ):
        """
        Initialize Lorentz MLR.
        
        Args:
            in_features: Size of each input sample
            num_classes: Number of classes
            k: Curvature of the Lorentz manifold (default: -1.0)
        """
        super().__init__()
        
        self.in_features = in_features
        self.num_classes = num_classes
        self.k = k
        self.manifold = Lorentz(k=k)
        
        # Create class embeddings
        self.class_embeddings = nn.Parameter(torch.Tensor(num_classes, in_features))
        
        # Initialize class embeddings
        self._initialize_class_embeddings()
    
    def _initialize_class_embeddings(self):
        """Initialize class embeddings on the Lorentz manifold."""
        with torch.no_grad():
            # Initialize in the tangent space at the origin
            tangent_vectors = torch.randn(self.num_classes, self.in_features)
            # Normalize to ensure valid Lorentz embeddings
            tangent_vectors = F.normalize(tangent_vectors, p=2, dim=1) * 0.1
            # Set the time component to ensure points lie on the manifold
            tangent_vectors[:, 0] = 0.0  # Zero out the time component
            
            # Map to the Lorentz manifold using the exponential map
            origin = torch.zeros(1, self.in_features, device=tangent_vectors.device)
            origin[:, 0] = 1.0  # Set the time component to 1
            
            # Use the exponential map to get points on the manifold
            self.class_embeddings.data = self.manifold.expmap(origin, tangent_vectors)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Lorentz MLR.
        
        Args:
            input: Input tensor of Lorentz embeddings
            
        Returns:
            Tensor of class logits
        """
        # Compute distances to class embeddings
        distances = torch.zeros(input.size(0), self.num_classes, device=input.device)
        
        for i in range(self.num_classes):
            class_embedding = self.class_embeddings[i].unsqueeze(0)
            distances[:, i] = self.manifold.dist(input, class_embedding.expand_as(input))
        
        # Convert distances to logits (negative distances)
        logits = -distances
        
        return logits


class LorentzFusion(nn.Module):
    """
    Fusion module for Lorentz embeddings.
    
    This module combines multiple Lorentz embeddings into a single embedding,
    preserving the manifold structure.
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        k: float = -1.0
    ):
        """
        Initialize Lorentz fusion module.
        
        Args:
            in_features: Size of each input embedding
            out_features: Size of the output embedding
            k: Curvature of the Lorentz manifold (default: -1.0)
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.manifold = Lorentz(k=k)
        
        # Create transformation layers
        self.transform = LorentzLinear(in_features, out_features, k=k)
        
        # Create attention mechanism for weighted fusion
        self.attention = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple Lorentz embeddings.
        
        Args:
            embeddings: List of Lorentz embedding tensors
            
        Returns:
            Fused Lorentz embedding
        """
        # Compute attention weights
        attention_weights = []
        for embedding in embeddings:
            # Map to tangent space for attention computation
            origin = torch.zeros(1, self.in_features, device=embedding.device)
            origin[:, 0] = 1.0  # Set the time component to 1
            tangent_vector = self.manifold.logmap(origin, embedding)
            
            # Compute attention weight
            weight = self.attention(tangent_vector)
            attention_weights.append(weight)
        
        # Normalize attention weights
        attention_weights = torch.cat(attention_weights, dim=1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted combination in tangent space
        origin = torch.zeros(1, self.in_features, device=embeddings[0].device)
        origin[:, 0] = 1.0  # Set the time component to 1
        
        tangent_vectors = []
        for embedding in embeddings:
            tangent_vector = self.manifold.logmap(origin, embedding)
            tangent_vectors.append(tangent_vector)
        
        tangent_vectors = torch.stack(tangent_vectors, dim=1)
        weighted_tangent = torch.sum(tangent_vectors * attention_weights.unsqueeze(-1), dim=1)
        
        # Map back to Lorentz manifold
        fused_embedding = self.manifold.expmap(origin, weighted_tangent)
        
        # Transform to output dimension
        output = self.transform(fused_embedding)
        
        return output


class HybridLorentzEuclideanEmbedding(nn.Module):
    """
    Hybrid embedding combining Lorentz and Euclidean spaces.
    
    This module maintains separate embeddings in Lorentz and Euclidean spaces,
    allowing for both hierarchical and similarity-based relationships.
    """
    
    def __init__(
        self, 
        num_embeddings: int, 
        lorentz_dim: int, 
        euclidean_dim: int, 
        k: float = -1.0,
        sparse: bool = False,
        padding_idx: Optional[int] = None,
        initialize: str = "random"
    ):
        """
        Initialize hybrid Lorentz-Euclidean embedding.
        
        Args:
            num_embeddings: Number of embeddings
            lorentz_dim: Dimension of Lorentz embeddings
            euclidean_dim: Dimension of Euclidean embeddings
            k: Curvature of the Lorentz manifold (default: -1.0)
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
            initialize: Initialization method for embeddings
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.lorentz_dim = lorentz_dim
        self.euclidean_dim = euclidean_dim
        self.k = k
        
        # Create Lorentz embedding
        self.lorentz_embedding = LorentzEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=lorentz_dim,
            k=k,
            sparse=sparse,
            padding_idx=padding_idx,
            initialize=initialize
        )
        
        # Create Euclidean embedding
        self.euclidean_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=euclidean_dim,
            sparse=sparse,
            padding_idx=padding_idx
        )
        
        # Initialize Euclidean embedding
        if initialize == "uniform":
            nn.init.uniform_(self.euclidean_embedding.weight, -0.1, 0.1)
        elif initialize == "normal":
            nn.init.normal_(self.euclidean_embedding.weight, mean=0.0, std=0.1)
        else:
            nn.init.xavier_uniform_(self.euclidean_embedding.weight)
    
    def forward(self, input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hybrid embedding.
        
        Args:
            input: Input tensor of indices
            
        Returns:
            Dictionary with 'lorentz' and 'euclidean' embeddings
        """
        lorentz_embeddings = self.lorentz_embedding(input)
        euclidean_embeddings = self.euclidean_embedding(input)
        
        return {
            'lorentz': lorentz_embeddings,
            'euclidean': euclidean_embeddings
        }


class HybridLorentzEuclideanDistance(nn.Module):
    """
    Distance module for hybrid Lorentz-Euclidean embeddings.
    
    This module computes a weighted combination of distances in
    Lorentz and Euclidean spaces.
    """
    
    def __init__(
        self, 
        lorentz_weight: float = 0.5, 
        euclidean_weight: float = 0.5,
        k: float = -1.0,
        learnable_weights: bool = False
    ):
        """
        Initialize hybrid distance module.
        
        Args:
            lorentz_weight: Weight for Lorentz distance
            euclidean_weight: Weight for Euclidean distance
            k: Curvature of the Lorentz manifold (default: -1.0)
            learnable_weights: If True, weights are learnable parameters
        """
        super().__init__()
        
        self.k = k
        self.manifold = Lorentz(k=k)
        
        if learnable_weights:
            self.lorentz_weight = nn.Parameter(torch.tensor(lorentz_weight))
            self.euclidean_weight = nn.Parameter(torch.tensor(euclidean_weight))
        else:
            self.register_buffer('lorentz_weight', torch.tensor(lorentz_weight))
            self.register_buffer('euclidean_weight', torch.tensor(euclidean_weight))
    
    def forward(
        self, 
        x: Dict[str, torch.Tensor], 
        y: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute hybrid distance between embeddings.
        
        Args:
            x: First dictionary with 'lorentz' and 'euclidean' embeddings
            y: Second dictionary with 'lorentz' and 'euclidean' embeddings
            
        Returns:
            Tensor of hybrid distances
        """
        # Compute Lorentz distance
        lorentz_dist = self.manifold.dist(x['lorentz'], y['lorentz'])
        
        # Compute Euclidean distance
        euclidean_dist = torch.norm(x['euclidean'] - y['euclidean'], p=2, dim=-1)
        
        # Normalize distances to [0, 1] range
        max_lorentz_dist = torch.max(lorentz_dist)
        if max_lorentz_dist > 0:
            lorentz_dist = lorentz_dist / max_lorentz_dist
        
        max_euclidean_dist = torch.max(euclidean_dist)
        if max_euclidean_dist > 0:
            euclidean_dist = euclidean_dist / max_euclidean_dist
        
        # Compute weighted combination
        hybrid_dist = self.lorentz_weight * lorentz_dist + self.euclidean_weight * euclidean_dist
        
        return hybrid_dist
