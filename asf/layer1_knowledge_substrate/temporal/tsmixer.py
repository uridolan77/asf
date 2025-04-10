"""
TSMixer Implementation

This module provides a TSMixer implementation for temporal modeling in the ASF framework.
TSMixer is a state-of-the-art architecture for time series forecasting that uses MLPs
to mix information across both time and feature dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tsmixer")

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for time series data.
    
    This normalization technique is specifically designed for time series data,
    allowing for normalization during training and denormalization during inference.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        """
        Initialize RevIN.
        
        Args:
            num_features: Number of features/variables in the time series
            eps: Small constant for numerical stability
            affine: Whether to apply learnable affine transformation
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Apply normalization or denormalization.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            mode: 'norm' for normalization, 'denorm' for denormalization
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
            )
            x = (x - self.mean) / self.stdev
            
            if self.affine:
                x = x * self.affine_weight.view(1, 1, -1) + self.affine_bias.view(1, 1, -1)
                
            return x
            
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias.view(1, 1, -1)) / self.affine_weight.view(1, 1, -1)
                
            x = x * self.stdev + self.mean
            
            return x
        
        else:
            raise ValueError(f"Mode {mode} not recognized. Use 'norm' or 'denorm'.")


class TimeMixer(nn.Module):
    """
    Time-mixing module for TSMixer.
    
    Applies MLP along the time dimension to capture temporal patterns.
    """
    
    def __init__(
        self, 
        seq_len: int, 
        hidden_dim: Optional[int] = None, 
        dropout: float = 0.1
    ):
        """
        Initialize TimeMixer.
        
        Args:
            seq_len: Length of the input sequence
            hidden_dim: Hidden dimension for the MLP (default: 2*seq_len)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 2 * seq_len
            
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # MLP for time mixing
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply time mixing.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Time-mixed tensor of the same shape
        """
        # Transpose to [batch_size, num_features, sequence_length]
        x_transposed = x.transpose(1, 2)
        
        # Apply MLP along time dimension
        output = self.time_mlp(x_transposed)
        
        # Transpose back to [batch_size, sequence_length, num_features]
        return output.transpose(1, 2)


class FeatureMixer(nn.Module):
    """
    Feature-mixing module for TSMixer.
    
    Applies MLP along the feature dimension to capture cross-feature interactions.
    """
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: Optional[int] = None, 
        dropout: float = 0.1
    ):
        """
        Initialize FeatureMixer.
        
        Args:
            num_features: Number of features/variables
            hidden_dim: Hidden dimension for the MLP (default: 2*num_features)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 2 * num_features
            
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # MLP for feature mixing
        self.feature_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature mixing.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Feature-mixed tensor of the same shape
        """
        # Apply MLP along feature dimension
        return self.feature_mlp(x)


class TSMixerBlock(nn.Module):
    """
    TSMixer block combining time mixing and feature mixing.
    
    Each block consists of:
    1. Layer normalization + Time mixing + Residual connection
    2. Layer normalization + Feature mixing + Residual connection
    """
    
    def __init__(
        self, 
        seq_len: int, 
        num_features: int, 
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize TSMixerBlock.
        
        Args:
            seq_len: Length of the input sequence
            num_features: Number of features/variables
            time_hidden_dim: Hidden dimension for time mixing MLP
            feature_hidden_dim: Hidden dimension for feature mixing MLP
            dropout: Dropout rate
        """
        super().__init__()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        
        # Time mixing
        self.time_mixer = TimeMixer(
            seq_len=seq_len,
            hidden_dim=time_hidden_dim,
            dropout=dropout
        )
        
        # Feature mixing
        self.feature_mixer = FeatureMixer(
            num_features=num_features,
            hidden_dim=feature_hidden_dim,
            dropout=dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply TSMixer block.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Processed tensor of the same shape
        """
        # Time mixing with residual connection
        x = x + self.time_mixer(self.norm1(x))
        
        # Feature mixing with residual connection
        x = x + self.feature_mixer(self.norm2(x))
        
        return x


class TSMixer(nn.Module):
    """
    TSMixer model for time series forecasting.
    
    This model applies multiple TSMixer blocks followed by a prediction head
    to forecast future values based on historical data.
    """
    
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        num_blocks: int = 3,
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        """
        Initialize TSMixer.
        
        Args:
            seq_len: Length of the input sequence
            num_features: Number of features/variables
            num_blocks: Number of TSMixer blocks
            time_hidden_dim: Hidden dimension for time mixing MLP
            feature_hidden_dim: Hidden dimension for feature mixing MLP
            output_dim: Output dimension (default: same as num_features)
            forecast_horizon: Number of future time steps to predict
            dropout: Dropout rate
            use_revin: Whether to use reversible instance normalization
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = num_features
            
        self.seq_len = seq_len
        self.num_features = num_features
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.use_revin = use_revin
        
        # Reversible Instance Normalization
        if use_revin:
            self.revin = RevIN(num_features=num_features)
        
        # TSMixer blocks
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                num_features=num_features,
                time_hidden_dim=time_hidden_dim,
                feature_hidden_dim=feature_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(num_features)
        
        # Prediction head
        self.prediction_head = nn.Linear(num_features, forecast_horizon * output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TSMixer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Predictions of shape [batch_size, forecast_horizon, output_dim]
        """
        batch_size = x.size(0)
        
        # Apply reversible instance normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Apply TSMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Global average pooling along the time dimension
        x = torch.mean(x, dim=1)
        
        # Apply prediction head
        x = self.prediction_head(x)
        
        # Reshape to [batch_size, forecast_horizon, output_dim]
        x = x.view(batch_size, self.forecast_horizon, self.output_dim)
        
        # Apply denormalization if using RevIN
        if self.use_revin:
            # Reshape for denormalization
            x = x.view(batch_size, self.forecast_horizon, self.output_dim)
            x = self.revin(x, mode='denorm')
        
        return x


class TSMixerEncoder(nn.Module):
    """
    TSMixer-based encoder for temporal data.
    
    This encoder uses TSMixer to process temporal sequences and produces
    fixed-size embeddings for downstream tasks.
    """
    
    def __init__(
        self,
        seq_len: int,
        num_features: int,
        embedding_dim: int,
        num_blocks: int = 3,
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        """
        Initialize TSMixerEncoder.
        
        Args:
            seq_len: Length of the input sequence
            num_features: Number of features/variables
            embedding_dim: Dimension of the output embedding
            num_blocks: Number of TSMixer blocks
            time_hidden_dim: Hidden dimension for time mixing MLP
            feature_hidden_dim: Hidden dimension for feature mixing MLP
            dropout: Dropout rate
            use_revin: Whether to use reversible instance normalization
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.use_revin = use_revin
        
        # Reversible Instance Normalization
        if use_revin:
            self.revin = RevIN(num_features=num_features)
        
        # TSMixer blocks
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                num_features=num_features,
                time_hidden_dim=time_hidden_dim,
                feature_hidden_dim=feature_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(num_features)
        
        # Projection to embedding space
        self.projection = nn.Linear(num_features, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TSMixerEncoder.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        # Apply reversible instance normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Apply TSMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Global average pooling along the time dimension
        x = torch.mean(x, dim=1)
        
        # Project to embedding space
        x = self.projection(x)
        
        return x


class AdaptiveTSMixer(nn.Module):
    """
    Adaptive TSMixer with dynamic sequence length handling.
    
    This variant of TSMixer can handle variable-length sequences by using
    adaptive pooling before applying the TSMixer blocks.
    """
    
    def __init__(
        self,
        max_seq_len: int,
        num_features: int,
        num_blocks: int = 3,
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        """
        Initialize AdaptiveTSMixer.
        
        Args:
            max_seq_len: Maximum sequence length to handle
            num_features: Number of features/variables
            num_blocks: Number of TSMixer blocks
            time_hidden_dim: Hidden dimension for time mixing MLP
            feature_hidden_dim: Hidden dimension for feature mixing MLP
            output_dim: Output dimension (default: same as num_features)
            forecast_horizon: Number of future time steps to predict
            dropout: Dropout rate
            use_revin: Whether to use reversible instance normalization
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = num_features
            
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.use_revin = use_revin
        
        # Adaptive pooling to handle variable sequence lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_seq_len)
        
        # Reversible Instance Normalization
        if use_revin:
            self.revin = RevIN(num_features=num_features)
        
        # TSMixer blocks
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=max_seq_len,
                num_features=num_features,
                time_hidden_dim=time_hidden_dim,
                feature_hidden_dim=feature_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(num_features)
        
        # Prediction head
        self.prediction_head = nn.Linear(num_features, forecast_horizon * output_dim)
    
    def forward(self, x: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of AdaptiveTSMixer.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            seq_lengths: Optional tensor of sequence lengths for each batch item
            
        Returns:
            Predictions of shape [batch_size, forecast_horizon, output_dim]
        """
        batch_size = x.size(0)
        
        # Apply adaptive pooling if sequence length is not max_seq_len
        if x.size(1) != self.max_seq_len:
            # Transpose to [batch_size, num_features, sequence_length]
            x = x.transpose(1, 2)
            
            # Apply adaptive pooling
            x = self.adaptive_pool(x)
            
            # Transpose back to [batch_size, sequence_length, num_features]
            x = x.transpose(1, 2)
        
        # Apply reversible instance normalization
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # Apply TSMixer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Global average pooling along the time dimension
        x = torch.mean(x, dim=1)
        
        # Apply prediction head
        x = self.prediction_head(x)
        
        # Reshape to [batch_size, forecast_horizon, output_dim]
        x = x.view(batch_size, self.forecast_horizon, self.output_dim)
        
        # Apply denormalization if using RevIN
        if self.use_revin:
            # Reshape for denormalization
            x = x.view(batch_size, self.forecast_horizon, self.output_dim)
            x = self.revin(x, mode='denorm')
        
        return x
