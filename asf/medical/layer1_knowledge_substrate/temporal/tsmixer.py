TSMixer Implementation

This module provides a TSMixer implementation for temporal modeling in the ASF framework.
TSMixer is a state-of-the-art architecture for time series forecasting that uses MLPs
to mix information across both time and feature dimensions.

import torch
import torch.nn as nn
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tsmixer")

class RevIN(nn.Module):
    Reversible Instance Normalization for time series data.
    
    This normalization technique is specifically designed for time series data,
    allowing for normalization during training and denormalization during inference.
    
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
    Time-mixing module for TSMixer.
    
    Applies MLP along the time dimension to capture temporal patterns.
    
    def __init__(
        self, 
        seq_len: int, 
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                seq_len: Description of seq_len
                hidden_dim: Description of hidden_dim
                dropout: Description of dropout
            """
        hidden_dim: Optional[int] = None, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 2 * seq_len
            
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
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
        x_transposed = x.transpose(1, 2)
        
        output = self.time_mlp(x_transposed)
        
        return output.transpose(1, 2)


class FeatureMixer(nn.Module):
    Feature-mixing module for TSMixer.
    
    Applies MLP along the feature dimension to capture cross-feature interactions.
    
    def __init__(
        self, 
        num_features: int, 
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                num_features: Description of num_features
                hidden_dim: Description of hidden_dim
                dropout: Description of dropout
            """
        hidden_dim: Optional[int] = None, 
        dropout: float = 0.1
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = 2 * num_features
            
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
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
        return self.feature_mlp(x)


class TSMixerBlock(nn.Module):
    TSMixer block combining time mixing and feature mixing.
    
    Each block consists of:
    1. Layer normalization + Time mixing + Residual connection
    2. Layer normalization + Feature mixing + Residual connection
    
    def __init__(
        self, 
        seq_len: int, 
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                seq_len: Description of seq_len
                num_features: Description of num_features
                time_hidden_dim: Description of time_hidden_dim
                feature_hidden_dim: Description of feature_hidden_dim
                dropout: Description of dropout
            """
        num_features: int, 
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(num_features)
        self.norm2 = nn.LayerNorm(num_features)
        
        self.time_mixer = TimeMixer(
            seq_len=seq_len,
            hidden_dim=time_hidden_dim,
            dropout=dropout
        )
        
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
        x = x + self.time_mixer(self.norm1(x))
        
        x = x + self.feature_mixer(self.norm2(x))
        
        return x


class TSMixer(nn.Module):
    TSMixer model for time series forecasting.
    
    This model applies multiple TSMixer blocks followed by a prediction head
    to forecast future values based on historical data.
    
    def __init__(
        self,
        seq_len: int,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                seq_len: Description of seq_len
                num_features: Description of num_features
                num_blocks: Description of num_blocks
                time_hidden_dim: Description of time_hidden_dim
                feature_hidden_dim: Description of feature_hidden_dim
                output_dim: Description of output_dim
                forecast_horizon: Description of forecast_horizon
                dropout: Description of dropout
                use_revin: Description of use_revin
            """
        num_features: int,
        num_blocks: int = 3,
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = num_features
            
        self.seq_len = seq_len
        self.num_features = num_features
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.use_revin = use_revin
        
        if use_revin:
            self.revin = RevIN(num_features=num_features)
        
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                num_features=num_features,
                time_hidden_dim=time_hidden_dim,
                feature_hidden_dim=feature_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(num_features)
        
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
        
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        
        x = torch.mean(x, dim=1)
        
        x = self.prediction_head(x)
        
        x = x.view(batch_size, self.forecast_horizon, self.output_dim)
        
        if self.use_revin:
            x = x.view(batch_size, self.forecast_horizon, self.output_dim)
            x = self.revin(x, mode='denorm')
        
        return x


class TSMixerEncoder(nn.Module):
    TSMixer-based encoder for temporal data.
    
    This encoder uses TSMixer to process temporal sequences and produces
    fixed-size embeddings for downstream tasks.
    
    def __init__(
        self,
        seq_len: int,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                seq_len: Description of seq_len
                num_features: Description of num_features
                embedding_dim: Description of embedding_dim
                num_blocks: Description of num_blocks
                time_hidden_dim: Description of time_hidden_dim
                feature_hidden_dim: Description of feature_hidden_dim
                dropout: Description of dropout
                use_revin: Description of use_revin
            """
        num_features: int,
        embedding_dim: int,
        num_blocks: int = 3,
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.use_revin = use_revin
        
        if use_revin:
            self.revin = RevIN(num_features=num_features)
        
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=seq_len,
                num_features=num_features,
                time_hidden_dim=time_hidden_dim,
                feature_hidden_dim=feature_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(num_features)
        
        self.projection = nn.Linear(num_features, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TSMixerEncoder.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, num_features]
            
        Returns:
            Embedding tensor of shape [batch_size, embedding_dim]
        """
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        
        x = torch.mean(x, dim=1)
        
        x = self.projection(x)
        
        return x


class AdaptiveTSMixer(nn.Module):
    Adaptive TSMixer with dynamic sequence length handling.
    
    This variant of TSMixer can handle variable-length sequences by using
    adaptive pooling before applying the TSMixer blocks.
    
    def __init__(
        self,
        max_seq_len: int,
            """
            __init__ function.
            
            This function provides functionality for...
            Args:
                max_seq_len: Description of max_seq_len
                num_features: Description of num_features
                num_blocks: Description of num_blocks
                time_hidden_dim: Description of time_hidden_dim
                feature_hidden_dim: Description of feature_hidden_dim
                output_dim: Description of output_dim
                forecast_horizon: Description of forecast_horizon
                dropout: Description of dropout
                use_revin: Description of use_revin
            """
        num_features: int,
        num_blocks: int = 3,
        time_hidden_dim: Optional[int] = None,
        feature_hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        forecast_horizon: int = 1,
        dropout: float = 0.1,
        use_revin: bool = True
    ):
        super().__init__()
        
        if output_dim is None:
            output_dim = num_features
            
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.output_dim = output_dim
        self.forecast_horizon = forecast_horizon
        self.use_revin = use_revin
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_seq_len)
        
        if use_revin:
            self.revin = RevIN(num_features=num_features)
        
        self.blocks = nn.ModuleList([
            TSMixerBlock(
                seq_len=max_seq_len,
                num_features=num_features,
                time_hidden_dim=time_hidden_dim,
                feature_hidden_dim=feature_hidden_dim,
                dropout=dropout
            ) for _ in range(num_blocks)
        ])
        
        self.final_norm = nn.LayerNorm(num_features)
        
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
        
        if x.size(1) != self.max_seq_len:
            x = x.transpose(1, 2)
            
            x = self.adaptive_pool(x)
            
            x = x.transpose(1, 2)
        
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        for block in self.blocks:
            x = block(x)
        
        x = self.final_norm(x)
        
        x = torch.mean(x, dim=1)
        
        x = self.prediction_head(x)
        
        x = x.view(batch_size, self.forecast_horizon, self.output_dim)
        
        if self.use_revin:
            x = x.view(batch_size, self.forecast_horizon, self.output_dim)
            x = self.revin(x, mode='denorm')
        
        return x
