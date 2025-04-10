"""
Temporal Graph Network (TGN) Implementation

This module provides TGN implementation for modeling the evolution of scientific literature
and citation networks over time. It extends the graph neural network capabilities with
temporal dynamics to capture how scientific claims and their authority evolve.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

# Try to import from torch_geometric_temporal, with fallback to custom implementation
try:
    from torch_geometric_temporal.nn.recurrent import TGNMemory
    HAS_TORCH_GEOMETRIC_TEMPORAL = True
except ImportError:
    HAS_TORCH_GEOMETRIC_TEMPORAL = False
    # We'll provide a simplified implementation if the library is not available


class TemporalGraphNetwork(nn.Module):
    """
    Temporal Graph Network for modeling evolving publication networks.
    
    This implementation can use torch_geometric_temporal if available,
    or fall back to a simplified custom implementation.
    
    TGNs capture how node representations evolve over time based on
    temporal interactions, making them ideal for modeling citation networks
    and the evolution of scientific claims.
    """
    
    def __init__(
        self, 
        node_features: int, 
        edge_features: int, 
        memory_dim: int = 128,
        time_dim: int = 16,
        num_neighbors: int = 10,
        message_dim: int = 128,
        message_function: str = "identity",
        aggregator_type: str = "mean"
    ):
        """
        Initialize the Temporal Graph Network.
        
        Args:
            node_features: Dimension of node features
            edge_features: Dimension of edge features
            memory_dim: Dimension of memory for each node
            time_dim: Dimension of temporal encoding
            num_neighbors: Number of neighbors to sample for each node
            message_dim: Dimension of messages passed between nodes
            message_function: Type of message function ('identity', 'mlp')
            aggregator_type: Type of neighborhood aggregator ('mean', 'sum', 'max')
        """
        super().__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        
        if HAS_TORCH_GEOMETRIC_TEMPORAL:
            self.tgn = TGNMemory(
                node_features=node_features,
                edge_features=edge_features,
                memory_dim=memory_dim,
                time_dim=time_dim,
                message_dim=message_dim,
                message_function=message_function,
                aggregator_type=aggregator_type
            )
        else:
            # Custom implementation if torch_geometric_temporal is not available
            self._init_custom_tgn(message_dim, message_function, aggregator_type)
        
        # Memory updater (GRU-based)
        self.memory_updater = nn.GRU(memory_dim, memory_dim)
        
        # Output projection
        self.output_projection = nn.Linear(memory_dim, node_features)
        
    def _init_custom_tgn(self, message_dim, message_function, aggregator_type):
        """Initialize custom TGN components if torch_geometric_temporal is not available."""
        # Message function
        if message_function == "identity":
            self.message_fn = nn.Identity()
        else:  # "mlp"
            self.message_fn = nn.Sequential(
                nn.Linear(self.node_features + self.edge_features + self.time_dim, message_dim),
                nn.ReLU(),
                nn.Linear(message_dim, self.memory_dim)
            )
        
        # Time encoder
        self.time_encoder = TimeEncoder(self.time_dim)
        
        # Memory
        self.memory = None
        
        # Aggregator
        self.aggregator_type = aggregator_type
        
    def _custom_forward(self, x, edge_index, edge_attr, t):
        """Custom forward pass implementation."""
        # Initialize memory if not exists
        batch_size = x.size(0)
        if self.memory is None:
            self.memory = torch.zeros(batch_size, self.memory_dim, device=x.device)
        
        # Encode time differences
        time_encoding = self.time_encoder(t)
        
        # Compute messages
        source_nodes, target_nodes = edge_index
        source_features = x[source_nodes]
        
        # Combine source features, edge features, and time encoding
        message_inputs = torch.cat([source_features, edge_attr, time_encoding], dim=1)
        messages = self.message_fn(message_inputs)
        
        # Aggregate messages for each target node
        aggregated_messages = self._aggregate_messages(messages, target_nodes, batch_size)
        
        # Update memory with GRU
        memory_input = aggregated_messages.unsqueeze(0)  # Add sequence dimension
        _, self.memory = self.memory_updater(memory_input, self.memory.unsqueeze(0))
        self.memory = self.memory.squeeze(0)  # Remove sequence dimension
        
        return self.memory
    
    def _aggregate_messages(self, messages, target_nodes, batch_size):
        """Aggregate messages for each target node based on aggregator type."""
        # Create empty tensor for aggregated messages
        aggregated = torch.zeros(batch_size, self.memory_dim, device=messages.device)
        
        # Aggregate based on type
        if self.aggregator_type == "mean":
            # Use scatter_mean (custom implementation since scatter_mean is not in PyTorch)
            count = torch.zeros(batch_size, 1, device=messages.device)
            count.scatter_add_(0, target_nodes.unsqueeze(1), torch.ones_like(target_nodes.unsqueeze(1), dtype=torch.float))
            aggregated.scatter_add_(0, target_nodes.unsqueeze(1).expand(-1, self.memory_dim), messages)
            # Avoid division by zero
            count = torch.clamp(count, min=1.0)
            aggregated = aggregated / count
        elif self.aggregator_type == "sum":
            # Use scatter_add
            aggregated.scatter_add_(0, target_nodes.unsqueeze(1).expand(-1, self.memory_dim), messages)
        elif self.aggregator_type == "max":
            # Use scatter_max (note: this returns a tuple (values, indices))
            aggregated, _ = torch.scatter_max(messages, 0, target_nodes.unsqueeze(1).expand(-1, self.memory_dim))
        
        return aggregated
        
    def forward(self, x, edge_index, edge_attr, t):
        """
        Forward pass of the TGN.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            t: Timestamps for each edge [num_edges]
            
        Returns:
            Updated node embeddings
        """
        if HAS_TORCH_GEOMETRIC_TEMPORAL:
            # Use torch_geometric_temporal implementation
            memory = self.tgn(x, edge_index, edge_attr, t)
        else:
            # Use custom implementation
            memory = self._custom_forward(x, edge_index, edge_attr, t)
        
        # Project memory to output space
        output = self.output_projection(memory)
        
        return output
    
    def reset_memory(self):
        """Reset the memory of the TGN."""
        if HAS_TORCH_GEOMETRIC_TEMPORAL:
            self.tgn.reset_memory()
        else:
            self.memory = None


class TimeEncoder(nn.Module):
    """
    Time encoder for encoding time differences.
    
    Uses a sinusoidal encoding similar to positional encoding in transformers,
    but adapted for continuous time values.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize the time encoder.
        
        Args:
            dimension: Dimension of the time encoding
        """
        super().__init__()
        self.dimension = dimension
        self.w = nn.Parameter(torch.Tensor(1, dimension))
        self.b = nn.Parameter(torch.Tensor(1, dimension))
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset the parameters of the time encoder."""
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)
        
    def forward(self, t):
        """
        Encode time differences.
        
        Args:
            t: Time differences [batch_size]
            
        Returns:
            Time encoding [batch_size, dimension]
        """
        # Reshape t to [batch_size, 1]
        t = t.view(-1, 1)
        
        # Apply sinusoidal encoding
        output = torch.cos(self.w * t + self.b)
        
        return output


class DynamicAuthorityScorer(nn.Module):
    """
    Dynamic authority scorer using TGN embeddings.
    
    This module computes dynamic authority scores for publications based on
    their temporal graph embeddings, citation patterns, and metadata.
    """
    
    def __init__(
        self, 
        embedding_dim: int, 
        metadata_dim: int,
        hidden_dim: int = 128
    ):
        """
        Initialize the dynamic authority scorer.
        
        Args:
            embedding_dim: Dimension of TGN embeddings
            metadata_dim: Dimension of metadata features (impact factor, etc.)
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        # Network for computing authority score
        self.authority_network = nn.Sequential(
            nn.Linear(embedding_dim + metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Scale to [0, 1]
        )
        
    def forward(
        self, 
        tgn_embedding: torch.Tensor, 
        metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic authority scores.
        
        Args:
            tgn_embedding: TGN embeddings [batch_size, embedding_dim]
            metadata: Metadata features [batch_size, metadata_dim]
                (impact factor, citation count, etc.)
            
        Returns:
            Authority scores [batch_size, 1]
        """
        # Concatenate TGN embedding and metadata
        features = torch.cat([tgn_embedding, metadata], dim=1)
        
        # Compute authority score
        authority_score = self.authority_network(features)
        
        return authority_score


class TemporalCitationAnalyzer:
    """
    Analyzer for temporal citation patterns using TGN.
    
    This class provides methods for analyzing citation patterns over time,
    identifying emerging trends, and computing dynamic authority scores.
    """
    
    def __init__(
        self,
        tgn_model: TemporalGraphNetwork,
        authority_scorer: DynamicAuthorityScorer
    ):
        """
        Initialize the temporal citation analyzer.
        
        Args:
            tgn_model: Trained TGN model
            authority_scorer: Dynamic authority scorer
        """
        self.tgn_model = tgn_model
        self.authority_scorer = authority_scorer
        
    def compute_dynamic_authority(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        timestamps: torch.Tensor,
        metadata: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute dynamic authority scores for publications.
        
        Args:
            node_features: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            timestamps: Timestamps for each edge [num_edges]
            metadata: Metadata features [num_nodes, metadata_dim]
            
        Returns:
            Authority scores [num_nodes, 1]
        """
        # Reset TGN memory
        self.tgn_model.reset_memory()
        
        # Get TGN embeddings
        with torch.no_grad():
            tgn_embeddings = self.tgn_model(node_features, edge_index, edge_attr, timestamps)
        
        # Compute authority scores
        authority_scores = self.authority_scorer(tgn_embeddings, metadata)
        
        return authority_scores
    
    def identify_emerging_trends(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        timestamps: torch.Tensor,
        time_window: int = 30,  # Days
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Identify emerging trends in the citation network.
        
        Args:
            node_features: Node features [num_nodes, node_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_features]
            timestamps: Timestamps for each edge [num_edges]
            time_window: Time window in days
            top_k: Number of top trends to return
            
        Returns:
            List of emerging trends with node indices and scores
        """
        # Reset TGN memory
        self.tgn_model.reset_memory()
        
        # Get TGN embeddings
        with torch.no_grad():
            tgn_embeddings = self.tgn_model(node_features, edge_index, edge_attr, timestamps)
        
        # Compute citation velocity (simplified)
        # In a real implementation, this would involve more sophisticated analysis
        # of citation patterns over time
        source_nodes, target_nodes = edge_index
        
        # Count citations in the recent time window
        current_time = timestamps.max()
        recent_mask = (current_time - timestamps) <= time_window
        recent_citations = target_nodes[recent_mask]
        
        # Count citations for each node
        citation_counts = torch.zeros(node_features.size(0), device=node_features.device)
        for node in recent_citations:
            citation_counts[node] += 1
        
        # Normalize by the time window
        citation_velocity = citation_counts / time_window
        
        # Combine with TGN embeddings to get trend score
        # This is a simplified approach; in practice, you might use a more
        # sophisticated model to identify emerging trends
        trend_scores = citation_velocity
        
        # Get top-k trends
        top_k_values, top_k_indices = torch.topk(trend_scores, min(top_k, trend_scores.size(0)))
        
        # Convert to list of dictionaries
        trends = []
        for i in range(top_k_indices.size(0)):
            trends.append({
                'node_index': top_k_indices[i].item(),
                'trend_score': top_k_values[i].item(),
                'citation_velocity': citation_velocity[top_k_indices[i]].item(),
                'embedding': tgn_embeddings[top_k_indices[i]].tolist()
            })
        
        return trends
