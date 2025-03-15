import torch
from typing import Dict, List, Optional

class SparseTensorSymbolNetwork:
    """
    Efficient sparse tensor representation of the symbol network.
    Uses Graph Tensor Convolution inspired approach for rich semantic propagation.
    Enhanced with hardware acceleration capabilities.
    """
    def __init__(self, max_symbols: int = 10000, device=None):
        self.max_symbols = max_symbols
        self.symbol_to_idx: Dict[str, int] = {}
        self.idx_to_symbol: Dict[int, str] = {}
        self.potential_to_idx: Dict[str, int] = {}
        self.idx_to_potential: Dict[int, str] = {}
        
        # Sparse adjacency data structures
        self._row_indices: List[int] = []
        self._col_indices: List[int] = []
        self._edge_values: List[float] = []
        self._edge_types: List[int] = []
        
        # Cached tensor representations
        self._adjacency_tensor: Optional[List[torch.Tensor]] = None
        self._feature_tensor: Optional[torch.Tensor] = None
        self._need_rebuild: bool = True
        
        # Configuration for tensor operations
        self.semantic_channels = 3  # Multiple channels for rich semantic representation
        
        # Hardware acceleration support
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    def add_symbol(self, symbol_id: str) -> int:
        """Add a symbol to the network and return its index."""
        if symbol_id in self.symbol_to_idx:
            return self.symbol_to_idx[symbol_id]
            
        idx = len(self.symbol_to_idx)
        self.symbol_to_idx[symbol_id] = idx
        self.idx_to_symbol[idx] = symbol_id
        self._need_rebuild = True
        return idx
    
    def add_potential(self, potential_id: str) -> int:
        """Add a potential to the network and return its index."""
        if potential_id in self.potential_to_idx:
            return self.potential_to_idx[potential_id]
            
        idx = len(self.potential_to_idx)
        self.potential_to_idx[potential_id] = idx
        self.idx_to_potential[idx] = potential_id
        self._need_rebuild = True
        return idx
    
    def add_relation(self, source_id: str, target_id: str, relation_type: int, strength: float) -> None:
        """Add a relation between symbols or potentials."""
        # Determine if source/target are symbols or potentials
        source_is_symbol = source_id in self.symbol_to_idx
        target_is_symbol = target_id in self.symbol_to_idx
        
        # Get or create indices
        if source_is_symbol:
            source_idx = self.symbol_to_idx[source_id]
        else:
            source_idx = self.add_potential(source_id) + self.max_symbols
            
        if target_is_symbol:
            target_idx = self.symbol_to_idx[target_id]
        else:
            target_idx = self.add_potential(target_id) + self.max_symbols
            
        # Add relation to sparse representation
        self._row_indices.append(source_idx)
        self._col_indices.append(target_idx)
        self._edge_values.append(strength)
        self._edge_types.append(relation_type)
        
        self._need_rebuild = True
        
    def _build_tensors(self) -> None:
        """Build sparse tensor representations of the network with hardware acceleration."""
        if not self._need_rebuild:
            return
            
        # Convert to PyTorch sparse tensor with hardware acceleration
        indices = torch.tensor([self._row_indices, self._col_indices], 
                              dtype=torch.long, device=self.device)
        values = torch.tensor(self._edge_values, dtype=torch.float, device=self.device)
        edge_types = torch.tensor(self._edge_types, dtype=torch.long, device=self.device)
        
        # Size includes both symbols and potentials
        size = (self.max_symbols * 2, self.max_symbols * 2)
        
        # Create multi-channel adjacency tensor (GTCN-inspired)
        # Each channel represents different semantic relationship aspects
        max_edge_type = max(self._edge_types) if self._edge_types else 0
        channels = min(max_edge_type + 1, self.semantic_channels)
        
        adjacency_tensors = []
        for channel in range(channels):
            # Filter edges for this channel/type
            channel_mask = edge_types == channel
            if not torch.any(channel_mask):
                # Empty channel
                channel_tensor = torch.sparse.FloatTensor(
                    torch.zeros((2, 0), dtype=torch.long, device=self.device),
                    torch.tensor([], dtype=torch.float, device=self.device),
                    size
                )
            else:
                channel_indices = indices[:, channel_mask]
                channel_values = values[channel_mask]
                channel_tensor = torch.sparse.FloatTensor(
                    channel_indices, channel_values, size,
                    device=self.device
                )
            adjacency_tensors.append(channel_tensor)
            
        # Store as list of sparse tensors (more efficient than 3D sparse tensor)
        self._adjacency_tensor = adjacency_tensors
        
        # Initialize feature tensor
        total_nodes = len(self.symbol_to_idx) + len(self.potential_to_idx)
        self._feature_tensor = torch.zeros(
            (total_nodes, 16),
            dtype=torch.float,
            device=self.device
        )
        
        self._need_rebuild = False
        
    def propagate_activations(self, initial_activations: Dict[str, float], iterations: int = 2) -> Dict[str, float]:
        """
        Propagate activations through the network using tensor operations.
        Uses GTCN-inspired multi-channel propagation for rich semantic transfer.
        """
        self._build_tensors()
        
        # Initialize activation vector
        activation_vector = torch.zeros(
            self.max_symbols * 2,
            dtype=torch.float,
            device=self.device
        )
        
        # Set initial activations
        for node_id, activation in initial_activations.items():
            if node_id in self.symbol_to_idx:
                idx = self.symbol_to_idx[node_id]
                activation_vector[idx] = activation
            elif node_id in self.potential_to_idx:
                idx = self.potential_to_idx[node_id] + self.max_symbols
                activation_vector[idx] = activation
        
        # Propagate activations through the network
        for _ in range(iterations):
            # Propagate through each channel
            channel_results = []
            for channel_tensor in self._adjacency_tensor:
                if channel_tensor._nnz() > 0:  # Check if tensor has any non-zero elements
                    # Sparse matrix multiplication for efficiency
                    channel_result = torch.sparse.mm(channel_tensor, 
                                                   activation_vector.unsqueeze(1)).squeeze(1)
                    channel_results.append(channel_result)
            
            if channel_results:
                # Stack results from all channels
                stacked_results = torch.stack(channel_results, dim=0)
                # Use maximization-based aggregation (MGC-inspired)
                # This prevents over-smoothing compared to averaging
                new_activations, _ = torch.max(stacked_results, dim=0)
                # Update activations (with residual connection)
                activation_vector = (activation_vector + new_activations) / 2
        
        # Extract results for symbols
        result = {}
        for symbol_id, idx in self.symbol_to_idx.items():
            result[symbol_id] = activation_vector[idx].item()
            
        return result
