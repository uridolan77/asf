import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss
import logging
import threading
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import asyncio

class AdvancedTensorOperations:
    """
    Enhanced tensor operations for the semantic network.
    Provides optimized operations for large-scale semantic processing.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("ASF.Layer3.TensorOps")
        self.logger.info(f"Initializing tensor operations on {self.device}")
        
    def normalize_embeddings(self, embeddings, dim=1, eps=1e-8):
        """
        Normalize embeddings to unit length with optimized performance.
        
        Args:
            embeddings: Embeddings tensor to normalize
            dim: Dimension along which to normalize
            eps: Small value to prevent division by zero
            
        Returns:
            Normalized embeddings
        """
        if isinstance(embeddings, np.ndarray):
            if embeddings.size == 0:
                return embeddings
                
            norm = np.linalg.norm(embeddings, axis=dim, keepdims=True)
            norm[norm < eps] = 1.0  # Avoid division by zero
            return embeddings / norm
            
        elif isinstance(embeddings, torch.Tensor):
            if embeddings.numel() == 0:
                return embeddings
                
            if embeddings.device != self.device:
                embeddings = embeddings.to(self.device)
                
            return F.normalize(embeddings, p=2, dim=dim, eps=eps)
            
        else:
            raise TypeError("Embeddings must be either numpy array or torch tensor")
    
    def batch_cosine_similarity(self, query, database, return_numpy=False):
        """
        Calculate cosine similarity between query and database vectors with optimized batching.
        
        Args:
            query: Query vectors [B, D] or single vector [D]
            database: Database of vectors [N, D]
            return_numpy: Whether to return numpy array (otherwise returns tensor)
            
        Returns:
            Similarity matrix [B, N] or vector [N]
        """
        is_single_query = False
        
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                query = query.reshape(1, -1)
                is_single_query = True
            query = torch.from_numpy(query).to(self.device).float()
            
        if isinstance(database, np.ndarray):
            database = torch.from_numpy(database).to(self.device).float()
        
        if isinstance(query, torch.Tensor):
            if query.dim() == 1:
                query = query.unsqueeze(0)
                is_single_query = True
                
            if query.device != self.device:
                query = query.to(self.device)
                
        if isinstance(database, torch.Tensor):
            if database.device != self.device:
                database = database.to(self.device)
        
        query_normalized = F.normalize(query, p=2, dim=1)
        database_normalized = F.normalize(database, p=2, dim=1)
        
        similarities = torch.mm(query_normalized, database_normalized.t())
        
        if return_numpy:
            similarities = similarities.cpu().numpy()
            
        if is_single_query:
            similarities = similarities.squeeze(0)
            
        return similarities
    
    def attention_weighted_graph(self, adjacency_matrix, node_features, attention_weights=None):
        """
        Apply attention mechanism to graph for focused semantic processing.
        
        Args:
            adjacency_matrix: Adjacency matrix [N, N]
            node_features: Node feature matrix [N, D]
            attention_weights: Optional pre-computed attention weights
            
        Returns:
            Attended node representations [N, D]
        """
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()
        
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).to(self.device).float()
            
        if adjacency_matrix.device != self.device:
            adjacency_matrix = adjacency_matrix.to(self.device)
            
        if node_features.device != self.device:
            node_features = node_features.to(self.device)
            
        if attention_weights is None:
            attention_logits = torch.matmul(node_features, node_features.t())
            
            attention_logits = attention_logits.masked_fill(adjacency_matrix == 0, -1e9)
            
            attention_weights = F.softmax(attention_logits, dim=-1)
        
        attended_features = torch.matmul(attention_weights, node_features)
        
        return attended_features, attention_weights
    
    def tensor_max_pool(self, tensor_list, dim=0):
        """
        Apply max pooling across a list of tensors.
        Implements Seth's principle of MGC-based maximization.
        
        Args:
            tensor_list: List of tensors with same shape
            dim: Dimension along which to stack tensors
            
        Returns:
            Tensor with maximum values
        """
        if not tensor_list:
            return None
            
        processed_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).to(self.device).float()
            elif isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                tensor = tensor.to(self.device)
                
            processed_tensors.append(tensor)
            
        stacked = torch.stack(processed_tensors, dim=dim)
        return torch.max(stacked, dim=dim)[0]
    
    def batch_project_to_hyperplane(self, vectors, normals, biases=None):
        """
        Project vectors onto multiple hyperplanes in a single batch operation.
        
        Args:
            vectors: Vectors to project [B, D] or [N, D]
            normals: Normal vectors of hyperplanes [K, D]
            biases: Optional hyperplane biases [K]
            
        Returns:
            Projected vectors [B, K, D] or [N, K, D]
        """
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors).to(self.device).float()
            
        if isinstance(normals, np.ndarray):
            normals = torch.from_numpy(normals).to(self.device).float()
            
        if biases is not None and isinstance(biases, np.ndarray):
            biases = torch.from_numpy(biases).to(self.device).float()
        elif biases is None:
            biases = torch.zeros(normals.size(0), device=self.device)
            
        if vectors.device != self.device:
            vectors = vectors.to(self.device)
            
        if normals.device != self.device:
            normals = normals.to(self.device)
            
        if biases.device != self.device:
            biases = biases.to(self.device)
            
        normals = F.normalize(normals, p=2, dim=1)
        
        vectors_expanded = vectors.unsqueeze(1)
        normals_expanded = normals.unsqueeze(0)
        biases_expanded = biases.unsqueeze(0)
        
        dots = torch.sum(vectors_expanded * normals_expanded, dim=2)
        
        distances = dots - biases_expanded
        
        distances = distances.unsqueeze(2)
        
        projections = vectors_expanded - distances * normals_expanded
        
        return projections
        
    def soft_attention_weighted_sum(self, query, keys, values, temperature=1.0):
        """
        Calculate attention-weighted sum of values with temperature scaling.
        
        Args:
            query: Query vector [D]
            keys: Key vectors [N, D]
            values: Value vectors [N, V]
            temperature: Controls softmax temperature
            
        Returns:
            Weighted sum of values [V]
        """
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query).to(self.device).float()
            
        if isinstance(keys, np.ndarray):
            keys = torch.from_numpy(keys).to(self.device).float()
            
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device).float()
            
        if query.device != self.device:
            query = query.to(self.device)
            
        if keys.device != self.device:
            keys = keys.to(self.device)
            
        if values.device != self.device:
            values = values.to(self.device)
            
        query_norm = F.normalize(query, p=2, dim=0)
        keys_norm = F.normalize(keys, p=2, dim=1)
        
        similarities = torch.matmul(keys_norm, query_norm)
        
        attention = F.softmax(similarities / temperature, dim=0)
        
        weighted_sum = torch.matmul(attention.unsqueeze(0), values).squeeze(0)
        
        return weighted_sum, attention


class OptimizedSemanticTensorNetwork:
    """
    Enhanced semantic network with optimized tensor operations.
    Implements efficient knowledge representation and querying.
    """
    def __init__(self, embedding_dim=1024, device=None):
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger("ASF.Layer3.SemanticTensorNetwork")
        
        self.nodes = {}  # Maps node ID to SemanticNode
        self.relations = {}  # Maps relation ID to SemanticRelation
        self.node_index = {}  # Maps node ID to matrix index
        self.index_to_node = {}  # Maps matrix index to node ID
        
        self.relation_index = {
            'source': defaultdict(list),  # Maps source ID to relation IDs
            'target': defaultdict(list),  # Maps target ID to relation IDs
            'type': defaultdict(list)  # Maps relation type to relation IDs
        }
        
        self.adjacency_tensors = {}  # By relation type
        self.node_embeddings = None  # [N, D]
        self.faiss_index = None
        self.embedding_buffer = []
        self.node_count = 0
        
        self.tensor_ops = AdvancedTensorOperations(device=self.device)
        self.relation_type_embeddings = {}  # Maps relation type to embedding
        self.rebuild_needed = True
        self.last_rebuild_time = 0
        
        self.similarity_cache = {}
        self.traversal_cache = {}
        self.subgraph_cache = {}
        
        self.lock = threading.RLock()
        
        self.logger.info(f"Initialized optimized semantic tensor network with {embedding_dim} dimensions on {self.device}")
        
    async def add_node(self, node, update_tensors=True):
        with self.lock:
            if relation.source_id not in self.nodes or relation.target_id not in self.nodes:
                return None
                
            if relation.id in self.relations:
                existing_relation = self.relations[relation.id]
                existing_relation.weight = relation.weight
                existing_relation.properties.update(relation.properties)
                existing_relation.confidence = max(existing_relation.confidence, relation.confidence)
                existing_relation.attention_weight = relation.attention_weight
                existing_relation.temporal_metadata.update_modification()
                
                self.rebuild_needed = True
                
                return relation.id
            
            self.relations[relation.id] = relation
            
            self.relation_index['source'][relation.source_id].append(relation.id)
            self.relation_index['target'][relation.target_id].append(relation.id)
            self.relation_index['type'][relation.relation_type].append(relation.id)
            
            self.rebuild_needed = True
            
            return relation.id
            
    async def get_node(self, node_id):
        with self.lock:
            relation = self.relations.get(relation_id)
            if relation:
                relation.temporal_metadata.update_access()
                return relation
            return None
            
    async def get_similar_nodes(self, node_id, k=5, threshold=0.7, 
                              by_embedding=True, relation_types=None):
        node = await self.get_node(node_id)
        if not node:
            return []
            
        with self.lock:
            cache_key = f"{node_id}_{k}_{threshold}_{by_embedding}_{relation_types}"
            if cache_key in self.similarity_cache:
                cache_entry = self.similarity_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < 60:  # 1 minute cache
                    return cache_entry['results']
            
            if by_embedding and hasattr(node, 'embeddings') and node.embeddings is not None:
                if self.rebuild_needed:
                    await self._update_tensors()
                    
                if self.faiss_index is not None:
                    embeddings = np.array([node.embeddings], dtype=np.float32)
                    distances, indices = self.faiss_index.search(embeddings, k*2)  # Get extra for filtering
                    
                    results = []
                    for i in range(len(indices[0])):
                        idx = indices[0][i]
                        if idx != -1 and idx < self.node_count:
                            similar_id = self.index_to_node.get(idx)
                            if similar_id and similar_id != node_id:
                                similarity = 1.0 - distances[0][i] / 2.0  # Convert to similarity
                                if similarity >= threshold:
                                    results.append((similar_id, similarity))
                    
                    self.similarity_cache[cache_key] = {
                        'results': results[:k],
                        'timestamp': time.time()
                    }
                    
                    return results[:k]
            
            return await self._find_similar_by_structure(node, k, threshold, relation_types)
            
    async def get_node_relations(self, node_id, direction="both", relation_type=None):
        Perform traversal search from a starting node.
        
        Args:
            start_node_id: Starting node ID
            max_depth: Maximum traversal depth
            relation_filter: Optional filter function for relations
            
        Returns:
            Dictionary mapping node IDs to paths and distances
        Extract a subgraph containing specified nodes and their relations.
        
        Args:
            node_ids: List of node IDs to include
            include_relations: Whether to include relations between nodes
            max_relations: Maximum number of relations to include
            
        Returns:
            Dictionary with nodes and relations
        Compute attention weights across the graph.
        
        Args:
            focus_node_ids: Optional list of nodes to focus attention on
            attention_weights: Optional predefined attention weights
            
        Returns:
            Attention-weighted embeddings
        adjacency = torch.zeros(self.node_count, self.node_count, device=self.device)
        
        for relation in self.relations.values():
            if relation.source_id in self.node_index and relation.target_id in self.node_index:
                source_idx = self.node_index[relation.source_id]
                target_idx = self.node_index[relation.target_id]
                
                edge_weight = relation.weight * relation.attention_weight
                
                adjacency[source_idx, target_idx] = edge_weight
                
                if relation.bidirectional:
                    adjacency[target_idx, source_idx] = edge_weight
                    
        self.global_adjacency = adjacency
    
    async def _update_tensors(self):
        if len(self.nodes) < 10:
            return  # Not enough nodes for indexing
            
        try:
            embeddings_np = self.node_embeddings.cpu().numpy()
            
            if self.faiss_index is None:
                if self.embedding_dim <= 64:
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                elif self.embedding_dim <= 256:
                    self.faiss_index = faiss.IndexIVFFlat(
                        faiss.IndexFlatL2(self.embedding_dim), 
                        self.embedding_dim, 
                        min(100, self.node_count // 10 + 1),
                        faiss.METRIC_L2
                    )
                    if self.node_count >= 1000:
                        self.faiss_index.train(embeddings_np)
                else:
                    self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                
                valid_mask = np.ones(self.node_count, dtype=bool)
                for idx in invalid_indices:
                    valid_mask[idx] = False
                    
                if np.any(valid_mask):
                    self.faiss_index.add(embeddings_np[valid_mask])
            else:
                for idx, embedding in self.embedding_buffer:
                    if idx not in invalid_indices:
                        if isinstance(self.faiss_index, faiss.IndexHNSWFlat):
                            if len(self.embedding_buffer) > self.node_count // 10:
                                valid_mask = np.ones(self.node_count, dtype=bool)
                                for inv_idx in invalid_indices:
                                    valid_mask[inv_idx] = False
                                
                                if np.any(valid_mask):
                                    self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                                    self.faiss_index.add(embeddings_np[valid_mask])
                        else:
                            self.faiss_index.add(np.array([embedding], dtype=np.float32))
            
            self.embedding_buffer = []
            
        except Exception as e:
            self.logger.error(f"Error updating FAISS index: {str(e)}")
            
    async def _find_similar_by_structure(self, node, k=5, threshold=0.7, relation_types=None):
    Optimized graph aggregation functions for semantic processing.
    Provides efficient implementations of common graph operations.
        Compute mean aggregation of node features across the graph.
        
        Args:
            node_features: Node feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            normalize: Whether to normalize by degree
            
        Returns:
            Aggregated features [N, D]
        Compute max aggregation of node features across the graph.
        
        Args:
            node_features: Node feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            
        Returns:
            Aggregated features [N, D]
        Compute attention-weighted aggregation of node features.
        
        Args:
            query_features: Query feature matrix [N, D]
            key_features: Key feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            value_features: Optional different value features [N, D']
            
        Returns:
            Aggregated features [N, D']
        Compute multi-hop aggregation with exponential decay.
        
        Args:
            node_features: Node feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            hops: Number of hops to aggregate
            alpha: Decay factor for distant hops
            
        Returns:
            Aggregated features [N, D]
    network = OptimizedSemanticTensorNetwork(embedding_dim=128)
    
    class MockNode:
        def __init__(self, id, embeddings=None):
            self.id = id
            self.label = f"Node {id}"
            self.embeddings = embeddings
            self.properties = {}
            self.confidence = 0.8
            
            class TemporalMetadata:
                def update_access(self):
                    pass
                    
                def update_modification(self):
                    pass
                    
            self.temporal_metadata = TemporalMetadata()
    
    class MockRelation:
        def __init__(self, id, source_id, target_id, relation_type="related_to"):
            self.id = id
            self.source_id = source_id
            self.target_id = target_id
            self.relation_type = relation_type
            self.weight = 0.9
            self.attention_weight = 1.0
            self.bidirectional = False
            self.properties = {}
            self.confidence = 0.8
            
            class TemporalMetadata:
                def update_access(self):
                    pass
                    
                def update_modification(self):
                    pass
                    
            self.temporal_metadata = TemporalMetadata()
    
    for i in range(10):
        embeddings = np.random.randn(128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings)
        
        node = MockNode(f"node_{i}", embeddings)
        await network.add_node(node)
    
    for i in range(5):
        source_id = f"node_{i}"
        target_id = f"node_{i+5}"
        
        relation = MockRelation(f"rel_{i}", source_id, target_id)
        await network.add_relation(relation)
    
    similar = await network.get_similar_nodes("node_0", k=3)
    print(f"Similar nodes to node_0: {similar}")
    
    traversal = await network.traversal_search("node_0", max_depth=2)
    print(f"Traversal from node_0: {list(traversal.keys())}")
    
    attention = await network.compute_graph_attention(focus_node_ids=["node_0"])
    print(f"Attention shape: {attention['attention'].shape}")
    
    agg = GraphAggregationFunctions()
    adj_matrix = network.global_adjacency.cpu().numpy()
    node_features = network.node_embeddings.cpu().numpy()
    
    mean_agg = agg.mean_aggregation(node_features, adj_matrix)
    print(f"Mean aggregation shape: {mean_agg.shape}")
    
    return "Tests completed successfully"

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_optimized_tensor_network())