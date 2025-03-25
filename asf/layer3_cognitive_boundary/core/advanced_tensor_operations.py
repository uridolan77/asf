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
            # Handle NumPy arrays
            if embeddings.size == 0:
                return embeddings
                
            norm = np.linalg.norm(embeddings, axis=dim, keepdims=True)
            norm[norm < eps] = 1.0  # Avoid division by zero
            return embeddings / norm
            
        elif isinstance(embeddings, torch.Tensor):
            # Handle PyTorch tensors with optimized implementation
            if embeddings.numel() == 0:
                return embeddings
                
            # Move to correct device if needed
            if embeddings.device != self.device:
                embeddings = embeddings.to(self.device)
                
            # Use PyTorch's built-in normalization for best performance
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
        
        # Handle numpy arrays
        if isinstance(query, np.ndarray):
            if query.ndim == 1:
                query = query.reshape(1, -1)
                is_single_query = True
            query = torch.from_numpy(query).to(self.device).float()
            
        if isinstance(database, np.ndarray):
            database = torch.from_numpy(database).to(self.device).float()
        
        # Handle torch tensors
        if isinstance(query, torch.Tensor):
            if query.dim() == 1:
                query = query.unsqueeze(0)
                is_single_query = True
                
            # Move to device if needed
            if query.device != self.device:
                query = query.to(self.device)
                
        if isinstance(database, torch.Tensor):
            if database.device != self.device:
                database = database.to(self.device)
        
        # Normalize
        query_normalized = F.normalize(query, p=2, dim=1)
        database_normalized = F.normalize(database, p=2, dim=1)
        
        # Calculate similarities
        similarities = torch.mm(query_normalized, database_normalized.t())
        
        # Convert back to numpy if requested
        if return_numpy:
            similarities = similarities.cpu().numpy()
            
        # Remove batch dimension for single query
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
        # Ensure tensors
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()
        
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).to(self.device).float()
            
        # Move to device if needed
        if adjacency_matrix.device != self.device:
            adjacency_matrix = adjacency_matrix.to(self.device)
            
        if node_features.device != self.device:
            node_features = node_features.to(self.device)
            
        # Compute attention weights if not provided
        if attention_weights is None:
            # Simple dot-product attention
            attention_logits = torch.matmul(node_features, node_features.t())
            
            # Apply adjacency mask (attention only flows along edges)
            attention_logits = attention_logits.masked_fill(adjacency_matrix == 0, -1e9)
            
            # Normalize with softmax
            attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention to propagate features
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
            
        # Convert all to same type and device
        processed_tensors = []
        for tensor in tensor_list:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).to(self.device).float()
            elif isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                tensor = tensor.to(self.device)
                
            processed_tensors.append(tensor)
            
        # Stack and max
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
        # Ensure tensors
        if isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors).to(self.device).float()
            
        if isinstance(normals, np.ndarray):
            normals = torch.from_numpy(normals).to(self.device).float()
            
        if biases is not None and isinstance(biases, np.ndarray):
            biases = torch.from_numpy(biases).to(self.device).float()
        elif biases is None:
            biases = torch.zeros(normals.size(0), device=self.device)
            
        # Move to device if needed
        if vectors.device != self.device:
            vectors = vectors.to(self.device)
            
        if normals.device != self.device:
            normals = normals.to(self.device)
            
        if biases.device != self.device:
            biases = biases.to(self.device)
            
        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=1)
        
        # Reshape for broadcasting
        # vectors: [B, 1, D]
        # normals: [1, K, D]
        # biases: [1, K]
        vectors_expanded = vectors.unsqueeze(1)
        normals_expanded = normals.unsqueeze(0)
        biases_expanded = biases.unsqueeze(0)
        
        # Compute dot products: [B, K]
        dots = torch.sum(vectors_expanded * normals_expanded, dim=2)
        
        # Subtract biases: [B, K]
        distances = dots - biases_expanded
        
        # Expand distances and normals for broadcasting
        # distances: [B, K, 1]
        # normals: [1, K, D]
        distances = distances.unsqueeze(2)
        
        # Compute projections: [B, K, D]
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
        # Ensure tensors
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query).to(self.device).float()
            
        if isinstance(keys, np.ndarray):
            keys = torch.from_numpy(keys).to(self.device).float()
            
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(self.device).float()
            
        # Move to device if needed
        if query.device != self.device:
            query = query.to(self.device)
            
        if keys.device != self.device:
            keys = keys.to(self.device)
            
        if values.device != self.device:
            values = values.to(self.device)
            
        # Normalize query and keys
        query_norm = F.normalize(query, p=2, dim=0)
        keys_norm = F.normalize(keys, p=2, dim=1)
        
        # Compute similarities
        similarities = torch.matmul(keys_norm, query_norm)
        
        # Apply temperature and softmax
        attention = F.softmax(similarities / temperature, dim=0)
        
        # Apply attention weights
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
        
        # Core data structures
        self.nodes = {}  # Maps node ID to SemanticNode
        self.relations = {}  # Maps relation ID to SemanticRelation
        self.node_index = {}  # Maps node ID to matrix index
        self.index_to_node = {}  # Maps matrix index to node ID
        
        # Relation indices for efficient lookup
        self.relation_index = {
            'source': defaultdict(list),  # Maps source ID to relation IDs
            'target': defaultdict(list),  # Maps target ID to relation IDs
            'type': defaultdict(list)  # Maps relation type to relation IDs
        }
        
        # Tensor components
        self.adjacency_tensors = {}  # By relation type
        self.node_embeddings = None  # [N, D]
        self.faiss_index = None
        self.embedding_buffer = []
        self.node_count = 0
        
        # Enhanced components
        self.tensor_ops = AdvancedTensorOperations(device=self.device)
        self.relation_type_embeddings = {}  # Maps relation type to embedding
        self.rebuild_needed = True
        self.last_rebuild_time = 0
        
        # Caches
        self.similarity_cache = {}
        self.traversal_cache = {}
        self.subgraph_cache = {}
        
        # Locks for thread safety
        self.lock = threading.RLock()
        
        self.logger.info(f"Initialized optimized semantic tensor network with {embedding_dim} dimensions on {self.device}")
        
    async def add_node(self, node, update_tensors=True):
        """Add a node to the semantic network with incremental tensor update."""
        with self.lock:
            if node.id in self.nodes:
                # Update existing node
                existing_node = self.nodes[node.id]
                existing_node.label = node.label
                existing_node.properties.update(node.properties)
                existing_node.confidence = max(existing_node.confidence, node.confidence)
                
                # Check for embedding updates
                embeddings_updated = False
                
                if isinstance(node.embeddings, np.ndarray) and hasattr(existing_node, 'embeddings'):
                    if np.linalg.norm(existing_node.embeddings - node.embeddings) > 0.1:
                        existing_node.embeddings = node.embeddings
                        embeddings_updated = True
                        
                elif hasattr(node, 'embeddings') and node.embeddings is not None:
                    existing_node.embeddings = node.embeddings
                    embeddings_updated = True
                
                # Update tensor representation if needed
                if embeddings_updated:
                    if isinstance(existing_node.embeddings, np.ndarray):
                        existing_node.tensor_representation = torch.tensor(
                            existing_node.embeddings, dtype=torch.float32
                        ).to(self.device)
                    elif isinstance(existing_node.embeddings, torch.Tensor):
                        existing_node.tensor_representation = existing_node.embeddings.to(self.device)
                    
                    # Mark for incremental FAISS update
                    if node.id in self.node_index:
                        self.embedding_buffer.append((self.node_index[node.id], existing_node.embeddings))
                        
                    self.rebuild_needed = True
                    existing_node.temporal_metadata.update_modification()
                    
                return node.id
            
            # Add new node
            self.nodes[node.id] = node
            
            # Prepare tensor representation
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                if isinstance(node.embeddings, np.ndarray):
                    node.tensor_representation = torch.tensor(
                        node.embeddings, dtype=torch.float32
                    ).to(self.device)
                elif isinstance(node.embeddings, torch.Tensor):
                    node.tensor_representation = node.embeddings.to(self.device)
            
            # Update indices
            self.node_index[node.id] = self.node_count
            self.index_to_node[self.node_count] = node.id
            
            # Mark for incremental FAISS update
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                self.embedding_buffer.append((self.node_count, node.embeddings))
                
            self.node_count += 1
            self.rebuild_needed = True
            
            # Check if we should update tensors
            if update_tensors and self.batched_operations and len(self.embedding_buffer) >= 10:
                await self._update_tensors()
                
            return node.id
            
    async def add_relation(self, relation):
        """Add a relation to the semantic network with tensor updates."""
        with self.lock:
            # Ensure source and target nodes exist
            if relation.source_id not in self.nodes or relation.target_id not in self.nodes:
                return None
                
            if relation.id in self.relations:
                # Update existing relation
                existing_relation = self.relations[relation.id]
                existing_relation.weight = relation.weight
                existing_relation.properties.update(relation.properties)
                existing_relation.confidence = max(existing_relation.confidence, relation.confidence)
                existing_relation.attention_weight = relation.attention_weight
                existing_relation.temporal_metadata.update_modification()
                
                # Mark as needing rebuild for tensor operations
                self.rebuild_needed = True
                
                return relation.id
            
            # Add new relation
            self.relations[relation.id] = relation
            
            # Update indices
            self.relation_index['source'][relation.source_id].append(relation.id)
            self.relation_index['target'][relation.target_id].append(relation.id)
            self.relation_index['type'][relation.relation_type].append(relation.id)
            
            # Update adjacency tensors if they exist
            self.rebuild_needed = True
            
            return relation.id
            
    async def get_node(self, node_id):
        """Get a node by ID with access tracking."""
        with self.lock:
            node = self.nodes.get(node_id)
            if node:
                node.temporal_metadata.update_access()
                return node
            return None
            
    async def get_relation(self, relation_id):
        """Get a relation by ID with access tracking."""
        with self.lock:
            relation = self.relations.get(relation_id)
            if relation:
                relation.temporal_metadata.update_access()
                return relation
            return None
            
    async def get_similar_nodes(self, node_id, k=5, threshold=0.7, 
                              by_embedding=True, relation_types=None):
        """
        Find semantically similar nodes using optimized similarity search.
        
        Args:
            node_id: Node ID to find similar nodes for
            k: Number of similar nodes to return
            threshold: Similarity threshold
            by_embedding: Whether to use embedding similarity (vs. structure)
            relation_types: Optional filter for relation types to consider
            
        Returns:
            List of (node_id, similarity) tuples
        """
        node = await self.get_node(node_id)
        if not node:
            return []
            
        with self.lock:
            # Check cache first
            cache_key = f"{node_id}_{k}_{threshold}_{by_embedding}_{relation_types}"
            if cache_key in self.similarity_cache:
                # Check if cache is fresh
                cache_entry = self.similarity_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < 60:  # 1 minute cache
                    return cache_entry['results']
            
            # Try FAISS for embedding-based similarity
            if by_embedding and hasattr(node, 'embeddings') and node.embeddings is not None:
                # Initialize FAISS if needed
                if self.rebuild_needed:
                    await self._update_tensors()
                    
                if self.faiss_index is not None:
                    embeddings = np.array([node.embeddings], dtype=np.float32)
                    distances, indices = self.faiss_index.search(embeddings, k*2)  # Get extra for filtering
                    
                    # Process results
                    results = []
                    for i in range(len(indices[0])):
                        idx = indices[0][i]
                        if idx != -1 and idx < self.node_count:
                            similar_id = self.index_to_node.get(idx)
                            if similar_id and similar_id != node_id:
                                similarity = 1.0 - distances[0][i] / 2.0  # Convert to similarity
                                if similarity >= threshold:
                                    results.append((similar_id, similarity))
                    
                    # Cache results
                    self.similarity_cache[cache_key] = {
                        'results': results[:k],
                        'timestamp': time.time()
                    }
                    
                    return results[:k]
            
            # Fallback to structure-based similarity
            return await self._find_similar_by_structure(node, k, threshold, relation_types)
            
    async def get_node_relations(self, node_id, direction="both", relation_type=None):
        """Get relations involving a node with efficient indexing."""
        with self.lock:
            results = []
            
            # Get outgoing relations
            if direction in ["outgoing", "both"]:
                for relation_id in self.relation_index['source'].get(node_id, []):
                    relation = self.relations.get(relation_id)
                    if relation and (relation_type is None or relation.relation_type == relation_type):
                        results.append(relation)
                        
            # Get incoming relations
            if direction in ["incoming", "both"]:
                for relation_id in self.relation_index['target'].get(node_id, []):
                    relation = self.relations.get(relation_id)
                    if relation and (relation_type is None or relation.relation_type == relation_type):
                        results.append(relation)
                        
            return results
            
    async def traversal_search(self, start_node_id, max_depth=2, relation_filter=None):
        """
        Perform traversal search from a starting node.
        
        Args:
            start_node_id: Starting node ID
            max_depth: Maximum traversal depth
            relation_filter: Optional filter function for relations
            
        Returns:
            Dictionary mapping node IDs to paths and distances
        """
        cache_key = f"{start_node_id}_{max_depth}_{id(relation_filter) if relation_filter else 'none'}"
        if cache_key in self.traversal_cache:
            cache_entry = self.traversal_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 300:  # 5 minute cache
                return cache_entry['results']
        
        # Breadth-first traversal
        visited = {start_node_id: {'distance': 0, 'path': [start_node_id]}}
        queue = [(start_node_id, 0)]  # (node_id, distance)
        
        while queue:
            current_id, distance = queue.pop(0)
            
            if distance >= max_depth:
                continue
                
            # Get relations for this node
            relations = await self.get_node_relations(current_id)
            
            for relation in relations:
                # Apply filter if provided
                if relation_filter and not relation_filter(relation):
                    continue
                    
                # Determine neighbor node
                neighbor_id = relation.target_id if relation.source_id == current_id else relation.source_id
                
                # Skip if already visited with shorter path
                if neighbor_id in visited and visited[neighbor_id]['distance'] <= distance + 1:
                    continue
                    
                # Add to visited
                visited[neighbor_id] = {
                    'distance': distance + 1,
                    'path': visited[current_id]['path'] + [neighbor_id],
                    'relation_path': visited[current_id].get('relation_path', []) + [relation.id]
                }
                
                # Add to queue
                queue.append((neighbor_id, distance + 1))
        
        # Cache results
        self.traversal_cache[cache_key] = {
            'results': visited,
            'timestamp': time.time()
        }
        
        return visited
        
    async def extract_subgraph(self, node_ids, include_relations=True, max_relations=None):
        """
        Extract a subgraph containing specified nodes and their relations.
        
        Args:
            node_ids: List of node IDs to include
            include_relations: Whether to include relations between nodes
            max_relations: Maximum number of relations to include
            
        Returns:
            Dictionary with nodes and relations
        """
        # Sort node IDs for consistent cache key
        sorted_ids = sorted(node_ids)
        cache_key = f"{'-'.join(sorted_ids)}_{include_relations}_{max_relations}"
        
        if cache_key in self.subgraph_cache:
            cache_entry = self.subgraph_cache[cache_key]
            if time.time() - cache_entry['timestamp'] < 300:  # 5 minute cache
                return cache_entry['results']
        
        # Extract nodes
        subgraph_nodes = {}
        for node_id in node_ids:
            node = await self.get_node(node_id)
            if node:
                subgraph_nodes[node_id] = node
                
        # Extract relations if requested
        subgraph_relations = {}
        if include_relations:
            relation_count = 0
            
            # Find relations between these nodes
            node_set = set(node_ids)
            
            for node_id in node_ids:
                relations = await self.get_node_relations(node_id)
                
                for relation in relations:
                    # Only include relations where both endpoints are in the subgraph
                    if relation.source_id in node_set and relation.target_id in node_set:
                        if relation.id not in subgraph_relations:
                            subgraph_relations[relation.id] = relation
                            relation_count += 1
                            
                            if max_relations and relation_count >= max_relations:
                                break
                
                if max_relations and relation_count >= max_relations:
                    break
                    
        # Prepare result
        result = {
            'nodes': subgraph_nodes,
            'relations': subgraph_relations,
            'node_count': len(subgraph_nodes),
            'relation_count': len(subgraph_relations)
        }
        
        # Cache result
        self.subgraph_cache[cache_key] = {
            'results': result,
            'timestamp': time.time()
        }
        
        return result
        
    async def compute_graph_attention(self, focus_node_ids=None, attention_weights=None):
        """
        Compute attention weights across the graph.
        
        Args:
            focus_node_ids: Optional list of nodes to focus attention on
            attention_weights: Optional predefined attention weights
            
        Returns:
            Attention-weighted embeddings
        """
        with self.lock:
            # Build tensors if needed
            if self.rebuild_needed:
                await self._update_tensors()
                
            # Get adjacency matrix and node embeddings
            if not hasattr(self, 'global_adjacency') or self.global_adjacency is None:
                self._build_global_adjacency()
                
            if self.node_embeddings is None:
                return None
                
            # Apply focus if specified
            if focus_node_ids:
                # Create focus mask
                focus_mask = torch.zeros(self.node_count, device=self.device)
                
                for node_id in focus_node_ids:
                    if node_id in self.node_index:
                        idx = self.node_index[node_id]
                        focus_mask[idx] = 1.0
                        
                # Apply focus to adjacency matrix
                focused_adjacency = self.global_adjacency * focus_mask.unsqueeze(0)
            else:
                focused_adjacency = self.global_adjacency
                
            # Compute attention-weighted representations
            attended_embeddings, attention = self.tensor_ops.attention_weighted_graph(
                focused_adjacency, 
                self.node_embeddings,
                attention_weights
            )
            
            return {
                'embeddings': attended_embeddings,
                'attention': attention
            }
            
    def _build_global_adjacency(self):
        """Build global adjacency matrix from relations."""
        # Initialize adjacency matrix
        adjacency = torch.zeros(self.node_count, self.node_count, device=self.device)
        
        # Fill with relations
        for relation in self.relations.values():
            if relation.source_id in self.node_index and relation.target_id in self.node_index:
                source_idx = self.node_index[relation.source_id]
                target_idx = self.node_index[relation.target_id]
                
                # Weight by relation weight and attention weight
                edge_weight = relation.weight * relation.attention_weight
                
                # Update adjacency
                adjacency[source_idx, target_idx] = edge_weight
                
                # Add reverse edge if bidirectional
                if relation.bidirectional:
                    adjacency[target_idx, source_idx] = edge_weight
                    
        self.global_adjacency = adjacency
    
    async def _update_tensors(self):
        """Update tensor representations for efficient operations."""
        with self.lock:
            start_time = time.time()
            
            # Build node embeddings tensor
            embeddings_list = []
            invalid_indices = []
            
            for i in range(self.node_count):
                node_id = self.index_to_node.get(i)
                if node_id and node_id in self.nodes:
                    node = self.nodes[node_id]
                    if hasattr(node, 'embeddings') and node.embeddings is not None:
                        # Convert to correct format
                        if isinstance(node.embeddings, np.ndarray):
                            embeddings_list.append(torch.tensor(node.embeddings, dtype=torch.float32))
                        else:
                            embeddings_list.append(node.embeddings)
                    else:
                        # Use zero embedding for nodes without embeddings
                        embeddings_list.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
                        invalid_indices.append(i)
                else:
                    # Use zero embedding for missing nodes
                    embeddings_list.append(torch.zeros(self.embedding_dim, dtype=torch.float32))
                    invalid_indices.append(i)
            
            # Stack embeddings
            if embeddings_list:
                self.node_embeddings = torch.stack(embeddings_list).to(self.device)
                
                # Update FAISS index
                await self._update_faiss_index(invalid_indices)
                
            # Build adjacency tensors by relation type
            self.adjacency_tensors = {}
            relation_types = set(self.relation_index['type'].keys())
            
            for rel_type in relation_types:
                # Initialize adjacency for this type
                adjacency = torch.zeros(self.node_count, self.node_count, device=self.device)
                
                # Fill with relations of this type
                for relation_id in self.relation_index['type'][rel_type]:
                    relation = self.relations.get(relation_id)
                    if relation and relation.source_id in self.node_index and relation.target_id in self.node_index:
                        source_idx = self.node_index[relation.source_id]
                        target_idx = self.node_index[relation.target_id]
                        
                        # Weight by relation weight
                        adjacency[source_idx, target_idx] = relation.weight
                        
                        # Add reverse edge if bidirectional
                        if relation.bidirectional:
                            adjacency[target_idx, source_idx] = relation.weight
                
                self.adjacency_tensors[rel_type] = adjacency
                
            # Build global adjacency
            self._build_global_adjacency()
                
            self.rebuild_needed = False
            self.last_rebuild_time = time.time()
            
            self.logger.debug(f"Tensor update completed in {time.time() - start_time:.3f} seconds")
    
    async def _update_faiss_index(self, invalid_indices=[]):
        """Update FAISS index for fast similarity search."""
        if len(self.nodes) < 10:
            return  # Not enough nodes for indexing
            
        try:
            # Convert embeddings to numpy for FAISS
            embeddings_np = self.node_embeddings.cpu().numpy()
            
            # Create index if it doesn't exist
            if self.faiss_index is None:
                # Create appropriate index based on dimensionality
                if self.embedding_dim <= 64:
                    # For low dimensions, exact L2 is fast enough
                    self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
                elif self.embedding_dim <= 256:
                    # For medium dimensions, IVF with 100 centroids
                    self.faiss_index = faiss.IndexIVFFlat(
                        faiss.IndexFlatL2(self.embedding_dim), 
                        self.embedding_dim, 
                        min(100, self.node_count // 10 + 1),
                        faiss.METRIC_L2
                    )
                    if self.node_count >= 1000:
                        self.faiss_index.train(embeddings_np)
                else:
                    # For high dimensions, use HNSW for better speed/accuracy tradeoff
                    self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                
                # Add all embeddings
                valid_mask = np.ones(self.node_count, dtype=bool)
                for idx in invalid_indices:
                    valid_mask[idx] = False
                    
                if np.any(valid_mask):
                    self.faiss_index.add(embeddings_np[valid_mask])
            else:
                # Update only modified embeddings
                for idx, embedding in self.embedding_buffer:
                    if idx not in invalid_indices:
                        # Remove and re-add for update
                        if isinstance(self.faiss_index, faiss.IndexHNSWFlat):
                            # HNSW doesn't support remove, so we can't update efficiently
                            # Instead, rebuild the whole index periodically
                            if len(self.embedding_buffer) > self.node_count // 10:
                                # Remove all batched updates as we're rebuilding
                                valid_mask = np.ones(self.node_count, dtype=bool)
                                for inv_idx in invalid_indices:
                                    valid_mask[inv_idx] = False
                                
                                if np.any(valid_mask):
                                    self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                                    self.faiss_index.add(embeddings_np[valid_mask])
                        else:
                            # For flat indices, we can efficiently update
                            self.faiss_index.add(np.array([embedding], dtype=np.float32))
            
            # Clear buffer after processing
            self.embedding_buffer = []
            
        except Exception as e:
            self.logger.error(f"Error updating FAISS index: {str(e)}")
            
    async def _find_similar_by_structure(self, node, k=5, threshold=0.7, relation_types=None):
        """Find similar nodes based on graph structure."""
        # Get connected nodes
        node_relations = await self.get_node_relations(node.id)
        
        # Filter by relation type if specified
        if relation_types:
            node_relations = [r for r in node_relations if r.relation_type in relation_types]
            
        # Get neighbor nodes
        neighbors = set()
        for relation in node_relations:
            if relation.source_id == node.id:
                neighbors.add(relation.target_id)
            else:
                neighbors.add(relation.source_id)
                
        # Find nodes with similar connections
        similarity_scores = {}
        
        # For each node in the network
        for candidate_id, candidate in self.nodes.items():
            if candidate_id == node.id:
                continue
                
            # Get candidate's relations
            candidate_relations = await self.get_node_relations(candidate_id)
            
            # Filter by relation type if specified
            if relation_types:
                candidate_relations = [r for r in candidate_relations if r.relation_type in relation_types]
                
            # Get candidate's neighbors
            candidate_neighbors = set()
            for relation in candidate_relations:
                if relation.source_id == candidate_id:
                    candidate_neighbors.add(relation.target_id)
                else:
                    candidate_neighbors.add(relation.source_id)
                    
            # Calculate Jaccard similarity
            intersection = len(neighbors.intersection(candidate_neighbors))
            union = len(neighbors.union(candidate_neighbors))
            
            if union > 0:
                similarity = intersection / union
                
                if similarity >= threshold:
                    similarity_scores[candidate_id] = similarity
        
        # Sort by similarity
        sorted_results = sorted(similarity_scores.items(), key=lambda x: -x[1])
        
        return sorted_results[:k]


class GraphAggregationFunctions:
    """
    Optimized graph aggregation functions for semantic processing.
    Provides efficient implementations of common graph operations.
    """
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_ops = AdvancedTensorOperations(device=self.device)
        
    def mean_aggregation(self, node_features, adjacency_matrix, normalize=True):
        """
        Compute mean aggregation of node features across the graph.
        
        Args:
            node_features: Node feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            normalize: Whether to normalize by degree
            
        Returns:
            Aggregated features [N, D]
        """
        # Ensure tensors are on correct device
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).to(self.device).float()
        elif node_features.device != self.device:
            node_features = node_features.to(self.device)
            
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()
        elif adjacency_matrix.device != self.device:
            adjacency_matrix = adjacency_matrix.to(self.device)
        
        # Compute degree matrix for normalization
        if normalize:
            # Sum across rows to get degrees
            degrees = torch.sum(adjacency_matrix, dim=1)
            # Avoid division by zero
            degrees = torch.clamp(degrees, min=1e-8)
            # Create degree matrix inverse
            degree_inv = torch.diag(1.0 / degrees)
            # Normalized adjacency
            norm_adjacency = torch.matmul(degree_inv, adjacency_matrix)
        else:
            norm_adjacency = adjacency_matrix
            
        # Perform aggregation
        aggregated = torch.matmul(norm_adjacency, node_features)
        
        return aggregated
        
    def max_aggregation(self, node_features, adjacency_matrix):
        """
        Compute max aggregation of node features across the graph.
        
        Args:
            node_features: Node feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            
        Returns:
            Aggregated features [N, D]
        """
        # Ensure tensors are on correct device
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).to(self.device).float()
        elif node_features.device != self.device:
            node_features = node_features.to(self.device)
            
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()
        elif adjacency_matrix.device != self.device:
            adjacency_matrix = adjacency_matrix.to(self.device)
            
        # Get feature dimensionality
        N, D = node_features.shape
        
        # Prepare output tensor
        aggregated = torch.zeros_like(node_features)
        
        # For each node, find max feature values across neighbors
        for i in range(N):
            # Get neighbor indices (where adjacency > 0)
            neighbors = torch.nonzero(adjacency_matrix[i]).squeeze(-1)
            
            if neighbors.numel() > 0:
                # Get neighbor features
                neighbor_features = node_features[neighbors]
                
                # Get max value for each feature dimension
                if neighbor_features.numel() > 0:
                    aggregated[i] = torch.max(neighbor_features, dim=0)[0]
        
        return aggregated
        
    def attention_aggregation(self, query_features, key_features, adjacency_matrix, value_features=None):
        """
        Compute attention-weighted aggregation of node features.
        
        Args:
            query_features: Query feature matrix [N, D]
            key_features: Key feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            value_features: Optional different value features [N, D']
            
        Returns:
            Aggregated features [N, D']
        """
        # Ensure tensors are on correct device
        if isinstance(query_features, np.ndarray):
            query_features = torch.from_numpy(query_features).to(self.device).float()
        elif query_features.device != self.device:
            query_features = query_features.to(self.device)
            
        if isinstance(key_features, np.ndarray):
            key_features = torch.from_numpy(key_features).to(self.device).float()
        elif key_features.device != self.device:
            key_features = key_features.to(self.device)
            
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()
        elif adjacency_matrix.device != self.device:
            adjacency_matrix = adjacency_matrix.to(self.device)
            
        # Use key features as value features if not provided
        if value_features is None:
            value_features = key_features
        elif isinstance(value_features, np.ndarray):
            value_features = torch.from_numpy(value_features).to(self.device).float()
        elif value_features.device != self.device:
            value_features = value_features.to(self.device)
            
        # Compute attention scores
        # (normalized dot product attention)
        attention_logits = torch.matmul(
            F.normalize(query_features, p=2, dim=1),
            F.normalize(key_features, p=2, dim=1).transpose(0, 1)
        )
        
        # Apply adjacency mask (attention only flows along edges)
        attention_mask = (adjacency_matrix == 0)
        attention_logits = attention_logits.masked_fill(attention_mask, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_logits, dim=-1)
        
        # Apply attention to aggregate features
        attended_features = torch.matmul(attention_weights, value_features)
        
        return attended_features, attention_weights
        
    def multi_hop_aggregation(self, node_features, adjacency_matrix, hops=2, alpha=0.5):
        """
        Compute multi-hop aggregation with exponential decay.
        
        Args:
            node_features: Node feature matrix [N, D]
            adjacency_matrix: Adjacency matrix [N, N]
            hops: Number of hops to aggregate
            alpha: Decay factor for distant hops
            
        Returns:
            Aggregated features [N, D]
        """
        # Ensure tensors are on correct device
        if isinstance(node_features, np.ndarray):
            node_features = torch.from_numpy(node_features).to(self.device).float()
        elif node_features.device != self.device:
            node_features = node_features.to(self.device)
            
        if isinstance(adjacency_matrix, np.ndarray):
            adjacency_matrix = torch.from_numpy(adjacency_matrix).to(self.device).float()
        elif adjacency_matrix.device != self.device:
            adjacency_matrix = adjacency_matrix.to(self.device)
            
        # Normalize adjacency matrix by row
        row_sum = torch.sum(adjacency_matrix, dim=1, keepdim=True)
        row_sum = torch.clamp(row_sum, min=1e-8)  # Avoid division by zero
        norm_adjacency = adjacency_matrix / row_sum
        
        # Initial features are the node's own features
        aggregated = node_features.clone()
        current_hop_features = node_features.clone()
        current_adjacency = norm_adjacency.clone()
        
        # Aggregate across multiple hops
        for hop in range(1, hops + 1):
            # Propagate features
            hop_features = torch.matmul(current_adjacency, current_hop_features)
            
            # Add with decay
            decay = alpha ** hop
            aggregated = aggregated + decay * hop_features
            
            # Update for next hop
            current_hop_features = hop_features
            current_adjacency = torch.matmul(current_adjacency, norm_adjacency)
            
        # Normalize final result
        total_weight = 1.0 + sum(alpha ** h for h in range(1, hops + 1))
        aggregated = aggregated / total_weight
        
        return aggregated


# Example usage
async def test_optimized_tensor_network():
    """Simple test for the optimized tensor network."""
    # Create tensor network
    network = OptimizedSemanticTensorNetwork(embedding_dim=128)
    
    # Create mock node class
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
    
    # Create mock relation class
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
    
    # Add nodes
    for i in range(10):
        # Create random embeddings
        embeddings = np.random.randn(128).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings)
        
        node = MockNode(f"node_{i}", embeddings)
        await network.add_node(node)
    
    # Add relations
    for i in range(5):
        source_id = f"node_{i}"
        target_id = f"node_{i+5}"
        
        relation = MockRelation(f"rel_{i}", source_id, target_id)
        await network.add_relation(relation)
    
    # Test similarity search
    similar = await network.get_similar_nodes("node_0", k=3)
    print(f"Similar nodes to node_0: {similar}")
    
    # Test traversal
    traversal = await network.traversal_search("node_0", max_depth=2)
    print(f"Traversal from node_0: {list(traversal.keys())}")
    
    # Test attention
    attention = await network.compute_graph_attention(focus_node_ids=["node_0"])
    print(f"Attention shape: {attention['attention'].shape}")
    
    # Test graph aggregation
    agg = GraphAggregationFunctions()
    adj_matrix = network.global_adjacency.cpu().numpy()
    node_features = network.node_embeddings.cpu().numpy()
    
    mean_agg = agg.mean_aggregation(node_features, adj_matrix)
    print(f"Mean aggregation shape: {mean_agg.shape}")
    
    return "Tests completed successfully"

if __name__ == "__main__":
    # Run test
    import asyncio
    asyncio.run(test_optimized_tensor_network())