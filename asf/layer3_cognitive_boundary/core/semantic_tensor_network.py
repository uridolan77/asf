import numpy as np
import torch
import torch.nn.functional as F
import threading
import logging
import faiss
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from asf.layer3_cognitive_boundary.core.semantic_node import SemanticNode
from asf.layer3_cognitive_boundary.core.semantic_relation import SemanticRelation

class SemanticTensorNetwork:
    """
    Core semantic network implementation with optimized tensor-based operations.
    Uses multi-channel sparse tensors for efficient graph algorithms and
    implements predictive mechanisms for anticipating network evolution.
    """
    def __init__(self, embedding_dim=1024, device=None):
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
        
        # Tensor-based representations
        self.embedding_dim = embedding_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-channel adjacency tensors (by relation type)
        self.adjacency_tensors = {}
        self.node_embeddings = None
        self.rebuild_needed = True
        
        # FAISS index for similarity search
        self.faiss_index = None
        self.embedding_buffer = []
        self.node_count = 0
        self.faiss_initialized = False
        
        # Performance optimization
        self.batched_operations = True
        self.batch_size = 64
        
        # Predictive components (Seth's Data Paradox integration)
        self.predicted_connections = {}  # (node1_id, node2_id) -> predicted relation
        self.connection_prediction_errors = defaultdict(list)  # Relation type -> errors
        self.connection_precision = {}  # Relation type -> precision
        self.predicted_network_changes = []  # List of anticipated changes
        
        # Locks for thread safety
        self.logger = logging.getLogger("ASF.Layer3.SemanticNetwork")
        self.lock = threading.RLock()

    async def add_node(self, node, update_tensors=True):
        """Add a node to the semantic network with incremental tensor update."""
        with self.lock:
            if node.id in self.nodes:
                # Update existing node
                existing_node = self.nodes[node.id]
                existing_node.label = node.label
                existing_node.properties.update(node.properties)
                existing_node.confidence = max(existing_node.confidence, node.confidence)
                
                # Only update embeddings if significantly different
                if isinstance(node.embeddings, np.ndarray) and np.linalg.norm(existing_node.embeddings - node.embeddings) > 0.1:
                    existing_node.embeddings = node.embeddings
                    existing_node.tensor_representation = torch.tensor(
                        node.embeddings, dtype=torch.float32
                    ).to(self.device)
                    
                    # Mark for incremental FAISS update
                    if node.id in self.node_index:
                        self.embedding_buffer.append((self.node_index[node.id], node.embeddings))
                    
                    existing_node.temporal_metadata.update_modification()
                    
                return node.id
            
            # Add new node
            self.nodes[node.id] = node
            
            # Move tensor to correct device
            if node.tensor_representation is not None:
                node.tensor_representation = node.tensor_representation.to(self.device)
                
            self.node_index[node.id] = self.node_count
            self.index_to_node[self.node_count] = node.id
            
            # Mark for incremental FAISS update
            if hasattr(node, 'embeddings') and node.embeddings is not None:
                self.embedding_buffer.append((self.node_count, node.embeddings))
                
            self.node_count += 1
            
            # Update hierarchical relationships
            if hasattr(node, 'parent_ids'):
                for parent_id in node.parent_ids:
                    if parent_id in self.nodes:
                        if node.id not in self.nodes[parent_id].child_ids:
                            self.nodes[parent_id].child_ids.append(node.id)
                            
            if hasattr(node, 'child_ids'):
                for child_id in node.child_ids:
                    if child_id in self.nodes:
                        if node.id not in self.nodes[child_id].parent_ids:
                            self.nodes[child_id].parent_ids.append(node.id)
            
            # Check if we should update tensors
            if update_tensors and self.batched_operations:
                if len(self.embedding_buffer) >= self.batch_size:
                    await self._update_tensors()
                    
            # Predictive component: anticipate connections for new node
            await self._predict_connections_for_node(node.id)
                
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
                
                # Evaluate prediction accuracy if applicable
                prediction_key = (relation.source_id, relation.target_id)
                if prediction_key in self.predicted_connections:
                    pred = self.predicted_connections[prediction_key]
                    if pred['relation_type'] == relation.relation_type:
                        # Calculate prediction error
                        weight_error = abs(pred['weight'] - relation.weight)
                        
                        # Update prediction errors
                        self.connection_prediction_errors[relation.relation_type].append(weight_error)
                        
                        # Limit history size
                        if len(self.connection_prediction_errors[relation.relation_type]) > 20:
                            self.connection_prediction_errors[relation.relation_type] = \
                                self.connection_prediction_errors[relation.relation_type][-20:]
                        
                        # Update precision
                        if len(self.connection_prediction_errors[relation.relation_type]) > 1:
                            variance = np.var(self.connection_prediction_errors[relation.relation_type])
                            precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                            self.connection_precision[relation.relation_type] = precision
                
                return relation.id
            
            # Add new relation
            self.relations[relation.id] = relation
            
            # Update indices
            self.relation_index['source'][relation.source_id].append(relation.id)
            self.relation_index['target'][relation.target_id].append(relation.id)
            self.relation_index['type'][relation.relation_type].append(relation.id)
            
            # Update adjacency tensors if they exist
            self.rebuild_needed = True
            
            # For hierarchical relations, update hierarchical structure
            if relation.relation_type in ["is_a", "part_of", "subclass_of"]:
                source_node = self.nodes[relation.source_id]
                target_node = self.nodes[relation.target_id]
                
                # Add hierarchical relationship
                if hasattr(source_node, 'parent_ids') and target_node.id not in source_node.parent_ids:
                    source_node.parent_ids.append(target_node.id)
                    
                if hasattr(target_node, 'child_ids') and source_node.id not in target_node.child_ids:
                    target_node.child_ids.append(source_node.id)
                
                # Update abstraction levels
                if hasattr(source_node, 'abstraction_level') and hasattr(target_node, 'abstraction_level'):
                    source_node.abstraction_level = max(
                        source_node.abstraction_level,
                        target_node.abstraction_level + 1
                    )
            
            # Evaluate prediction accuracy if applicable
            prediction_key = (relation.source_id, relation.target_id)
            if prediction_key in self.predicted_connections:
                pred = self.predicted_connections[prediction_key]
                if pred['relation_type'] == relation.relation_type:
                    # Calculate prediction error
                    weight_error = abs(pred['weight'] - relation.weight)
                    
                    # Update prediction errors
                    self.connection_prediction_errors[relation.relation_type].append(weight_error)
                    
                    # Limit history size
                    if len(self.connection_prediction_errors[relation.relation_type]) > 20:
                        self.connection_prediction_errors[relation.relation_type] = \
                            self.connection_prediction_errors[relation.relation_type][-20:]
                    
                    # Update precision
                    if len(self.connection_prediction_errors[relation.relation_type]) > 1:
                        variance = np.var(self.connection_prediction_errors[relation.relation_type])
                        precision = 1.0 / (variance + 1e-6)  # Avoid division by zero
                        self.connection_precision[relation.relation_type] = precision
            
            return relation.id

    async def get_node(self, node_id):
        """Get a node by ID."""
        with self.lock:
            node = self.nodes.get(node_id)
            if node:
                node.temporal_metadata.update_access()
                return node
            return None

    async def get_relation(self, relation_id):
        """Get a relation by ID."""
        with self.lock:
            relation = self.relations.get(relation_id)
            if relation:
                relation.temporal_metadata.update_access()
                return relation
            return None

    async def get_similar_nodes(self, node_id, k=5, threshold=0.7, by_embedding=True):
        """Find semantically similar nodes using FAISS or tensor operations."""
        node = await self.get_node(node_id)
        if not node:
            return []
            
        with self.lock:
            if by_embedding and (not self.faiss_initialized or self.rebuild_needed):
                await self._initialize_faiss_index()
                
            if by_embedding and self.faiss_index is not None:
                # Search using FAISS (faster for large networks)
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
                
                return sorted(results, key=lambda x: -x[1])[:k]
            else:
                # Fallback to direct tensor similarity
                return await self._compute_tensor_similarities(node, k, threshold)

    async def get_node_relations(self, node_id, direction="both", relation_type=None):
        """Get relations involving a node."""
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

    async def predict_connections(self, node_id, max_predictions=5):
        """
        Predict potential new connections for a node.
        Implements Seth's "controlled hallucination" principle for network structure.
        
        Args:
            node_id: Node to predict connections for
            max_predictions: Maximum number of predictions to return
            
        Returns:
            List of predicted connections
        """
        node = await self.get_node(node_id)
        if not node:
            return []
            
        # Get similar nodes
        similar_nodes = await self.get_similar_nodes(node_id, k=10, threshold=0.6)
        
        # Track predictions
        predictions = []
        
        # Look at relations of similar nodes
        for similar_id, similarity in similar_nodes:
            # Get relations of similar node
            relations = await self.get_node_relations(similar_id)
            
            for relation in relations:
                # Determine if this relation pattern could apply to our node
                if relation.source_id == similar_id:
                    # Check if target is already connected to our node
                    target_id = relation.target_id
                    existing_relation = False
                    
                    for rel_id in self.relation_index['source'].get(node_id, []):
                        if self.relations[rel_id].target_id == target_id:
                            existing_relation = True
                            break
                            
                    if not existing_relation:
                        # This is a potential new connection
                        # Get precision for this relation type
                        precision = self.connection_precision.get(relation.relation_type, 1.0)
                        
                        # Calculate predicted weight
                        predicted_weight = relation.weight * similarity * precision
                        
                        predictions.append({
                            'source_id': node_id,
                            'target_id': target_id,
                            'relation_type': relation.relation_type,
                            'weight': predicted_weight,
                            'confidence': similarity * precision,
                            'based_on': similar_id
                        })
                        
                elif relation.target_id == similar_id:
                    # Check if source is already connected to our node
                    source_id = relation.source_id
                    existing_relation = False
                    
                    for rel_id in self.relation_index['target'].get(node_id, []):
                        if self.relations[rel_id].source_id == source_id:
                            existing_relation = True
                            break
                            
                    if not existing_relation:
                        # This is a potential new connection
                        # Get precision for this relation type
                        precision = self.connection_precision.get(relation.relation_type, 1.0)
                        
                        # Calculate predicted weight
                        predicted_weight = relation.weight * similarity * precision
                        
                        predictions.append({
                            'source_id': source_id,
                            'target_id': node_id,
                            'relation_type': relation.relation_type,
                            'weight': predicted_weight,
                            'confidence': similarity * precision,
                            'based_on': similar_id
                        })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Store predictions
        for pred in predictions[:max_predictions]:
            prediction_key = (pred['source_id'], pred['target_id'])
            self.predicted_connections[prediction_key] = pred
            
        return predictions[:max_predictions]

    async def predict_network_evolution(self, time_horizon=10, confidence_threshold=0.7):
        """
        Predict how the network might evolve over time.
        Implements Seth's predictive processing principle for anticipating future states.
        
        Args:
            time_horizon: Time steps to predict ahead
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            List of predicted changes
        """
        predicted_changes = []
        
        # Predict new relations
        for node_id in self.nodes:
            # Predict connections for each node
            node_predictions = await self.predict_connections(node_id)
            
            # Filter by confidence
            high_confidence_predictions = [
                p for p in node_predictions if p['confidence'] >= confidence_threshold
            ]
            
            # Add to predicted changes
            for pred in high_confidence_predictions:
                predicted_changes.append({
                    'type': 'new_relation',
                    'source_id': pred['source_id'],
                    'target_id': pred['target_id'],
                    'relation_type': pred['relation_type'],
                    'weight': pred['weight'],
                    'confidence': pred['confidence'],
                    'time_step': np.random.randint(1, time_horizon + 1)  # Random time step
                })
        
        # Predict weight changes
        for relation_id, relation in self.relations.items():
            # Skip if relation has very low weight
            if relation.weight < 0.1:
                continue
                
            # Predict temporal decay
            decay_factor = 0.95  # 5% decay per time step
            final_weight = relation.weight * (decay_factor ** time_horizon)
            
            # Only include significant changes
            if abs(relation.weight - final_weight) > 0.1:
                predicted_changes.append({
                    'type': 'weight_change',
                    'relation_id': relation_id,
                    'initial_weight': relation.weight,
                    'final_weight': final_weight,
                    'confidence': 0.8,
                    'time_step': time_horizon
                })
        
        # Sort by time step
        predicted_changes.sort(key=lambda x: x['time_step'])
        
        # Store predictions
        self.predicted_network_changes = predicted_changes
        
        return predicted_changes

    # The remaining private methods (_build_tensors, _initialize_faiss_index, etc.)
    # would be included but are omitted for brevity

    async def _predict_connections_for_node(self, node_id):
        """Predict potential connections for a newly added node."""
        # Get node
        node = self.nodes.get(node_id)
        if not node:
            return
            
        # Find similar nodes
        similar_nodes = await self.get_similar_nodes(node_id, k=5)
        
        # If no similar nodes found, return
        if not similar_nodes:
            return
            
        # Look at relations of similar nodes
        for similar_id, similarity in similar_nodes:
            if similarity < 0.6:  # Threshold for similarity
                continue
                
            relations = await self.get_node_relations(similar_id)
            
            for relation in relations:
                # Check if this relation type could apply to our node
                target_id = relation.target_id if relation.source_id == similar_id else relation.source_id
                
                # Skip if target is the node itself
                if target_id == node_id:
                    continue
                    
                # Skip if relation already exists
                existing_relation = False
                
                for rel_id in self.relation_index['source'].get(node_id, []):
                    if self.relations[rel_id].target_id == target_id:
                        existing_relation = True
                        break
                        
                for rel_id in self.relation_index['target'].get(node_id, []):
                    if self.relations[rel_id].source_id == target_id:
                        existing_relation = True
                        break
                        
                if not existing_relation:
                    # This is a potential connection
                    precision = self.connection_precision.get(relation.relation_type, 1.0)
                    
                    # Determine direction
                    if relation.source_id == similar_id:
                        # Similar node is source, so our node should be source too
                        source_id = node_id
                        target_id = relation.target_id
                    else:
                        # Similar node is target, so our node should be target too
                        source_id = relation.source_id
                        target_id = node_id
                        
                    # Store prediction
                    prediction_key = (source_id, target_id)
                    self.predicted_connections[prediction_key] = {
                        'source_id': source_id,
                        'target_id': target_id,
                        'relation_type': relation.relation_type,
                        'weight': relation.weight * similarity,
                        'confidence': similarity * precision,
                        'based_on': similar_id
                    }
async def _build_tensors(self, full_rebuild=False):
    """
    Build tensor representations of the semantic network.
    Creates adjacency tensors and node embeddings for efficient operations.
    
    Args:
        full_rebuild (bool): Whether to perform a full rebuild or incremental update
        
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    self.logger.info(f"Building tensor representations (full_rebuild={full_rebuild})")
    
    try:
        # Determine if we need to rebuild
        if not self.rebuild_needed and not full_rebuild:
            return True
            
        # Get node count for dimensions
        node_count = self.node_count
        if node_count == 0:
            self.logger.warning("No nodes available to build tensors")
            return False
            
        # === 1. Build node embeddings tensor ===
        if full_rebuild or self.node_embeddings is None:
            # Initialize with zeros
            embeddings_list = []
            valid_indices = []
            
            # Collect embeddings from nodes
            for node_id, idx in self.node_index.items():
                node = self.nodes.get(node_id)
                if node is None:
                    continue
                    
                # Skip nodes without embeddings
                if not hasattr(node, 'embeddings') or node.embeddings is None:
                    continue
                    
                # Convert numpy array to tensor if needed
                if isinstance(node.embeddings, np.ndarray):
                    if node.tensor_representation is None:
                        node.tensor_representation = torch.tensor(
                            node.embeddings, dtype=torch.float32
                        ).to(self.device)
                    embeddings_list.append(node.tensor_representation)
                elif isinstance(node.embeddings, torch.Tensor):
                    node.tensor_representation = node.embeddings.to(self.device)
                    embeddings_list.append(node.tensor_representation)
                else:
                    continue  # Skip invalid embeddings
                    
                valid_indices.append(idx)
                
            # Create padded embeddings tensor
            if embeddings_list:
                try:
                    # Ensure consistent dimensions
                    embedding_dim = embeddings_list[0].shape[0]
                    
                    # Create full tensor with padding for nodes without embeddings
                    self.node_embeddings = torch.zeros(
                        (node_count, embedding_dim), dtype=torch.float32
                    ).to(self.device)
                    
                    # Fill with actual embeddings where available
                    for i, idx in enumerate(valid_indices):
                        self.node_embeddings[idx] = embeddings_list[i]
                        
                    self.logger.debug(f"Built embeddings tensor with shape {self.node_embeddings.shape}")
                except Exception as e:
                    self.logger.error(f"Error building node embeddings tensor: {str(e)}", exc_info=True)
                    # Create a minimal valid tensor to avoid future errors
                    self.node_embeddings = torch.zeros(
                        (node_count, self.embedding_dim), dtype=torch.float32
                    ).to(self.device)
        
        # === 2. Build adjacency tensors by relation type ===
        # Get unique relation types
        relation_types = set()
        for rel in self.relations.values():
            relation_types.add(rel.relation_type)
            
        # Create sparse adjacency tensors for each relation type
        for rel_type in relation_types:
            # Skip if already built and not doing full rebuild
            if rel_type in self.adjacency_tensors and not full_rebuild:
                continue
                
            # Get relations of this type
            rel_ids = self.relation_index['type'].get(rel_type, [])
            if not rel_ids:
                continue
                
            # Build COO format sparse tensor
            indices = []
            values = []
            
            for rel_id in rel_ids:
                rel = self.relations.get(rel_id)
                if rel is None:
                    continue
                    
                # Get indices from node mapping
                source_idx = self.node_index.get(rel.source_id)
                target_idx = self.node_index.get(rel.target_id)
                
                if source_idx is None or target_idx is None:
                    continue
                    
                # Add edge with weight
                indices.append([source_idx, target_idx])
                values.append(rel.weight)
                
                # Add reverse edge if bidirectional
                if rel.bidirectional:
                    indices.append([target_idx, source_idx])
                    values.append(rel.weight)
            
            if not indices:
                # No valid relations of this type
                continue
                
            # Create sparse tensor
            try:
                indices_tensor = torch.tensor(indices, dtype=torch.long).t().to(self.device)
                values_tensor = torch.tensor(values, dtype=torch.float32).to(self.device)
                
                sparse_tensor = torch.sparse_coo_tensor(
                    indices_tensor, 
                    values_tensor,
                    (node_count, node_count),
                    device=self.device
                )
                
                # Store in adjacency tensors
                self.adjacency_tensors[rel_type] = sparse_tensor
                self.logger.debug(f"Built adjacency tensor for relation type '{rel_type}' with {len(indices)} edges")
            except Exception as e:
                self.logger.error(f"Error building adjacency tensor for relation type '{rel_type}': {str(e)}", exc_info=True)
        
        # === 3. Build combined adjacency tensor ===
        if relation_types:
            try:
                # Combine all relation types for a general adjacency tensor
                all_indices = []
                all_values = []
                
                for rel_id, rel in self.relations.items():
                    # Get indices from node mapping
                    source_idx = self.node_index.get(rel.source_id)
                    target_idx = self.node_index.get(rel.target_id)
                    
                    if source_idx is None or target_idx is None:
                        continue
                        
                    # Calculate effective weight
                    effective_weight = rel.compute_effective_weight()
                    
                    # Add edge with weight
                    all_indices.append([source_idx, target_idx])
                    all_values.append(effective_weight)
                    
                    # Add reverse edge if bidirectional
                    if rel.bidirectional:
                        all_indices.append([target_idx, source_idx])
                        all_values.append(effective_weight)
                
                if all_indices:
                    # Create combined sparse tensor
                    indices_tensor = torch.tensor(all_indices, dtype=torch.long).t().to(self.device)
                    values_tensor = torch.tensor(all_values, dtype=torch.float32).to(self.device)
                    
                    combined_tensor = torch.sparse_coo_tensor(
                        indices_tensor, 
                        values_tensor,
                        (node_count, node_count),
                        device=self.device
                    )
                    
                    # Store as special 'combined' type
                    self.adjacency_tensors['combined'] = combined_tensor
                    self.logger.debug(f"Built combined adjacency tensor with {len(all_indices)} edges")
            except Exception as e:
                self.logger.error(f"Error building combined adjacency tensor: {str(e)}", exc_info=True)
        
        # Mark as built
        self.rebuild_needed = False
        
        # Log performance information
        build_time = time.time() - start_time
        self.logger.info(f"Tensor building completed in {build_time:.3f}s")
        
        return True
        
    except Exception as e:
        self.logger.error(f"Error in _build_tensors: {str(e)}", exc_info=True)
        return False

async def _initialize_faiss_index(self):
    """
    Initialize FAISS index for efficient similarity search.
    Creates and populates the index with node embeddings.
    
    Returns:
        bool: True if successful, False otherwise
    """
    start_time = time.time()
    self.logger.info("Initializing FAISS index")
    
    try:
        # Skip if no embeddings
        if self.node_count == 0:
            self.logger.warning("No nodes available for FAISS index")
            return False
            
        # Make sure embeddings are up to date
        if self.rebuild_needed or self.node_embeddings is None:
            success = await self._build_tensors(full_rebuild=True)
            if not success:
                self.logger.error("Failed to build tensors for FAISS index")
                return False
        
        # Collect embeddings into a numpy array for FAISS
        embedding_matrix = np.zeros((self.node_count, self.embedding_dim), dtype=np.float32)
        valid_count = 0
        
        # First check if we have a valid node_embeddings tensor
        if self.node_embeddings is not None and self.node_embeddings.shape[0] > 0:
            # Convert to numpy for FAISS
            embedding_matrix = self.node_embeddings.cpu().numpy()
            valid_count = embedding_matrix.shape[0]
        else:
            # Manual collection from nodes
            for node_id, idx in self.node_index.items():
                node = self.nodes.get(node_id)
                if node is None or not hasattr(node, 'embeddings') or node.embeddings is None:
                    continue
                    
                # Handle both numpy arrays and torch tensors
                if isinstance(node.embeddings, np.ndarray):
                    # Ensure proper shape
                    if len(node.embeddings.shape) == 1 and node.embeddings.shape[0] == self.embedding_dim:
                        embedding_matrix[idx] = node.embeddings
                        valid_count += 1
                elif isinstance(node.embeddings, torch.Tensor):
                    # Convert tensor to numpy
                    emb_np = node.embeddings.cpu().detach().numpy()
                    if len(emb_np.shape) == 1 and emb_np.shape[0] == self.embedding_dim:
                        embedding_matrix[idx] = emb_np
                        valid_count += 1
        
        if valid_count == 0:
            self.logger.warning("No valid embeddings found for FAISS index")
            return False
            
        # Normalize embeddings (required for cosine similarity)
        # First replace NaN and Inf values
        embedding_matrix = np.nan_to_num(embedding_matrix)
        
        # Normalize
        norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        normalized_embeddings = embedding_matrix / norms
        
        # Choose appropriate index type
        # For smaller datasets, use flat index for exact search
        # For larger datasets, use IVF index for approximate search
        index_type = self.config.get('faiss_index_type', 'auto')
        use_gpu = torch.cuda.is_available() and self.config.get('faiss_use_gpu', True)
        
        if index_type == 'auto':
            if self.node_count < 10000:
                index_type = 'flat'
            else:
                index_type = 'ivf'
        
        # Create the index
        try:
            if index_type == 'flat':
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity with normalized vectors
            elif index_type == 'ivf':
                # For IVF, we need to determine the number of centroids
                # Rule of thumb: sqrt(n) for n data points
                n_centroids = min(int(np.sqrt(self.node_count)), 256)
                n_centroids = max(n_centroids, 8)  # At least 8 centroids
                
                # Create quantizer and IVF index
                quantizer = faiss.IndexFlatIP(self.embedding_dim)
                self.faiss_index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, n_centroids, faiss.METRIC_INNER_PRODUCT)
                
                # Train the index with the normalized embeddings
                self.faiss_index.train(normalized_embeddings)
            else:
                self.logger.warning(f"Unknown FAISS index type: {index_type}, falling back to flat index")
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                
            # Move index to GPU if available and requested
            if use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                    self.logger.info("FAISS index moved to GPU")
                except Exception as e:
                    self.logger.warning(f"Failed to move FAISS index to GPU: {str(e)}, using CPU index")
            
            # Add embeddings to the index
            self.faiss_index.add(normalized_embeddings)
            
            # Mark as initialized
            self.faiss_initialized = True
            
            # Clear embedding buffer
            self.embedding_buffer = []
            
            # Log performance information
            init_time = time.time() - start_time
            self.logger.info(f"FAISS index initialized with {valid_count} embeddings in {init_time:.3f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing FAISS index: {str(e)}", exc_info=True)
            return False
            
    except Exception as e:
        self.logger.error(f"Error in _initialize_faiss_index: {str(e)}", exc_info=True)
        return False

async def _compute_tensor_similarities(self, node, k=5, threshold=0.7):
    """
    Compute similarities between a node and all other nodes using tensor operations.
    Falls back to direct computation when FAISS is not available.
    
    Args:
        node: Query node
        k: Number of similar nodes to return
        threshold: Minimum similarity threshold
        
    Returns:
        List[Tuple[str, float]]: List of (node_id, similarity) pairs
    """
    if node is None:
        return []
        
    try:
        # Get node embedding
        node_embedding = None
        
        if hasattr(node, 'tensor_representation') and node.tensor_representation is not None:
            node_embedding = node.tensor_representation
        elif hasattr(node, 'embeddings') and node.embeddings is not None:
            if isinstance(node.embeddings, np.ndarray):
                node_embedding = torch.tensor(node.embeddings, dtype=torch.float32).to(self.device)
            elif isinstance(node.embeddings, torch.Tensor):
                node_embedding = node.embeddings.to(self.device)
                
        if node_embedding is None:
            self.logger.warning(f"Node {node.id} has no valid embedding for similarity computation")
            return []
            
        # Ensure embeddings are built
        if self.node_embeddings is None or self.rebuild_needed:
            await self._build_tensors()
            
        if self.node_embeddings is None or self.node_embeddings.shape[0] == 0:
            self.logger.warning("No valid embeddings available for similarity computation")
            return []
            
        # Normalize query embedding
        query_embedding = F.normalize(node_embedding, p=2, dim=0)
        
        # Approach 1: Tensor-based computation (faster for smaller networks)
        if self.node_count < 10000 or self.faiss_index is None:
            # Compute cosine similarities using matrix multiplication
            # Ensure node_embeddings are normalized
            normalized_embeddings = F.normalize(self.node_embeddings, p=2, dim=1)
            
            # Compute similarities using matrix multiplication
            similarities = torch.matmul(normalized_embeddings, query_embedding)
            
            # Convert to numpy for processing
            similarities_np = similarities.cpu().numpy()
            
            # Get top k indices (excluding the query node)
            query_idx = self.node_index.get(node.id, -1)
            
            # Create a mask for valid similarities
            valid_mask = similarities_np >= threshold
            
            # If query node is in the index, exclude it
            if query_idx >= 0 and query_idx < len(valid_mask):
                valid_mask[query_idx] = False
                
            # Get indices of valid similarities
            valid_indices = np.where(valid_mask)[0]
            
            # Sort by similarity (descending)
            sorted_indices = valid_indices[np.argsort(-similarities_np[valid_indices])]
            
            # Limit to top k
            top_indices = sorted_indices[:k]
            
            # Create result list
            results = []
            for idx in top_indices:
                if idx < len(similarities_np) and idx in self.index_to_node:
                    node_id = self.index_to_node[idx]
                    similarity = float(similarities_np[idx])
                    if similarity >= threshold and node_id != node.id:
                        results.append((node_id, similarity))
            
            return results
        else:
            # Approach 2: Use FAISS for larger networks
            # Normalize query embedding (again, for safety)
            query_np = query_embedding.cpu().numpy().astype(np.float32)
            query_np = query_np.reshape(1, -1)  # Reshape to (1, D) for FAISS
            
            # Ensure norms
            query_norm = np.linalg.norm(query_np)
            if query_norm > 0:
                query_np = query_np / query_norm
            
            # Search using FAISS
            distances, indices = self.faiss_index.search(query_np, k + 1)  # +1 to account for self
            
            # Process results
            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                if idx == -1 or idx >= self.node_count:
                    continue
                    
                similar_id = self.index_to_node.get(idx)
                if similar_id and similar_id != node.id:
                    # For inner product with normalized vectors, 
                    # similarity = 1 - distance/2
                    # Note: distances are negative (inner product), so we negate again
                    similarity = float(distances[0][i])
                    
                    if similarity >= threshold:
                        results.append((similar_id, similarity))
            
            return results
            
    except Exception as e:
        self.logger.error(f"Error in _compute_tensor_similarities: {str(e)}", exc_info=True)
        return []

async def _update_tensors(self):
    """
    Update tensor representations with buffered changes.
    More efficient than full rebuilds for incremental updates.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not self.embedding_buffer:
        return True
        
    self.logger.debug(f"Updating tensors with {len(self.embedding_buffer)} buffered embeddings")
    
    try:
        # Update node embeddings tensor if it exists
        if self.node_embeddings is not None:
            for node_idx, embedding in self.embedding_buffer:
                # Skip invalid indices
                if node_idx >= self.node_embeddings.shape[0]:
                    continue
                    
                # Convert numpy to tensor if needed
                if isinstance(embedding, np.ndarray):
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
                elif isinstance(embedding, torch.Tensor):
                    embedding_tensor = embedding.to(self.device)
                else:
                    continue
                    
                # Update the embeddings tensor
                if embedding_tensor.shape[0] == self.node_embeddings.shape[1]:
                    self.node_embeddings[node_idx] = embedding_tensor
        
        # Update FAISS index if initialized
        if self.faiss_initialized and self.faiss_index is not None:
            # For FAISS, we need to remove and re-add or do direct update
            # Depending on the index type
            faiss_updates = []
            
            for node_idx, embedding in self.embedding_buffer:
                # Skip invalid indices
                if node_idx >= self.node_count:
                    continue
                    
                # Prepare embedding for FAISS
                if isinstance(embedding, np.ndarray):
                    emb_np = embedding
                elif isinstance(embedding, torch.Tensor):
                    emb_np = embedding.cpu().detach().numpy()
                else:
                    continue
                    
                # Reshape if needed
                if len(emb_np.shape) == 1:
                    emb_np = emb_np.reshape(1, -1)
                    
                # Normalize for cosine similarity
                norm = np.linalg.norm(emb_np)
                if norm > 0:
                    emb_np = emb_np / norm
                    
                faiss_updates.append((node_idx, emb_np))
            
            # Check if index supports direct update
            if hasattr(self.faiss_index, 'remove_ids') and hasattr(self.faiss_index, 'add_with_ids'):
                # For indices that support removal and targeted addition
                for node_idx, emb_np in faiss_updates:
                    # Create ID array for this index
                    ids = np.array([node_idx], dtype=np.int64)
                    
                    # Remove existing vector
                    self.faiss_index.remove_ids(ids)
                    
                    # Add updated vector
                    self.faiss_index.add_with_ids(emb_np, ids)
            else:
                # For indices that don't support direct updates,
                # we may need to rebuild the index
                # This is less efficient but ensures correctness
                if len(faiss_updates) > 0:
                    self.logger.info(f"FAISS index type doesn't support direct updates, scheduling rebuild")
                    self.rebuild_needed = True
        
        # Clear buffer
        self.embedding_buffer = []
        
        return True
        
    except Exception as e:
        self.logger.error(f"Error updating tensors: {str(e)}", exc_info=True)
        return False