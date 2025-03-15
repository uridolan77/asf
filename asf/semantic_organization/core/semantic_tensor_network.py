import numpy as np
import torch
import torch.nn.functional as F
import threading
import logging
import faiss
import time
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

from asf.semantic_organization.core.semantic_node import SemanticNode
from asf.semantic_organization.core.semantic_relation import SemanticRelation

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
