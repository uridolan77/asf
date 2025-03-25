"""
Performance optimization implementation for SemanticTensorNetwork.
Includes batched processing, incremental updates, and caching mechanisms.
"""

import time
import asyncio
import functools
from typing import Dict, List, Set, Tuple, Any, Optional, Union, Callable
from collections import OrderedDict, defaultdict
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import logging
from asf.layer3_cognitive_boundary.core.semantic_tensor_network import SemanticTensorNetwork

# ---------------------------------------------------------
# 1. BATCHED PROCESSING IMPLEMENTATION
# ---------------------------------------------------------

class BatchedTensorOperations:
    """
    Provides batched processing for computationally intensive tensor operations.
    Improves performance by processing data in optimized chunks.
    """
    
    def __init__(self, tensor_network, config=None):
        """
        Initialize batched processing with configuration.
        
        Args:
            tensor_network: Parent SemanticTensorNetwork instance
            config: Configuration dictionary
        """
        self.tensor_network = tensor_network
        self.config = config or {}
        self.device = tensor_network.device
        self.logger = logging.getLogger("ASF.Layer3.BatchedOperations")
        
        # Configure batch sizes for different operations
        self.similarity_batch_size = self.config.get('similarity_batch_size', 128)
        self.embedding_batch_size = self.config.get('embedding_batch_size', 64)
        self.relation_batch_size = self.config.get('relation_batch_size', 256)
        
        # Performance tracking
        self.operation_times = defaultdict(list)
        self.batch_counts = defaultdict(int)
        
        # Maximum history for performance tracking
        self.max_history = 100
        
        # Adaptive batch sizing
        self.adaptive_batch_sizing = self.config.get('adaptive_batch_sizing', True)
        self.min_batch_size = self.config.get('min_batch_size', 16)
        self.max_batch_size = self.config.get('max_batch_size', 512)
    
    async def batch_process_similarities(self, query_embeddings, target_embeddings=None, 
                                         k=5, threshold=0.7):
        """
        Perform batched similarity calculations between query embeddings and targets.
        
        Args:
            query_embeddings: Tensor or array of query embeddings [Q, D]
            target_embeddings: Optional tensor of target embeddings [T, D] (uses node_embeddings if None)
            k: Number of top results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of List of (idx, similarity) pairs for each query
        """
        start_time = time.time()
        
        # Ensure we have tensors
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32).to(self.device)
        
        if target_embeddings is None:
            # Use the network's node embeddings
            if self.tensor_network.node_embeddings is None:
                return [[]]
            target_embeddings = self.tensor_network.node_embeddings
        elif isinstance(target_embeddings, np.ndarray):
            target_embeddings = torch.tensor(target_embeddings, dtype=torch.float32).to(self.device)
        
        # Normalize embeddings (if not already normalized)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        
        # Get dimensions
        num_queries = query_embeddings.shape[0]
        
        # Calculate optimal batch size (memory optimization)
        if self.adaptive_batch_sizing:
            # Adjust batch size based on embedding dimensions and device memory
            embed_dim = query_embeddings.shape[1]
            if torch.cuda.is_available() and self.device.type == 'cuda':
                # For GPU: limit batch size based on available memory
                free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
                # Estimate memory needed per query and adjust batch size accordingly
                # Each similarity calculation needs memory proportional to embed_dim * num_targets
                memory_per_query = 4 * embed_dim * target_embeddings.shape[0]  # 4 bytes per float32
                adjusted_batch_size = max(self.min_batch_size, min(self.max_batch_size, 
                                                                   int(free_memory * 0.7 / memory_per_query)))
                batch_size = min(adjusted_batch_size, self.similarity_batch_size)
            else:
                # For CPU: use smaller batches to avoid memory pressure
                batch_size = min(self.similarity_batch_size, 64)
        else:
            batch_size = self.similarity_batch_size
        
        # Limit batch size to number of queries
        batch_size = min(batch_size, num_queries)
        
        # Process in batches
        all_results = []
        
        for i in range(0, num_queries, batch_size):
            # Get batch
            end_idx = min(i + batch_size, num_queries)
            query_batch = query_embeddings[i:end_idx]
            
            # Calculate similarities
            similarities = torch.mm(query_batch, target_embeddings.t())
            
            # Process each query in the batch
            for j in range(query_batch.shape[0]):
                query_similarities = similarities[j]
                
                # Find top-k indices
                values, indices = torch.topk(query_similarities, min(k, len(query_similarities)))
                
                # Convert to list of tuples, filtering by threshold
                results = []
                for idx, val in zip(indices.cpu().numpy(), values.cpu().numpy()):
                    if val >= threshold:
                        results.append((int(idx), float(val)))
                
                all_results.append(results)
        
        # Track performance
        elapsed = time.time() - start_time
        self.operation_times['batch_similarities'].append(elapsed)
        self.batch_counts['batch_similarities'] += 1
        
        # Keep history limited
        if len(self.operation_times['batch_similarities']) > self.max_history:
            self.operation_times['batch_similarities'] = self.operation_times['batch_similarities'][-self.max_history:]
        
        # Log performance info
        self.logger.debug(f"Batch processed {num_queries} queries in {elapsed:.3f}s (batch_size={batch_size})")
        
        return all_results
    
    async def batch_update_embeddings(self, nodes_with_embeddings):
        """
        Update embeddings for multiple nodes in batches.
        
        Args:
            nodes_with_embeddings: List of (node_id, embeddings) pairs
            
        Returns:
            int: Number of successfully updated nodes
        """
        start_time = time.time()
        
        # Skip if empty
        if not nodes_with_embeddings:
            return 0
        
        # Process in batches
        batch_size = self.embedding_batch_size
        total_updates = len(nodes_with_embeddings)
        successful_updates = 0
        
        for i in range(0, total_updates, batch_size):
            # Get batch
            end_idx = min(i + batch_size, total_updates)
            batch = nodes_with_embeddings[i:end_idx]
            
            # Process batch
            batch_updates = []
            
            for node_id, embeddings in batch:
                try:
                    # Convert embeddings to proper format
                    if isinstance(embeddings, list):
                        embeddings = np.array(embeddings, dtype=np.float32)
                    
                    # Get node index
                    if node_id in self.tensor_network.node_index:
                        node_idx = self.tensor_network.node_index[node_id]
                        batch_updates.append((node_idx, embeddings))
                        
                        # Update node in tensor_network
                        node = self.tensor_network.nodes.get(node_id)
                        if node:
                            node.embeddings = embeddings
                            # Create tensor representation
                            if isinstance(embeddings, np.ndarray):
                                node.tensor_representation = torch.tensor(
                                    embeddings, dtype=torch.float32
                                ).to(self.device)
                            elif isinstance(embeddings, torch.Tensor):
                                node.tensor_representation = embeddings.to(self.device)
                            
                            successful_updates += 1
                except Exception as e:
                    self.logger.warning(f"Error updating embeddings for node {node_id}: {str(e)}")
            
            # Add to buffer for tensor updates
            self.tensor_network.embedding_buffer.extend(batch_updates)
            
            # Update tensors if buffer is large enough
            if len(self.tensor_network.embedding_buffer) >= self.tensor_network.batch_size:
                await self.tensor_network._update_tensors()
        
        # Track performance
        elapsed = time.time() - start_time
        self.operation_times['batch_embeddings'].append(elapsed)
        self.batch_counts['batch_embeddings'] += 1
        
        # Log performance info
        self.logger.debug(f"Batch updated {successful_updates}/{total_updates} embeddings in {elapsed:.3f}s")
        
        return successful_updates
    
    async def batch_process_relations(self, relation_operations):
        """
        Process multiple relation operations in batches.
        
        Args:
            relation_operations: List of relation operations (dicts with operation details)
            
        Returns:
            Dict: Results with success/failure counts
        """
        start_time = time.time()
        
        # Skip if empty
        if not relation_operations:
            return {'success': 0, 'failure': 0, 'total': 0}
        
        # Process in batches
        batch_size = self.relation_batch_size
        total_operations = len(relation_operations)
        success_count = 0
        failure_count = 0
        
        for i in range(0, total_operations, batch_size):
            # Get batch
            end_idx = min(i + batch_size, total_operations)
            batch = relation_operations[i:end_idx]
            
            # Process each operation in the batch
            for operation in batch:
                try:
                    op_type = operation.get('type')
                    
                    if op_type == 'add_relation':
                        # Create relation object
                        relation = self._create_relation_from_operation(operation)
                        if relation:
                            # Add to network
                            await self.tensor_network.add_relation(relation)
                            success_count += 1
                        else:
                            failure_count += 1
                            
                    elif op_type == 'update_relation':
                        # Get relation ID
                        relation_id = operation.get('relation_id')
                        if not relation_id or relation_id not in self.tensor_network.relations:
                            failure_count += 1
                            continue
                            
                        # Get relation
                        relation = self.tensor_network.relations[relation_id]
                        
                        # Update fields
                        if 'weight' in operation:
                            relation.weight = operation['weight']
                        if 'properties' in operation:
                            relation.properties.update(operation['properties'])
                        if 'confidence' in operation:
                            relation.confidence = operation['confidence']
                        
                        # Update timestamp
                        relation.temporal_metadata.update_modification()
                        
                        # Add back to network
                        await self.tensor_network.add_relation(relation)
                        success_count += 1
                        
                    elif op_type == 'delete_relation':
                        # Not implemented in original tensor network
                        # Would require additional functionality
                        failure_count += 1
                        
                    else:
                        # Unknown operation
                        failure_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing relation operation: {str(e)}")
                    failure_count += 1
        
        # Mark tensors for rebuild
        if success_count > 0:
            self.tensor_network.rebuild_needed = True
        
        # Track performance
        elapsed = time.time() - start_time
        self.operation_times['batch_relations'].append(elapsed)
        self.batch_counts['batch_relations'] += 1
        
        # Log performance info
        self.logger.debug(f"Batch processed {total_operations} relation operations in {elapsed:.3f}s " + 
                       f"(success: {success_count}, failure: {failure_count})")
        
        return {
            'success': success_count,
            'failure': failure_count,
            'total': total_operations
        }
    
    def _create_relation_from_operation(self, operation):
        """Helper to create a relation object from operation dict"""
        from asf.layer3_cognitive_boundary.core.semantic_relation import SemanticRelation
        from asf.layer3_cognitive_boundary.temporal import AdaptiveTemporalMetadata
        
        try:
            # Extract required fields
            source_id = operation.get('source_id')
            target_id = operation.get('target_id')
            relation_type = operation.get('relation_type', 'generic')
            
            # Validate nodes exist
            if source_id not in self.tensor_network.nodes or target_id not in self.tensor_network.nodes:
                return None
                
            # Extract optional fields
            relation_id = operation.get('relation_id', f"rel_{hash(source_id + target_id + relation_type) % 10000:04d}")
            weight = operation.get('weight', 0.8)
            bidirectional = operation.get('bidirectional', False)
            properties = operation.get('properties', {})
            confidence = operation.get('confidence', 0.7)
            
            # Create temporal metadata
            temporal_metadata = AdaptiveTemporalMetadata()
            
            # Create relation
            relation = SemanticRelation(
                id=relation_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight,
                bidirectional=bidirectional,
                properties=properties,
                confidence=confidence,
                temporal_metadata=temporal_metadata
            )
            
            return relation
        except Exception as e:
            self.logger.warning(f"Error creating relation: {str(e)}")
            return None
    
    def get_performance_stats(self):
        """
        Get performance statistics for batched operations.
        
        Returns:
            Dict: Performance statistics
        """
        stats = {}
        
        for op_name, times in self.operation_times.items():
            if not times:
                continue
                
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            
            batch_count = self.batch_counts.get(op_name, 0)
            
            stats[op_name] = {
                'avg_time': avg_time,
                'max_time': max_time,
                'min_time': min_time,
                'batch_count': batch_count,
                'samples': len(times)
            }
            
        return stats


# ---------------------------------------------------------
# 2. INCREMENTAL UPDATE IMPLEMENTATION
# ---------------------------------------------------------

class IncrementalUpdateManager:
    """
    Manages incremental updates to tensor representations.
    Tracks changes and applies them efficiently without full rebuilds.
    """
    
    def __init__(self, tensor_network, config=None):
        """
        Initialize incremental update manager.
        
        Args:
            tensor_network: Parent SemanticTensorNetwork instance
            config: Configuration dictionary
        """
        self.tensor_network = tensor_network
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3.IncrementalUpdates")
        
        # Change tracking
        self.pending_node_changes = set()  # Set of node IDs with changes
        self.pending_relation_changes = set()  # Set of relation IDs with changes
        self.pending_relation_type_changes = set()  # Set of relation types with changes
        
        # Deletion tracking
        self.deleted_nodes = set()  # Set of deleted node IDs
        self.deleted_relations = set()  # Set of deleted relation IDs
        
        # Configuration
        self.rebuild_threshold = self.config.get('rebuild_threshold', 0.2)  # Fraction of total items changed before full rebuild
        self.rebuild_check_frequency = self.config.get('rebuild_check_frequency', 100)
        self.change_count_since_rebuild = 0
        
        # Performance tracking
        self.update_times = []
        self.full_rebuild_times = []
        self.max_history = 20
    
    def track_node_change(self, node_id):
        """
        Track a node change for incremental updates.
        
        Args:
            node_id: ID of the changed node
        """
        self.pending_node_changes.add(node_id)
        self.change_count_since_rebuild += 1
        self._check_rebuild_needed()
    
    def track_relation_change(self, relation_id, relation_type=None):
        """
        Track a relation change for incremental updates.
        
        Args:
            relation_id: ID of the changed relation
            relation_type: Type of the relation (optional)
        """
        self.pending_relation_changes.add(relation_id)
        if relation_type:
            self.pending_relation_type_changes.add(relation_type)
        self.change_count_since_rebuild += 1
        self._check_rebuild_needed()
    
    def track_node_deletion(self, node_id):
        """
        Track a node deletion for cleanup.
        
        Args:
            node_id: ID of the deleted node
        """
        self.deleted_nodes.add(node_id)
        if node_id in self.pending_node_changes:
            self.pending_node_changes.remove(node_id)
        self.change_count_since_rebuild += 1
        self._check_rebuild_needed()
    
    def track_relation_deletion(self, relation_id, relation_type=None):
        """
        Track a relation deletion for cleanup.
        
        Args:
            relation_id: ID of the deleted relation
            relation_type: Type of the relation (optional)
        """
        self.deleted_relations.add(relation_id)
        if relation_id in self.pending_relation_changes:
            self.pending_relation_changes.remove(relation_id)
        if relation_type:
            self.pending_relation_type_changes.add(relation_type)
        self.change_count_since_rebuild += 1
        self._check_rebuild_needed()
    
    def _check_rebuild_needed(self):
        """Check if full rebuild should be triggered instead of incremental updates"""
        if self.change_count_since_rebuild % self.rebuild_check_frequency != 0:
            return
            
        # Calculate change ratios
        node_count = len(self.tensor_network.nodes)
        relation_count = len(self.tensor_network.relations)
        
        if node_count == 0 or relation_count == 0:
            return
            
        node_change_ratio = len(self.pending_node_changes) / node_count
        relation_change_ratio = len(self.pending_relation_changes) / relation_count
        
        # Check if thresholds exceeded
        if node_change_ratio > self.rebuild_threshold or relation_change_ratio > self.rebuild_threshold:
            self.logger.info(f"Change ratio exceeded rebuild threshold (nodes: {node_change_ratio:.3f}, " + 
                          f"relations: {relation_change_ratio:.3f}), scheduling full rebuild")
            self.tensor_network.rebuild_needed = True
    
    async def apply_incremental_updates(self):
        """
        Apply all pending incremental updates.
        
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        
        if not self.pending_node_changes and not self.pending_relation_changes and not self.deleted_nodes and not self.deleted_relations:
            return True
            
        # If full rebuild is needed, don't do incremental updates
        if self.tensor_network.rebuild_needed:
            self.logger.info("Full rebuild needed, skipping incremental updates")
            return await self._perform_full_rebuild()
            
        self.logger.info(f"Applying incremental updates: {len(self.pending_node_changes)} nodes, " + 
                      f"{len(self.pending_relation_changes)} relations, " +
                      f"{len(self.deleted_nodes)} deleted nodes, " +
                      f"{len(self.deleted_relations)} deleted relations")
        
        try:
            # 1. Update node embeddings tensor
            if self.tensor_network.node_embeddings is not None and self.pending_node_changes:
                await self._update_node_embeddings()
                
            # 2. Update FAISS index if needed
            if self.tensor_network.faiss_initialized and self.pending_node_changes:
                await self._update_faiss_index()
                
            # 3. Update relation adjacency tensors
            if self.pending_relation_changes or self.deleted_relations:
                await self._update_relation_tensors()
                
            # 4. Process deletions if not already handled
            if self.deleted_nodes or self.deleted_relations:
                await self._process_deletions()
                
            # Clear pending changes
            self.pending_node_changes.clear()
            self.pending_relation_changes.clear()
            self.pending_relation_type_changes.clear()
            self.deleted_nodes.clear()
            self.deleted_relations.clear()
            
            # Reset change counter
            self.change_count_since_rebuild = 0
            
            # Track performance
            elapsed = time.time() - start_time
            self.update_times.append(elapsed)
            if len(self.update_times) > self.max_history:
                self.update_times = self.update_times[-self.max_history:]
                
            self.logger.info(f"Incremental updates completed in {elapsed:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during incremental updates: {str(e)}", exc_info=True)
            
            # Fall back to full rebuild
            self.logger.info("Falling back to full rebuild after incremental update failure")
            self.tensor_network.rebuild_needed = True
            return await self._perform_full_rebuild()
    
    async def _update_node_embeddings(self):
        """Update the node embeddings tensor incrementally"""
        # Skip if no embeddings tensor
        if self.tensor_network.node_embeddings is None:
            return
            
        embedding_updates = []
        
        # Collect embeddings for changed nodes
        for node_id in self.pending_node_changes:
            if node_id in self.deleted_nodes:
                continue  # Skip deleted nodes
                
            node = self.tensor_network.nodes.get(node_id)
            if not node or not hasattr(node, 'embeddings') or node.embeddings is None:
                continue
                
            node_idx = self.tensor_network.node_index.get(node_id)
            if node_idx is None or node_idx >= self.tensor_network.node_embeddings.shape[0]:
                continue
                
            # Get embedding
            if isinstance(node.embeddings, np.ndarray):
                emb = node.embeddings
            elif isinstance(node.embeddings, torch.Tensor):
                emb = node.embeddings.cpu().detach().numpy()
            else:
                continue
                
            embedding_updates.append((node_idx, emb))
        
        # Apply updates in batch
        for node_idx, emb in embedding_updates:
            if isinstance(emb, np.ndarray):
                emb_tensor = torch.tensor(emb, dtype=torch.float32).to(self.tensor_network.device)
            elif isinstance(emb, torch.Tensor):
                emb_tensor = emb.to(self.tensor_network.device)
            else:
                continue
                
            # Update the tensor if shapes match
            if emb_tensor.shape[0] == self.tensor_network.node_embeddings.shape[1]:
                self.tensor_network.node_embeddings[node_idx] = emb_tensor
    
    async def _update_faiss_index(self):
        """Update the FAISS index incrementally"""
        # Skip if no FAISS index
        if not self.tensor_network.faiss_initialized or self.tensor_network.faiss_index is None:
            return
            
        # Check if index supports direct updates
        supports_direct_update = hasattr(self.tensor_network.faiss_index, 'remove_ids') and \
                             hasattr(self.tensor_network.faiss_index, 'add_with_ids')
        
        if not supports_direct_update:
            # Fall back to full rebuild of FAISS index
            self.logger.info("FAISS index doesn't support direct updates, performing full index rebuild")
            await self.tensor_network._initialize_faiss_index()
            return
            
        # Process updates
        for node_id in self.pending_node_changes:
            if node_id in self.deleted_nodes:
                continue  # Skip deleted nodes
                
            node = self.tensor_network.nodes.get(node_id)
            if not node or not hasattr(node, 'embeddings') or node.embeddings is None:
                continue
                
            node_idx = self.tensor_network.node_index.get(node_id)
            if node_idx is None:
                continue
                
            # Get embedding
            if isinstance(node.embeddings, np.ndarray):
                emb = node.embeddings
            elif isinstance(node.embeddings, torch.Tensor):
                emb = node.embeddings.cpu().detach().numpy()
            else:
                continue
                
            # Reshape and normalize
            if len(emb.shape) == 1:
                emb = emb.reshape(1, -1)
                
            # Normalize for cosine similarity
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
            # Create ID array
            ids = np.array([node_idx], dtype=np.int64)
            
            # Remove and re-add
            try:
                self.tensor_network.faiss_index.remove_ids(ids)
                self.tensor_network.faiss_index.add_with_ids(emb, ids)
            except Exception as e:
                self.logger.warning(f"Error updating FAISS index for node {node_id}: {str(e)}")
                # If direct update fails, mark for full rebuild
                self.tensor_network.rebuild_needed = True
    
    async def _update_relation_tensors(self):
        """Update the relation adjacency tensors incrementally"""
        # Skip if no changes
        if not self.pending_relation_changes and not self.deleted_relations:
            return
            
        # Get relation types that need updating
        types_to_update = self.pending_relation_type_changes.copy()
        
        # Add types from changed relations not already tracked
        for rel_id in self.pending_relation_changes:
            rel = self.tensor_network.relations.get(rel_id)
            if rel:
                types_to_update.add(rel.relation_type)
                
        # Add types from deleted relations
        for rel_id in self.deleted_relations:
            # We can't get the type directly, so we rebuild all
            # This could be improved by tracking relation types during deletion
            self.tensor_network.rebuild_needed = True
            break
            
        # If too many types need updates, do full rebuild
        if len(types_to_update) > 5:  # Arbitrary threshold
            self.logger.info(f"Many relation types changed ({len(types_to_update)}), performing full rebuild")
            self.tensor_network.rebuild_needed = True
            return
            
        # Rebuild specific relation type tensors
        for rel_type in types_to_update:
            await self._rebuild_relation_type_tensor(rel_type)
            
        # Always rebuild combined tensor
        await self._rebuild_combined_tensor()
    
    async def _rebuild_relation_type_tensor(self, rel_type):
        """Rebuild the adjacency tensor for a specific relation type"""
        # Get relations of this type
        rel_ids = self.tensor_network.relation_index['type'].get(rel_type, [])
        
        # Get dimensions
        node_count = self.tensor_network.node_count
        
        # Build COO format sparse tensor
        indices = []
        values = []
        
        for rel_id in rel_ids:
            rel = self.tensor_network.relations.get(rel_id)
            if rel is None or rel_id in self.deleted_relations:
                continue
                
            # Get indices from node mapping
            source_idx = self.tensor_network.node_index.get(rel.source_id)
            target_idx = self.tensor_network.node_index.get(rel.target_id)
            
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
            if rel_type in self.tensor_network.adjacency_tensors:
                del self.tensor_network.adjacency_tensors[rel_type]
            return
            
        # Create sparse tensor
        try:
            indices_tensor = torch.tensor(indices, dtype=torch.long).t().to(self.tensor_network.device)
            values_tensor = torch.tensor(values, dtype=torch.float32).to(self.tensor_network.device)
            
            sparse_tensor = torch.sparse_coo_tensor(
                indices_tensor, 
                values_tensor,
                (node_count, node_count),
                device=self.tensor_network.device
            )
            
            # Store in adjacency tensors
            self.tensor_network.adjacency_tensors[rel_type] = sparse_tensor
        except Exception as e:
            self.logger.error(f"Error rebuilding tensor for relation type '{rel_type}': {str(e)}")
            # Mark for full rebuild on failure
            self.tensor_network.rebuild_needed = True
    
    async def _rebuild_combined_tensor(self):
        """Rebuild the combined adjacency tensor"""
        # Get dimensions
        node_count = self.tensor_network.node_count
        
        try:
            # Combine all relation types for a general adjacency tensor
            all_indices = []
            all_values = []
            
            for rel_id, rel in self.tensor_network.relations.items():
                if rel_id in self.deleted_relations:
                    continue
                    
                # Get indices from node mapping
                source_idx = self.tensor_network.node_index.get(rel.source_id)
                target_idx = self.tensor_network.node_index.get(rel.target_id)
                
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
                indices_tensor = torch.tensor(all_indices, dtype=torch.long).t().to(self.tensor_network.device)
                values_tensor = torch.tensor(all_values, dtype=torch.float32).to(self.tensor_network.device)
                
                combined_tensor = torch.sparse_coo_tensor(
                    indices_tensor, 
                    values_tensor,
                    (node_count, node_count),
                    device=self.tensor_network.device
                )
                
                # Store as special 'combined' type
                self.tensor_network.adjacency_tensors['combined'] = combined_tensor
        except Exception as e:
            self.logger.error(f"Error rebuilding combined tensor: {str(e)}")
            # Mark for full rebuild on failure
            self.tensor_network.rebuild_needed = True
    
    async def _process_deletions(self):
        """Process node and relation deletions"""
        # Process node deletions
        for node_id in self.deleted_nodes:
            # Check if node was already removed from main storage
            if node_id not in self.tensor_network.nodes:
                continue
                
            # Remove from main storage
            node_idx = self.tensor_network.node_index.pop(node_id, None)
            self.tensor_network.nodes.pop(node_id, None)
            
            # Remove from index_to_node if we have an index
            if node_idx is not None:
                if node_idx in self.tensor_network.index_to_node:
                    self.tensor_network.index_to_node.pop(node_idx)
                    
            # We can't easily update tensors for deletions
            # Mark for full rebuild
            self.tensor_network.rebuild_needed = True
            
        # Process relation deletions
        for rel_id in self.deleted_relations:
            # Check if relation was already removed
            if rel_id not in self.tensor_network.relations:
                continue
                
            # Get relation for type information (before removal)
            rel = self.tensor_network.relations.get(rel_id)
            rel_type = rel.relation_type if rel else None
            
            # Remove from main storage
            self.tensor_network.relations.pop(rel_id, None)
            
            # Remove from indices
            if rel and rel.source_id in self.tensor_network.relation_index['source']:
                if rel_id in self.tensor_network.relation_index['source'][rel.source_id]:
                    self.tensor_network.relation_index['source'][rel.source_id].remove(rel_id)
                    
            if rel and rel.target_id in self.tensor_network.relation_index['target']:
                if rel_id in self.tensor_network.relation_index['target'][rel.target_id]:
                    self.tensor_network.relation_index['target'][rel.target_id].remove(rel_id)
                    
            if rel_type and rel_type in self.tensor_network.relation_index['type']:
                if rel_id in self.tensor_network.relation_index['type'][rel_type]:
                    self.tensor_network.relation_index['type'][rel_type].remove(rel_id)
                    
            # Add to relation type changes for tensor updates
            if rel_type:
                self.pending_relation_type_changes.add(rel_type)
    
    async def _perform_full_rebuild(self):
        """Perform a full rebuild of all tensors"""
        start_time = time.time()
        self.logger.info("Performing full tensor rebuild")
        
        success = await self.tensor_network._build_tensors(full_rebuild=True)
        
        # Rebuild FAISS index if initialized
        if success and self.tensor_network.faiss_initialized:
            await self.tensor_network._initialize_faiss_index()
            
        # Clear pending changes
        self.pending_node_changes.clear()
        self.pending_relation_changes.clear()
        self.pending_relation_type_changes.clear()
        self.deleted_nodes.clear()
        self.deleted_relations.clear()
        
        # Reset change counter
        self.change_count_since_rebuild = 0
        
        # Reset rebuild flag
        self.tensor_network.rebuild_needed = False
        
        # Track performance
        elapsed = time.time() - start_time
        self.full_rebuild_times.append(elapsed)
        if len(self.full_rebuild_times) > self.max_history:
            self.full_rebuild_times = self.full_rebuild_times[-self.max_history:]
            
        self.logger.info(f"Full rebuild completed in {elapsed:.3f}s")
        return success
    
    def get_update_stats(self):
        """
        Get statistics about incremental updates and rebuilds.
        
        Returns:
            Dict: Update statistics
        """
        stats = {
            'pending_node_changes': len(self.pending_node_changes),
            'pending_relation_changes': len(self.pending_relation_changes),
            'pending_relation_type_changes': len(self.pending_relation_type_changes),
            'deleted_nodes': len(self.deleted_nodes),
            'deleted_relations': len(self.deleted_relations),
            'changes_since_rebuild': self.change_count_since_rebuild
        }
        
        if self.update_times:
            stats['incremental_update_time'] = {
                'avg': sum(self.update_times) / len(self.update_times),
                'min': min(self.update_times),
                'max': max(self.update_times),
                'samples': len(self.update_times)
            }
            
        if self.full_rebuild_times:
            stats['full_rebuild_time'] = {
                'avg': sum(self.full_rebuild_times) / len(self.full_rebuild_times),
                'min': min(self.full_rebuild_times),
                'max': max(self.full_rebuild_times),
                'samples': len(self.full_rebuild_times)
            }
            
        return stats


# ---------------------------------------------------------
# 3. CACHING MECHANISMS IMPLEMENTATION
# ---------------------------------------------------------

class SimilarityCacheManager:
    """
    Caches similarity search results to avoid redundant computations.
    Implements an LRU (Least Recently Used) cache with time-based expiration.
    """
    
    def __init__(self, tensor_network, config=None):
        """
        Initialize cache manager with configuration.
        
        Args:
            tensor_network: Parent SemanticTensorNetwork instance
            config: Configuration dictionary
        """
        self.tensor_network = tensor_network
        self.config = config or {}
        self.logger = logging.getLogger("ASF.Layer3.SimilarityCache")
        
        # Cache configuration
        self.cache_enabled = self.config.get('similarity_cache_enabled', True)
        self.cache_size = self.config.get('similarity_cache_size', 1000)
        self.cache_expiration = self.config.get('similarity_cache_expiration', 300)  # 5 minutes
        self.min_query_count = self.config.get('min_query_count', 2)  # Min queries before caching
        
        # Create LRU cache
        self.similarity_cache = OrderedDict()
        self.expiration_times = {}
        
        # Query tracking for adaptive caching
        self.query_counts = defaultdict(int)
        self.last_cleanup = time.time()
        self.cleanup_interval = 60  # Cleanup every minute
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_insertions = 0
        self.cache_expirations = 0
    
    async def get_cached_similarity(self, node_id, k=5, threshold=0.7):
        """
        Retrieve cached similarity results if available.
        
        Args:
            node_id: Query node ID
            k: Number of similar nodes to return
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple[bool, Optional[List]]: (cache_hit, results)
        """
        if not self.cache_enabled:
            return False, None
            
        # Track this query
        self.query_counts[node_id] += 1
        
        # Check if cleanup is needed
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            await self._cleanup_expired()
            self.last_cleanup = current_time
            
        # Create cache key
        cache_key = self._create_cache_key(node_id, k, threshold)
        
        # Check if in cache and not expired
        if cache_key in self.similarity_cache:
            # Check expiration
            if cache_key in self.expiration_times and current_time > self.expiration_times[cache_key]:
                # Expired
                del self.similarity_cache[cache_key]
                if cache_key in self.expiration_times:
                    del self.expiration_times[cache_key]
                self.cache_expirations += 1
                return False, None
                
            # Update LRU order
            results = self.similarity_cache.pop(cache_key)
            self.similarity_cache[cache_key] = results
            
            # Increment hit counter
            self.cache_hits += 1
            
            return True, results
            
        # Cache miss
        self.cache_misses += 1
        return False, None
    
    async def cache_similarity_results(self, node_id, results, k=5, threshold=0.7):
        """
        Cache similarity search results.
        
        Args:
            node_id: Query node ID
            results: Similarity results to cache
            k: Number of similar nodes returned
            threshold: Similarity threshold used
            
        Returns:
            bool: True if cached, False if skipped
        """
        if not self.cache_enabled:
            return False
            
        # Skip caching for infrequent queries
        if self.query_counts[node_id] < self.min_query_count:
            return False
            
        # Create cache key
        cache_key = self._create_cache_key(node_id, k, threshold)
        
        # Maintain cache size limit (LRU eviction)
        while len(self.similarity_cache) >= self.cache_size:
            # Remove oldest item
            if self.similarity_cache:
                oldest_key, _ = next(iter(self.similarity_cache.items()))
                del self.similarity_cache[oldest_key]
                if oldest_key in self.expiration_times:
                    del self.expiration_times[oldest_key]
            else:
                break
                
        # Add to cache with expiration
        self.similarity_cache[cache_key] = results
        self.expiration_times[cache_key] = time.time() + self.cache_expiration
        
        # Increment insertion counter
        self.cache_insertions += 1
        
        return True
    
    async def invalidate_node_cache(self, node_id):
        """
        Invalidate cache entries related to a specific node.
        
        Args:
            node_id: Node ID to invalidate
            
        Returns:
            int: Number of invalidated entries
        """
        if not self.cache_enabled:
            return 0
            
        # Find keys to invalidate
        keys_to_remove = []
        for cache_key in self.similarity_cache.keys():
            if node_id in cache_key:
                keys_to_remove.append(cache_key)
                
        # Also invalidate results containing this node
        for cache_key, results in self.similarity_cache.items():
            if any(similar_id == node_id for similar_id, _ in results):
                if cache_key not in keys_to_remove:
                    keys_to_remove.append(cache_key)
                    
        # Remove from cache
        for key in keys_to_remove:
            if key in self.similarity_cache:
                del self.similarity_cache[key]
            if key in self.expiration_times:
                del self.expiration_times[key]
                
        if keys_to_remove:
            self.logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for node {node_id}")
            
        return len(keys_to_remove)
    
    async def clear_cache(self):
        """
        Clear the entire similarity cache.
        
        Returns:
            int: Number of cleared entries
        """
        count = len(self.similarity_cache)
        self.similarity_cache.clear()
        self.expiration_times.clear()
        self.logger.info(f"Cleared similarity cache ({count} entries)")
        return count
    
    async def _cleanup_expired(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        keys_to_remove = []
        
        # Find expired entries
        for cache_key, expiration_time in self.expiration_times.items():
            if current_time > expiration_time:
                keys_to_remove.append(cache_key)
                
        # Remove expired entries
        for key in keys_to_remove:
            if key in self.similarity_cache:
                del self.similarity_cache[key]
            if key in self.expiration_times:
                del self.expiration_times[key]
                
        self.cache_expirations += len(keys_to_remove)
        
        if keys_to_remove:
            self.logger.debug(f"Cleaned up {len(keys_to_remove)} expired cache entries")
    
    def _create_cache_key(self, node_id, k, threshold):
        """Create a stable cache key"""
        return f"{node_id}_{k}_{threshold:.4f}"
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            Dict: Cache statistics
        """
        stats = {
            'enabled': self.cache_enabled,
            'size': len(self.similarity_cache),
            'max_size': self.cache_size,
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'insertions': self.cache_insertions,
            'expirations': self.cache_expirations,
            'frequent_queries': sum(1 for count in self.query_counts.values() if count >= self.min_query_count),
            'total_tracked_queries': len(self.query_counts)
        }
        
        # Calculate hit rate
        total_queries = self.cache_hits + self.cache_misses
        if total_queries > 0:
            stats['hit_rate'] = self.cache_hits / total_queries
        else:
            stats['hit_rate'] = 0.0
            
        return stats

    def get_most_queried(self, limit=10):
        """
        Get most frequently queried nodes.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            List[Tuple[str, int]]: List of (node_id, query_count) pairs
        """
        sorted_queries = sorted(
            self.query_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_queries[:limit]


# ---------------------------------------------------------
# 4. SEMANTIC TENSOR NETWORK OPTIMIZATION INTEGRATION
# ---------------------------------------------------------

class SemanticTensorNetworkOptimized(SemanticTensorNetwork):
    """
    Enhanced version of SemanticTensorNetwork with integrated performance optimizations.
    Implements batched processing, incremental updates, and caching mechanisms.
    """
    
    def __init__(self, embedding_dim=1024, device=None, config=None):
        """
        Initialize optimized semantic tensor network.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            device: Computation device (CPU/GPU)
            config: Configuration dictionary
        """
        # Call original initialization
        super().__init__(embedding_dim, device)
        
        # Store configuration
        self.config = config or {}
        
        # Initialize optimization components
        self.batched_ops = BatchedTensorOperations(self, self.config)
        self.update_manager = IncrementalUpdateManager(self, self.config)
        self.cache_manager = SimilarityCacheManager(self, self.config)
        
        # Enable optimizations based on config
        self.enable_batching = self.config.get('enable_batching', True)
        self.enable_incremental_updates = self.config.get('enable_incremental_updates', True)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Performance monitoring
        self.perf_metrics = defaultdict(list)
        self.max_metrics_history = 100
    
    async def add_node(self, node, update_tensors=True):
        """
        Add a node to the semantic network with optimized tensor updates.
        
        Args:
            node: Node to add
            update_tensors: Whether to update tensors
            
        Returns:
            str: Node ID
        """
        start_time = time.time()
        
        node_id = await super().add_node(node, update_tensors and not self.enable_incremental_updates)
        
        # Track for incremental updates if enabled
        if self.enable_incremental_updates and node_id:
            self.update_manager.track_node_change(node_id)
            
            # Apply incremental updates if buffer is large enough
            if len(self.embedding_buffer) >= self.batch_size:
                await self.update_manager.apply_incremental_updates()
                
        # Invalidate cache for this node
        if self.enable_caching and node_id:
            await self.cache_manager.invalidate_node_cache(node_id)
            
        # Track performance
        elapsed = time.time() - start_time
        self.perf_metrics['add_node'].append(elapsed)
        if len(self.perf_metrics['add_node']) > self.max_metrics_history:
            self.perf_metrics['add_node'] = self.perf_metrics['add_node'][-self.max_metrics_history:]
            
        return node_id
    
    async def add_relation(self, relation):
        """
        Add a relation to the semantic network with optimized tensor updates.
        
        Args:
            relation: Relation to add
            
        Returns:
            str: Relation ID
        """
        start_time = time.time()
        
        relation_id = await super().add_relation(relation)
        
        # Track for incremental updates if enabled
        if self.enable_incremental_updates and relation_id:
            self.update_manager.track_relation_change(relation_id, relation.relation_type)
            
            # Note: We don't immediately apply updates for relations
            # This allows batching multiple relation changes
            
        # Track performance
        elapsed = time.time() - start_time
        self.perf_metrics['add_relation'].append(elapsed)
        if len(self.perf_metrics['add_relation']) > self.max_metrics_history:
            self.perf_metrics['add_relation'] = self.perf_metrics['add_relation'][-self.max_metrics_history:]
            
        return relation_id
    
    async def batch_add_nodes(self, nodes):
        """
        Add multiple nodes in a batch.
        
        Args:
            nodes: List of nodes to add
            
        Returns:
            Dict: Results with node IDs
        """
        if not self.enable_batching or not nodes:
            # Fall back to individual adds
            results = {}
            for node in nodes:
                node_id = await self.add_node(node, update_tensors=False)
                if node_id:
                    results[node_id] = True
                    
            # Update tensors once at the end
            if results and (self.enable_incremental_updates or self.embedding_buffer):
                await self.update_manager.apply_incremental_updates()
                
            return results
            
        start_time = time.time()
        
        # Add nodes without tensor updates
        results = {}
        for node in nodes:
            with self.lock:
                # Modified version of add_node without tensor updates
                if node.id in self.nodes:
                    # Update existing node
                    existing_node = self.nodes[node.id]
                    existing_node.label = node.label
                    existing_node.properties.update(node.properties)
                    existing_node.confidence = max(existing_node.confidence, node.confidence)
                    
                    if hasattr(node, 'embeddings') and node.embeddings is not None:
                        # Update embeddings
                        existing_node.embeddings = node.embeddings
                        if isinstance(node.embeddings, np.ndarray):
                            existing_node.tensor_representation = torch.tensor(
                                node.embeddings, dtype=torch.float32
                            ).to(self.device)
                            
                        # Track for update
                        if self.enable_incremental_updates:
                            self.update_manager.track_node_change(node.id)
                        elif node.id in self.node_index:
                            self.embedding_buffer.append((self.node_index[node.id], node.embeddings))
                            
                    results[node.id] = True
                else:
                    # Add new node
                    self.nodes[node.id] = node
                    
                    # Move tensor to correct device
                    if hasattr(node, 'tensor_representation') and node.tensor_representation is not None:
                        node.tensor_representation = node.tensor_representation.to(self.device)
                        
                    # Update indices
                    self.node_index[node.id] = self.node_count
                    self.index_to_node[self.node_count] = node.id
                    
                    # Track for embedding updates
                    if hasattr(node, 'embeddings') and node.embeddings is not None:
                        if self.enable_incremental_updates:
                            self.update_manager.track_node_change(node.id)
                        else:
                            self.embedding_buffer.append((self.node_count, node.embeddings))
                            
                    self.node_count += 1
                    results[node.id] = True
                    
                    # Invalidate cache
                    if self.enable_caching:
                        await self.cache_manager.invalidate_node_cache(node.id)
        
        # Apply batch updates
        if self.enable_incremental_updates:
            await self.update_manager.apply_incremental_updates()
        elif self.embedding_buffer:
            await self._update_tensors()
            
        # Track performance
        elapsed = time.time() - start_time
        self.perf_metrics['batch_add_nodes'].append(elapsed)
        if len(self.perf_metrics['batch_add_nodes']) > self.max_metrics_history:
            self.perf_metrics['batch_add_nodes'] = self.perf_metrics['batch_add_nodes'][-self.max_metrics_history:]
            
        self.logger.info(f"Batch added {len(results)}/{len(nodes)} nodes in {elapsed:.3f}s")
        
        return results
    
    async def batch_add_relations(self, relations):
        """
        Add multiple relations in a batch.
        
        Args:
            relations: List of relations to add
            
        Returns:
            Dict: Results with relation IDs
        """
        if not self.enable_batching or not relations:
            # Fall back to individual adds
            results = {}
            for relation in relations:
                relation_id = await self.add_relation(relation)
                if relation_id:
                    results[relation_id] = True
                    
            return results
            
        start_time = time.time()
        
        # Use batch processing for relations
        if self.enable_batching:
            # Convert to operation format
            operations = []
            for relation in relations:
                operations.append({
                    'type': 'add_relation',
                    'source_id': relation.source_id,
                    'target_id': relation.target_id,
                    'relation_type': relation.relation_type,
                    'relation_id': relation.id,
                    'weight': relation.weight,
                    'bidirectional': relation.bidirectional,
                    'properties': relation.properties,
                    'confidence': relation.confidence
                })
                
            # Process in batch
            batch_results = await self.batched_ops.batch_process_relations(operations)
            
            # Update tensors
            if self.enable_incremental_updates:
                await self.update_manager.apply_incremental_updates()
            else:
                self.rebuild_needed = True
                
            # Track performance
            elapsed = time.time() - start_time
            self.perf_metrics['batch_add_relations'].append(elapsed)
            if len(self.perf_metrics['batch_add_relations']) > self.max_metrics_history:
                self.perf_metrics['batch_add_relations'] = self.perf_metrics['batch_add_relations'][-self.max_metrics_history:]
                
            self.logger.info(f"Batch added {batch_results['success']}/{len(relations)} relations in {elapsed:.3f}s")
            
            # Create results dict
            results = {}
            for relation in relations:
                results[relation.id] = True
                
            return results
        else:
            # Fall back to individual adds
            results = {}
            for relation in relations:
                relation_id = await self.add_relation(relation)
                if relation_id:
                    results[relation_id] = True
                    
            return results
    
    async def get_similar_nodes(self, node_id, k=5, threshold=0.7, by_embedding=True):
        """
        Find semantically similar nodes with caching optimization.
        
        Args:
            node_id: Node ID to find similar nodes for
            k: Number of similar nodes to return
            threshold: Minimum similarity threshold
            by_embedding: Whether to use embeddings
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, similarity) pairs
        """
        start_time = time.time()
        
        # Try cache first if enabled
        if self.enable_caching and by_embedding:
            cache_hit, cached_results = await self.cache_manager.get_cached_similarity(node_id, k, threshold)
            
            if cache_hit:
                # Cache hit - return cached results
                elapsed = time.time() - start_time
                self.perf_metrics['cached_similarity'].append(elapsed)
                if len(self.perf_metrics['cached_similarity']) > self.max_metrics_history:
                    self.perf_metrics['cached_similarity'] = self.perf_metrics['cached_similarity'][-self.max_metrics_history:]
                
                self.logger.debug(f"Cache hit for similar_nodes({node_id}, k={k})")
                return cached_results
        
        # Cache miss or caching disabled - compute similarity
        results = await super().get_similar_nodes(node_id, k, threshold, by_embedding)
        
        # Cache results if enabled
        if self.enable_caching and by_embedding and results:
            await self.cache_manager.cache_similarity_results(node_id, results, k, threshold)
            
        # Track performance
        elapsed = time.time() - start_time
        self.perf_metrics['similar_nodes'].append(elapsed)
        if len(self.perf_metrics['similar_nodes']) > self.max_metrics_history:
            self.perf_metrics['similar_nodes'] = self.perf_metrics['similar_nodes'][-self.max_metrics_history:]
            
        return results
    
    async def batch_get_similar_nodes(self, node_ids, k=5, threshold=0.7):
        """
        Find similar nodes for multiple query nodes in batch.
        
        Args:
            node_ids: List of node IDs to find similar nodes for
            k: Number of similar nodes to return per query
            threshold: Minimum similarity threshold
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary mapping node IDs to similarity results
        """
        if not self.enable_batching or not node_ids:
            # Fall back to individual queries
            results = {}
            for node_id in node_ids:
                results[node_id] = await self.get_similar_nodes(node_id, k, threshold)
                
            return results
            
        start_time = time.time()
        
        # Check cache for all queries if enabled
        cached_results = {}
        query_nodes = []
        query_embeddings = []
        query_indices = []
        
        # Process each query node
        for i, node_id in enumerate(node_ids):
            if self.enable_caching:
                # Try cache first
                cache_hit, result = await self.cache_manager.get_cached_similarity(node_id, k, threshold)
                
                if cache_hit:
                    # Cache hit
                    cached_results[node_id] = result
                    continue
            
            # Cache miss or caching disabled - need to compute
            node = await self.get_node(node_id)
            if node:
                # Get embedding
                embedding = None
                if hasattr(node, 'embeddings') and node.embeddings is not None:
                    embedding = node.embeddings
                elif hasattr(node, 'tensor_representation') and node.tensor_representation is not None:
                    embedding = node.tensor_representation.cpu().numpy() if hasattr(node.tensor_representation, 'cpu') else node.tensor_representation
                
                if embedding is not None:
                    query_nodes.append(node)
                    query_embeddings.append(embedding)
                    query_indices.append(i)
                    
        # If all results were cached, return early
        if len(cached_results) == len(node_ids):
            return cached_results
            
        # Process non-cached queries in batch
        if query_embeddings:
            # Convert embeddings to appropriate format
            if isinstance(query_embeddings[0], np.ndarray):
                # Stack numpy arrays
                stacked_embeddings = np.stack(query_embeddings)
            elif isinstance(query_embeddings[0], torch.Tensor):
                # Stack torch tensors
                stacked_embeddings = torch.stack(query_embeddings)
            else:
                # Unsupported format, fall back to individual queries
                for node_id in node_ids:
                    if node_id not in cached_results:
                        cached_results[node_id] = await self.get_similar_nodes(node_id, k, threshold)
                        
                return cached_results
                
            # Perform batch similarity computation
            batch_results = await self.batched_ops.batch_process_similarities(
                stacked_embeddings, self.node_embeddings, k, threshold
            )
            
            # Convert batch results to node IDs and cache
            for i, batch_result in enumerate(batch_results):
                if i < len(query_nodes):
                    node = query_nodes[i]
                    node_id = node.id
                    
                    # Convert indices to node IDs
                    node_results = []
                    for idx, similarity in batch_result:
                        if idx < len(self.index_to_node) and idx >= 0:
                            similar_id = self.index_to_node.get(idx)
                            if similar_id and similar_id != node_id:
                                node_results.append((similar_id, similarity))
                    
                    # Store result
                    cached_results[node_id] = node_results
                    
                    # Cache result if enabled
                    if self.enable_caching:
                        await self.cache_manager.cache_similarity_results(node_id, node_results, k, threshold)
        
        # Track performance
        elapsed = time.time() - start_time
        self.perf_metrics['batch_similar_nodes'].append(elapsed)
        if len(self.perf_metrics['batch_similar_nodes']) > self.max_metrics_history:
            self.perf_metrics['batch_similar_nodes'] = self.perf_metrics['batch_similar_nodes'][-self.max_metrics_history:]
            
        self.logger.info(f"Batch similarity for {len(node_ids)} nodes in {elapsed:.3f}s " + 
                      f"(cached: {len(cached_results) - len(query_nodes)})")
        
        return cached_results
    
    async def apply_updates(self):
        """
        Apply pending updates to tensors and indices.
        
        Returns:
            bool: True if successful, False otherwise
        """
        start_time = time.time()
        
        if self.enable_incremental_updates:
            success = await self.update_manager.apply_incremental_updates()
        else:
            success = await self._update_tensors()
            
        # Track performance
        elapsed = time.time() - start_time
        self.perf_metrics['apply_updates'].append(elapsed)
        if len(self.perf_metrics['apply_updates']) > self.max_metrics_history:
            self.perf_metrics['apply_updates'] = self.perf_metrics['apply_updates'][-self.max_metrics_history:]
            
        return success
    
    async def clear_caches(self):
        """
        Clear all caches.
        
        Returns:
            Dict: Results with counts of cleared items
        """
        results = {}
        
        if self.enable_caching:
            similarity_count = await self.cache_manager.clear_cache()
            results['similarity_cache'] = similarity_count
            
        # More caches can be added here
        
        self.logger.info(f"Cleared caches: {results}")
        return results
    
    def get_performance_stats(self):
        """
        Get comprehensive performance statistics.
        
        Returns:
            Dict: Performance statistics
        """
        stats = {
            'config': {
                'batching_enabled': self.enable_batching,
                'incremental_updates_enabled': self.enable_incremental_updates,
                'caching_enabled': self.enable_caching
            },
            'node_count': len(self.nodes),
            'relation_count': len(self.relations),
            'faiss_initialized': self.faiss_initialized,
            'rebuild_needed': self.rebuild_needed
        }
        
        # Add operation timing statistics
        timing_stats = {}
        for op_name, timings in self.perf_metrics.items():
            if not timings:
                continue
                
            timing_stats[op_name] = {
                'avg': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'samples': len(timings)
            }
            
        stats['timing'] = timing_stats
        
        # Add component-specific stats
        if self.enable_batching:
            stats['batched_ops'] = self.batched_ops.get_performance_stats()
            
        if self.enable_incremental_updates:
            stats['incremental_updates'] = self.update_manager.get_update_stats()
            
        if self.enable_caching:
            stats['similarity_cache'] = self.cache_manager.get_cache_stats()
            
        return stats


# ---------------------------------------------------------
# EXAMPLE INTEGRATION WITH SEMANTIC TENSOR NETWORK
# ---------------------------------------------------------

# Example with type hints for replacing the original SemanticTensorNetwork implementation
"""
# In layer3_cognitive_boundary/core/semantic_tensor_network.py

# Import the optimized implementation
from asf.layer3_cognitive_boundary.optimizations.tensor_optimizations import SemanticTensorNetworkOptimized

# Override the original class
class SemanticTensorNetwork(SemanticTensorNetworkOptimized):
    # This inherits all optimized implementations
    pass
"""