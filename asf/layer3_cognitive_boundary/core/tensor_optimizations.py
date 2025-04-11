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
        
        self.similarity_batch_size = self.config.get('similarity_batch_size', 128)
        self.embedding_batch_size = self.config.get('embedding_batch_size', 64)
        self.relation_batch_size = self.config.get('relation_batch_size', 256)
        
        self.operation_times = defaultdict(list)
        self.batch_counts = defaultdict(int)
        
        self.max_history = 100
        
        self.adaptive_batch_sizing = self.config.get('adaptive_batch_sizing', True)
        self.min_batch_size = self.config.get('min_batch_size', 16)
        self.max_batch_size = self.config.get('max_batch_size', 512)
    
    async def batch_process_similarities(self, query_embeddings, target_embeddings=None, 
                                         k=5, threshold=0.7):
        start_time = time.time()
        
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = torch.tensor(query_embeddings, dtype=torch.float32).to(self.device)
        
        if target_embeddings is None:
            if self.tensor_network.node_embeddings is None:
                return [[]]
            target_embeddings = self.tensor_network.node_embeddings
        elif isinstance(target_embeddings, np.ndarray):
            target_embeddings = torch.tensor(target_embeddings, dtype=torch.float32).to(self.device)
        
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        
        num_queries = query_embeddings.shape[0]
        
        if self.adaptive_batch_sizing:
            embed_dim = query_embeddings.shape[1]
            if torch.cuda.is_available() and self.device.type == 'cuda':
                free_memory = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
                memory_per_query = 4 * embed_dim * target_embeddings.shape[0]  # 4 bytes per float32
                adjusted_batch_size = max(self.min_batch_size, min(self.max_batch_size, 
                                                                   int(free_memory * 0.7 / memory_per_query)))
                batch_size = min(adjusted_batch_size, self.similarity_batch_size)
            else:
                batch_size = min(self.similarity_batch_size, 64)
        else:
            batch_size = self.similarity_batch_size
        
        batch_size = min(batch_size, num_queries)
        
        all_results = []
        
        for i in range(0, num_queries, batch_size):
            end_idx = min(i + batch_size, num_queries)
            query_batch = query_embeddings[i:end_idx]
            
            similarities = torch.mm(query_batch, target_embeddings.t())
            
            for j in range(query_batch.shape[0]):
                query_similarities = similarities[j]
                
                values, indices = torch.topk(query_similarities, min(k, len(query_similarities)))
                
                results = []
                for idx, val in zip(indices.cpu().numpy(), values.cpu().numpy()):
                    if val >= threshold:
                        results.append((int(idx), float(val)))
                
                all_results.append(results)
        
        elapsed = time.time() - start_time
        self.operation_times['batch_similarities'].append(elapsed)
        self.batch_counts['batch_similarities'] += 1
        
        if len(self.operation_times['batch_similarities']) > self.max_history:
            self.operation_times['batch_similarities'] = self.operation_times['batch_similarities'][-self.max_history:]
        
        self.logger.debug(f"Batch processed {num_queries} queries in {elapsed:.3f}s (batch_size={batch_size})")
        
        return all_results
    
    async def batch_update_embeddings(self, nodes_with_embeddings):
        start_time = time.time()
        
        if not nodes_with_embeddings:
            return 0
        
        batch_size = self.embedding_batch_size
        total_updates = len(nodes_with_embeddings)
        successful_updates = 0
        
        for i in range(0, total_updates, batch_size):
            end_idx = min(i + batch_size, total_updates)
            batch = nodes_with_embeddings[i:end_idx]
            
            batch_updates = []
            
            for node_id, embeddings in batch:
                try:
                    if isinstance(embeddings, list):
                        embeddings = np.array(embeddings, dtype=np.float32)
                    
                    if node_id in self.tensor_network.node_index:
                        node_idx = self.tensor_network.node_index[node_id]
                        batch_updates.append((node_idx, embeddings))
                        
                        node = self.tensor_network.nodes.get(node_id)
                        if node:
                            node.embeddings = embeddings
                            if isinstance(embeddings, np.ndarray):
                                node.tensor_representation = torch.tensor(
                                    embeddings, dtype=torch.float32
                                ).to(self.device)
                            elif isinstance(embeddings, torch.Tensor):
                                node.tensor_representation = embeddings.to(self.device)
                            
                            successful_updates += 1
                except Exception as e:
                    self.logger.warning(f"Error updating embeddings for node {node_id}: {str(e)}")
            
            self.tensor_network.embedding_buffer.extend(batch_updates)
            
            if len(self.tensor_network.embedding_buffer) >= self.tensor_network.batch_size:
                await self.tensor_network._update_tensors()
        
        elapsed = time.time() - start_time
        self.operation_times['batch_embeddings'].append(elapsed)
        self.batch_counts['batch_embeddings'] += 1
        
        self.logger.debug(f"Batch updated {successful_updates}/{total_updates} embeddings in {elapsed:.3f}s")
        
        return successful_updates
    
    async def batch_process_relations(self, relation_operations):
        start_time = time.time()
        
        if not relation_operations:
            return {'success': 0, 'failure': 0, 'total': 0}
        
        batch_size = self.relation_batch_size
        total_operations = len(relation_operations)
        success_count = 0
        failure_count = 0
        
        for i in range(0, total_operations, batch_size):
            end_idx = min(i + batch_size, total_operations)
            batch = relation_operations[i:end_idx]
            
            for operation in batch:
                try:
                    op_type = operation.get('type')
                    
                    if op_type == 'add_relation':
                        relation = self._create_relation_from_operation(operation)
                        if relation:
                            await self.tensor_network.add_relation(relation)
                            success_count += 1
                        else:
                            failure_count += 1
                            
                    elif op_type == 'update_relation':
                        relation_id = operation.get('relation_id')
                        if not relation_id or relation_id not in self.tensor_network.relations:
                            failure_count += 1
                            continue
                            
                        relation = self.tensor_network.relations[relation_id]
                        
                        if 'weight' in operation:
                            relation.weight = operation['weight']
                        if 'properties' in operation:
                            relation.properties.update(operation['properties'])
                        if 'confidence' in operation:
                            relation.confidence = operation['confidence']
                        
                        relation.temporal_metadata.update_modification()
                        
                        await self.tensor_network.add_relation(relation)
                        success_count += 1
                        
                    elif op_type == 'delete_relation':
                        failure_count += 1
                        
                    else:
                        failure_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error processing relation operation: {str(e)}")
                    failure_count += 1
        
        if success_count > 0:
            self.tensor_network.rebuild_needed = True
        
        elapsed = time.time() - start_time
        self.operation_times['batch_relations'].append(elapsed)
        self.batch_counts['batch_relations'] += 1
        
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
    Manages incremental updates to tensor representations.
    Tracks changes and applies them efficiently without full rebuilds.
        Initialize incremental update manager.
        
        Args:
            tensor_network: Parent SemanticTensorNetwork instance
            config: Configuration dictionary
        Track a node change for incremental updates.
        
        Args:
            node_id: ID of the changed node
        Track a relation change for incremental updates.
        
        Args:
            relation_id: ID of the changed relation
            relation_type: Type of the relation (optional)
        Track a node deletion for cleanup.
        
        Args:
            node_id: ID of the deleted node
        Track a relation deletion for cleanup.
        
        Args:
            relation_id: ID of the deleted relation
            relation_type: Type of the relation (optional)
        if self.change_count_since_rebuild % self.rebuild_check_frequency != 0:
            return
            
        node_count = len(self.tensor_network.nodes)
        relation_count = len(self.tensor_network.relations)
        
        if node_count == 0 or relation_count == 0:
            return
            
        node_change_ratio = len(self.pending_node_changes) / node_count
        relation_change_ratio = len(self.pending_relation_changes) / relation_count
        
        if node_change_ratio > self.rebuild_threshold or relation_change_ratio > self.rebuild_threshold:
            self.logger.info(f"Change ratio exceeded rebuild threshold (nodes: {node_change_ratio:.3f}, " + 
                          f"relations: {relation_change_ratio:.3f}), scheduling full rebuild")
            self.tensor_network.rebuild_needed = True
    
    async def apply_incremental_updates(self):
        start_time = time.time()
        
        if not self.pending_node_changes and not self.pending_relation_changes and not self.deleted_nodes and not self.deleted_relations:
            return True
            
        if self.tensor_network.rebuild_needed:
            self.logger.info("Full rebuild needed, skipping incremental updates")
            return await self._perform_full_rebuild()
            
        self.logger.info(f"Applying incremental updates: {len(self.pending_node_changes)} nodes, " + 
                      f"{len(self.pending_relation_changes)} relations, " +
                      f"{len(self.deleted_nodes)} deleted nodes, " +
                      f"{len(self.deleted_relations)} deleted relations")
        
        try:
            if self.tensor_network.node_embeddings is not None and self.pending_node_changes:
                await self._update_node_embeddings()
                
            if self.tensor_network.faiss_initialized and self.pending_node_changes:
                await self._update_faiss_index()
                
            if self.pending_relation_changes or self.deleted_relations:
                await self._update_relation_tensors()
                
            if self.deleted_nodes or self.deleted_relations:
                await self._process_deletions()
                
            self.pending_node_changes.clear()
            self.pending_relation_changes.clear()
            self.pending_relation_type_changes.clear()
            self.deleted_nodes.clear()
            self.deleted_relations.clear()
            
            self.change_count_since_rebuild = 0
            
            elapsed = time.time() - start_time
            self.update_times.append(elapsed)
            if len(self.update_times) > self.max_history:
                self.update_times = self.update_times[-self.max_history:]
                
            self.logger.info(f"Incremental updates completed in {elapsed:.3f}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error during incremental updates: {str(e)}", exc_info=True)
            
            self.logger.info("Falling back to full rebuild after incremental update failure")
            self.tensor_network.rebuild_needed = True
            return await self._perform_full_rebuild()
    
    async def _update_node_embeddings(self):
        if not self.tensor_network.faiss_initialized or self.tensor_network.faiss_index is None:
            return
            
        supports_direct_update = hasattr(self.tensor_network.faiss_index, 'remove_ids') and \
                             hasattr(self.tensor_network.faiss_index, 'add_with_ids')
        
        if not supports_direct_update:
            self.logger.info("FAISS index doesn't support direct updates, performing full index rebuild")
            await self.tensor_network._initialize_faiss_index()
            return
            
        for node_id in self.pending_node_changes:
            if node_id in self.deleted_nodes:
                continue  # Skip deleted nodes
                
            node = self.tensor_network.nodes.get(node_id)
            if not node or not hasattr(node, 'embeddings') or node.embeddings is None:
                continue
                
            node_idx = self.tensor_network.node_index.get(node_id)
            if node_idx is None:
                continue
                
            if isinstance(node.embeddings, np.ndarray):
                emb = node.embeddings
            elif isinstance(node.embeddings, torch.Tensor):
                emb = node.embeddings.cpu().detach().numpy()
            else:
                continue
                
            if len(emb.shape) == 1:
                emb = emb.reshape(1, -1)
                
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
            ids = np.array([node_idx], dtype=np.int64)
            
            try:
                self.tensor_network.faiss_index.remove_ids(ids)
                self.tensor_network.faiss_index.add_with_ids(emb, ids)
            except Exception as e:
                self.logger.warning(f"Error updating FAISS index for node {node_id}: {str(e)}")
                self.tensor_network.rebuild_needed = True
    
    async def _update_relation_tensors(self):
        rel_ids = self.tensor_network.relation_index['type'].get(rel_type, [])
        
        node_count = self.tensor_network.node_count
        
        indices = []
        values = []
        
        for rel_id in rel_ids:
            rel = self.tensor_network.relations.get(rel_id)
            if rel is None or rel_id in self.deleted_relations:
                continue
                
            source_idx = self.tensor_network.node_index.get(rel.source_id)
            target_idx = self.tensor_network.node_index.get(rel.target_id)
            
            if source_idx is None or target_idx is None:
                continue
                
            indices.append([source_idx, target_idx])
            values.append(rel.weight)
            
            if rel.bidirectional:
                indices.append([target_idx, source_idx])
                values.append(rel.weight)
        
        if not indices:
            if rel_type in self.tensor_network.adjacency_tensors:
                del self.tensor_network.adjacency_tensors[rel_type]
            return
            
        try:
            indices_tensor = torch.tensor(indices, dtype=torch.long).t().to(self.tensor_network.device)
            values_tensor = torch.tensor(values, dtype=torch.float32).to(self.tensor_network.device)
            
            sparse_tensor = torch.sparse_coo_tensor(
                indices_tensor, 
                values_tensor,
                (node_count, node_count),
                device=self.tensor_network.device
            )
            
            self.tensor_network.adjacency_tensors[rel_type] = sparse_tensor
        except Exception as e:
            self.logger.error(f"Error rebuilding tensor for relation type '{rel_type}': {str(e)}")
            self.tensor_network.rebuild_needed = True
    
    async def _rebuild_combined_tensor(self):
        for node_id in self.deleted_nodes:
            if node_id not in self.tensor_network.nodes:
                continue
                
            node_idx = self.tensor_network.node_index.pop(node_id, None)
            self.tensor_network.nodes.pop(node_id, None)
            
            if node_idx is not None:
                if node_idx in self.tensor_network.index_to_node:
                    self.tensor_network.index_to_node.pop(node_idx)
                    
            self.tensor_network.rebuild_needed = True
            
        for rel_id in self.deleted_relations:
            if rel_id not in self.tensor_network.relations:
                continue
                
            rel = self.tensor_network.relations.get(rel_id)
            rel_type = rel.relation_type if rel else None
            
            self.tensor_network.relations.pop(rel_id, None)
            
            if rel and rel.source_id in self.tensor_network.relation_index['source']:
                if rel_id in self.tensor_network.relation_index['source'][rel.source_id]:
                    self.tensor_network.relation_index['source'][rel.source_id].remove(rel_id)
                    
            if rel and rel.target_id in self.tensor_network.relation_index['target']:
                if rel_id in self.tensor_network.relation_index['target'][rel.target_id]:
                    self.tensor_network.relation_index['target'][rel.target_id].remove(rel_id)
                    
            if rel_type and rel_type in self.tensor_network.relation_index['type']:
                if rel_id in self.tensor_network.relation_index['type'][rel_type]:
                    self.tensor_network.relation_index['type'][rel_type].remove(rel_id)
                    
            if rel_type:
                self.pending_relation_type_changes.add(rel_type)
    
    async def _perform_full_rebuild(self):
        Get statistics about incremental updates and rebuilds.
        
        Returns:
            Dict: Update statistics
    Caches similarity search results to avoid redundant computations.
    Implements an LRU (Least Recently Used) cache with time-based expiration.
        Initialize cache manager with configuration.
        
        Args:
            tensor_network: Parent SemanticTensorNetwork instance
            config: Configuration dictionary
        Retrieve cached similarity results if available.
        
        Args:
            node_id: Query node ID
            k: Number of similar nodes to return
            threshold: Minimum similarity threshold
            
        Returns:
            Tuple[bool, Optional[List]]: (cache_hit, results)
        Cache similarity search results.
        
        Args:
            node_id: Query node ID
            results: Similarity results to cache
            k: Number of similar nodes returned
            threshold: Similarity threshold used
            
        Returns:
            bool: True if cached, False if skipped
        Invalidate cache entries related to a specific node.
        
        Args:
            node_id: Node ID to invalidate
            
        Returns:
            int: Number of invalidated entries
        Clear the entire similarity cache.
        
        Returns:
            int: Number of cleared entries
        current_time = time.time()
        keys_to_remove = []
        
        for cache_key, expiration_time in self.expiration_times.items():
            if current_time > expiration_time:
                keys_to_remove.append(cache_key)
                
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
        Get most frequently queried nodes.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            List[Tuple[str, int]]: List of (node_id, query_count) pairs
    Enhanced version of SemanticTensorNetwork with integrated performance optimizations.
    Implements batched processing, incremental updates, and caching mechanisms.
        Initialize optimized semantic tensor network.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            device: Computation device (CPU/GPU)
            config: Configuration dictionary
        Add a node to the semantic network with optimized tensor updates.
        
        Args:
            node: Node to add
            update_tensors: Whether to update tensors
            
        Returns:
            str: Node ID
        Add a relation to the semantic network with optimized tensor updates.
        
        Args:
            relation: Relation to add
            
        Returns:
            str: Relation ID
        Add multiple nodes in a batch.
        
        Args:
            nodes: List of nodes to add
            
        Returns:
            Dict: Results with node IDs
        Add multiple relations in a batch.
        
        Args:
            relations: List of relations to add
            
        Returns:
            Dict: Results with relation IDs
        Find semantically similar nodes with caching optimization.
        
        Args:
            node_id: Node ID to find similar nodes for
            k: Number of similar nodes to return
            threshold: Minimum similarity threshold
            by_embedding: Whether to use embeddings
            
        Returns:
            List[Tuple[str, float]]: List of (node_id, similarity) pairs
        Find similar nodes for multiple query nodes in batch.
        
        Args:
            node_ids: List of node IDs to find similar nodes for
            k: Number of similar nodes to return per query
            threshold: Minimum similarity threshold
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Dictionary mapping node IDs to similarity results
        Apply pending updates to tensors and indices.
        
        Returns:
            bool: True if successful, False otherwise
        Clear all caches.
        
        Returns:
            Dict: Results with counts of cleared items
        Get comprehensive performance statistics.
        
        Returns:
            Dict: Performance statistics

from asf.layer3_cognitive_boundary.optimizations.tensor_optimizations import SemanticTensorNetworkOptimized

class SemanticTensorNetwork(SemanticTensorNetworkOptimized):
    pass