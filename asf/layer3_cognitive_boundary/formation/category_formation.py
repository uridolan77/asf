import torch
import torch.nn.functional as F
import uuid
import time
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from asf.layer3_cognitive_boundary.core.semantic_node import SemanticNode
from asf.layer3_cognitive_boundary.core.semantic_relation import SemanticRelation
from asf.layer3_cognitive_boundary.enums import SemanticConfidenceState

class CategoryFormationSystem:
    """
    Forms and maintains categories with active inference capabilities.
    Implements Seth's principle of minimizing prediction error through actions.
    """
    def __init__(self, semantic_network):
        self.semantic_network = semantic_network
        self.category_formation_history = []
        self.logger = logging.getLogger("ASF.Layer3.CategoryFormation")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.category_predictions = {}  # Node set hash -> predicted categories
        self.category_errors = defaultdict(list)  # Input hash -> prediction errors
        self.category_precision = {}  # Input hash -> precision
        
    async def predict_categories(self, partial_nodes, method="similarity", params=None):
        params = params or {}
        node_set_hash = self._hash_node_set(partial_nodes.keys(), method)
        
        if node_set_hash in self.category_predictions:
            return self.category_predictions[node_set_hash]
        
        if method == "similarity":
            node_ids = list(partial_nodes.keys())
            
            embeddings_list = []
            valid_node_ids = []
            
            for node_id in node_ids:
                node = partial_nodes[node_id]
                if isinstance(node.embeddings, np.ndarray):
                    embeddings_list.append(torch.tensor(node.embeddings, dtype=torch.float32))
                    valid_node_ids.append(node_id)
                elif isinstance(node.embeddings, torch.Tensor):
                    embeddings_list.append(node.embeddings)
                    valid_node_ids.append(node_id)
            
            if not embeddings_list:
                return {"status": "error", "message": "No valid embeddings for prediction"}
            
            embeddings_tensor = torch.stack(embeddings_list).to(self.device)
            
            norms = torch.norm(embeddings_tensor, dim=1, keepdim=True)
            norms[norms == 0] = 1.0  # Avoid division by zero
            normalized_embeddings = embeddings_tensor / norms
            
            similarity_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t()).cpu().numpy()
            
            min_cluster_size = params.get('min_cluster_size', 3)
            similarity_threshold = params.get('similarity_threshold', 0.7)
            
            predicted_clusters = self._hierarchical_clustering(
                similarity_matrix, 
                min_cluster_size, 
                max_clusters=params.get('max_clusters', 10),
                similarity_threshold=similarity_threshold
            )
            
            predicted_categories = []
            
            for i, cluster in enumerate(predicted_clusters):
                cluster_nodes = [partial_nodes[valid_node_ids[idx]] for idx in cluster if idx < len(valid_node_ids)]
                
                if len(cluster_nodes) < min_cluster_size:
                    continue
                
                common_properties = self._find_common_properties(cluster_nodes)
                
                category_label = self._generate_category_label(common_properties, cluster_nodes)
                
                predicted_categories.append({
                    'id': f"predicted_category_{i}",
                    'label': category_label,
                    'size': len(cluster_nodes),
                    'members': [node.id for node in cluster_nodes],
                    'common_properties': common_properties,
                    'confidence': 0.7  # Default confidence for predictions
                })
            
            prediction_result = {
                "status": "success",
                "categories": predicted_categories,
                "total_nodes": len(partial_nodes),
                "categorized_nodes": sum(len(cat['members']) for cat in predicted_categories)
            }
            
            self.category_predictions[node_set_hash] = prediction_result
            
            return prediction_result
            
        elif method == "property":
            all_properties = set()
            for node in partial_nodes.values():
                all_properties.update(node.properties.keys())
                
            property_to_nodes = defaultdict(list)
            for node_id, node in partial_nodes.items():
                for prop in node.properties:
                    property_to_nodes[prop].append(node_id)
            
            predicted_categories = []
            
            for prop, members in property_to_nodes.items():
                if len(members) >= params.get('min_members', 3):
                    member_nodes = [partial_nodes[m] for m in members if m in partial_nodes]
                    common_props = self._find_common_properties(member_nodes)
                    
                    if len(common_props) >= params.get('min_shared_props', 2):
                        category_label = self._generate_category_label(common_props, member_nodes)
                        
                        predicted_categories.append({
                            'id': f"predicted_category_{prop}",
                            'label': category_label,
                            'size': len(member_nodes),
                            'members': [node.id for node in member_nodes],
                            'common_properties': common_props,
                            'key_property': prop,
                            'confidence': 0.7
                        })
            
            prediction_result = {
                "status": "success",
                "categories": predicted_categories,
                "total_nodes": len(partial_nodes),
                "categorized_nodes": sum(len(cat['members']) for cat in predicted_categories)
            }
            
            self.category_predictions[node_set_hash] = prediction_result
            
            return prediction_result
        
        else:
            return await self.predict_categories(partial_nodes, "similarity", params)
    
    async def refine_categories_via_active_inference(self, initial_categories, source_nodes=None):
        if not initial_categories or 'categories' not in initial_categories:
            return initial_categories
        
        if not source_nodes:
            return initial_categories
        
        categories = initial_categories['categories']
        
        initial_error = self._calculate_category_prediction_error(categories, source_nodes)
        
        candidates = self._generate_category_refinements(categories, source_nodes)
        
        best_categories = categories
        best_error = initial_error
        
        for candidate in candidates:
            error = self._calculate_category_prediction_error(candidate, source_nodes)
            
            if error < best_error:
                best_error = error
                best_categories = candidate
        
        if best_error < initial_error:
            refined_result = dict(initial_categories)
            refined_result['categories'] = best_categories
            refined_result['refinement'] = {
                'initial_error': initial_error,
                'refined_error': best_error,
                'improvement': initial_error - best_error
            }
            
            return refined_result
        
        return initial_categories
    
    def _calculate_category_prediction_error(self, categories, source_nodes):
        """Calculate prediction error for a set of categories."""
        # Collect category memberships
        node_categories = defaultdict(list)
        
        for cat in categories:
            for member_id in cat['members']:
                node_categories[member_id].append(cat['id'])
        
        # Calculate how well categories predict node similarities
        error = 0.0
        
        # For each pair of nodes, check if they should be in same categories
        node_ids = list(source_nodes.keys())
        
        for i, node1_id in enumerate(node_ids):
            node1 = source_nodes[node1_id]
            
            for j in range(i+1, len(node_ids)):
                node2_id = node_ids[j]
                node2 = source_nodes[node2_id]
                
                # Calculate actual similarity
                if isinstance(node1.embeddings, np.ndarray) and isinstance(node2.embeddings, np.ndarray):
                    actual_similarity = np.dot(node1.embeddings, node2.embeddings) / (
                        np.linalg.norm(node1.embeddings) * np.linalg.norm(node2.embeddings)
                    )
                else:
                    # Fallback
                    actual_similarity = 0.5
                
                # Calculate predicted similarity based on shared categories
                cat1 = set(node_categories[node1_id])
                cat2 = set(node_categories[node2_id])
                
                shared_categories = len(cat1.intersection(cat2))
                total_categories = len(cat1.union(cat2))
                
                if total_categories > 0:
                    predicted_similarity = shared_categories / total_categories
                else:
                    predicted_similarity = 0.0
                
                # Add squared error
                pair_error = (predicted_similarity - actual_similarity) ** 2
                error += pair_error
        
        # Normalize by number of pairs
        n_pairs = (len(node_ids) * (len(node_ids) - 1)) // 2
        if n_pairs > 0:
            error /= n_pairs
        
        return error
    
    def _generate_category_refinements(self, categories, source_nodes):
        """Generate candidate category refinements."""
        candidates = []
        
        node_categories = defaultdict(list)
        category_nodes = {}
        
        for i, cat in enumerate(categories):
            category_nodes[i] = cat['members']
            for member_id in cat['members']:
                node_categories[member_id].append(i)
        
        for node_id, cat_indices in node_categories.items():
            if len(cat_indices) == 1 and node_id in source_nodes:
                current_cat_idx = cat_indices[0]
                
                node = source_nodes[node_id]
                best_similarity = -1
                best_target_idx = -1
                
                for target_idx, members in category_nodes.items():
                    if target_idx == current_cat_idx:
                        continue
                    
                    similarities = []
                    for member_id in members:
                        if member_id in source_nodes:
                            member = source_nodes[member_id]
                            if isinstance(node.embeddings, np.ndarray) and isinstance(member.embeddings, np.ndarray):
                                sim = np.dot(node.embeddings, member.embeddings) / (
                                    np.linalg.norm(node.embeddings) * np.linalg.norm(member.embeddings)
                                )
                                similarities.append(sim)
                    
                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        if avg_similarity > best_similarity:
                            best_similarity = avg_similarity
                            best_target_idx = target_idx
                
                if best_similarity > 0.5:  # Only if reasonably similar
                    candidate = []
                    
                    for i, cat in enumerate(categories):
                        new_cat = dict(cat)
                        
                        if i == current_cat_idx:
                            new_cat['members'] = [m for m in cat['members'] if m != node_id]
                            new_cat['size'] = len(new_cat['members'])
                        elif i == best_target_idx:
                            new_cat['members'] = cat['members'] + [node_id]
                            new_cat['size'] = len(new_cat['members'])
                        
                        candidate.append(new_cat)
                    
                    candidates.append(candidate)
        
        for i in range(len(categories)):
            for j in range(i+1, len(categories)):
                cat1 = categories[i]
                cat2 = categories[j]
                
                members1 = set(cat1['members'])
                members2 = set(cat2['members'])
                
                overlap = len(members1.intersection(members2))
                
                if overlap > 0 or self._are_categories_similar(cat1, cat2):
                    merged_members = list(members1.union(members2))
                    
                    common_properties = {}
                    for prop in cat1['common_properties']:
                        if prop in cat2['common_properties'] and cat1['common_properties'][prop] == cat2['common_properties'][prop]:
                            common_properties[prop] = cat1['common_properties'][prop]
                    
                    merged_cat = {
                        'id': f"merged_{cat1['id']}_{cat2['id']}",
                        'label': f"Merged: {cat1['label']} + {cat2['label']}",
                        'size': len(merged_members),
                        'members': merged_members,
                        'common_properties': common_properties
                    }
                    
                    candidate = []
                    for k, cat in enumerate(categories):
                        if k != i and k != j:
                            candidate.append(cat)
                    
                    candidate.append(merged_cat)
                    candidates.append(candidate)
        
        return candidates
    
    def _are_categories_similar(self, cat1, cat2):
        """Check if two categories are similar enough to be merged."""
        # Check for similar properties
        props1 = set(cat1['common_properties'].keys())
        props2 = set(cat2['common_properties'].keys())
        
        if not props1 or not props2:
            return False
        
        # Check property overlap
        overlap = props1.intersection(props2)
        if len(overlap) >= min(2, min(len(props1), len(props2))):
            return True
        
        # Check for similar labels
        label1 = cat1['label'].lower()
        label2 = cat2['label'].lower()
        
        # Simple string similarity
        words1 = set(label1.split())
        words2 = set(label2.split())
        
        word_overlap = words1.intersection(words2)
        if len(word_overlap) >= min(2, min(len(words1), len(words2))):
            return True
        
        return False
    
    def _hash_node_set(self, node_ids, method):
        """Create a stable hash for node sets."""
        sorted_ids = sorted(str(nid) for nid in node_ids)
        return hash((tuple(sorted_ids), method))
