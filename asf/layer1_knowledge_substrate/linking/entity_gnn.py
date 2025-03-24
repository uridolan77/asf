import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

from asf.__core.enums import PerceptualInputType

class EntityLinkingGNN:
    """
    Links entities across modalities using Graph Neural Networks.
    """
    def __init__(self, feature_dim=128, hidden_dim=64):
        # Graph Attention Network for entity linking
        self.gat = GATConv(feature_dim, hidden_dim, heads=4)
        self.output_layer = nn.Linear(hidden_dim * 4, feature_dim)
        
        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(10, hidden_dim)  # 10 relation types
        
        # Entity cache to avoid redundant processing
        self.entity_embedding_cache = {}
    
    def create_entity_graph(self, entities, relations):
        """
        Creates graph representation of entities and their relations
        Parameters:
        - entities: List of PerceptualEntity objects
        - relations: List of (source_idx, target_idx, relation_type) tuples
        """
        # Extract entity features
        x = torch.stack([torch.tensor(e.get_feature_vector(), dtype=torch.float32) 
                         for e in entities])
        
        # Create edge index and edge attributes
        edge_index = []
        edge_attr = []
        for source, target, rel_type in relations:
            # Add bidirectional edges for undirected graph
            edge_index.append([source, target])
            edge_index.append([target, source])
            # Edge attributes include relation type embedding
            edge_attr.append(rel_type)
            edge_attr.append(rel_type)
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_attr = torch.tensor(edge_attr)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def update_entity_embeddings(self, entities, relations):
        """
        Updates entity embeddings using graph neural network
        """
        # Create graph data
        graph_data = self.create_entity_graph(entities, relations)
        
        # Apply GAT layer
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        
        # Get relation embeddings
        rel_embeddings = self.relation_embeddings(edge_attr)
        
        # Apply GAT with relation-aware attention
        h = self.gat(x, edge_index, edge_attr=rel_embeddings)
        h = F.relu(h)
        
        # Final embeddings
        output = self.output_layer(h)
        
        # Update entity embeddings in cache
        for i, entity in enumerate(entities):
            self.entity_embedding_cache[entity.id] = output[i].detach()
        
        return output
    
    def get_entity_embedding(self, entity_id):
        """
        Retrieves cached embedding for an entity
        """
        return self.entity_embedding_cache.get(entity_id, None)
    
    def find_similar_entities(self, query_entity, all_entities, top_k=5):
        """
        Finds entities similar to query entity across all modalities
        """
        query_embedding = self.get_entity_embedding(query_entity.id)
        if query_embedding is None:
            return []
        
        # Compute similarities
        similarities = []
        for entity in all_entities:
            entity_embedding = self.get_entity_embedding(entity.id)
            if entity_embedding is not None:
                sim = torch.cosine_similarity(query_embedding, entity_embedding, dim=0)
                similarities.append((entity, sim.item()))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def suggest_cross_modal_links(self, entities, similarity_threshold=0.7):
        """
        Suggests potential cross-modal links between entities
        """
        # Group entities by modality
        entities_by_modality = {}
        for entity in entities:
            if entity.input_type not in entities_by_modality:
                entities_by_modality[entity.input_type] = []
            entities_by_modality[entity.input_type].append(entity)
        
        # Build graph of entities with existing cross-modal links
        existing_relations = []
        for i, entity_i in enumerate(entities):
            for j, entity_j in enumerate(entities):
                if i != j and entity_j.id in entity_i.cross_modal_links:
                    # Find relation type based on modality pair
                    rel_type = self._get_relation_type(entity_i.input_type, entity_j.input_type)
                    existing_relations.append((i, j, rel_type))
        
        # Update embeddings using existing relations
        self.update_entity_embeddings(entities, existing_relations)
        
        # Find potential new links across modalities
        suggested_links = []
        for mod1, entities1 in entities_by_modality.items():
            for mod2, entities2 in entities_by_modality.items():
                if mod1 != mod2:  # Only link across modalities
                    for entity1 in entities1:
                        for entity2 in entities2:
                            # Skip if already linked
                            if entity2.id in entity1.cross_modal_links:
                                continue
                            
                            # Get embeddings
                            emb1 = self.get_entity_embedding(entity1.id)
                            emb2 = self.get_entity_embedding(entity2.id)
                            
                            if emb1 is not None and emb2 is not None:
                                # Calculate similarity
                                similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
                                
                                # Suggest if above threshold
                                if similarity > similarity_threshold:
                                    suggested_links.append((entity1, entity2, similarity))
        
        # Sort by similarity
        suggested_links.sort(key=lambda x: x[2], reverse=True)
        return suggested_links
    
    def _get_relation_type(self, modality1, modality2):
        """
        Determine relation type based on modality pair
        Returns an integer representing the relation type
        """
        # Create a dictionary mapping modality pairs to relation types
        modality_pairs = {
            (PerceptualInputType.TEXT, PerceptualInputType.IMAGE): 0,
            (PerceptualInputType.IMAGE, PerceptualInputType.TEXT): 0,
            (PerceptualInputType.TEXT, PerceptualInputType.AUDIO): 1,
            (PerceptualInputType.AUDIO, PerceptualInputType.TEXT): 1,
            (PerceptualInputType.IMAGE, PerceptualInputType.AUDIO): 2,
            (PerceptualInputType.AUDIO, PerceptualInputType.IMAGE): 2,
            # Default relation type for other pairs
            "default": 9
        }
        
        # Get the relation type for this modality pair
        key = (modality1, modality2)
        return modality_pairs.get(key, modality_pairs["default"])
