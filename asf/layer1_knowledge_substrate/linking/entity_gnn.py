import asyncio
import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from pydantic import BaseModel, Field, validator

# Assuming asf.__core.enums, ChronographMiddleware, ChronoGnosisLayer
# from asf.__core.enums import PerceptualInputType
# from chronograph_middleware_layer import ChronographMiddleware
# from chronograph_gnosis_layer import ChronoGnosisLayer

# Mocked for standalone execution
class PerceptualInputType(str):  # Mock
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class ChronographMiddleware:  # Mock
    async def get_entity(self, entity_id: str, include_history: bool = False) -> Optional[Dict]:
        print(f"Mock Chronograph: get_entity({entity_id}, {include_history})")
        return {"id": entity_id, "features": {}, "input_type": PerceptualInputType.TEXT, "cross_modal_links": []} # Added input type

    async def record_entity_state(self, entity_id: str, state_data: Dict, confidence: float):
        print(f"Mock Chronograph: record({entity_id}, {state_data})")

    async def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, float]]:
      #returns neighbors, relationship type, and strength.
      print(f"MOCK: Getting neighbors for {entity_id}")
      return []

    async def update_entity(self, entity_id:str, updates:Dict):
      print(f"Mock Chronograph: update_entity({entity_id}, {updates})")
    async def get_entity_confidence(self, entity_id:str) -> Optional[Dict]:
      print(f"MOCK: get_entity_confidence({entity_id})")
      return None # Mock return
    async def update_entity_confidence(self, entity_id:str, alpha:float, beta:float, last_updated:datetime.datetime):
      print(f"MOCK: update_entity_confidence({entity_id}, {alpha}, {beta}, {last_updated})")
    async def get_all_entities(self) -> List[Dict]:
        print("MOCK: get_all_entities")
        # Return some mock entities
        return [
          {"id": "entity_1", "input_type": PerceptualInputType.TEXT, "features": {"embedding": [0.1]*128}, "cross_modal_links": ["entity_2"]},
          {"id": "entity_2", "input_type": PerceptualInputType.IMAGE, "features": {"embedding": [0.2]*128}, "cross_modal_links": ["entity_1"]},
          {"id": "entity_3", "input_type": PerceptualInputType.AUDIO, "features": {"embedding": [0.3]*128}, "cross_modal_links": []},
          {"id": "entity_4", "input_type": PerceptualInputType.TEXT, "features": {"embedding": [0.4]*128}, "cross_modal_links": []},
        ]
    async def get_entity_relations(self, entity_id:str) -> List[Tuple[str, int]]: # (target_id, relation_type)
      print(f"MOCK: get_entity_relations({entity_id})")
      if entity_id == "entity_1":
        return [("entity_2", 0)] # Example relation
      return []


    async def add_entity_link(self, source_entity_id: str, target_entity_id: str, relation_type: int, similarity:float):
        print(f"Mock Chronograph: add_cross_modal_link({source_entity_id}, {target_entity_id}, {relation_type}, {similarity})")
    async def update_entity_embedding(self, entity_id:str, embedding:List[float]):
        print(f"MOCK: update_entity_embedding({entity_id}, {embedding[:3]}...)") # Print a slice

class ChronoGnosisLayer:  # Mock
    async def generate_embeddings(self, entity_ids: List[str]) -> Dict[str, Dict]:
      return {entity_id: {"embedding": [0.1 + i*0.01]*128} for i, entity_id in enumerate(entity_ids)} # 128 dim, unique
    async def generate_context_embedding(self, context:Dict):
      print(f"MOCK: generate_context_embedding({context})")
      return [0.4,0.5,0.6] # Mock
    async def predict_relevance_from_embedding(self, entity_embedding:List[float], context_embedding:List[float]) -> float:
        print(f"MOCK: predict_relevance_from_embedding({entity_embedding}, {context_embedding})")
        return 0.7 # Mock prediction

class PerceptualEntity:  # Mock (but more complete)
    def __init__(self, entity_id: str, input_type: str, features: Dict, cross_modal_links:Optional[List]=None):
        self.id = entity_id
        self.input_type = input_type
        self.features = features # Dictionary of features
        self.cross_modal_links = cross_modal_links if cross_modal_links is not None else []

    def get_feature_vector(self) -> List[float]:
      return self.features.get("embedding", [])

    def __repr__(self):
        return f"PerceptualEntity(id={self.id}, type={self.input_type}, links={self.cross_modal_links})"

class EntityLinkingGNNConfig(BaseModel):
    feature_dim: int = Field(128, description="Dimension of entity feature vectors.")
    hidden_dim: int = Field(64, description="Dimension of hidden layers in the GNN.")
    num_relation_types: int = Field(10, description="Number of distinct relation types.")
    similarity_threshold: float = Field(
        0.7, description="Minimum similarity for suggesting a link."
    )
    gat_heads: int = Field(4, description="Number of attention heads in GATConv.")

class EntityLinkingGNN:
    """
    Links entities across modalities using Graph Neural Networks.
    """

    def __init__(self, chronograph: ChronographMiddleware, gnosis: ChronoGnosisLayer, config: Optional[EntityLinkingGNNConfig] = None):
        self.config = config or EntityLinkingGNNConfig()
        self.chronograph = chronograph
        self.gnosis = gnosis
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Use GPU if available

        # Graph Attention Network
        self.gat = GATConv(self.config.feature_dim, self.config.hidden_dim, heads=self.config.gat_heads).to(self.device)
        self.output_layer = nn.Linear(self.config.hidden_dim * self.config.gat_heads, self.config.feature_dim).to(self.device)

        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(self.config.num_relation_types, self.config.hidden_dim).to(self.device)

        # Entity cache (stores tensors)
        self.entity_embedding_cache: Dict[str, torch.Tensor] = {}

    async def _create_entity_graph(self, entities: List[PerceptualEntity], relations: List[Tuple[int, int, int]]) -> Data:
      """Creates a PyTorch Geometric Data object representing the entity graph."""
      # Create mapping of entity ID to index
      entity_id_to_index = {entity.id: i for i, entity in enumerate(entities)}

      # Extract entity features (using cached embeddings if available)
      x = []
      for entity in entities:
          embedding = self.entity_embedding_cache.get(entity.id)
          if embedding is None:
              embedding = torch.tensor(entity.get_feature_vector(), dtype=torch.float32, device=self.device)
          else:
              embedding = embedding.to(self.device)  # Ensure correct device
          x.append(embedding)
      x = torch.stack(x)

      # Create edge index and edge attributes
      edge_index = []
      edge_attr = []
      for source_id, target_id, rel_type in relations:
          if source_id in entity_id_to_index and target_id in entity_id_to_index: #Handle missing
            source_index = entity_id_to_index[source_id]
            target_index = entity_id_to_index[target_id]
            # Bidirectional edges
            edge_index.append([source_index, target_index])
            edge_index.append([target_index, source_index])
            edge_attr.append(rel_type)
            edge_attr.append(rel_type)  # Duplicate for bidirectional

      edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t().contiguous()
      edge_attr = torch.tensor(edge_attr, dtype=torch.long, device=self.device)

      return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    async def _update_entity_embeddings(self, entities: List[PerceptualEntity], relations:  List[Tuple[str, str, int]]):
        """Updates entity embeddings using the GNN."""

        # Create graph
        graph_data = await self._create_entity_graph(entities, relations)

        # Apply GAT layer
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        rel_embeddings = self.relation_embeddings(edge_attr)
        h = self.gat(x, edge_index, edge_attr=rel_embeddings)
        h = F.relu(h)
        output = self.output_layer(h)

        # Update entity embeddings in cache and database
        for i, entity in enumerate(entities):
            self.entity_embedding_cache[entity.id] = output[i].detach().cpu() # Move back to CPU
            await self.chronograph.update_entity_embedding(entity.id, output[i].detach().cpu().tolist()) #Persist

    async def get_entity_embedding(self, entity_id: str) -> Optional[torch.Tensor]:
        """Retrieves the cached embedding for an entity, or loads from db"""
        if entity_id in self.entity_embedding_cache:
            return self.entity_embedding_cache[entity_id]

        # If not in cache, try to get from database
        entity_data = await self.chronograph.get_entity(entity_id)
        if entity_data and "embedding" in entity_data:
            embedding = torch.tensor(entity_data["embedding"], dtype=torch.float32)
            self.entity_embedding_cache[entity_id] = embedding # Cache it
            return embedding
        return None


    async def find_similar_entities(
        self, query_entity_id: str, all_entities: List[PerceptualEntity], top_k: int = 5
    ) -> List[Tuple[PerceptualEntity, float]]:
        """Finds entities similar to the query entity."""

        query_embedding = await self.get_entity_embedding(query_entity_id)
        if query_embedding is None:
            return []

        similarities = []
        for entity in all_entities:
            entity_embedding = await self.get_entity_embedding(entity.id)
            if entity_embedding is not None and entity.id != query_entity_id: # Don't compare to self
                sim = torch.cosine_similarity(query_embedding, entity_embedding, dim=0).item()
                similarities.append((entity, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    async def _get_relation_type(self, modality1: str, modality2: str) -> int:
        """Determine relation type based on modality pair."""
        #Simplified for the mock
        modality_pairs = {
            (PerceptualInputType.TEXT, PerceptualInputType.IMAGE): 0,
            (PerceptualInputType.IMAGE, PerceptualInputType.TEXT): 0,
            (PerceptualInputType.TEXT, PerceptualInputType.AUDIO): 1,
            (PerceptualInputType.AUDIO, PerceptualInputType.TEXT): 1,
            (PerceptualInputType.IMAGE, PerceptualInputType.AUDIO): 2,
            (PerceptualInputType.AUDIO, PerceptualInputType.IMAGE): 2,
            # Default
            "default": 9,
        }
        key = (modality1, modality2)
        return modality_pairs.get(key, modality_pairs["default"])


    async def suggest_cross_modal_links(self, similarity_threshold: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """Suggests potential cross-modal links between entities."""

        if similarity_threshold is None:
          similarity_threshold = self.config.similarity_threshold

        all_entities_data = await self.chronograph.get_all_entities()
        # Convert to PerceptualEntity objects, fetching embeddings if needed.
        all_entities = []
        for entity_data in all_entities_data:
            entity = PerceptualEntity(
                entity_data["id"],
                entity_data["input_type"],
                entity_data["features"],
                entity_data.get("cross_modal_links", []) # Handle potential missing key
            )
            # Pre-populate the cache to avoid redundant database calls
            if "embedding" in entity_data:
                self.entity_embedding_cache[entity.id] = torch.tensor(entity_data["embedding"], dtype=torch.float32)

            all_entities.append(entity)



        # Group entities by modality
        entities_by_modality: Dict[str, List[PerceptualEntity]] = {}
        for entity in all_entities:
            if entity.input_type not in entities_by_modality:
                entities_by_modality[entity.input_type] = []
            entities_by_modality[entity.input_type].append(entity)

        # Build graph of entities with existing cross-modal links
        existing_relations = []
        for i, entity_i in enumerate(all_entities):
            for j, entity_j in enumerate(all_entities):
                if i != j and entity_j.id in entity_i.cross_modal_links:
                    rel_type = await self._get_relation_type(
                        entity_i.input_type, entity_j.input_type
                    )
                    existing_relations.append((entity_i.id, entity_j.id, rel_type))

        # Update embeddings *once* using existing relations
        if existing_relations:  # Only update if there are existing relations
           await self._update_entity_embeddings(all_entities, existing_relations)

        # Find potential new links across modalities
        suggested_links: List[Tuple[str, str, float]] = []
        for mod1, entities1 in entities_by_modality.items():
            for mod2, entities2 in entities_by_modality.items():
                if mod1 != mod2:  # Only link across modalities
                    for entity1 in entities1:
                        for entity2 in entities2:
                            if entity2.id not in entity1.cross_modal_links:
                                emb1 = await self.get_entity_embedding(entity1.id)
                                emb2 = await self.get_entity_embedding(entity2.id)
                                if emb1 is not None and emb2 is not None:
                                    similarity = (
                                        torch.cosine_similarity(emb1, emb2, dim=0).item()
                                    )
                                    if similarity > similarity_threshold:
                                        rel_type = await self._get_relation_type(entity1.input_type, entity2.input_type)
                                        suggested_links.append((entity1.id, entity2.id, similarity))
                                        # Add to Chronograph *immediately*
                                        await self.chronograph.add_entity_link(entity1.id, entity2.id, rel_type, similarity)


        suggested_links.sort(key=lambda x: x[2], reverse=True)  # Sort by similarity
        return suggested_links

async def main():
    # Initialize
    chronograph = ChronographMiddleware()
    gnosis = ChronoGnosisLayer()
    config = EntityLinkingGNNConfig()
    linker = EntityLinkingGNN(chronograph, gnosis, config)


    # Suggest links
    suggested_links = await linker.suggest_cross_modal_links()
    print("\nSuggested Cross-Modal Links:")
    for link in suggested_links:
        print(f"  {link[0]} <--> {link[1]} (Similarity: {link[2]:.4f})")

    # Find similar entities for entity_1
    all_entities_data = await chronograph.get_all_entities()
    all_entities = [PerceptualEntity(e["id"], e["input_type"], e["features"], e.get("cross_modal_links",[])) for e in all_entities_data]

    similar_entities = await linker.find_similar_entities("entity_1", all_entities)
    print("\nEntities Similar to entity_1:")
    for entity, similarity in similar_entities:
        print(f"  {entity.id} (Similarity: {similarity:.4f})")


if __name__ == "__main__":
    asyncio.run(main())