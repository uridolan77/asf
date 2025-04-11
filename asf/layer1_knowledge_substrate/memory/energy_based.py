import asyncio
import datetime
import hashlib
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator


class PerceptualInputType(str):  # Mock
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"

class ChronographMiddleware:  # Mock
    async def get_entity(self, entity_id: str, include_history: bool = False) -> Optional[Dict]:
        print(f"Mock Chronograph: get_entity({entity_id}, {include_history})")
        if entity_id in _mock_db:  # Check if entity exists in mock DB
            return _mock_db[entity_id]
        return None

    async def record_entity_state(self, entity_id: str, state_data: Dict, confidence: float):
        print(f"Mock Chronograph: record({entity_id}, {state_data})")

    async def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, float]]:
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
        return list(_mock_db.values()) # Return all entities

    async def get_entity_relations(self, entity_id:str) -> List[Tuple[str, int]]: # (target_id, relation_type)
      print(f"MOCK: get_entity_relations({entity_id})")
      if entity_id == "entity_1":
        return [("entity_2", 0)] # Example relation
      return []


    async def add_entity_link(self, source_entity_id: str, target_entity_id: str, relation_type: int, similarity:float):
        print(f"Mock Chronograph: add_cross_modal_link({source_entity_id}, {target_entity_id}, {relation_type}, {similarity})")
    async def update_entity_embedding(self, entity_id:str, embedding:List[float]):
        print(f"MOCK: update_entity_embedding({entity_id}, {embedding[:3]}...)") # Print a slice
    async def add_or_update_entity(self, entity_data: Dict):
        print(f"MOCK: add_or_update_entity({entity_data['id']})")
        _mock_db[entity_data["id"]] = entity_data # Update mock DB

    async def delete_entity(self, entity_id:str):
        print(f"MOCK: delete_entity({entity_id})")
        if entity_id in _mock_db:
            del _mock_db[entity_id]


class ChronoGnosisLayer:  # Mock
    async def generate_embeddings(self, entity_ids: List[str]) -> Dict[str, Dict]:
      return {entity_id: {"embedding": [0.1 + i*0.01]*128} for i, entity_id in enumerate(entity_ids)} # 128 dim, unique
    async def generate_context_embedding(self, context:Dict):
      print(f"MOCK: generate_context_embedding({context})")
      return [0.4, 0.5, 0.6] * 43  # Example 129-dim embedding (for consistency).
    async def predict_relevance_from_embedding(self, entity_embedding:List[float], context_embedding:List[float]) -> float:
        print(f"MOCK: predict_relevance_from_embedding({entity_embedding}, {context_embedding})")
        return 0.7 # Mock prediction

class Feature(BaseModel): # More robust Feature
    value: Any
    confidence: float = Field(1.0, ge=0.0, le=1.0)

class TemporalMetadata(BaseModel):
    creation_time: datetime.datetime = Field(default_factory=datetime.datetime.now)
    last_access_time: datetime.datetime = Field(default_factory=datetime.datetime.now)

    def get_temporal_relevance(self) -> float:
        """Calculate temporal relevance (higher value = more recent)."""
        time_since_last_access = (datetime.datetime.now() - self.last_access_time).total_seconds()
        # Exponential decay: relevance halves every 1 hour (3600 seconds)
        return np.exp(-time_since_last_access / 3600)

    def update_access_time(self):
         self.last_access_time = datetime.datetime.now()


class PerceptualEntity(BaseModel):  # More complete
    id: str
    input_type: str
    features: Dict[str, Feature]  # Use Feature class
    cross_modal_links: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    temporal_metadata: TemporalMetadata = Field(default_factory=TemporalMetadata)

    def __repr__(self):
        return f"PerceptualEntity(id={self.id}, type={self.input_type}, links={self.cross_modal_links})"

class EnergyBasedMemoryManagerConfig(BaseModel):
    capacity: int = Field(1000, description="Maximum number of entities to store.")
    decay_rate: float = Field(
        0.95, description="Rate at which energy decays on access.", ge=0.0, le=1.0
    )
    sampling_temp: float = Field(
        0.1, description="Temperature for softmax sampling.", gt=0.0
    )
    base_energy_weight: float = Field(0.3, description="Weight for base energy.")
    temporal_weight: float = Field(0.3, description="Weight for temporal factor.")
    complexity_weight: float = Field(0.2, description="Weight for complexity.")
    surprise_weight: float = Field(0.2, description="Weight for surprise.")
    learning_rate: float = Field(0.2, description="Learning rate for prediction updates.")
    max_features_for_complexity: int = Field(20, description="Used to normalize complexity")
    max_surprise_history: int = Field(100, description="Maximum number of surprise values")

class EnergyBasedMemoryManager:
    """
    Manages memory allocation and retention using principles inspired by Friston's Free Energy.
        entity_id = entity.id

        initial_energy = await self.calculate_energy(entity, context)

        if len(_mock_db) >= self.config.capacity and entity_id not in _mock_db:
            await self._free_capacity()

        self.energy_values[entity_id] = initial_energy
        await self.chronograph.add_or_update_entity(entity.dict()) # Persist

        if context is not None:
            await self._update_predictions(entity, context)

        return True
    async def calculate_energy(self, entity: PerceptualEntity, context: Optional[Dict] = None) -> float:

        context_id = self._get_context_id(context)

        if context_id not in self.predicted_patterns:
            return 1.0  # Maximum surprise if no predictions

        predictions = self.predicted_patterns[context_id]
        surprise_values = []

        for feature_name, feature in entity.features.items():
            if feature_name in predictions:
                predicted_value, prediction_confidence = predictions[feature_name]

                if isinstance(feature.value, (int, float, np.number)) and isinstance(predicted_value, (int, float, np.number)):
                    diff = abs(feature.value - predicted_value) / (1.0 + abs(predicted_value))  if predicted_value else abs(feature.value)
                    surprise = diff * prediction_confidence
                    surprise_values.append(surprise)

                elif isinstance(feature.value, (list, np.ndarray)) and isinstance(predicted_value, (list, np.ndarray)):

                    actual = np.array(feature.value)
                    predicted = np.array(predicted_value)
                    if actual.size > 0 and predicted.size>0:
                      actual = actual / (np.linalg.norm(actual) + 1e-8)  # Normalize + epsilon
                      predicted = predicted / (np.linalg.norm(predicted) + 1e-8)

                    if actual.shape == predicted.shape and actual.size > 0: # Check for valid comparison
                        similarity = np.dot(actual, predicted)
                        surprise = (1.0 - max(0, similarity)) * prediction_confidence # Clip similarity
                        surprise_values.append(surprise)



        if surprise_values:
            avg_surprise = sum(surprise_values) / len(surprise_values)

            self.surprise_history.append(avg_surprise)
            if len(self.surprise_history) > self.config.max_surprise_history:
                self.surprise_history = self.surprise_history[-self.config.max_surprise_history:]

            return avg_surprise
        return 0.5  # Default medium surprise


    async def _update_predictions(self, entity: PerceptualEntity, context: Dict):
        if not _mock_db: # Use the mock database
            return

        sorted_entities = sorted(
            self.energy_values.items(), key=lambda item: item[1], reverse=True
        )

        if sorted_entities:
            entity_id_to_remove, _ = sorted_entities[0]
            await self.chronograph.delete_entity(entity_id_to_remove)  # Use mock DB
            self.energy_values.pop(entity_id_to_remove, None) # Remove from energy values
            self.access_counts.pop(entity_id_to_remove, None) # Remove from access counts


    async def get_entity(self, entity_id: str) -> Optional[PerceptualEntity]:
        if not _mock_db:
            return []

        entity_ids = list(_mock_db.keys())
        energies = np.array([self.energy_values.get(eid, 1.0) for eid in entity_ids])

        if context is not None:
            context_id = self._get_context_id(context)
            if context_id in self.predicted_patterns:
                for i, entity_id in enumerate(entity_ids):
                    entity_data = await self.chronograph.get_entity(entity_id) #From DB
                    if entity_data: # Check if entity still exists
                        entity = PerceptualEntity(**entity_data)
                        surprise = await self._calculate_surprise(entity, context)
                        energies[i] *= (0.5 + 0.5 * surprise)

        inverted_energies = 1.0 / (energies + 1e-8)
        probabilities = np.exp(inverted_energies / self.config.sampling_temp)
        probabilities = probabilities / np.sum(probabilities)

        try:
            sampled_indices = np.random.choice(
                len(entity_ids), size=min(n, len(entity_ids)), replace=False, p=probabilities
            )
            sampled_entities = []
            for idx in sampled_indices:
              entity_data = await self.chronograph.get_entity(entity_ids[idx]) # Get from database
              if entity_data: #Ensure it still exists
                sampled_entities.append(PerceptualEntity(**entity_data))

            return sampled_entities
        except ValueError as e:
            logging.error(f"Error during sampling: {e}, Probabilities: {probabilities}, Energies: {energies}") # Log
            return [PerceptualEntity(**_mock_db[eid]) for eid in list(_mock_db.keys())[:n] if eid in _mock_db]


    async def forget_entity(self, entity_id: str) -> bool:
        for entity_id in list(_mock_db.keys()): # Iterate over a copy to allow deletion
            entity_data = await self.chronograph.get_entity(entity_id) # From database.
            if entity_data:
              entity = PerceptualEntity(**entity_data)
              self.energy_values[entity_id] = await self.calculate_energy(entity, context)

    async def get_memory_statistics(self) -> Dict:
        context_str = str(sorted(context.items()))
        context_hash = hashlib.sha256(context_str.encode("utf-8")).hexdigest()
        return context_hash
_mock_db: Dict[str, Dict] = {}

async def main():
    chronograph = ChronographMiddleware()
    gnosis = ChronoGnosisLayer()
    config = EnergyBasedMemoryManagerConfig()
    memory_manager = EnergyBasedMemoryManager(chronograph, gnosis, config)

    entity1 = PerceptualEntity(
        id="entity_1",
        input_type=PerceptualInputType.TEXT,
        features={"text_embedding": Feature(value=[0.1] * 128, confidence=0.95)},
        cross_modal_links=["entity_2"],
        confidence_score=0.9
    )
    entity2 = PerceptualEntity(
        id="entity_2",
        input_type=PerceptualInputType.IMAGE,
        features={"image_embedding": Feature(value=[0.2] * 128, confidence=0.8)},
        cross_modal_links=["entity_1"],
        confidence_score = 0.8
    )
    entity3 = PerceptualEntity(
        id="entity_3",
        input_type=PerceptualInputType.AUDIO,
        features={"audio_embedding": Feature(value=[0.3] * 128, confidence=0.7)},
        confidence_score = 0.7
    )
    await memory_manager.add_entity(entity1, context={"task": "classification", "domain": "images"})
    await memory_manager.add_entity(entity2, context={"task": "retrieval", "domain": "text"})
    await memory_manager.add_entity(entity3)

    retrieved_entity = await memory_manager.get_entity("entity_1")
    if retrieved_entity:
        print(f"\nRetrieved entity: {retrieved_entity}")

    sampled_entities = await memory_manager.sample_entities(context={"task": "classification", "domain": "images"}, n=2)
    print("\nSampled entities:")
    for entity in sampled_entities:
        print(f"  {entity}")

    stats = await memory_manager.get_memory_statistics()
    print("\nMemory Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    await memory_manager.forget_entity("entity_3")
    print(f"\nEntity 3 forgotten: {await memory_manager.get_entity('entity_3')}") # Should be None

    for i in range(4, 1005): # Go over the capacity limit
      new_entity = PerceptualEntity(
          id=f"entity_{i}",
          input_type=PerceptualInputType.TEXT,
          features={"text_embedding": Feature(value=[0.1 + i * 0.001] * 128, confidence=0.9)},
          confidence_score= 0.9
      )
      await memory_manager.add_entity(new_entity)

    print(f"\nMemory stats after adding many entities: {await memory_manager.get_memory_statistics()}")


if __name__ == "__main__":
    asyncio.run(main())