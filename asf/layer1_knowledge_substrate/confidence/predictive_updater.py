import asyncio
import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import beta

# Assuming fully implemented and asynchronous Chronograph and Gnosis layers
# from chronograph_middleware_layer import ChronographMiddleware
# from chronograph_gnosis_layer import ChronoGnosisLayer

# For standalone testing (replace with actual imports when ready)
class ChronographMiddleware:  # Mock
    async def get_entity(self, entity_id: str, include_history: bool = False) -> Optional[Dict]:
        print(f"Mock Chronograph: get_entity({entity_id}, {include_history})")
        return {"id": entity_id, "features": {}}

    async def record_entity_state(self, entity_id: str, state_data: Dict, confidence: float):
        print(f"Mock Chronograph: record({entity_id}, {state_data})")

    async def get_neighbors(self, entity_id: str) -> List[Tuple[str, str, float]]:
      #returns neighbors, relationship type, and strength.
      print(f"MOCK: Getting neighbors for {entity_id}")
      return []

    async def update_entity(self, entity_id:str, updates:Dict):
      print(f"Mock Chronograph: update_entity({entity_id}, {updates})")
class ChronoGnosisLayer:  # Mock
    async def generate_embeddings(self, entity_ids: List[str]) -> Dict[str, Dict]:
      return {entity_id: {"embedding": [0.1, 0.2, 0.3]} for entity_id in entity_ids}


# --- Configuration ---
class ConfidenceUpdaterConfig(BaseModel):
    initial_alpha: float = Field(1.0, description="Initial alpha for Beta distribution.")
    initial_beta: float = Field(1.0, description="Initial beta for Beta distribution.")
    learning_rate: float = Field(0.1, description="Base learning rate.")
    time_decay_factor: float = Field(
        0.0001, description="Factor for time-based decay."
    )
    relevance_threshold: float = Field(
        0.5, description="Threshold for considering a prediction relevant."
    )
    propagation_factor: float = Field(
        0.5, description="How much to propagate confidence updates (0-1)."
    )
    relationship_effects: Dict[str, float] = Field(
        {  # Define how different relationship types affect confidence
            "supports": 1.0,
            "contradicts": -1.0,
            "is_part_of": 0.5,
            "is_a": 0.8,
            "related": 0.2,
        },
        description="Influence of relationship types on confidence propagation.",
    )

class PredictiveConfidenceUpdater:
    def __init__(
        self,
        entity_id: str, # Add entity_id
        chronograph: ChronographMiddleware,
        gnosis: Optional[ChronoGnosisLayer] = None, # Make gnosis optional
        config: Optional[ConfidenceUpdaterConfig] = None,
    ):
        """
        Initializes the PredictiveConfidenceUpdater for a specific entity.
        """
        self.config = config or ConfidenceUpdaterConfig()  # Use default if None
        self.entity_id = entity_id
        self.chronograph = chronograph
        self.gnosis = gnosis  # Optional, for prediction
        self.alpha = self.config.initial_alpha
        self.beta = self.config.initial_beta
        self.last_updated = datetime.datetime.now()
        self.decay_type = "linear"  # Can also be exponential

        # These are no longer instance variables, get them from the config
        # self.learning_rate = learning_rate
        # self.time_decay_factor = time_decay_factor
        # self.relevance_threshold = relevance_threshold
        # self.network = network if network else {}
        # self.propagation_factor = propagation_factor
        # self.relationship_effects = relationship_effects

    async def _predict_relevance(self, context_features: Dict[str, Any]) -> float:
        """
        Predicts the relevance using the gnosis layer (if available), otherwise falls back to a simple mock.
        """
        if self.gnosis:
            # Use the Gnosis layer for prediction (assuming predict_relevance exists)
            try:
              return await self.gnosis.predict_relevance(
                  entity_id=self.entity_id, context=context_features
              )
            except Exception as e: #Catch gnosis errors
              print(f"Error predicting relevance using Gnosis: {e}, falling back.")
              return 0.0 #Return a reasonable default.
        else:
            # Fallback to a simple mock prediction (same as before)
            feature_weights = {"feature_1": 0.6, "feature_2": 0.3, "feature_3": 0.1}
            relevance_score = 0
            for feature, weight in feature_weights.items():
                if feature in context_features:
                    relevance_score += context_features[feature] * weight
            return relevance_score

    async def update_confidence(
        self,
        predicted_relevance: float,
        actual_relevance: float,
        context_features: Dict,
        current_time: Optional[datetime.datetime] = None,
    ):
        """
        Updates the confidence based on prediction error, Bayesian updating, and propagates.
        """
        current_time = current_time or datetime.datetime.now()

        # 1. Time Decay
        time_elapsed = (current_time - self.last_updated).total_seconds()

        if self.decay_type == "linear":
            self.beta += self.config.time_decay_factor * time_elapsed
        elif self.decay_type == "exponential":
            self.beta += self.beta * self.config.time_decay_factor * time_elapsed
        else:
            raise ValueError("decay_type must be one of 'linear' or 'exponential'")

        self.last_updated = current_time

        # 2. Prediction Error
        prediction_error = actual_relevance - predicted_relevance

        # 3. Adaptive Learning Rate
        precision = self.alpha / (self.alpha + self.beta)
        adaptive_learning_rate = self.config.learning_rate * (1 - precision)

        # 4. Bayesian Update
        if predicted_relevance > self.config.relevance_threshold:
            if prediction_error > 0:
                self.alpha += adaptive_learning_rate * prediction_error * actual_relevance
            else:
                self.beta -= (
                    adaptive_learning_rate * prediction_error * actual_relevance
                )  # prediction_error is negative

        # 5. Propagate Update to Neighbors
        await self._propagate_update(prediction_error)

        # Normalize to avoid overflow
        total = self.alpha + self.beta
        self.alpha = (self.alpha / total) * 100
        self.beta = (self.beta / total) * 100

        # 6. Persist updated confidence (CRUCIAL - using Chronograph)
        await self.chronograph.update_entity(
            self.entity_id, {"confidence": self.get_confidence()}
        )


    async def _propagate_update(self, prediction_error: float):
        """Propagates confidence updates to neighboring entities."""
        neighbors = await self.chronograph.get_neighbors(self.entity_id)  # Get neighbors

        for neighbor_id, rel_type, strength in neighbors:
            # Fetch the neighbor's updater (or create one if it doesn't exist)

            neighbor_updater = await get_or_create_updater(neighbor_id, self.chronograph, self.gnosis, self.config)

            relationship_effect = self.config.relationship_effects.get(rel_type, 0.0)
            propagated_change = (
                self.config.propagation_factor * strength * relationship_effect * prediction_error
            )

            # Update neighbor's confidence (simplified - no recursion)
            if propagated_change > 0:
                neighbor_updater.alpha += propagated_change
            else:
                neighbor_updater.beta -= propagated_change # change is negative

            # Normalize and persist neighbor's confidence
            total = neighbor_updater.alpha + neighbor_updater.beta
            neighbor_updater.alpha = (neighbor_updater.alpha/ total) * 100
            neighbor_updater.beta = (neighbor_updater.beta / total) * 100
            await self.chronograph.update_entity(neighbor_id, {"confidence": neighbor_updater.get_confidence()})


    def get_confidence(self) -> float:
        """Returns the current confidence (mean of the Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)

    def get_distribution_params(self) -> Tuple[float, float]:
        """Returns the parameters of the Beta confidence distribution."""
        return self.alpha, self.beta

# --- Helper Function (Outside the Class) ---
_updater_cache: Dict[str, PredictiveConfidenceUpdater] = {} #Global cache

async def get_or_create_updater(
    entity_id: str,
    chronograph: ChronographMiddleware,
    gnosis: Optional[ChronoGnosisLayer],
    config: Optional[ConfidenceUpdaterConfig] = None
) -> PredictiveConfidenceUpdater:
    """
    Retrieves an existing updater from the cache or creates a new one.
    Loads initial confidence from the database if the entity exists.
    """
    global _updater_cache

    if entity_id in _updater_cache:
        return _updater_cache[entity_id]

    updater = PredictiveConfidenceUpdater(entity_id, chronograph, gnosis, config)

    # Load initial confidence from the database (if the entity exists)
    entity_data = await chronograph.get_entity(entity_id)
    if entity_data and "confidence" in entity_data: # Check if the data exists
        # Convert confidence to alpha/beta (inverse of the initialization heuristic)
        confidence = entity_data["confidence"]
        updater.alpha = confidence * 10 + 1
        updater.beta = (1 - confidence) * 10 + 1
    # Add to cache
    _updater_cache[entity_id] = updater
    return updater

# --- Example Usage (Illustrative) ---
async def main():
    # Mock Chronograph and Gnosis
    chronograph = ChronographMiddleware()
    gnosis = ChronoGnosisLayer()
    config = ConfidenceUpdaterConfig()

    # Create an updater for an entity
    entity_id = "entity_1"
    updater = await get_or_create_updater(entity_id, chronograph, gnosis, config)


    # Simulate an observation and update confidence
    context_features = {"feature_1": 0.8, "feature_2": 0.3, "feature_3": 0.6}
    predicted_relevance = await updater._predict_relevance(context_features)
    actual_relevance = 0.7  # Example actual relevance

    await updater.update_confidence(
        predicted_relevance, actual_relevance, context_features
    )

    print(f"Entity {entity_id} Confidence: {updater.get_confidence()}")

    # Example of getting distribution parameters
    alpha, beta = updater.get_distribution_params()
    print(f"Entity {entity_id} Beta Distribution: alpha={alpha}, beta={beta}")

if __name__ == "__main__":
  asyncio.run(main())