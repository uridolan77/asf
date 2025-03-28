import asyncio
import datetime
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field, validator
from scipy.stats import beta

# Assuming asf.__core.enums and fully implemented Chronograph/Gnosis
# from asf.__core.enums import EntityConfidenceState
# from chronograph_middleware_layer import ChronographMiddleware
# from chronograph_gnosis_layer import ChronoGnosisLayer

# Mocked for standalone execution
class EntityConfidenceState(str):  # Mock
    CANONICAL = "canonical"
    PROVISIONAL = "provisional"
    UNVERIFIED = "unverified"

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
    async def get_entity_confidence(self, entity_id:str) -> Optional[Dict]:
      print(f"MOCK: get_entity_confidence({entity_id})")
      return None # Mock return
    async def update_entity_confidence(self, entity_id:str, alpha:float, beta:float, last_updated:datetime.datetime):
      print(f"MOCK: update_entity_confidence({entity_id}, {alpha}, {beta}, {last_updated})")

class ChronoGnosisLayer:  # Mock
    async def generate_embeddings(self, entity_ids: List[str]) -> Dict[str, Dict]:
      return {entity_id: {"embedding": [0.1, 0.2, 0.3]} for entity_id in entity_ids}
    async def generate_context_embedding(self, context:Dict):
      print(f"MOCK: generate_context_embedding({context})")
      return [0.4,0.5,0.6] # Mock
    async def predict_relevance_from_embedding(self, entity_embedding:List[float], context_embedding:List[float]) -> float:
        print(f"MOCK: predict_relevance_from_embedding({entity_embedding}, {context_embedding})")
        return 0.7 # Mock prediction

class BayesianConfidenceUpdaterConfig(BaseModel):
    prior_alpha: float = Field(1.0, description="Initial alpha for Beta distribution.", gt=0)
    prior_beta: float = Field(1.0, description="Initial beta for Beta distribution.", gt=0)
    decay_rate: float = Field(
        0.999, description="Rate at which confidence decays towards the prior.", ge=0, le=1
    )
    max_context_samples: int = Field(
        100, description="Maximum number of context samples to store.", gt=0
    )
    context_similarity_weight: float = Field(
        0.7, description="Weight for context similarity in relevance prediction.", ge=0, le=1
    )
    min_similarity_for_relevance: float = Field(
        0.1, description="Minimum similarity to consider context relevant.", ge=0, le=1
    )
    learning_rate: float = Field(0.1, description="Learning rate for context model updates.", ge=0, le=1)
    context_model_type: str = Field("knn", description="Type of context model ('knn' or 'gmm').")
    knn_k: int = Field(5, description="Number of neighbors for k-NN context model.", gt=0)
    propagation_factor: float = Field(0.5, description="Factor for confidence propagation.")
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

    @validator("context_model_type")
    def context_model_type_valid(cls, value):
        if value not in ["knn", "gmm"]:  # Expand as needed
            raise ValueError("context_model_type must be 'knn' or 'gmm'")
        return value


class BayesianConfidenceUpdater:
    """
    Updates entity confidence using Bayesian inference and context modeling.
    """

    def __init__(
        self,
        entity_id: str,
        chronograph: ChronographMiddleware,
        gnosis: ChronoGnosisLayer,
        config: Optional[BayesianConfidenceUpdaterConfig] = None,
    ):
        self.config = config or BayesianConfidenceUpdaterConfig()
        self.entity_id = entity_id
        self.chronograph = chronograph
        self.gnosis = gnosis
        self.prior_alpha = self.config.prior_alpha
        self.prior_beta = self.config.prior_beta
        self.entity_posterior = (
            self.prior_alpha,
            self.prior_beta,
        )  # Single tuple, not a dict

        self.context_model: Dict[str, Any] = {}  # Initialize empty context model

        self.last_updated = datetime.datetime.now()


    async def initialize(self):
        """Loads confidence and context model from the database."""
        db_conf = await self.chronograph.get_entity_confidence(self.entity_id)
        if db_conf:
            self.entity_posterior = (db_conf['alpha'], db_conf['beta'])
            self.last_updated = db_conf.get("last_updated", datetime.datetime.now())

        # Consider adding context model loading (if you persist it)

        # Initialize the context model (k-NN or GMM) after loading data
        await self._initialize_context_model()


    async def _initialize_context_model(self):
        """Initializes the context model (k-NN or GMM)."""
        if self.config.context_model_type == "knn":
            self.context_model = {"contexts": [], "relevance": []} # List based KNN.
        elif self.config.context_model_type == "gmm":
            # Placeholder:  You would initialize a Gaussian Mixture Model here.
            # This requires a library like scikit-learn.  For now, we'll raise an error.
            raise NotImplementedError("GMM context model not yet implemented.") #Important
            # Example (using scikit-learn):
            # from sklearn.mixture import GaussianMixture
            # self.context_model = GaussianMixture(n_components=..., covariance_type=...)
        else:
            raise ValueError(
                f"Invalid context_model_type: {self.config.context_model_type}"
            )


    async def update_confidence(
        self, observation_relevant: bool, context_features: Optional[Dict] = None
    ):
        """
        Updates entity confidence using Bayesian inference and context.

        Args:
            observation_relevant: Whether the entity was relevant.
            context_features:  Dictionary representing the context.
        """
        alpha, beta = self.entity_posterior

        # Apply time decay (exponential decay towards the prior)
        time_elapsed = (datetime.datetime.now() - self.last_updated).total_seconds()
        alpha = self.prior_alpha + (alpha - self.prior_alpha) * (
            self.config.decay_rate ** time_elapsed
        )
        beta = self.prior_beta + (beta - self.prior_beta) * (
            self.config.decay_rate ** time_elapsed
        )

        # Bayesian Update
        if observation_relevant:
            alpha += 1
        else:
            beta += 1

        self.entity_posterior = (alpha, beta)
        self.last_updated = datetime.datetime.now()


        # Update context model if context provided
        if context_features:
          await self._update_context_model(context_features, observation_relevant)

        # Normalize and persist updated confidence
        await self._normalize_and_persist()
        await self._propagate_update(observation_relevant) #Propagate changes
        return self.get_confidence()

    async def _normalize_and_persist(self):
        """Normalizes alpha/beta and persists to the database."""
        alpha, beta = self.entity_posterior
        total = alpha + beta
        normalized_alpha = (alpha / total) * 100  # Scale to 100 for stability
        normalized_beta = (beta / total) * 100
        self.entity_posterior = (normalized_alpha, normalized_beta)

        # Persist updated confidence to database
        await self.chronograph.update_entity_confidence(
            self.entity_id, normalized_alpha, normalized_beta, self.last_updated
        )

    async def _update_context_model(
        self, context_features: Dict, was_relevant: bool
    ):
        """Updates the context-based prediction model for the entity."""

        context_embedding = await self.gnosis.generate_context_embedding(
            context_features
        )
        context_embedding = np.array(context_embedding)  # Ensure it's a NumPy array


        if self.config.context_model_type == "knn":
            # Add to k-NN model (list-based)
            self.context_model["contexts"].append(context_embedding)
            self.context_model["relevance"].append(1.0 if was_relevant else 0.0)

            # Limit context samples
            if len(self.context_model["contexts"]) > self.config.max_context_samples:
                self.context_model["contexts"] = self.context_model["contexts"][
                    -self.config.max_context_samples :
                ]
                self.context_model["relevance"] = self.context_model["relevance"][
                    -self.config.max_context_samples :
                ]


        elif self.config.context_model_type == "gmm":
            # Update GMM (Placeholder - requires scikit-learn or similar)
              raise NotImplementedError("GMM context model is not supported yet.")
        # Example (using scikit-learn - needs fitting and updating)
        #   relevance = 1.0 if was_relevant else 0.0
        #   self.context_model.fit(np.array([context_embedding]), [relevance])  # Simplified


    async def predict_relevance(self, context_features: Dict) -> float:
        """
        Predicts entity relevance in a given context.
        """
        alpha, beta = self.entity_posterior
        base_rate = alpha / (alpha + beta)

        context_embedding = await self.gnosis.generate_context_embedding(
            context_features
        )
        context_embedding = np.array(context_embedding)  # Ensure NumPy array

        if not self.context_model:  # No context model yet
              return base_rate

        if self.config.context_model_type == "knn":
            if not self.context_model["contexts"]: # Empty model
                return base_rate
            # k-NN prediction
            contexts = np.array(self.context_model["contexts"])
            relevance_scores = np.array(self.context_model["relevance"])

            similarities = self._compute_similarities(context_embedding, contexts)
            # Get top-k similar contexts
            top_k_indices = np.argsort(similarities)[-self.config.knn_k :]
            top_k_similarities = similarities[top_k_indices]
            top_k_relevance = relevance_scores[top_k_indices]

            # Weighted average based on similarity
            if np.sum(top_k_similarities) > 0:
                weighted_relevance = np.sum(top_k_similarities * top_k_relevance) / np.sum(
                    top_k_similarities
                )
            else:  # Handle case of no similar contexts
                weighted_relevance = base_rate

            # Combine with base rate
            max_similarity = (
                np.max(top_k_similarities) if len(top_k_similarities) > 0 else 0
            )
            combined_prediction = (
                max_similarity * weighted_relevance + (1 - max_similarity) * base_rate
            )
            return combined_prediction

        elif self.config.context_model_type == "gmm":
          # GMM prediction (Placeholder)
          raise NotImplementedError("GMM is not implemented")
            # Example (using scikit-learn)
            # return self.context_model.predict_proba(np.array([context_embedding]))[0][1]  # Probability of relevance

        return base_rate  # Fallback

    def _compute_similarities(self, query_vector: np.ndarray, context_vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarities between query and context vectors."""
        if len(context_vectors) == 0:
            return np.array([])

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return np.zeros(len(context_vectors))
        query_vector = query_vector / query_norm

        similarities = []
        for ctx in context_vectors:
            ctx_norm = np.linalg.norm(ctx)
            if ctx_norm == 0:
                similarities.append(0.0)
            else:
                ctx_normalized = ctx / ctx_norm
                similarity = np.dot(query_vector, ctx_normalized)
                similarities.append(max(0, similarity))

        return np.array(similarities)

    async def _propagate_update(self, observation_relevant:bool):
        """Propagates confidence updates to neighboring entities."""
        neighbors = await self.chronograph.get_neighbors(self.entity_id)

        for neighbor_id, rel_type, strength in neighbors:
            neighbor_updater = await get_or_create_bayesian_updater(
                neighbor_id, self.chronograph, self.gnosis, self.config
            )

            relationship_effect = self.config.relationship_effects.get(rel_type, 0.0)
            # Propagate based on observation (simplified - could use prediction error)
            if observation_relevant:
                change = self.config.propagation_factor * strength * relationship_effect
            else:
                change = -self.config.propagation_factor * strength * relationship_effect

            # Apply the change to neighbor's alpha/beta (similar to update)
            alpha, beta = neighbor_updater.entity_posterior
            if change > 0:
                alpha += change
            else:
                beta -= change  # change is negative

            neighbor_updater.entity_posterior = (alpha, beta)
            await neighbor_updater._normalize_and_persist() # Persist changes

    def get_confidence(self) -> float:
        """Returns the current confidence (mean of the Beta distribution)."""
        alpha, beta = self.entity_posterior
        return alpha / (alpha + beta)

    def get_confidence_state(self) -> EntityConfidenceState:
        """Convert confidence score to EntityConfidenceState."""
        confidence_score = self.get_confidence()
        if confidence_score > 0.8:
            return EntityConfidenceState.CANONICAL
        elif confidence_score > 0.5:
            return EntityConfidenceState.PROVISIONAL
        else:
            return EntityConfidenceState.UNVERIFIED

# --- Helper Function (Outside the Class) ---
_updater_cache: Dict[str, BayesianConfidenceUpdater] = {}  # Global cache

async def get_or_create_bayesian_updater(
    entity_id: str,
    chronograph: ChronographMiddleware,
    gnosis: ChronoGnosisLayer,
    config: Optional[BayesianConfidenceUpdaterConfig] = None,
) -> BayesianConfidenceUpdater:
    """
    Retrieves or creates a BayesianConfidenceUpdater, loading initial state.
    """
    global _updater_cache
    if entity_id in _updater_cache:
        return _updater_cache[entity_id]

    updater = BayesianConfidenceUpdater(entity_id, chronograph, gnosis, config)
    await updater.initialize()  # Load from database
    _updater_cache[entity_id] = updater
    return updater

# --- Example Usage ---
async def main():
    # Mock Chronograph and Gnosis
    chronograph = ChronographMiddleware()
    gnosis = ChronoGnosisLayer()
    config = BayesianConfidenceUpdaterConfig()

    # Create an updater for an entity
    entity_id = "entity_1"
    updater = await get_or_create_bayesian_updater(entity_id, chronograph, gnosis, config)

    # Simulate an observation and update confidence
    context_features = {"feature_1": 0.8, "feature_2": 0.3, "feature_3": 0.6}
    await updater.update_confidence(True, context_features)
    print(f"Entity {entity_id} Confidence: {updater.get_confidence()}")
    print(f"Entity {entity_id} State: {updater.get_confidence_state()}")

    # Predict relevance in a new context
    new_context_features = {"feature_1": 0.1, "feature_2": 0.9, "feature_3": 0.2}
    predicted_relevance = await updater.predict_relevance(new_context_features)
    print(f"Predicted Relevance in new context: {predicted_relevance}")

    # Simulate another entity and propagate
    entity_id2 = "entity_2"
    updater2 = await get_or_create_bayesian_updater(entity_id2, chronograph, gnosis, config)
    context_features2 = {"feature_A": 0.9, "feature_B":0.7}
    await updater2.update_confidence(False, context_features2)  # Entity 2 irrelevant
    print(f"Entity {entity_id2} Confidence: {updater2.get_confidence()}")
    print(f"Entity {entity_id} Confidence (After Propagation): {updater.get_confidence()}") # See effects of propagation

if __name__ == "__main__":
    asyncio.run(main())