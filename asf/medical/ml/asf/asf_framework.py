"""
ASF Framework Module

This module implements the main ASF (Autopoietic Semantic Fields) framework class
that integrates all components into a unified system.
"""

import time
import uuid
import re
import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from .confidence_ecosystem import ConfidenceEcosystem
from .permeability_gate import PermeabilityGate
from .contradiction_engine import ContradictionAssimilationEngine
from .hybrid_memory import HybridMemoryEngine
from .predictive_processor import PredictiveProcessor


class ASFFramework:
    """
    The main ASF framework class that integrates all components.
    
    The ASF Framework provides a unified interface to the Autopoietic Semantic Fields
    system, integrating the Dynamic Confidence Ecosystem, Contextual Permeability Gates,
    Contradiction Assimilation Engine, Hybrid Memory Engine, and Predictive Processor
    into a cohesive whole.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ASF framework.
        
        Args:
            config: Configuration for the framework (optional)
        """
        self.config = config or {}
        
        # Initialize core components
        self.confidence_ecosystem = ConfidenceEcosystem(
            decay_rate=self.config.get("confidence_decay_rate", 0.01),
            min_confidence=self.config.get("min_confidence", 0.1),
            max_confidence=self.config.get("max_confidence", 1.0)
        )
        
        self.permeability_gate = PermeabilityGate(
            default_threshold=self.config.get("default_threshold", 0.5)
        )
        
        self.contradiction_engine = ContradictionAssimilationEngine(
            confidence_ecosystem=self.confidence_ecosystem
        )
        
        self.hybrid_memory = HybridMemoryEngine(
            cache_size=self.config.get("cache_size", 10000)
        )
        
        self.predictive_processor = PredictiveProcessor()
        
        # Knowledge store
        self.knowledge_store: Dict[str, Dict[str, Any]] = {}  # Map of knowledge_id -> knowledge_content
        
        # Integration components
        self.llm_integration = None  # Will be set up later
        self.peft_integration = None  # Will be set up later
        
        # Framework state
        self.initialized = False
        self.created_at = time.time()
        self.last_updated = time.time()
        
        # Register default validation criteria and resolution strategies
        self._register_default_components()
    
    async def initialize(self) -> bool:
        """
        Initialize the ASF framework.
        
        Returns:
            Success flag
        """
        # Perform any necessary initialization
        self.initialized = True
        return True
    
    async def process_input(
        self, 
        input_text: str, 
        context_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through ASF components.
        
        Args:
            input_text: Input text
            context_id: Context ID (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            Processing result
        """
        # Check if input passes permeability gate
        if not self.permeability_gate.evaluate(input_text, context_id):
            return {
                "status": "rejected",
                "reason": "Input did not pass permeability gate",
                "input": input_text,
                "context_id": context_id
            }
        
        # Extract knowledge claims from input
        knowledge_claims = await self._extract_knowledge_claims(input_text)
        
        # Process each knowledge claim
        processed_claims = []
        for claim in knowledge_claims:
            # Check for contradictions with existing knowledge
            contradictions = []
            for existing_id, existing_claim in self.knowledge_store.items():
                if self.contradiction_engine.detect_contradiction(claim, existing_claim):
                    resolution = self.contradiction_engine.resolve_contradiction(
                        claim, existing_claim
                    )
                    contradictions.append({
                        "existing_id": existing_id,
                        "resolution": resolution
                    })
            
            # Generate unique ID for the claim
            claim_id = str(uuid.uuid4())
            
            # Add metadata
            if metadata:
                claim["metadata"] = metadata
                
            # Add timestamp
            claim["timestamp"] = time.time()
            
            # Add source
            claim["source"] = "input"
            
            # Store the claim
            self.knowledge_store[claim_id] = claim
            
            # Initialize confidence
            self.confidence_ecosystem.initialize_confidence(claim_id, 0.5)  # Initial confidence
            
            # Store in hybrid memory
            await self.hybrid_memory.store_entity(claim_id, claim)
            
            processed_claims.append({
                "id": claim_id,
                "claim": claim,
                "contradictions": contradictions,
                "confidence": self.confidence_ecosystem.get_confidence(claim_id)
            })
        
        # Update last updated timestamp
        self.last_updated = time.time()
        
        return {
            "status": "processed",
            "processed_claims": processed_claims,
            "input": input_text,
            "context_id": context_id,
            "timestamp": time.time()
        }
    
    async def query_knowledge(
        self, 
        query: str, 
        top_k: int = 10,
        min_confidence: float = 0.0,
        context_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query knowledge based on text similarity.
        
        Args:
            query: Query text
            top_k: Maximum number of results
            min_confidence: Minimum confidence threshold
            context_id: Context ID (optional)
            
        Returns:
            List of matching knowledge items
        """
        # Check if query passes permeability gate
        if not self.permeability_gate.evaluate(query, context_id):
            return []
        
        # Extract features from query
        query_features = {"text": query}
        
        # Query hybrid memory
        results = await self.hybrid_memory.query(query_features, top_k * 2)  # Get more results than needed for filtering
        
        # Filter by confidence
        filtered_results = []
        for entity_id, score in results:
            confidence = self.confidence_ecosystem.get_confidence(entity_id)
            if confidence >= min_confidence:
                entity = await self.hybrid_memory.get_entity(entity_id)
                if entity:
                    filtered_results.append({
                        "id": entity_id,
                        "entity": entity["data"],
                        "score": score,
                        "confidence": confidence,
                        "combined_score": score * confidence
                    })
        
        # Sort by combined score
        filtered_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return filtered_results[:top_k]
    
    async def update_knowledge(
        self, 
        knowledge_id: str, 
        update_data: Dict[str, Any],
        update_confidence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update existing knowledge.
        
        Args:
            knowledge_id: Knowledge ID
            update_data: Update data
            update_confidence: New confidence value (optional)
            
        Returns:
            Update result
        """
        if knowledge_id not in self.knowledge_store:
            return {
                "status": "error",
                "reason": "Knowledge not found",
                "knowledge_id": knowledge_id
            }
        
        # Get existing knowledge
        existing_knowledge = self.knowledge_store[knowledge_id]
        
        # Update knowledge
        if isinstance(existing_knowledge, dict) and isinstance(update_data, dict):
            # Merge dictionaries
            existing_knowledge.update(update_data)
        else:
            # Replace data
            existing_knowledge = update_data
            
        # Update timestamp
        existing_knowledge["updated_at"] = time.time()
        
        # Store updated knowledge
        self.knowledge_store[knowledge_id] = existing_knowledge
        
        # Update in hybrid memory
        await self.hybrid_memory.update_entity(knowledge_id, existing_knowledge)
        
        # Update confidence if provided
        confidence_update = None
        if update_confidence is not None:
            confidence_update = self.confidence_ecosystem.update_confidence(
                knowledge_id, update_confidence
            )
        
        # Update last updated timestamp
        self.last_updated = time.time()
        
        return {
            "status": "updated",
            "knowledge_id": knowledge_id,
            "confidence_update": confidence_update
        }
    
    async def delete_knowledge(self, knowledge_id: str) -> Dict[str, Any]:
        """
        Delete knowledge.
        
        Args:
            knowledge_id: Knowledge ID
            
        Returns:
            Delete result
        """
        if knowledge_id not in self.knowledge_store:
            return {
                "status": "error",
                "reason": "Knowledge not found",
                "knowledge_id": knowledge_id
            }
        
        # Delete from knowledge store
        del self.knowledge_store[knowledge_id]
        
        # Delete from hybrid memory
        await self.hybrid_memory.delete_entity(knowledge_id)
        
        # Update last updated timestamp
        self.last_updated = time.time()
        
        return {
            "status": "deleted",
            "knowledge_id": knowledge_id
        }
    
    async def register_validation_criterion(
        self, 
        criteria_id: str, 
        validation_function: Callable
    ) -> bool:
        """
        Register a validation criterion for the permeability gate.
        
        Args:
            criteria_id: Criterion ID
            validation_function: Function to evaluate information against this criterion
            
        Returns:
            Success flag
        """
        return self.permeability_gate.register_validation_criterion(criteria_id, validation_function)
    
    async def register_resolution_strategy(
        self, 
        strategy_id: str, 
        resolution_function: Callable
    ) -> bool:
        """
        Register a resolution strategy for the contradiction engine.
        
        Args:
            strategy_id: Strategy ID
            resolution_function: Function to resolve contradictions
            
        Returns:
            Success flag
        """
        return self.contradiction_engine.register_resolution_strategy(strategy_id, resolution_function)
    
    async def register_contradiction_pattern(
        self, 
        pattern_id: str, 
        pattern_type: str, 
        pattern_data: Any,
        description: str = ""
    ) -> bool:
        """
        Register a pattern for detecting contradictions.
        
        Args:
            pattern_id: Pattern ID
            pattern_type: Type of pattern ('regex', 'semantic', 'logical', etc.)
            pattern_data: Pattern data (depends on pattern_type)
            description: Description of the pattern
            
        Returns:
            Success flag
        """
        return self.contradiction_engine.register_contradiction_pattern(
            pattern_id, pattern_type, pattern_data, description
        )
    
    async def register_prediction_model(
        self, 
        model_id: str, 
        model_type: str, 
        model_config: Dict[str, Any]
    ) -> bool:
        """
        Register a prediction model.
        
        Args:
            model_id: Model ID
            model_type: Type of model (e.g., 'bayesian', 'neural')
            model_config: Configuration for the model
            
        Returns:
            Success flag
        """
        return await self.predictive_processor.register_model(model_id, model_type, model_config)
    
    async def generate_prediction(
        self, 
        entity_id: str, 
        model_id: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a prediction for an entity.
        
        Args:
            entity_id: Entity ID
            model_id: Model ID to use
            context: Additional context for the prediction
            
        Returns:
            Prediction result
        """
        return await self.predictive_processor.generate_prediction(entity_id, model_id, context)
    
    async def update_prediction(
        self, 
        prediction_id: str, 
        actual_value: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Update a prediction with the actual value.
        
        Args:
            prediction_id: Prediction ID
            actual_value: Actual value
            
        Returns:
            Updated prediction
        """
        return await self.predictive_processor.update_prediction(prediction_id, actual_value)
    
    async def _extract_knowledge_claims(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract knowledge claims from text.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of knowledge claims
        """
        # This is a placeholder - in a real implementation, this might use
        # NLP techniques to extract structured claims
        
        # For now, we'll just treat each sentence as a separate claim
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        for sentence in sentences:
            claims.append({
                "text": sentence,
                "source": "extracted",
                "timestamp": time.time()
            })
        
        return claims
    
    def _register_default_components(self) -> None:
        """
        Register default validation criteria and resolution strategies.
        """
        # Register default validation criteria
        self.permeability_gate.register_validation_criterion(
            "length",
            lambda text: min(1.0, len(text) / 10)  # Simple length-based criterion
        )
        
        self.permeability_gate.register_validation_criterion(
            "complexity",
            lambda text: min(1.0, len(set(text.split())) / 5)  # Simple vocabulary-based criterion
        )
        
        # Register default resolution strategies
        self.contradiction_engine.register_resolution_strategy(
            "confidence_based",
            lambda claim_a, claim_b, conf_a, conf_b: {
                "resolution": "keep_higher_confidence",
                "kept": "a" if conf_a > conf_b else "b",
                "confidence_update": 0.1  # Boost confidence of kept claim
            }
        )
        
        # Register default contradiction patterns
        self.contradiction_engine.register_contradiction_pattern(
            "negation",
            "regex",
            {
                "pattern_a": r"(is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|must) (not|n't)",
                "pattern_b": r"(is|are|was|were|has|have|had|do|does|did|can|could|will|would|should|must) "
            },
            "Detects simple negation contradictions"
        )
    
    async def batch_decay_confidence(self) -> int:
        """
        Apply temporal decay to all confidence scores.
        
        Returns:
            Number of entities updated
        """
        return self.confidence_ecosystem.batch_decay()
    
    async def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an entity by ID.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Entity data or None if not found
        """
        return await self.hybrid_memory.get_entity(entity_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about ASF framework performance.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "confidence_ecosystem": self.confidence_ecosystem.get_metrics(),
            "permeability_gate": self.permeability_gate.get_metrics(),
            "contradiction_engine": self.contradiction_engine.get_metrics(),
            "hybrid_memory": self.hybrid_memory.get_metrics(),
            "predictive_processor": self.predictive_processor.get_metrics(),
            "knowledge_store_size": len(self.knowledge_store),
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "uptime": time.time() - self.created_at
        }
