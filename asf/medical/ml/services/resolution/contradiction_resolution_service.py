"""
Medical Contradiction Resolution Service.

This module provides the main service for resolving contradictions in medical literature
based on evidence-based medicine principles.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from asf.medical.ml.services.enhanced_contradiction_classifier import (
    ContradictionType,
    ClinicalSignificance,
    EvidenceQuality
)

from asf.medical.ml.services.resolution.resolution_models import (
    ResolutionStrategy,
    ResolutionConfidence,
    RecommendationType
)

from asf.medical.ml.services.resolution.resolution_strategies import (
    resolve_by_evidence_hierarchy,
    resolve_by_sample_size,
    resolve_by_recency,
    resolve_by_population_specificity
)

from asf.medical.ml.services.resolution.more_strategies import (
    resolve_by_methodological_quality,
    resolve_by_statistical_significance,
    resolve_by_combined_evidence
)

from asf.medical.ml.services.resolution.explanation_generator import (
    generate_explanation
)

# Set up logging
logger = logging.getLogger(__name__)

class MedicalContradictionResolutionService:
    """
    Service for resolving contradictions in medical literature.
    
    This service provides strategies for resolving contradictions based on
    evidence-based medicine principles, including evidence hierarchy,
    sample size, publication recency, population specificity, and methodological quality.
    """
    
    def __init__(self):
        """Initialize the medical contradiction resolution service."""
        # Initialize resolution strategies
        self.resolution_strategies = {
            ResolutionStrategy.EVIDENCE_HIERARCHY: resolve_by_evidence_hierarchy,
            ResolutionStrategy.SAMPLE_SIZE_WEIGHTING: resolve_by_sample_size,
            ResolutionStrategy.RECENCY_PREFERENCE: resolve_by_recency,
            ResolutionStrategy.POPULATION_SPECIFICITY: resolve_by_population_specificity,
            ResolutionStrategy.METHODOLOGICAL_QUALITY: resolve_by_methodological_quality,
            ResolutionStrategy.STATISTICAL_SIGNIFICANCE: resolve_by_statistical_significance
        }
        
        # Resolution history for learning
        self.resolution_history = []
        
        logger.info("Medical contradiction resolution service initialized")
    
    async def resolve_contradiction(
        self,
        contradiction: Dict[str, Any],
        strategy: Optional[ResolutionStrategy] = None
    ) -> Dict[str, Any]:
        """
        Resolve a medical contradiction using the specified or automatically selected strategy.
        
        Args:
            contradiction: Classified contradiction
            strategy: Resolution strategy to use (optional)
            
        Returns:
            Resolution result
        """
        # Select strategy if not specified
        if not strategy:
            strategy = self._select_resolution_strategy(contradiction)
        
        # Get resolution function
        resolution_func = self.resolution_strategies.get(
            strategy, resolve_by_evidence_hierarchy
        )
        
        # Apply resolution strategy
        resolution = await resolution_func(contradiction)
        
        # Add metadata
        resolution["strategy"] = strategy
        resolution["timestamp"] = datetime.now().isoformat()
        
        # Generate explanation
        explanation = generate_explanation(contradiction, resolution, strategy)
        resolution["explanation"] = explanation
        
        # Update resolution history
        self._update_resolution_history(contradiction, resolution, strategy)
        
        return resolution
    
    def _select_resolution_strategy(self, contradiction: Dict[str, Any]) -> ResolutionStrategy:
        """
        Select the most appropriate resolution strategy for a medical contradiction.
        
        Args:
            contradiction: Classified contradiction
            
        Returns:
            Selected resolution strategy
        """
        # Extract classification
        classification = contradiction.get("classification", {})
        
        # Check for significant methodological differences
        methodological_difference = classification.get("methodological_difference", {})
        if methodological_difference.get("detected", False) and methodological_difference.get("score", 0) > 0.7:
            return ResolutionStrategy.METHODOLOGICAL_QUALITY
        
        # Check for significant evidence quality differential
        evidence_quality = classification.get("evidence_quality", {})
        if abs(evidence_quality.get("differential", 0)) > 0.3:
            return ResolutionStrategy.EVIDENCE_HIERARCHY
        
        # Check for significant sample size differences
        metadata1 = contradiction.get("metadata1", {})
        metadata2 = contradiction.get("metadata2", {})
        sample_size1 = metadata1.get("sample_size", 0)
        sample_size2 = metadata2.get("sample_size", 0)
        
        if sample_size1 > 0 and sample_size2 > 0:
            ratio = max(sample_size1, sample_size2) / max(1, min(sample_size1, sample_size2))
            if ratio > 5:  # One study has 5x more participants
                return ResolutionStrategy.SAMPLE_SIZE_WEIGHTING
        
        # Check for significant temporal differences
        temporal_factor = classification.get("temporal_factor", {})
        if temporal_factor.get("detected", False) and temporal_factor.get("score", 0) > 0.5:
            return ResolutionStrategy.RECENCY_PREFERENCE
        
        # Check for population differences
        population_difference = classification.get("population_difference", {})
        if population_difference.get("detected", False) and population_difference.get("score", 0) > 0.5:
            return ResolutionStrategy.POPULATION_SPECIFICITY
        
        # Check for statistical significance differences
        p_value1 = metadata1.get("p_value")
        p_value2 = metadata2.get("p_value")
        if p_value1 is not None and p_value2 is not None:
            if (p_value1 <= 0.05 and p_value2 > 0.05) or (p_value1 > 0.05 and p_value2 <= 0.05):
                return ResolutionStrategy.STATISTICAL_SIGNIFICANCE
        
        # Default to combined evidence
        return ResolutionStrategy.COMBINED_EVIDENCE
    
    async def resolve_contradiction_with_combined_evidence(
        self,
        contradiction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve a medical contradiction using the combined evidence approach.
        
        Args:
            contradiction: Classified contradiction
            
        Returns:
            Resolution result
        """
        # Apply combined evidence strategy
        resolution = await resolve_by_combined_evidence(contradiction, self.resolution_strategies)
        
        # Add metadata
        resolution["strategy"] = ResolutionStrategy.COMBINED_EVIDENCE
        resolution["timestamp"] = datetime.now().isoformat()
        
        # Generate explanation
        explanation = generate_explanation(contradiction, resolution, ResolutionStrategy.COMBINED_EVIDENCE)
        resolution["explanation"] = explanation
        
        # Update resolution history
        self._update_resolution_history(contradiction, resolution, ResolutionStrategy.COMBINED_EVIDENCE)
        
        return resolution
    
    def _update_resolution_history(
        self,
        contradiction: Dict[str, Any],
        resolution: Dict[str, Any],
        strategy: ResolutionStrategy
    ) -> None:
        """
        Update resolution history with new resolution.
        
        Args:
            contradiction: Classified contradiction
            resolution: Resolution result
            strategy: Resolution strategy used
        """
        # Create history entry
        history_entry = {
            "contradiction_id": contradiction.get("id", str(len(self.resolution_history))),
            "contradiction_type": contradiction.get("contradiction_type", ContradictionType.UNKNOWN),
            "clinical_significance": contradiction.get("classification", {}).get("clinical_significance", ClinicalSignificance.UNKNOWN),
            "strategy": strategy,
            "recommendation": resolution.get("recommendation", RecommendationType.INCONCLUSIVE),
            "confidence": resolution.get("confidence", ResolutionConfidence.LOW),
            "timestamp": datetime.now().isoformat(),
            "feedback": None  # To be filled in later if feedback is received
        }
        
        # Add to history
        self.resolution_history.append(history_entry)
    
    def get_resolution_history(self) -> List[Dict[str, Any]]:
        """
        Get the resolution history.
        
        Returns:
            List of resolution history entries
        """
        return self.resolution_history
    
    def add_resolution_feedback(
        self,
        contradiction_id: str,
        feedback: Dict[str, Any]
    ) -> bool:
        """
        Add feedback to a resolution in the history.
        
        Args:
            contradiction_id: ID of the contradiction
            feedback: Feedback data
            
        Returns:
            True if feedback was added successfully, False otherwise
        """
        # Find the resolution in the history
        for entry in self.resolution_history:
            if entry["contradiction_id"] == contradiction_id:
                entry["feedback"] = feedback
                return True
        
        return False
