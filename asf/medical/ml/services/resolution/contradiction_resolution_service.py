"""Medical Contradiction Resolution Service.

This module provides the main service for resolving contradictions in medical literature
based on evidence-based medicine principles.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import contradiction classifier service if needed

from asf.medical.ml.services.resolution.resolution_models import ResolutionStrategy

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

logger = logging.getLogger(__name__)

class MedicalContradictionResolutionService:
    """Service for resolving contradictions in medical literature.
    
    This service provides strategies for resolving contradictions based on
    evidence-based medicine principles, including evidence hierarchy,
    sample size, publication recency, population specificity, and methodological quality.
    """

    def __init__(self):
        """Initialize the medical contradiction resolution service.

    Args:
        # TODO: Add parameter descriptions

    Returns:
        # TODO: Add return description
    """
        self.resolution_strategies = {
            ResolutionStrategy.EVIDENCE_HIERARCHY: resolve_by_evidence_hierarchy,
            ResolutionStrategy.SAMPLE_SIZE_WEIGHTING: resolve_by_sample_size,
            ResolutionStrategy.RECENCY_PREFERENCE: resolve_by_recency,
            ResolutionStrategy.POPULATION_SPECIFICITY: resolve_by_population_specificity,
            ResolutionStrategy.METHODOLOGICAL_QUALITY: resolve_by_methodological_quality,
            ResolutionStrategy.STATISTICAL_SIGNIFICANCE: resolve_by_statistical_significance
        }

        self.resolution_history = []

        logger.info("Medical contradiction resolution service initialized")

    async def resolve_contradiction(
        self,
        contradiction: Dict[str, Any],
        strategy: Optional[ResolutionStrategy] = None
    ) -> Dict[str, Any]:
        """Resolve a contradiction using the specified strategy.

        Args:
            contradiction: The contradiction to resolve
            strategy: The resolution strategy to use

        Returns:
            Dictionary with resolution results
        """
        if not contradiction:
            return {
                "resolved": False,
                "error": "No contradiction provided"
            }

        # Use the specified strategy or determine the best one
        if not strategy:
            strategy = self._determine_best_strategy(contradiction)

        # Apply the selected strategy
        if strategy == ResolutionStrategy.EVIDENCE_HIERARCHY:
            result = await resolve_by_evidence_hierarchy(contradiction)
        elif strategy == ResolutionStrategy.SAMPLE_SIZE:
            result = await resolve_by_sample_size(contradiction)
        elif strategy == ResolutionStrategy.RECENCY:
            result = await resolve_by_recency(contradiction)
        elif strategy == ResolutionStrategy.POPULATION_SPECIFICITY:
            result = await resolve_by_population_specificity(contradiction)
        elif strategy == ResolutionStrategy.METHODOLOGICAL_QUALITY:
            result = await resolve_by_methodological_quality(contradiction)
        elif strategy == ResolutionStrategy.STATISTICAL_SIGNIFICANCE:
            result = await resolve_by_statistical_significance(contradiction)
        elif strategy == ResolutionStrategy.COMBINED_EVIDENCE:
            result = await resolve_by_combined_evidence(contradiction)
        else:
            # Default to combined evidence if strategy is not recognized
            result = await resolve_by_combined_evidence(contradiction)

        # Generate explanation
        explanation = generate_explanation(contradiction, result, strategy)
        result["explanation"] = explanation

        return result

    def _determine_best_strategy(self, contradiction: Dict[str, Any]) -> ResolutionStrategy:
        """Determine the best resolution strategy for a contradiction.

        Args:
            contradiction: The contradiction to resolve

        Returns:
            The best resolution strategy
        """
        # Default to combined evidence
        if not contradiction:
            return ResolutionStrategy.COMBINED_EVIDENCE

        # Check if we have evidence quality information
        metadata1 = contradiction.get("metadata1", {})
        metadata2 = contradiction.get("metadata2", {})

        if metadata1.get("study_type") and metadata2.get("study_type"):
            return ResolutionStrategy.EVIDENCE_HIERARCHY

        # Check if we have sample size information
        if metadata1.get("sample_size") and metadata2.get("sample_size"):
            return ResolutionStrategy.SAMPLE_SIZE

        # Check if we have publication date information
        if metadata1.get("publication_date") and metadata2.get("publication_date"):
            return ResolutionStrategy.RECENCY

        # Check if we have population information
        if metadata1.get("population") and metadata2.get("population"):
            return ResolutionStrategy.POPULATION_SPECIFICITY

        # Default to combined evidence
        return ResolutionStrategy.COMBINED_EVIDENCE

    def get_resolution_history(self) -> List[Dict[str, Any]]:
        """Get the resolution history.

        Returns:
            List of resolution history entries
        """
        return self.resolution_history

    def add_resolution_feedback(self, contradiction_id: str, feedback: Dict[str, Any]) -> bool:
        """Add feedback for a contradiction resolution.

        Args:
            contradiction_id: ID of the contradiction
            feedback: Feedback data

        Returns:
            True if feedback was added successfully, False otherwise
        """
        if not contradiction_id or not feedback:
            return False

        # Find the resolution in history
        for resolution in self.resolution_history:
            if resolution.get("contradiction_id") == contradiction_id:
                # Add feedback to the resolution
                resolution["feedback"] = feedback
                resolution["feedback_timestamp"] = datetime.now().isoformat()
                return True

        return False

    async def batch_resolve_contradictions(
        self,
        contradictions: List[Dict[str, Any]],
        strategy: Optional[ResolutionStrategy] = None
    ) -> List[Dict[str, Any]]:
        """Resolve multiple contradictions.

        Args:
            contradictions: List of contradictions to resolve
            strategy: Resolution strategy to use

        Returns:
            List of resolution results
        """
        if not contradictions:
            return []

        results = []
        for contradiction in contradictions:
            result = await self.resolve_contradiction(contradiction, strategy)
            results.append(result)

        return results