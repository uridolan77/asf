"""Contradiction Detection Service for the Medical Research Synthesizer.

This module provides a comprehensive service for detecting contradictions in medical literature,
integrating multiple methods and models for accurate contradiction detection including:
- BioMedLM for direct contradiction detection
- TSMixer for temporal contradiction analysis
- Lorentz embeddings for hierarchical contradiction detection
- SHAP for explainability
"""
import logging
from typing import Dict, Optional, Any
from asf.medical.core.enhanced_cache import enhanced_cached
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.lorentz_embeddings import LorentzEmbeddingService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionClassifierService,
    ContradictionType
)
from asf.medical.ml.services.temporal_service import TemporalService

from asf.medical.core.exceptions import OperationError

logger = logging.getLogger(__name__)
class ContradictionService:
    """Contradiction detection service for medical literature.

    This service integrates multiple methods and models for accurate contradiction detection,
    including BioMedLM for direct contradiction detection, TSMixer for temporal contradiction analysis,
    Lorentz embeddings for hierarchical contradiction detection, and SHAP for explainability.
    """
    def __init__(
        self,
        biomedlm_service: Optional[BioMedLMService] = None,
        tsmixer_service: Optional[TSMixerService] = None,
        lorentz_service: Optional[LorentzEmbeddingService] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
        temporal_service: Optional[TemporalService] = None,
        classifier_service: Optional[ContradictionClassifierService] = None
    ):
        """Initialize the contradiction service.
        Args:
            biomedlm_service: BioMedLM service for contradiction detection
            tsmixer_service: TSMixer service for temporal sequence analysis
            lorentz_service: Lorentz embedding service for hierarchical contradiction detection
            shap_explainer: SHAP explainer for contradiction explanation
            temporal_service: Temporal service for temporal contradiction detection
            enhanced_classifier: Enhanced contradiction classifier
        """
        # Initialize services
        self.biomedlm_service = biomedlm_service or BioMedLMService()
        self.tsmixer_service = tsmixer_service or TSMixerService()
        self.lorentz_service = lorentz_service or LorentzEmbeddingService()
        self.shap_explainer = shap_explainer or SHAPExplainer()
        self.temporal_service = temporal_service or TemporalService(tsmixer_service=self.tsmixer_service)
        self.classifier_service = classifier_service or ContradictionClassifierService()
        # Set thresholds for different contradiction types
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.STATISTICAL: 0.65,
            ContradictionType.TEMPORAL: 0.6,
            ContradictionType.HIERARCHICAL: 0.75,
            ContradictionType.NEGATION: 0.8
        }
        logger.info("Contradiction service initialized")
    @enhanced_cached(ttl=3600)
    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        use_temporal: bool = False
    ) -> Dict[str, Any]:
        """Detect contradictions between two claims.
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for contradiction detection
            use_temporal: Whether to use temporal analysis for contradiction detection
        Returns:
            Dictionary containing contradiction detection results
        """
        result = {
            "claim1": claim1,
            "claim2": claim2,
            "metadata1": metadata1,
            "metadata2": metadata2,
            "contradiction_detected": False,
            "contradiction_type": None,
            "contradiction_confidence": None,
            "contradiction_score": 0.0,
            "explanation": None,
            "evidence": [],
            "analysis": {}
        }
        # Use enhanced classifier for multi-dimensional contradiction analysis
        enhanced_result = await self.enhanced_classifier.classify_contradiction(
            claim1=claim1,
            claim2=claim2,
            metadata1=metadata1,
            metadata2=metadata2
        )
        result["contradiction_detected"] = enhanced_result["is_contradiction"]
        result["contradiction_type"] = enhanced_result["contradiction_type"]
        result["contradiction_confidence"] = enhanced_result["confidence"]
        result["contradiction_score"] = enhanced_result["score"]
        result["explanation"] = enhanced_result["explanation"]
        result["evidence"] = enhanced_result["evidence"]
        result["analysis"] = enhanced_result["analysis"]
        # Add individual model results if requested
        if use_biomedlm:
            biomedlm_result = await self._detect_biomedlm_contradiction(claim1, claim2)
            result["models"] = result.get("models", {})
            result["models"]["biomedlm"] = biomedlm_result
        if use_tsmixer:
            tsmixer_result = await self._detect_tsmixer_contradiction(claim1, claim2, metadata1, metadata2)
            result["models"] = result.get("models", {})
            result["models"]["tsmixer"] = tsmixer_result
        if use_lorentz:
            lorentz_result = await self._detect_lorentz_contradiction(claim1, claim2)
            result["models"] = result.get("models", {})
            result["models"]["lorentz"] = lorentz_result
        if use_temporal:
            temporal_result = await self._detect_temporal_contradiction(claim1, claim2, metadata1, metadata2)
            result["models"] = result.get("models", {})
            result["models"]["temporal"] = temporal_result
        # Generate SHAP explanation if contradiction detected
        if result["contradiction_detected"] and result["contradiction_score"] > 0.7:
            try:
                explanation = await self.shap_explainer.explain_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    contradiction_type=result["contradiction_type"],
                    contradiction_score=result["contradiction_score"]
                )
                result["shap_explanation"] = explanation
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {str(e)}")
                raise OperationError(f"Operation failed: {str(e)}")
        return result
    async def _detect_biomedlm_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """Detect contradictions using BioMedLM.
        Args:
            claim1: First claim
            claim2: Second claim
        Returns:
            Dictionary containing BioMedLM contradiction detection results
        """
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None
        }
        try:
            _, score = await self.biomedlm_service.detect_contradiction(claim1, claim2)
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)
            result["explanation"] = (
                f"BioMedLM detected a contradiction with score {score:.2f}"
                if result["is_contradiction"]
                else f"BioMedLM did not detect a contradiction (score: {score:.2f})"
            )
        except Exception as e:
            logger.error(f"Error detecting contradiction with BioMedLM: {str(e)}")
            raise OperationError(f"Operation failed: {str(e)}")
        return result
    async def _detect_tsmixer_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect contradictions using TSMixer.
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
        Returns:
            Dictionary containing TSMixer contradiction detection results
        """
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None,
            "temporal_analysis": None
        }
        try:
            date1 = metadata1.get("publication_date", "") if metadata1 else ""
            date2 = metadata2.get("publication_date", "") if metadata2 else ""
            _, score, explanation = await self.tsmixer_service.detect_temporal_contradiction(
                claim1, claim2, date1, date2
            )
            result["is_contradiction"] = score > self.thresholds[ContradictionType.TEMPORAL]
            result["score"] = float(score)
            result["explanation"] = explanation
            result["temporal_analysis"] = {
                "date1": date1,
                "date2": date2,
                "time_difference": self.tsmixer_service.calculate_time_difference(date1, date2)
            }
        except Exception as e:
            logger.error(f"Error detecting contradiction with TSMixer: {str(e)}")
            raise OperationError(f"Operation failed: {str(e)}")
        return result
    async def _detect_lorentz_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """Detect contradictions using Lorentz embeddings.
        Args:
            claim1: First claim
            claim2: Second claim
        Returns:
            Dictionary containing Lorentz embedding contradiction detection results
        """
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None
        }
        try:
            _, score = await self.lorentz_service.detect_contradiction(claim1, claim2)
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)
            result["explanation"] = (
                f"Lorentz embeddings detected a contradiction with score {score:.2f}"
                if result["is_contradiction"]
                else f"Lorentz embeddings did not detect a contradiction (score: {score:.2f})"
            )
        except Exception as e:
            logger.error(f"Error detecting contradiction with Lorentz embeddings: {str(e)}")
            raise OperationError(f"Operation failed: {str(e)}")
        return result
    async def _detect_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect temporal contradictions.
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
        Returns:
            Dictionary containing temporal contradiction detection results
        """
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None,
            "temporal_analysis": None
        }
        try:
            date1 = metadata1.get("publication_date", "") if metadata1 else ""
            date2 = metadata2.get("publication_date", "") if metadata2 else ""
            _, score, explanation = await self.temporal_service.analyze_temporal_contradiction(
                claim1, claim2, date1, date2
            )
            result["is_contradiction"] = score > self.thresholds[ContradictionType.TEMPORAL]
            result["score"] = float(score)
            result["explanation"] = explanation
            result["temporal_analysis"] = {
                "date1": date1,
                "date2": date2,
                "time_difference": self.temporal_service.calculate_time_difference(date1, date2)
            }
        except Exception as e:
            logger.error(f"Error detecting temporal contradiction: {str(e)}")
            raise OperationError(f"Operation failed: {str(e)}")
        return result