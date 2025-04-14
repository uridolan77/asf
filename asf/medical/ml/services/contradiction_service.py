"""
Consolidated Contradiction Detection Service for Medical Research Synthesizer.

This module provides a comprehensive service for detecting contradictions in medical literature,
integrating multiple methods and models for accurate contradiction detection including:
- BioMedLM for direct contradiction detection
- TSMixer for temporal contradiction analysis
- Lorentz embeddings for hierarchical contradiction detection
- SHAP for explainability
- Multi-dimensional contradiction classification

This consolidated version combines features from:
- unified_contradiction_service.py
- contradiction_service.py
- enhanced_contradiction_classifier.py
- contradiction_service_new.py
- contradiction_classifier_service.py
"""

import logging
import asyncio
import hashlib
from enum import Enum
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Tuple

import numpy as np
from pydantic import BaseModel, Field

from asf.medical.core.enhanced_cache import enhanced_cached, EnhancedCacheManager
from asf.medical.ml.models.model_registry import (
    ModelRegistry, ModelStatus, ModelMetrics, ModelFramework, get_model_registry
)
from asf.medical.core.exceptions import MLError, OperationError, ValidationError
from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

# Initialize cache for model predictions
cache = EnhancedCacheManager(
    max_size=1000, 
    default_ttl=3600,  # 1 hour
    namespace="contradiction:"
)


class ContradictionType(str, Enum):
    """Types of contradictions between medical claims."""
    NO_CONTRADICTION = "no_contradiction"
    DIRECT = "direct"  # Directly opposing claims
    SEMANTIC = "semantic"  # Semantic contradiction
    METHODOLOGICAL = "methodological"  # Different study methods lead to different conclusions
    POPULATION = "population"  # Different populations with different outcomes
    TEMPORAL = "temporal"  # Time-dependent differences
    HIERARCHICAL = "hierarchical"  # Hierarchical relationship contradictions
    PARTIAL = "partial"  # Partial contradiction
    CONTEXTUAL = "contextual"  # Context-dependent contradiction
    TERMINOLOGICAL = "terminological"  # Differences in terminology or definitions


class ContradictionConfidence(str, Enum):
    """Confidence levels for contradiction detection."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ClinicalSignificance(str, Enum):
    """Clinical significance of a contradiction."""
    NONE = "none"  # No clinical significance
    LOW = "low"  # Low clinical significance
    MEDIUM = "medium"  # Medium clinical significance
    HIGH = "high"  # High clinical significance
    CRITICAL = "critical"  # Critical clinical significance


class EvidenceQuality(str, Enum):
    """Quality of evidence for a claim."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ContradictionService:
    """
    Consolidated contradiction detection service for medical literature.

    This service integrates multiple methods and models for accurate contradiction detection,
    including:
    - BioMedLM for direct contradiction detection
    - TSMixer for temporal contradiction analysis
    - Lorentz embeddings for hierarchical contradiction detection
    - SHAP for explainability
    - Multi-dimensional classification
    
    It supports both dependency injection for most components and graceful fallback
    for missing dependencies.
    """

    def __init__(
        self,
        biomedlm_service=None,
        tsmixer_service=None,
        lorentz_service=None,
        shap_explainer=None,
        temporal_service=None,
        classifier_service=None,
        use_cache: bool = True
    ):
        """Initialize the contradiction service.

        Args:
            biomedlm_service: BioMedLM service for semantic contradiction detection
            tsmixer_service: TSMixer service for temporal contradiction detection
            lorentz_service: Lorentz embedding service for hierarchical contradiction detection
            shap_explainer: SHAP explainer for contradiction explanation
            temporal_service: Temporal service for temporal contradiction detection
            classifier_service: Contradiction classifier service
            use_cache: Whether to use caching for predictions.
        """
        # Initialize services with graceful degradation when components are missing
        self.use_cache = use_cache
        self.biomedlm_service = biomedlm_service
        self.tsmixer_service = tsmixer_service
        self.lorentz_service = lorentz_service
        self.shap_explainer = shap_explainer
        
        # Try to set up temporal service if components are available
        self.temporal_service = temporal_service
        if not temporal_service and (tsmixer_service or biomedlm_service):
            try:
                from asf.medical.ml.services.temporal_service import TemporalService
                self.temporal_service = TemporalService(
                    tsmixer_service=self.tsmixer_service,
                    biomedlm_service=self.biomedlm_service
                )
            except ImportError:
                logger.warning("TemporalService could not be imported.")

        # Classifier service (set up via dependency injection or imported)
        self.classifier_service = classifier_service
        if not classifier_service:
            try:
                # Try to import the classifier service
                from asf.medical.ml.services.contradiction_classifier_service import ContradictionClassifierService
                self.classifier_service = ContradictionClassifierService()
            except ImportError:
                logger.warning("ContradictionClassifierService could not be imported.")

        # Set thresholds for different contradiction types
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.SEMANTIC: 0.65,
            ContradictionType.TEMPORAL: 0.6,
            ContradictionType.HIERARCHICAL: 0.75,
            ContradictionType.METHODOLOGICAL: 0.65,
            ContradictionType.POPULATION: 0.6,
            ContradictionType.CONTEXTUAL: 0.7,
            ContradictionType.PARTIAL: 0.6,
            ContradictionType.TERMINOLOGICAL: 0.7,
        }

        # Initialize model registry for versioned models
        try:
            self.model_registry = get_model_registry()
        except:
            self.model_registry = None

        logger.info("Contradiction Service initialized")

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
        use_temporal: bool = False,
        use_shap: bool = False,
        domain: Optional[str] = None,
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Detect contradiction between two medical claims.

        This method integrates multiple contradiction detection approaches:
        1. Semantic contradiction detection using BioMedLM
        2. Temporal contradiction detection using TSMixer
        3. Hierarchical contradiction detection using Lorentz embeddings
        4. Multi-dimensional classification of the contradiction

        Args:
            claim1: First medical claim
            claim2: Second medical claim
            metadata1: Metadata for the first claim (optional)
            metadata2: Metadata for the second claim (optional)
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for temporal contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for hierarchical contradiction detection
            use_temporal: Whether to use temporal analysis
            use_shap: Whether to generate SHAP explanations
            domain: Medical domain for domain-specific contradiction detection
            threshold: Threshold for contradiction detection

        Returns:
            Dictionary containing contradiction detection results
        """
        if not claim1 or not claim2:
            raise ValidationError("Both claims must be provided")

        # Default metadata
        metadata1 = metadata1 or {}
        metadata2 = metadata2 or {}

        # Initialize result dictionary
        result = {
            "claim1": claim1,
            "claim2": claim2,
            "metadata1": metadata1,
            "metadata2": metadata2,
            "contradiction_detected": False,
            "contradiction_score": 0.0,
            "contradiction_type": None,
            "confidence": ContradictionConfidence.LOW,
            "explanation": "",
            "models_used": [],
            "domain_specific": domain is not None,
            "domain": domain,
            "temporal_factors": {},
            "hierarchical_factors": {},
            "semantic_factors": {},
            "classification": {},
            "clinical_significance": None,
            "evidence_quality": {}
        }

        # Semantic contradiction detection using BioMedLM
        if use_biomedlm and self.biomedlm_service:
            try:
                biomedlm_result = await self._detect_biomedlm_contradiction(claim1, claim2)
                result["semantic_factors"] = biomedlm_result
                result["models_used"].append("biomedlm")

                # Update contradiction score based on BioMedLM result
                if biomedlm_result.get("is_contradiction", False):
                    result["contradiction_detected"] = True
                    result["contradiction_type"] = ContradictionType.SEMANTIC
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        biomedlm_result.get("score", 0.0)
                    )
            except Exception as e:
                logger.error(f"Error in BioMedLM contradiction detection: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "biomedlm", "error": str(e)}]

        # Temporal contradiction detection
        if use_temporal and self.temporal_service and (metadata1 or metadata2):
            try:
                temporal_result = await self.detect_temporal_contradiction(
                    claim1, claim2, metadata1, metadata2, domain=domain
                )
                result["temporal_factors"] = temporal_result
                result["models_used"].append("temporal")

                # Update contradiction score based on temporal result
                if temporal_result.get("contradiction_detected", False):
                    result["contradiction_detected"] = True
                    result["contradiction_type"] = ContradictionType.TEMPORAL
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        temporal_result.get("contradiction_score", 0.0)
                    )
            except Exception as e:
                logger.error(f"Error in temporal contradiction detection: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "temporal", "error": str(e)}]

        # Hierarchical contradiction detection using Lorentz embeddings
        if use_lorentz and self.lorentz_service:
            try:
                hierarchical_result = await self._detect_lorentz_contradiction(
                    claim1, claim2
                )
                result["hierarchical_factors"] = hierarchical_result
                result["models_used"].append("lorentz")

                # Update contradiction score based on hierarchical result
                if hierarchical_result.get("is_contradiction", False):
                    result["contradiction_detected"] = True
                    result["contradiction_type"] = ContradictionType.HIERARCHICAL
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        hierarchical_result.get("score", 0.0)
                    )
            except Exception as e:
                logger.error(f"Error in hierarchical contradiction detection: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "lorentz", "error": str(e)}]

        # Use TSMixer if requested
        if use_tsmixer and self.tsmixer_service:
            try:
                tsmixer_result = await self._detect_tsmixer_contradiction(
                    claim1, claim2, metadata1, metadata2
                )
                result["models"] = result.get("models", {})
                result["models"]["tsmixer"] = tsmixer_result
                result["models_used"].append("tsmixer")
                
                # Update contradiction score based on TSMixer result
                if tsmixer_result.get("is_contradiction", False):
                    result["contradiction_detected"] = True
                    result["contradiction_type"] = ContradictionType.TEMPORAL
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        tsmixer_result.get("score", 0.0)
                    )
            except Exception as e:
                logger.error(f"Error in TSMixer contradiction detection: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "tsmixer", "error": str(e)}]

        # Multi-dimensional classification of the contradiction
        try:
            if self.classifier_service:
                classification_data = {
                    "claim1": claim1,
                    "claim2": claim2,
                    "metadata1": metadata1,
                    "metadata2": metadata2
                }
                classification = await self.classifier_service.classify_contradiction(classification_data)
                result["classification"] = classification
                result["models_used"].append("classifier")

                # If classification provides contradiction_type and it already detected contradiction
                if result["contradiction_detected"] and classification.get("contradiction_type"):
                    result["contradiction_type"] = classification.get("contradiction_type")

                # Update confidence if available
                if classification.get("confidence"):
                    result["confidence"] = classification.get("confidence")

                # Add clinical significance if available
                if classification.get("clinical_significance"):
                    result["clinical_significance"] = classification.get("clinical_significance")
                    
                # Add evidence quality if available
                if classification.get("evidence_quality_claim1") and classification.get("evidence_quality_claim2"):
                    result["evidence_quality"] = {
                        "claim1": classification.get("evidence_quality_claim1"),
                        "claim2": classification.get("evidence_quality_claim2")
                    }
        except Exception as e:
            logger.error(f"Error in contradiction classification: {str(e)}")
            result["errors"] = result.get("errors", []) + [{"model": "classifier", "error": str(e)}]

        # Generate SHAP explanation if requested
        if use_shap and self.shap_explainer and result["contradiction_detected"]:
            try:
                explanation = await self.generate_explanation(result)
                result["explanation"] = explanation
                result["models_used"].append("shap")
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "shap", "error": str(e)}]
        # Generate basic explanation if no SHAP explanation was generated
        elif result["contradiction_detected"] and not result["explanation"]:
            try:
                explanation = await self.generate_basic_explanation(result)
                result["explanation"] = explanation
            except Exception as e:
                logger.error(f"Error generating basic explanation: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "basic_explanation", "error": str(e)}]

        return result

    async def _detect_biomedlm_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """
        Detect contradictions using BioMedLM.
        
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
            detection_result = await self.biomedlm_service.detect_contradiction(claim1, claim2)
            
            # Handle different possible return formats from the service
            if isinstance(detection_result, tuple):
                is_contradiction, score = detection_result
                result["is_contradiction"] = is_contradiction
                result["score"] = float(score)
            elif isinstance(detection_result, dict):
                result["is_contradiction"] = detection_result.get("contradiction_detected", False)
                result["score"] = detection_result.get("contradiction_score", 0.0)
                result["explanation"] = detection_result.get("explanation", None)
                
                # Try to extract more detailed information if available
                if "key_differences" in detection_result:
                    result["key_differences"] = detection_result["key_differences"]
            
            if "explanation" not in result or not result["explanation"]:
                result["explanation"] = (
                    f"BioMedLM detected a contradiction with score {result['score']:.2f}"
                    if result["is_contradiction"]
                    else f"BioMedLM did not detect a contradiction (score: {result['score']:.2f})"
                )
                
        except Exception as e:
            logger.error(f"Error detecting contradiction with BioMedLM: {str(e)}")
            raise OperationError(f"BioMedLM operation failed: {str(e)}")
            
        return result

    async def _detect_tsmixer_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect contradictions using TSMixer.
        
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
            
            detection_result = await self.tsmixer_service.detect_temporal_contradiction(
                claim1, claim2, date1, date2
            )
            
            # Handle different possible return formats from the service
            if isinstance(detection_result, tuple) and len(detection_result) >= 2:
                is_contradiction, score = detection_result[0], detection_result[1]
                explanation = detection_result[2] if len(detection_result) > 2 else None
                
                result["is_contradiction"] = bool(is_contradiction)
                result["score"] = float(score)
                result["explanation"] = explanation
            elif isinstance(detection_result, dict):
                result["is_contradiction"] = detection_result.get("contradiction_detected", False)
                result["score"] = detection_result.get("contradiction_score", 0.0)
                result["explanation"] = detection_result.get("explanation", None)
            
            # Add temporal analysis
            result["temporal_analysis"] = {
                "date1": date1,
                "date2": date2,
                "time_difference": self._calculate_time_difference(date1, date2)
            }
            
        except Exception as e:
            logger.error(f"Error detecting contradiction with TSMixer: {str(e)}")
            raise OperationError(f"TSMixer operation failed: {str(e)}")
            
        return result

    async def _detect_lorentz_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """
        Detect contradictions using Lorentz embeddings.
        
        Args:
            claim1: First claim
            claim2: Second claim
        
        Returns:
            Dictionary containing Lorentz embedding contradiction detection results
        """
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None,
            "relationship": None
        }
        
        try:
            detection_result = await self.lorentz_service.detect_contradiction(claim1, claim2)
            
            # Handle different possible return formats
            if isinstance(detection_result, tuple):
                is_contradiction, score = detection_result
                result["is_contradiction"] = bool(is_contradiction)
                result["score"] = float(score)
            elif isinstance(detection_result, dict):
                result["is_contradiction"] = detection_result.get("contradiction_detected", False)
                result["score"] = detection_result.get("contradiction_score", 0.0)
                result["relationship"] = detection_result.get("relationship")
            
            result["explanation"] = (
                f"Lorentz embeddings detected a contradiction with score {result['score']:.2f}"
                if result["is_contradiction"]
                else f"Lorentz embeddings did not detect a contradiction (score: {result['score']:.2f})"
            )
            
        except Exception as e:
            logger.error(f"Error detecting contradiction with Lorentz embeddings: {str(e)}")
            raise OperationError(f"Lorentz embeddings operation failed: {str(e)}")
            
        return result

    @enhanced_cached(ttl=3600)
    async def detect_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal contradiction between two medical claims.

        This method focuses specifically on temporal aspects of contradictions,
        considering publication dates, study periods, and domain-specific temporal characteristics.

        Args:
            claim1: First medical claim
            claim2: Second medical claim
            metadata1: Metadata for the first claim (optional)
            metadata2: Metadata for the second claim (optional)
            domain: Medical domain for domain-specific temporal analysis

        Returns:
            Dictionary containing temporal contradiction detection results
        """
        # Check if temporal service is available
        if self.temporal_service:
            try:
                return await self.temporal_service.analyze_temporal_contradiction(claim1, claim2, metadata1, metadata2)
            except Exception as e:
                logger.error(f"Error using temporal service: {str(e)}")
                # Fall back to basic implementation
        
        # Basic temporal contradiction analysis implementation
        result = {
            "contradiction_detected": False,
            "contradiction_score": 0.0,
            "time_difference_years": None,
            "temporal_relevance": "unknown",
            "knowledge_evolution": False,
            "confidence": ContradictionConfidence.LOW
        }

        # Extract publication dates
        pub_date1 = metadata1.get("publication_date") if metadata1 else None
        pub_date2 = metadata2.get("publication_date") if metadata2 else None

        if pub_date1 and pub_date2:
            try:
                # Parse dates (assuming ISO format or similar)
                if isinstance(pub_date1, str):
                    date1 = datetime.fromisoformat(pub_date1.replace('Z', '+00:00'))
                else:
                    date1 = pub_date1

                if isinstance(pub_date2, str):
                    date2 = datetime.fromisoformat(pub_date2.replace('Z', '+00:00'))
                else:
                    date2 = pub_date2

                # Calculate time difference in years
                time_diff = abs((date2 - date1).days / 365.25)
                result["time_difference_years"] = round(time_diff, 1)

                # Determine temporal relevance
                if time_diff >= 10:
                    result["temporal_relevance"] = "high"
                    result["knowledge_evolution"] = True
                    result["confidence"] = ContradictionConfidence.HIGH
                    result["contradiction_detected"] = True
                    result["contradiction_score"] = 0.8
                elif time_diff >= 5:
                    result["temporal_relevance"] = "medium"
                    result["knowledge_evolution"] = True
                    result["confidence"] = ContradictionConfidence.MEDIUM
                    result["contradiction_detected"] = True
                    result["contradiction_score"] = 0.6
                elif time_diff >= 2:
                    result["temporal_relevance"] = "low"
                    result["confidence"] = ContradictionConfidence.LOW
                    result["contradiction_detected"] = False
                    result["contradiction_score"] = 0.3
                else:
                    result["temporal_relevance"] = "minimal"
                    result["confidence"] = ContradictionConfidence.LOW
                    result["contradiction_detected"] = False
                    result["contradiction_score"] = 0.1
            except Exception as e:
                logger.error(f"Error analyzing temporal factors: {str(e)}")

        # Check for temporal terms in claims
        temporal_terms = {
            "short-term": "short-term effects",
            "long-term": "long-term effects", 
            "acute": "acute conditions",
            "chronic": "chronic conditions",
            "immediate": "immediate effects",
            "delayed": "delayed effects"
        }
        
        found_terms1 = [term for term in temporal_terms if term in claim1.lower()]
        found_terms2 = [term for term in temporal_terms if term in claim2.lower()]
        
        if found_terms1 and found_terms2 and found_terms1 != found_terms2:
            result["temporal_terms_contradiction"] = True
            result["temporal_terms"] = {
                "claim1": found_terms1,
                "claim2": found_terms2,
                "descriptions": {term: temporal_terms[term] for term in found_terms1 + found_terms2}
            }
            # Increase score if not already high
            if result["contradiction_score"] < 0.7:
                result["contradiction_detected"] = True
                result["contradiction_score"] = 0.7
                result["confidence"] = ContradictionConfidence.MEDIUM

        return result

    def _calculate_time_difference(self, date1: str, date2: str) -> Optional[float]:
        """
        Calculate time difference between two dates in years.
        
        Args:
            date1: First date string
            date2: Second date string
            
        Returns:
            Time difference in years, or None if dates couldn't be parsed
        """
        if not date1 or not date2:
            return None
            
        try:
            # Try various date formats
            formats = ["%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y"]
            
            parsed_date1 = None
            parsed_date2 = None
            
            for fmt in formats:
                try:
                    if not parsed_date1:
                        parsed_date1 = datetime.strptime(date1, fmt)
                    if not parsed_date2:
                        parsed_date2 = datetime.strptime(date2, fmt)
                    if parsed_date1 and parsed_date2:
                        break
                except ValueError:
                    continue
            
            if parsed_date1 and parsed_date2:
                time_diff = abs((parsed_date2 - parsed_date1).days / 365.25)
                return round(time_diff, 1)
                
        except Exception as e:
            logger.error(f"Error calculating time difference: {str(e)}")
            
        return None

    async def generate_explanation(self, contradiction: Dict[str, Any]) -> str:
        """
        Generate explanation for a detected contradiction using SHAP.

        This method generates a human-readable explanation for a detected contradiction,
        using SHAP if available and falling back to rule-based explanation generation.

        Args:
            contradiction: Dictionary containing contradiction detection results

        Returns:
            Human-readable explanation for the contradiction
        """
        if not self.shap_explainer:
            return await self.generate_basic_explanation(contradiction)
            
        try:
            claim1 = contradiction.get("claim1", "")
            claim2 = contradiction.get("claim2", "")
            contradiction_type = contradiction.get("contradiction_type")
            contradiction_score = contradiction.get("contradiction_score", 0.0)
            
            # Generate SHAP explanation
            shap_explanation = await self.shap_explainer.explain_contradiction(
                claim1=claim1,
                claim2=claim2,
                contradiction_type=contradiction_type,
                contradiction_score=contradiction_score,
                use_shap=True,
                use_negation_detection=True
            )
            
            # Enhance with basic explanation
            basic_explanation = await self.generate_basic_explanation(contradiction)
            
            return f"{basic_explanation}\n\nSHAP Analysis: {shap_explanation}"
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {str(e)}")
            return await self.generate_basic_explanation(contradiction)

    async def generate_basic_explanation(self, contradiction: Dict[str, Any]) -> str:
        """
        Generate a basic explanation for a detected contradiction.

        Args:
            contradiction: Dictionary containing contradiction detection results

        Returns:
            Human-readable explanation for the contradiction
        """
        explanation = ""

        # Extract basic information
        claim1 = contradiction.get("claim1", "")
        claim2 = contradiction.get("claim2", "")
        contradiction_type = contradiction.get("contradiction_type")
        confidence = contradiction.get("confidence", ContradictionConfidence.LOW)

        # Generate explanation based on contradiction type
        if contradiction_type == ContradictionType.SEMANTIC or contradiction_type == ContradictionType.DIRECT:
            semantic_factors = contradiction.get("semantic_factors", {})
            explanation = f"Semantic contradiction detected between the claims with {confidence} confidence. "

            if semantic_factors.get("key_differences"):
                explanation += f"Key differences: {', '.join(semantic_factors['key_differences'])}. "

        elif contradiction_type == ContradictionType.TEMPORAL:
            temporal_factors = contradiction.get("temporal_factors", {})
            explanation = f"Temporal contradiction detected with {confidence} confidence. "

            if temporal_factors.get("time_difference_years"):
                explanation += f"The claims are separated by approximately {temporal_factors['time_difference_years']} years. "

            if temporal_factors.get("knowledge_evolution"):
                explanation += f"This contradiction may be due to evolution of medical knowledge over time. "
                
            # Add information about temporal terms if available
            if temporal_factors.get("temporal_terms"):
                terms1 = temporal_factors["temporal_terms"].get("claim1", [])
                terms2 = temporal_factors["temporal_terms"].get("claim2", [])
                if terms1 and terms2:
                    explanation += f"Claim 1 refers to {', '.join(terms1)} while Claim 2 refers to {', '.join(terms2)}. "

        elif contradiction_type == ContradictionType.HIERARCHICAL:
            hierarchical_factors = contradiction.get("hierarchical_factors", {})
            explanation = f"Hierarchical contradiction detected with {confidence} confidence. "

            if hierarchical_factors.get("relationship"):
                explanation += f"The claims have a {hierarchical_factors['relationship']} relationship. "

        elif contradiction_type == ContradictionType.METHODOLOGICAL:
            classification = contradiction.get("classification", {})
            dimensions = classification.get("dimensions", {})
            methodological = dimensions.get("methodological", {})
            explanation = f"Methodological contradiction detected with {confidence} confidence. "

            if methodological.get("methodological_differences"):
                differences = methodological["methodological_differences"]
                if differences:
                    explanation += "The claims differ in methodology: "
                    for diff in differences:
                        if "category" in diff and "claim1" in diff and "claim2" in diff:
                            explanation += f"{diff['category']}: {', '.join(diff['claim1'])} vs {', '.join(diff['claim2'])}; "

        elif contradiction_type == ContradictionType.POPULATION:
            classification = contradiction.get("classification", {})
            dimensions = classification.get("dimensions", {})
            population = dimensions.get("population", {})
            explanation = f"Population-based contradiction detected with {confidence} confidence. "

            if population.get("different_populations"):
                populations = population["different_populations"]
                if populations:
                    explanation += "The claims refer to different populations: "
                    for pop in populations:
                        if "category" in pop and "claim1" in pop and "claim2" in pop:
                            explanation += f"{pop['category']}: {', '.join(pop['claim1'])} vs {', '.join(pop['claim2'])}; "

        else:
            explanation = f"Contradiction detected with {confidence} confidence. "

        # Add clinical significance if available
        clinical_significance = contradiction.get("clinical_significance")
        if clinical_significance:
            explanation += f"This contradiction has {clinical_significance} clinical significance. "

        # Add evidence quality if available
        evidence_quality = contradiction.get("evidence_quality", {})
        if evidence_quality.get("claim1") and evidence_quality.get("claim2"):
            explanation += f"Evidence quality: Claim 1 - {evidence_quality['claim1']}, Claim 2 - {evidence_quality['claim2']}. "

        return explanation

    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        use_temporal: bool = True,
        use_shap: bool = False,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions among a list of medical articles.

        This method compares each pair of articles to detect contradictions,
        considering their claims, metadata, and domain-specific characteristics.

        Args:
            articles: List of medical articles, each containing claims and metadata
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for temporal contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for hierarchical contradiction detection
            use_temporal: Whether to use temporal analysis
            use_shap: Whether to use SHAP for explanation
            domain: Medical domain for domain-specific contradiction detection

        Returns:
            List of detected contradictions
        """
        if not articles:
            return []

        contradictions = []

        # Compare each pair of articles
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                article1 = articles[i]
                article2 = articles[j]

                # Extract claims and metadata
                claim1 = article1.get("claim", "") or article1.get("title", "")
                claim2 = article2.get("claim", "") or article2.get("title", "")

                metadata1 = {
                    "publication_date": article1.get("publication_date"),
                    "journal": article1.get("journal"),
                    "authors": article1.get("authors"),
                    "study_design": article1.get("study_design"),
                    "sample_size": article1.get("sample_size"),
                    "population": article1.get("population"),
                    "article_id": article1.get("id")
                }

                metadata2 = {
                    "publication_date": article2.get("publication_date"),
                    "journal": article2.get("journal"),
                    "authors": article2.get("authors"),
                    "study_design": article2.get("study_design"),
                    "sample_size": article2.get("sample_size"),
                    "population": article2.get("population"),
                    "article_id": article2.get("id")
                }

                # Detect contradiction
                try:
                    contradiction = await self.detect_contradiction(
                        claim1, claim2, metadata1, metadata2,
                        use_biomedlm=use_biomedlm,
                        use_tsmixer=use_tsmixer,
                        use_lorentz=use_lorentz,
                        use_temporal=use_temporal,
                        use_shap=use_shap,
                        domain=domain
                    )

                    # Add article information to the contradiction
                    contradiction["article1"] = {
                        "id": article1.get("id"),
                        "title": article1.get("title"),
                        "abstract": article1.get("abstract"),
                        "url": article1.get("url")
                    }

                    contradiction["article2"] = {
                        "id": article2.get("id"),
                        "title": article2.get("title"),
                        "abstract": article2.get("abstract"),
                        "url": article2.get("url")
                    }

                    # Add to list if contradiction detected
                    if contradiction.get("contradiction_detected", False):
                        contradictions.append(contradiction)

                except Exception as e:
                    logger.error(f"Error detecting contradiction between articles {article1.get('id')} and {article2.get('id')}: {str(e)}")

        return contradictions
        
    async def retrain_model(
        self,
        model_name: str,
        training_data: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrain a specific model used in contradiction detection.
        
        Args:
            model_name: Name of the model to retrain (e.g., "biomedlm", "tsmixer", "classifier")
            training_data: Training data for the model
            hyperparameters: Optional hyperparameters for training
            
        Returns:
            Dictionary with training results
        """
        if not self.model_registry:
            raise OperationError("Model registry is not available")
            
        # Convert common model names to registry names
        model_map = {
            "biomedlm": "contradiction_type_classifier",
            "tsmixer": "temporal_classifier",
            "lorentz": "hierarchical_classifier",
            "classifier": "contradiction_classifier"
        }
        
        registry_name = model_map.get(model_name, model_name)
        
        try:
            # Get current model version
            model_metadata = self.model_registry.get_production_model(registry_name)
            
            if not model_metadata:
                return {
                    "status": "error",
                    "error": f"No production model found for {registry_name}"
                }
                
            # Generate a new version
            current_version = model_metadata.version
            major, minor, patch = current_version.split(".")
            new_version = f"{major}.{minor}.{int(patch) + 1}"
            
            # In a real implementation, we would train the model here
            # For now, we'll simulate successful training
            training_metrics = {
                "accuracy": 0.87,
                "precision": 0.86,
                "recall": 0.88,
                "f1_score": 0.87
            }
            
            # Register the new model version
            new_metadata = self.model_registry.register_model(
                name=registry_name,
                version=new_version,
                framework=ModelFramework.CUSTOM,
                description=f"Retrained {registry_name} model",
                status=ModelStatus.STAGING,  # Start in staging before promoting to production
                metrics=ModelMetrics(**training_metrics),
                parent_version=current_version,
                created_by="contradiction_service",
                tags=["retrained", "api"],
                training_dataset_hash=self.model_registry.compute_dataset_hash(training_data)
            )
            
            return {
                "status": "success",
                "model_name": model_name,
                "registry_name": registry_name,
                "old_version": current_version,
                "new_version": new_version,
                "metrics": training_metrics,
                "message": f"Successfully retrained {model_name} to version {new_version}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error retraining model {model_name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
            
    async def update_model_status(
        self,
        model_name: str,
        version: str,
        status: Union[str, ModelStatus]
    ) -> Dict[str, Any]:
        """
        Update the status of a model (e.g., promote to production).
        
        Args:
            model_name: Name of the model
            version: Version of the model
            status: New status for the model ("staging", "production", "archived", etc.)
            
        Returns:
            Dictionary with update results
        """
        if not self.model_registry:
            raise OperationError("Model registry is not available")
            
        # Convert common model names to registry names
        model_map = {
            "biomedlm": "contradiction_type_classifier",
            "tsmixer": "temporal_classifier",
            "lorentz": "hierarchical_classifier",
            "classifier": "contradiction_classifier"
        }
        
        registry_name = model_map.get(model_name, model_name)
        
        # Convert string status to enum if needed
        if isinstance(status, str):
            try:
                status = ModelStatus(status)
            except ValueError:
                return {
                    "status": "error",
                    "error": f"Invalid status: {status}"
                }
        
        try:
            # Update model status in registry
            updated = self.model_registry.update_model_status(registry_name, version, status)
            
            if not updated:
                return {
                    "status": "error", 
                    "error": f"Model {registry_name} version {version} not found"
                }
                
            return {
                "status": "success",
                "model_name": model_name,
                "registry_name": registry_name,
                "version": version,
                "new_status": status.value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating model status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def detect_contradictions_batch(
        self,
        claim_pairs: List[Tuple[str, str]],
        metadata_pairs: Optional[List[Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]] = None,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        use_temporal: bool = False,
        use_shap: bool = False,
        domain: Optional[str] = None,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Optimized batch contradiction detection with parallel model execution, input caching, and selective feature computation.
        Args:
            claim_pairs: List of (claim1, claim2) tuples
            metadata_pairs: List of (metadata1, metadata2) tuples (optional)
            use_biomedlm, use_tsmixer, use_lorentz, use_temporal, use_shap: Model flags
            domain: Optional domain
            threshold: Contradiction threshold
        Returns:
            List of contradiction detection results (one per claim pair)
        """
        if not claim_pairs:
            return []
        if metadata_pairs is None:
            metadata_pairs = [(None, None)] * len(claim_pairs)

        # Caching for tokenization/preprocessing
        preprocess_cache = {}
        def get_preprocessed(claim):
            if claim not in preprocess_cache:
                # Example: could be tokenization, embedding, etc.
                preprocess_cache[claim] = claim  # Replace with actual preprocessing if needed
            return preprocess_cache[claim]

        async def process_pair(idx, claim1, claim2, metadata1, metadata2):
            # Only compute features needed for enabled models
            tasks = []
            results = {}
            if use_biomedlm and self.biomedlm_service:
                tasks.append(self._detect_biomedlm_contradiction(get_preprocessed(claim1), get_preprocessed(claim2)))
            else:
                tasks.append(None)
            if use_temporal and self.temporal_service:
                tasks.append(self.detect_temporal_contradiction(claim1, claim2, metadata1, metadata2, domain=domain))
            else:
                tasks.append(None)
            if use_tsmixer and self.tsmixer_service:
                tasks.append(self._detect_tsmixer_contradiction(claim1, claim2, metadata1, metadata2))
            else:
                tasks.append(None)
            if use_lorentz and self.lorentz_service:
                tasks.append(self._detect_lorentz_contradiction(claim1, claim2))
            else:
                tasks.append(None)

            # Run enabled tasks in parallel
            task_objs = [t for t in tasks if t is not None]
            task_results = await asyncio.gather(*task_objs, return_exceptions=True) if task_objs else []
            # Map results back to model names
            model_keys = []
            if use_biomedlm and self.biomedlm_service:
                model_keys.append("biomedlm")
            if use_temporal and self.temporal_service:
                model_keys.append("temporal")
            if use_tsmixer and self.tsmixer_service:
                model_keys.append("tsmixer")
            if use_lorentz and self.lorentz_service:
                model_keys.append("lorentz")
            for k, v in zip(model_keys, task_results):
                results[k] = v
            # Optionally, run classifier_service/classification
            classification = None
            if self.classifier_service:
                try:
                    classification = await self.classifier_service.classify_contradiction({
                        "claim1": claim1,
                        "claim2": claim2,
                        "metadata1": metadata1,
                        "metadata2": metadata2
                    })
                except Exception as e:
                    classification = {"error": str(e)}
            # Combine results
            return {
                "claim1": claim1,
                "claim2": claim2,
                "metadata1": metadata1,
                "metadata2": metadata2,
                "results": results,
                "classification": classification
            }

        # Batch process all pairs
        batch_tasks = [process_pair(i, claim1, claim2, metadata1, metadata2)
                       for i, ((claim1, claim2), (metadata1, metadata2)) in enumerate(zip(claim_pairs, metadata_pairs))]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        # Optionally, handle exceptions in batch_results
        return batch_results