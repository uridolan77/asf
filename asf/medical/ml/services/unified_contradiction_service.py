"""
Unified Contradiction Detection Service for the Medical Research Synthesizer.

This module provides a unified service for detecting contradictions in medical literature,
integrating multiple methods and models for more accurate contradiction detection.
"""

import logging
from typing import Dict, List, Optional, Any, Union

from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.lorentz_embeddings import LorentzEmbeddingService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.enhanced_contradiction_classifier import (
    EnhancedContradictionClassifier, 
    ContradictionType, 
    ContradictionConfidence
)
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.core.enhanced_cache import enhanced_cache_manager, enhanced_cached

# Set up logging
logger = logging.getLogger(__name__)

class UnifiedContradictionService:
    """
    Unified service for detecting contradictions in medical literature.

    This service integrates multiple methods and models for more accurate contradiction detection,
    including BioMedLM, TSMixer, Lorentz embeddings, and SHAP explainability.
    """

    def __init__(
        self,
        biomedlm_service: Optional[BioMedLMService] = None,
        tsmixer_service: Optional[TSMixerService] = None,
        lorentz_service: Optional[LorentzEmbeddingService] = None,
        temporal_service: Optional[TemporalService] = None,
        shap_explainer: Optional[SHAPExplainer] = None,
        thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the unified contradiction service.

        Args:
            biomedlm_service: BioMedLM service for semantic contradiction detection
            tsmixer_service: TSMixer service for temporal contradiction detection
            lorentz_service: Lorentz embeddings service for hyperbolic contradiction detection
            temporal_service: Temporal service for temporal contradiction detection
            shap_explainer: SHAP explainer for contradiction explanation
            thresholds: Contradiction detection thresholds for different methods
        """
        # Initialize services
        self.biomedlm_service = biomedlm_service or BioMedLMService()
        self.tsmixer_service = tsmixer_service or TSMixerService()
        self.lorentz_service = lorentz_service or LorentzEmbeddingService()
        self.temporal_service = temporal_service or TemporalService()
        self.shap_explainer = shap_explainer or SHAPExplainer()
        
        # Initialize contradiction classifier
        self.contradiction_classifier = EnhancedContradictionClassifier()
        
        # Set thresholds
        self.thresholds = thresholds or {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.TEMPORAL: 0.7,
            ContradictionType.METHODOLOGICAL: 0.7,
            ContradictionType.POPULATION: 0.7,
            ContradictionType.EVIDENCE_QUALITY: 0.7
        }
        
        logger.info("Unified contradiction service initialized")

    @enhanced_cached(prefix="detect_contradiction", data_type="analysis")
    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        use_temporal: bool = False,
        skip_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
            threshold: Contradiction detection threshold
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for contradiction detection
            use_temporal: Whether to use temporal analysis for contradiction detection
            skip_cache: Whether to skip cache

        Returns:
            Contradiction detection result
        """
        # Initialize result
        result = {
            "claim1": claim1,
            "claim2": claim2,
            "is_contradiction": False,
            "contradiction_score": 0.0,
            "contradiction_type": ContradictionType.NONE,
            "confidence": ContradictionConfidence.NONE,
            "explanation": "",
            "methods_used": [],
            "details": {}
        }
        
        # Update thresholds based on input
        if threshold != 0.7:
            for key in self.thresholds:
                self.thresholds[key] = threshold
        
        # Detect direct contradiction using BioMedLM
        if use_biomedlm and self.biomedlm_service:
            direct_result = await self._detect_direct_contradiction(claim1, claim2)
            result["methods_used"].append("biomedlm")
            result["details"]["direct"] = direct_result

            # Update result if direct contradiction is detected
            if direct_result["is_contradiction"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = direct_result["score"]
                result["contradiction_type"] = ContradictionType.DIRECT
                result["confidence"] = direct_result["confidence"]
                result["explanation"] = direct_result["explanation"]
        
        # Detect temporal contradiction using TSMixer
        if use_tsmixer and self.tsmixer_service:
            temporal_result = await self._detect_temporal_contradiction(claim1, claim2, metadata1, metadata2)
            result["methods_used"].append("tsmixer")
            result["details"]["temporal"] = temporal_result

            # Update result if temporal contradiction is detected and has higher score
            if temporal_result["is_contradiction"] and (
                not result["is_contradiction"] or 
                temporal_result["score"] > result["contradiction_score"]
            ):
                result["is_contradiction"] = True
                result["contradiction_score"] = temporal_result["score"]
                result["contradiction_type"] = ContradictionType.TEMPORAL
                result["confidence"] = temporal_result["confidence"]
                result["explanation"] = temporal_result["explanation"]
        
        # Detect contradiction using Lorentz embeddings
        if use_lorentz and self.lorentz_service:
            lorentz_result = await self._detect_lorentz_contradiction(claim1, claim2)
            result["methods_used"].append("lorentz")
            result["details"]["lorentz"] = lorentz_result

            # Update result if Lorentz contradiction is detected and has higher score
            if lorentz_result["is_contradiction"] and (
                not result["is_contradiction"] or 
                lorentz_result["score"] > result["contradiction_score"]
            ):
                result["is_contradiction"] = True
                result["contradiction_score"] = lorentz_result["score"]
                result["contradiction_type"] = ContradictionType.DIRECT
                result["confidence"] = lorentz_result["confidence"]
                result["explanation"] = lorentz_result["explanation"]
        
        # Detect temporal contradiction using temporal analysis
        if use_temporal and self.temporal_service:
            temporal_analysis_result = await self._detect_temporal_analysis_contradiction(claim1, claim2, metadata1, metadata2)
            result["methods_used"].append("temporal_analysis")
            result["details"]["temporal_analysis"] = temporal_analysis_result

            # Update result if temporal analysis contradiction is detected and has higher score
            if temporal_analysis_result["is_contradiction"] and (
                not result["is_contradiction"] or 
                temporal_analysis_result["score"] > result["contradiction_score"]
            ):
                result["is_contradiction"] = True
                result["contradiction_score"] = temporal_analysis_result["score"]
                result["contradiction_type"] = ContradictionType.TEMPORAL
                result["confidence"] = temporal_analysis_result["confidence"]
                result["explanation"] = temporal_analysis_result["explanation"]
        
        # Generate explanation using SHAP if available and not already set
        if self.shap_explainer and result["is_contradiction"] and not result["explanation"]:
            explanation = await self._generate_explanation(claim1, claim2, result["contradiction_type"])
            result["explanation"] = explanation

        # Apply enhanced classification if contradiction is detected
        if result["is_contradiction"]:
            result = await self._apply_enhanced_classification(result, metadata1, metadata2)

        logger.info(f"Contradiction detection result: {result['is_contradiction']} (score: {result['contradiction_score']}, type: {result['contradiction_type']})")
        return result

    @enhanced_cached(prefix="detect_contradictions_in_articles", data_type="analysis")
    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        use_temporal: bool = False,
        skip_cache: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions in a list of articles.

        Args:
            articles: List of articles to analyze
            threshold: Contradiction detection threshold
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for contradiction detection
            use_temporal: Whether to use temporal analysis for contradiction detection
            skip_cache: Whether to skip cache

        Returns:
            List of contradictions
        """
        # Validate input
        if not articles or len(articles) < 2:
            logger.warning("Not enough articles to detect contradictions")
            return []
        
        # Initialize contradictions
        contradictions = []
        
        # Compare each pair of articles
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                # Extract claims
                claim1 = articles[i].get("title", "") + ". " + articles[i].get("abstract", "")
                claim2 = articles[j].get("title", "") + ". " + articles[j].get("abstract", "")
                
                # Skip if either claim is empty
                if not claim1.strip() or not claim2.strip():
                    continue
                
                # Extract metadata
                metadata1 = {
                    "publication_date": articles[i].get("publication_date", ""),
                    "journal": articles[i].get("journal", ""),
                    "authors": articles[i].get("authors", []),
                    "pmid": articles[i].get("pmid", ""),
                    "doi": articles[i].get("doi", "")
                }
                
                metadata2 = {
                    "publication_date": articles[j].get("publication_date", ""),
                    "journal": articles[j].get("journal", ""),
                    "authors": articles[j].get("authors", []),
                    "pmid": articles[j].get("pmid", ""),
                    "doi": articles[j].get("doi", "")
                }
                
                # Detect contradiction
                result = await self.detect_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    metadata1=metadata1,
                    metadata2=metadata2,
                    threshold=threshold,
                    use_biomedlm=use_biomedlm,
                    use_tsmixer=use_tsmixer,
                    use_lorentz=use_lorentz,
                    use_temporal=use_temporal,
                    skip_cache=skip_cache
                )
                
                # Add to contradictions if contradiction detected
                if result.get("is_contradiction", False):
                    contradiction = {
                        "article1": articles[i],
                        "article2": articles[j],
                        "contradiction_score": result.get("contradiction_score"),
                        "contradiction_type": result.get("contradiction_type"),
                        "confidence": result.get("confidence"),
                        "explanation": result.get("explanation"),
                        "classification": result.get("classification", {})
                    }
                    
                    contradictions.append(contradiction)
        
        logger.info(f"Found {len(contradictions)} contradictions in {len(articles)} articles")
        return contradictions

    async def _detect_direct_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """
        Detect direct contradiction using BioMedLM.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Direct contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.NONE,
            "explanation": ""
        }
        
        try:
            # Use BioMedLM to detect contradiction
            is_contradiction, score = await self.biomedlm_service.detect_contradiction(claim1, claim2)

            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)

            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW

            # Generate explanation
            if result["is_contradiction"]:
                result["explanation"] = f"The claims directly contradict each other with a score of {score:.2f}."

            return result
        except Exception as e:
            logger.error(f"Error detecting direct contradiction: {str(e)}")
            return result

    async def _detect_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal contradiction using TSMixer.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Temporal contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.NONE,
            "explanation": ""
        }
        
        try:
            # Extract publication dates from metadata
            date1 = metadata1.get("publication_date", "") if metadata1 else ""
            date2 = metadata2.get("publication_date", "") if metadata2 else ""
            
            # Use TSMixer to detect temporal contradiction
            is_contradiction, score, explanation = await self.tsmixer_service.detect_temporal_contradiction(
                claim1, claim2, date1, date2
            )

            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.TEMPORAL]
            result["score"] = float(score)

            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW

            # Set explanation
            if result["is_contradiction"]:
                result["explanation"] = explanation or f"The claims contradict each other temporally with a score of {score:.2f}."

            return result
        except Exception as e:
            logger.error(f"Error detecting temporal contradiction: {str(e)}")
            return result

    async def _detect_lorentz_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """
        Detect contradiction using Lorentz embeddings.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Lorentz contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.NONE,
            "explanation": ""
        }
        
        try:
            # Use Lorentz embeddings to detect contradiction
            is_contradiction, score = await self.lorentz_service.detect_contradiction(claim1, claim2)

            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)

            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW

            # Generate explanation
            if result["is_contradiction"]:
                result["explanation"] = f"The claims contradict each other in hyperbolic space with a score of {score:.2f}."

            return result
        except Exception as e:
            logger.error(f"Error detecting Lorentz contradiction: {str(e)}")
            return result

    async def _detect_temporal_analysis_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal contradiction using temporal analysis.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Temporal analysis contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.NONE,
            "explanation": ""
        }
        
        try:
            # Extract publication dates from metadata
            date1 = metadata1.get("publication_date", "") if metadata1 else ""
            date2 = metadata2.get("publication_date", "") if metadata2 else ""
            
            # Use temporal service to detect temporal contradiction
            is_contradiction, score, explanation = await self.temporal_service.analyze_temporal_contradiction(
                claim1, claim2, date1, date2
            )

            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.TEMPORAL]
            result["score"] = float(score)

            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW

            # Set explanation
            if result["is_contradiction"]:
                result["explanation"] = explanation or f"The claims contradict each other temporally with a score of {score:.2f}."

            return result
        except Exception as e:
            logger.error(f"Error detecting temporal analysis contradiction: {str(e)}")
            return result

    async def _generate_explanation(
        self,
        claim1: str,
        claim2: str,
        contradiction_type: str
    ) -> str:
        """
        Generate explanation for contradiction.

        Args:
            claim1: First claim
            claim2: Second claim
            contradiction_type: Type of contradiction

        Returns:
            Explanation for contradiction
        """
        try:
            # Use SHAP explainer to generate explanation
            explanation = await self.shap_explainer.explain_contradiction(claim1, claim2)
            
            # Extract explanation text
            explanation_text = explanation.get("summary", "")
            
            # Add contradiction type to explanation
            if contradiction_type == ContradictionType.DIRECT:
                explanation_text = f"Direct contradiction: {explanation_text}"
            elif contradiction_type == ContradictionType.TEMPORAL:
                explanation_text = f"Temporal contradiction: {explanation_text}"
            elif contradiction_type == ContradictionType.METHODOLOGICAL:
                explanation_text = f"Methodological contradiction: {explanation_text}"
            elif contradiction_type == ContradictionType.POPULATION:
                explanation_text = f"Population contradiction: {explanation_text}"
            elif contradiction_type == ContradictionType.EVIDENCE_QUALITY:
                explanation_text = f"Evidence quality contradiction: {explanation_text}"
            
            return explanation_text
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"The claims contradict each other ({contradiction_type})."

    async def _apply_enhanced_classification(
        self,
        result: Dict[str, Any],
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply enhanced classification to contradiction result.

        Args:
            result: Contradiction result
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Enhanced contradiction result with classification
        """
        try:
            # Add metadata to the contradiction result
            contradiction_with_metadata = {
                **result,
                "metadata1": metadata1 or {},
                "metadata2": metadata2 or {}
            }

            # Classify the contradiction
            classified_contradiction = await self.contradiction_classifier.classify_contradiction(contradiction_with_metadata)

            # Add classification to the result
            result["classification"] = classified_contradiction["classification"]

            logger.info(f"Contradiction classified with clinical significance: {result['classification']['clinical_significance']}")

            return result
        except Exception as e:
            logger.error(f"Error applying enhanced classification: {str(e)}")
            return result
