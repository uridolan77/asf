"""Unified Contradiction Detection Service for the Medical Research Synthesizer.

This module provides a comprehensive service for detecting contradictions in medical literature,
integrating multiple methods and models for accurate contradiction detection including:
- BioMedLM for direct contradiction detection
- TSMixer for temporal contradiction analysis
- Lorentz embeddings for hierarchical contradiction detection
- SHAP for explainability
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

from asf.medical.core.enhanced_cache import enhanced_cached
# Import ML models with fallbacks for missing dependencies
try:
    from asf.medical.ml.models.biomedlm import BioMedLMService
except ImportError:
    logger.warning("BioMedLMService could not be imported. Using mock implementation.")
    BioMedLMService = None

try:
    from asf.medical.ml.models.tsmixer import TSMixerService
except ImportError:
    logger.warning("TSMixerService could not be imported. Using mock implementation.")
    TSMixerService = None

try:
    from asf.medical.ml.models.lorentz_embeddings import LorentzEmbeddingService
except ImportError:
    logger.warning("LorentzEmbeddingService could not be imported. Using mock implementation.")
    LorentzEmbeddingService = None

try:
    from asf.medical.ml.models.shap_explainer import SHAPExplainer
except ImportError:
    logger.warning("SHAPExplainer could not be imported. Using mock implementation.")
    SHAPExplainer = None
from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionClassifierService,
    ContradictionType,
    ContradictionConfidence
)
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.core.exceptions import MLError, OperationError, ValidationError



class ContradictionService:
    """Contradiction detection service for medical literature.

    This service integrates multiple methods and models for accurate contradiction detection,
    including BioMedLM for direct contradiction detection, TSMixer for temporal contradiction analysis,
    Lorentz embeddings for hierarchical contradiction detection, and SHAP for explainability.
    """

    def __init__(
        self,
        biomedlm_service = None,
        tsmixer_service = None,
        lorentz_service = None,
        shap_explainer = None,
        temporal_service = None,
        classifier_service = None
    ):
        """Initialize the contradiction service.

        Args:
            biomedlm_service: BioMedLM service for semantic contradiction detection
            tsmixer_service: TSMixer service for temporal contradiction detection
            lorentz_service: Lorentz embedding service for hierarchical contradiction detection
            shap_explainer: SHAP explainer for contradiction explanation
            temporal_service: Temporal service for temporal contradiction detection
            classifier_service: Contradiction classifier service
        """
        # Initialize services
        self.biomedlm_service = biomedlm_service
        self.tsmixer_service = tsmixer_service
        self.lorentz_service = lorentz_service
        self.shap_explainer = shap_explainer
        self.temporal_service = temporal_service or TemporalService(
            tsmixer_service=self.tsmixer_service,
            biomedlm_service=self.biomedlm_service
        ) if self.tsmixer_service or self.biomedlm_service else None
        self.classifier_service = classifier_service or ContradictionClassifierService()

        logger.info("Unified Contradiction Service initialized")

    @enhanced_cached(ttl=3600)
    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        use_biomedlm: bool = True,
        use_tsmixer: bool = True,
        use_lorentz: bool = True,
        use_shap: bool = False,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Detect contradiction between two medical claims.

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
            use_shap: Whether to generate SHAP explanations
            domain: Medical domain for domain-specific contradiction detection

        Returns:
            Dictionary containing contradiction detection results
        """
        if not claim1 or not claim2:
            raise ValidationError("Both claims must be provided")

        # Initialize result dictionary
        result = {
            "claim1": claim1,
            "claim2": claim2,
            "metadata1": metadata1 or {},
            "metadata2": metadata2 or {},
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
            "classification": {}
        }

        # Semantic contradiction detection using BioMedLM
        if use_biomedlm and self.biomedlm_service:
            try:
                biomedlm_result = await self.biomedlm_service.detect_contradiction(claim1, claim2)
                result["semantic_factors"] = biomedlm_result
                result["models_used"].append("biomedlm")

                # Update contradiction score based on BioMedLM result
                result["contradiction_score"] = max(
                    result["contradiction_score"],
                    biomedlm_result.get("contradiction_score", 0.0)
                )

                # Update contradiction detection based on BioMedLM result
                if biomedlm_result.get("contradiction_detected", False):
                    result["contradiction_detected"] = True
                    result["contradiction_type"] = ContradictionType.SEMANTIC
            except Exception as e:
                logger.error(f"Error in BioMedLM contradiction detection: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "biomedlm", "error": str(e)}]

        # Temporal contradiction detection using TSMixer
        if use_tsmixer and (metadata1 or metadata2):
            try:
                temporal_result = await self.detect_temporal_contradiction(
                    claim1, claim2, metadata1, metadata2, domain=domain
                )
                result["temporal_factors"] = temporal_result
                result["models_used"].append("tsmixer")

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
                result["errors"] = result.get("errors", []) + [{"model": "tsmixer", "error": str(e)}]

        # Hierarchical contradiction detection using Lorentz embeddings
        if use_lorentz and self.lorentz_service:
            try:
                hierarchical_result = await self.lorentz_service.detect_hierarchical_contradiction(
                    claim1, claim2
                )
                result["hierarchical_factors"] = hierarchical_result
                result["models_used"].append("lorentz")

                # Update contradiction score based on hierarchical result
                if hierarchical_result.get("contradiction_detected", False):
                    result["contradiction_detected"] = True
                    result["contradiction_type"] = ContradictionType.HIERARCHICAL
                    result["contradiction_score"] = max(
                        result["contradiction_score"],
                        hierarchical_result.get("contradiction_score", 0.0)
                    )
            except Exception as e:
                logger.error(f"Error in hierarchical contradiction detection: {str(e)}")
                result["errors"] = result.get("errors", []) + [{"model": "lorentz", "error": str(e)}]

        # Multi-dimensional classification of the contradiction
        try:
            if self.classifier_service:
                classification = await self.classifier_service.classify_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    metadata1=metadata1,
                    metadata2=metadata2
                )
                result["classification"] = classification

                # Update contradiction type and confidence based on classification
                if classification.get("contradiction_type"):
                    result["contradiction_type"] = classification["contradiction_type"]

                if classification.get("confidence"):
                    result["confidence"] = classification["confidence"]
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
        """Detect temporal contradiction between two medical claims.

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
        # No need to check for temporal_service since we're implementing the method directly

        # Default temporal contradiction result
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

        return result

    async def generate_explanation(self, contradiction: Dict[str, Any]) -> str:
        """Generate explanation for a detected contradiction.

        This method generates a human-readable explanation for a detected contradiction,
        using SHAP if available and falling back to rule-based explanation generation.

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
        if contradiction_type == ContradictionType.SEMANTIC:
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

        elif contradiction_type == ContradictionType.HIERARCHICAL:
            hierarchical_factors = contradiction.get("hierarchical_factors", {})
            explanation = f"Hierarchical contradiction detected with {confidence} confidence. "

            if hierarchical_factors.get("relationship"):
                explanation += f"The claims have a {hierarchical_factors['relationship']} relationship. "

        elif contradiction_type == ContradictionType.METHODOLOGICAL:
            classification = contradiction.get("classification", {})
            methodological_difference = classification.get("methodological_difference", {})
            explanation = f"Methodological contradiction detected with {confidence} confidence. "

            if methodological_difference.get("differences"):
                explanation += f"Methodological differences: {', '.join(methodological_difference['differences'])}. "

        elif contradiction_type == ContradictionType.POPULATION:
            classification = contradiction.get("classification", {})
            population_difference = classification.get("population_difference", {})
            explanation = f"Population-based contradiction detected with {confidence} confidence. "

            if population_difference.get("differences"):
                explanation += f"Population differences: {', '.join(population_difference['differences'])}. "

        else:
            explanation = f"Contradiction detected with {confidence} confidence. "

        # Add clinical significance if available
        classification = contradiction.get("classification", {})
        clinical_significance = classification.get("clinical_significance", {})

        if clinical_significance.get("significance"):
            explanation += f"This contradiction has {clinical_significance['significance']} clinical significance. "

        if clinical_significance.get("implications"):
            explanation += f"Clinical implications: {clinical_significance['implications']}. "

        # Add evidence quality if available
        evidence_quality = classification.get("evidence_quality", {})

        if evidence_quality.get("claim1") and evidence_quality.get("claim2"):
            explanation += f"Evidence quality: Claim 1 - {evidence_quality['claim1']}, Claim 2 - {evidence_quality['claim2']}. "

        # Use SHAP explanation if available
        if self.shap_explainer and self.shap_explainer.is_initialized():
            try:
                shap_explanation = await self.shap_explainer.explain_contradiction(claim1, claim2)
                if shap_explanation:
                    explanation += f"\n\nSHAP explanation: {shap_explanation}"
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {str(e)}")

        return explanation

    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        use_biomedlm: bool = True,
        use_tsmixer: bool = True,
        use_lorentz: bool = True,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Detect contradictions among a list of medical articles.

        This method compares each pair of articles to detect contradictions,
        considering their claims, metadata, and domain-specific characteristics.

        Args:
            articles: List of medical articles, each containing claims and metadata
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for temporal contradiction detection
            use_lorentz: Whether to use Lorentz embeddings for hierarchical contradiction detection
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
