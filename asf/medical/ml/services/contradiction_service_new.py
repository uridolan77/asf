Contradiction Detection Service for the Medical Research Synthesizer.

This module provides a comprehensive service for detecting contradictions in medical literature,
integrating multiple methods and models for accurate contradiction detection including:
- BioMedLM for direct contradiction detection
- TSMixer for temporal contradiction analysis
- Lorentz embeddings for hierarchical contradiction detection
- SHAP for explainability

import logging
from typing import Dict, List, Optional, Any

from asf.medical.core.enhanced_cache import enhanced_cached
from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.tsmixer import TSMixerService
from asf.medical.ml.models.lorentz_embeddings import LorentzEmbeddingService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.contradiction_classifier_service import (
    ContradictionClassifierService,
    ContradictionType,
    ContradictionConfidence
)
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.core.exceptions import MLError, OperationError


logger = logging.getLogger(__name__)

class ContradictionService:
    Contradiction detection service for medical literature.
    
    This service integrates multiple methods and models for accurate contradiction detection,
    including BioMedLM for direct contradiction detection, TSMixer for temporal contradiction analysis,
    Lorentz embeddings for hierarchical contradiction detection, and SHAP for explainability.

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
            biomedlm_service: BioMedLM service for semantic contradiction detection
            tsmixer_service: TSMixer service for temporal contradiction detection
            lorentz_service: Lorentz embedding service for hierarchical contradiction detection
            shap_explainer: SHAP explainer for contradiction explanation
            temporal_service: Temporal service for temporal contradiction detection
            classifier_service: Contradiction classifier service for multi-dimensional classification
        """
        self.biomedlm_service = biomedlm_service or BioMedLMService()
        self.tsmixer_service = tsmixer_service or TSMixerService()
        self.lorentz_service = lorentz_service or LorentzEmbeddingService()
        self.shap_explainer = shap_explainer or SHAPExplainer()
        self.temporal_service = temporal_service or TemporalService(tsmixer_service=self.tsmixer_service)
        self.classifier_service = classifier_service or ContradictionClassifierService()

        # Set thresholds for different contradiction types
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.NEGATION: 0.8,
            ContradictionType.TEMPORAL: 0.6,
            ContradictionType.HIERARCHICAL: 0.65,
            ContradictionType.METHODOLOGICAL: 0.75,
            ContradictionType.STATISTICAL: 0.8,
            ContradictionType.POPULATION: 0.7
        }

        logger.info("Contradiction service initialized")

    @enhanced_cached(prefix="detect_contradiction", data_type="contradiction")
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
            claim1: First claim text
            claim2: Second claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
            use_biomedlm: Whether to use BioMedLM for contradiction detection
            use_tsmixer: Whether to use TSMixer for temporal contradiction detection
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

        # Use classifier service for multi-dimensional contradiction analysis
        classifier_result = await self.classifier_service.classify_contradiction(
            claim1=claim1,
            claim2=claim2,
            metadata1=metadata1,
            metadata2=metadata2
        )

        # Extract results from classifier
        result["contradiction_detected"] = classifier_result["is_contradiction"]
        result["contradiction_type"] = classifier_result["contradiction_type"]
        result["contradiction_confidence"] = classifier_result["confidence"]
        result["contradiction_score"] = classifier_result["score"]
        result["explanation"] = classifier_result["explanation"]
        result["evidence"] = classifier_result["evidence"]
        result["analysis"] = classifier_result["analysis"]

        # Add SHAP explanation if available
        if self.shap_explainer and result["contradiction_detected"]:
            try:
                shap_explanation = await self.shap_explainer.explain_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    contradiction_type=result["contradiction_type"]
                )
                result["shap_explanation"] = shap_explanation
            except Exception as e:
                logger.warning(f"Error generating SHAP explanation: {e}")
                result["explanation"] = f"Error generating SHAP explanation: {str(e)}"

        # Add temporal analysis if requested and available
        if use_temporal and self.tsmixer_service and metadata1 and metadata2:
            try:
                # Extract temporal data if available
                temporal_data1 = metadata1.get("temporal_data", [])
                temporal_data2 = metadata2.get("temporal_data", [])

                if temporal_data1 and temporal_data2:
                    temporal_analysis = await self.tsmixer_service.analyze_temporal_sequence(
                        sequence1=temporal_data1,
                        sequence2=temporal_data2
                    )

                    if "models" not in result:
                        result["models"] = {}

                    result["models"]["tsmixer"] = temporal_analysis

                    # Update contradiction detection if temporal analysis indicates contradiction
                    if temporal_analysis.get("contradiction_scores", [0])[0] > 0.7:
                        result["contradiction_detected"] = True
                        result["contradiction_type"] = ContradictionType.TEMPORAL
                        result["contradiction_confidence"] = ContradictionConfidence.HIGH
                        result["contradiction_score"] = max(
                            result["contradiction_score"],
                            temporal_analysis.get("contradiction_scores", [0])[0]
                        )
            except Exception as e:
                logger.warning(f"Error performing temporal analysis: {e}")
                result["models"] = {"tsmixer": {"error": f"Error performing temporal analysis: {str(e)}"}} if "models" not in result else result["models"]

        return result

    async def detect_biomedlm_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect contradictions using BioMedLM.

        Args:
            claim1: First claim text
            claim2: Second claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing contradiction detection results
        """
        result = {
            "method": "biomedlm",
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None
        }

        try:
            contradiction_result, score = await self.biomedlm_service.detect_contradiction(claim1, claim2)

            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)

            if result["is_contradiction"]:
                result["explanation"] = "BioMedLM detected a direct contradiction between the claims."
            else:
                result["explanation"] = "BioMedLM did not detect a significant contradiction between the claims."
        except Exception as e:
            logger.warning(f"Error detecting contradiction with BioMedLM: {e}")
            result["explanation"] = f"Error detecting contradiction with BioMedLM: {str(e)}"

        return result

    async def detect_tsmixer_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect temporal contradictions using TSMixer.

        Args:
            claim1: First claim text
            claim2: Second claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing contradiction detection results
        """
        result = {
            "method": "tsmixer",
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None
        }

        try:
            date1 = metadata1.get("publication_date", "") if metadata1 else ""
            date2 = metadata2.get("publication_date", "") if metadata2 else ""

            contradiction_result, score, explanation = await self.tsmixer_service.detect_temporal_contradiction(
                claim1, claim2, date1, date2
            )

            result["is_contradiction"] = score > self.thresholds[ContradictionType.TEMPORAL]
            result["score"] = float(score)
            result["explanation"] = explanation
        except Exception as e:
            logger.warning(f"Error detecting temporal contradiction with TSMixer: {e}")
            result["explanation"] = f"Error detecting temporal contradiction with TSMixer: {str(e)}"

        return result

    async def detect_lorentz_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect hierarchical contradictions using Lorentz embeddings.

        Args:
            claim1: First claim text
            claim2: Second claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing contradiction detection results
        """
        result = {
            "method": "lorentz",
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None
        }

        try:
            contradiction_result, score = await self.lorentz_service.detect_contradiction(claim1, claim2)

            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = float(score)

            if result["is_contradiction"]:
                result["explanation"] = "Lorentz embeddings detected a hierarchical contradiction between the claims."
            else:
                result["explanation"] = "Lorentz embeddings did not detect a significant contradiction between the claims."
        except Exception as e:
            logger.warning(f"Error detecting contradiction with Lorentz embeddings: {e}")
            result["explanation"] = f"Error detecting contradiction with Lorentz embeddings: {str(e)}"

        return result

    async def detect_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect temporal contradictions using the temporal service.

        Args:
            claim1: First claim text
            claim2: Second claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing contradiction detection results
        """
        result = {
            "method": "temporal",
            "is_contradiction": False,
            "score": 0.0,
            "explanation": None,
            "temporal_factors": {}
        }

        try:
            date1 = metadata1.get("publication_date", "") if metadata1 else ""
            date2 = metadata2.get("publication_date", "") if metadata2 else ""

            contradiction_result, score, explanation = await self.temporal_service.analyze_temporal_contradiction(
                claim1, claim2, date1, date2
            )

            result["is_contradiction"] = score > self.thresholds[ContradictionType.TEMPORAL]
            result["score"] = float(score)
            result["explanation"] = explanation

            # Add temporal factors if available
            if metadata1 and metadata2:
                temporal_factors = self.classifier_service._assess_temporal_factor(metadata1, metadata2)
                result["temporal_factors"] = temporal_factors
        except Exception as e:
            logger.warning(f"Error detecting temporal contradiction: {e}")
            result["explanation"] = f"Error detecting temporal contradiction: {str(e)}"

        return result

    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        threshold: float = 0.7,
        use_all_methods: bool = False
    ) -> List[Dict[str, Any]]:
        """Detect contradictions in a list of articles.

        Args:
            articles: List of articles to check for contradictions
            threshold: Threshold for contradiction detection
            use_all_methods: Whether to use all available methods

        Returns:
            List of detected contradictions
        """
        if not articles or len(articles) < 2:
            return []

        contradictions = []

        # Compare each article with every other article
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                article1 = articles[i]
                article2 = articles[j]

                # Extract claims from articles
                claim1 = article1.get("abstract", "")
                claim2 = article2.get("abstract", "")

                if not claim1 or not claim2:
                    continue

                # Extract metadata from articles
                metadata1 = {
                    "publication_date": article1.get("publication_date", ""),
                    "source": article1.get("journal", ""),
                    "authors": article1.get("authors", []),
                    "pmid": article1.get("pmid", ""),
                    "doi": article1.get("doi", "")
                }

                metadata2 = {
                    "publication_date": article2.get("publication_date", ""),
                    "source": article2.get("journal", ""),
                    "authors": article2.get("authors", []),
                    "pmid": article2.get("pmid", ""),
                    "doi": article2.get("doi", "")
                }

                # Detect contradiction
                result = await self.detect_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    metadata1=metadata1,
                    metadata2=metadata2,
                    use_biomedlm=True,
                    use_tsmixer=use_all_methods,
                    use_lorentz=use_all_methods,
                    use_temporal=use_all_methods
                )

                # Add to contradictions if score exceeds threshold
                if result["contradiction_score"] >= threshold:
                    contradictions.append({
                        "article1": article1,
                        "article2": article2,
                        "contradiction": result
                    })

        return contradictions
