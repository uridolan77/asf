"""
Enhanced contradiction detection service for the Medical Research Synthesizer.

This module provides an enhanced service for detecting contradictions in medical literature,
integrating multiple methods and models for more accurate contradiction detection.
"""

import logging
from typing import Dict, List, Optional, Any

from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.models.shap_explainer import SHAPExplainer
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.ml.services.enhanced_contradiction_classifier import EnhancedContradictionClassifier, ContradictionType, ContradictionConfidence
# Import settings if needed
# from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

# ContradictionType and ContradictionConfidence are now imported from enhanced_contradiction_classifier

class EnhancedContradictionService:
    """
    Enhanced service for detecting contradictions in medical literature.

    This service integrates multiple methods and models for more accurate contradiction detection,
    including BioMedLM, TSMixer, Lorentz embeddings, and SHAP explainability.
    """

    def __init__(
        self,
        biomedlm_service: Optional[BioMedLMService] = None,
        temporal_service: Optional[TemporalService] = None,
        shap_explainer: Optional[SHAPExplainer] = None
    ):
        """
        Initialize the enhanced contradiction service.

        Args:
            biomedlm_service: BioMedLM service for semantic analysis
            temporal_service: Temporal service for temporal analysis
            shap_explainer: SHAP explainer for explainability
        """
        self.biomedlm_service = biomedlm_service
        self.temporal_service = temporal_service
        self.shap_explainer = shap_explainer

        # Initialize the enhanced contradiction classifier
        self.contradiction_classifier = EnhancedContradictionClassifier()

        # Configure thresholds
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.NEGATION: 0.8,
            ContradictionType.TEMPORAL: 0.6,
            ContradictionType.HIERARCHICAL: 0.65,
            ContradictionType.METHODOLOGICAL: 0.75,
            ContradictionType.STATISTICAL: 0.8,
            ContradictionType.POPULATION: 0.7
        }

        logger.info("Enhanced contradiction service initialized")

    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        use_all_methods: bool = True
    ) -> Dict[str, Any]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim
            use_all_methods: Whether to use all available methods

        Returns:
            Contradiction detection result
        """
        logger.info(f"Detecting contradiction between '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "claim1": claim1,
            "claim2": claim2,
            "is_contradiction": False,
            "contradiction_score": 0.0,
            "contradiction_type": ContradictionType.UNKNOWN,
            "confidence": ContradictionConfidence.LOW,
            "methods_used": [],
            "explanation": "",
            "details": {}
        }

        # Detect direct contradiction using BioMedLM
        if self.biomedlm_service:
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

        # If using all methods, continue with other detection methods
        if use_all_methods:
            # Detect negation contradiction
            negation_result = await self._detect_negation_contradiction(claim1, claim2)
            result["methods_used"].append("negation")
            result["details"]["negation"] = negation_result

            # Update result if negation contradiction is detected and has higher score
            if negation_result["is_contradiction"] and negation_result["score"] > result["contradiction_score"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = negation_result["score"]
                result["contradiction_type"] = ContradictionType.NEGATION
                result["confidence"] = negation_result["confidence"]
                result["explanation"] = negation_result["explanation"]

            # Detect temporal contradiction if temporal service is available
            if self.temporal_service:
                temporal_result = await self._detect_temporal_contradiction(
                    claim1, claim2, metadata1, metadata2
                )
                result["methods_used"].append("temporal")
                result["details"]["temporal"] = temporal_result

                # Update result if temporal contradiction is detected and has higher score
                if temporal_result["is_contradiction"] and temporal_result["score"] > result["contradiction_score"]:
                    result["is_contradiction"] = True
                    result["contradiction_score"] = temporal_result["score"]
                    result["contradiction_type"] = ContradictionType.TEMPORAL
                    result["confidence"] = temporal_result["confidence"]
                    result["explanation"] = temporal_result["explanation"]

            # Detect methodological contradiction if metadata is available
            if metadata1 and metadata2:
                methodological_result = await self._detect_methodological_contradiction(
                    claim1, claim2, metadata1, metadata2
                )
                result["methods_used"].append("methodological")
                result["details"]["methodological"] = methodological_result

                # Update result if methodological contradiction is detected and has higher score
                if methodological_result["is_contradiction"] and methodological_result["score"] > result["contradiction_score"]:
                    result["is_contradiction"] = True
                    result["contradiction_score"] = methodological_result["score"]
                    result["contradiction_type"] = ContradictionType.METHODOLOGICAL
                    result["confidence"] = methodological_result["confidence"]
                    result["explanation"] = methodological_result["explanation"]

            # Detect statistical contradiction if metadata is available
            if metadata1 and metadata2:
                statistical_result = await self._detect_statistical_contradiction(
                    claim1, claim2, metadata1, metadata2
                )
                result["methods_used"].append("statistical")
                result["details"]["statistical"] = statistical_result

                # Update result if statistical contradiction is detected and has higher score
                if statistical_result["is_contradiction"] and statistical_result["score"] > result["contradiction_score"]:
                    result["is_contradiction"] = True
                    result["contradiction_score"] = statistical_result["score"]
                    result["contradiction_type"] = ContradictionType.STATISTICAL
                    result["confidence"] = statistical_result["confidence"]
                    result["explanation"] = statistical_result["explanation"]

        # Generate explanation using SHAP if available and not already set
        if self.shap_explainer and result["is_contradiction"] and not result["explanation"]:
            explanation = await self._generate_explanation(claim1, claim2, result["contradiction_type"])
            result["explanation"] = explanation

        # Apply enhanced classification if contradiction is detected
        if result["is_contradiction"]:
            result = await self._apply_enhanced_classification(result, metadata1, metadata2)

        logger.info(f"Contradiction detection result: {result['is_contradiction']} (score: {result['contradiction_score']}, type: {result['contradiction_type']})")
        return result

    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        threshold: float = 0.7,
        use_all_methods: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions in a list of articles.

        Args:
            articles: List of articles
            threshold: Threshold for contradiction detection
            use_all_methods: Whether to use all available methods

        Returns:
            List of detected contradictions
        """
        logger.info(f"Detecting contradictions in {len(articles)} articles")

        contradictions = []

        # Compare each pair of articles
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                article1 = articles[i]
                article2 = articles[j]

                # Extract claims from articles
                claim1 = self._extract_main_claim(article1)
                claim2 = self._extract_main_claim(article2)

                if not claim1 or not claim2:
                    continue

                # Extract metadata
                metadata1 = self._extract_metadata(article1)
                metadata2 = self._extract_metadata(article2)

                # Detect contradiction
                result = await self.detect_contradiction(
                    claim1, claim2, metadata1, metadata2, use_all_methods
                )

                # Add to contradictions if above threshold
                if result["is_contradiction"] and result["contradiction_score"] >= threshold:
                    contradiction = {
                        "article1": {
                            "id": article1.get("pmid", ""),
                            "title": article1.get("title", ""),
                            "claim": claim1
                        },
                        "article2": {
                            "id": article2.get("pmid", ""),
                            "title": article2.get("title", ""),
                            "claim": claim2
                        },
                        "contradiction_score": result["contradiction_score"],
                        "contradiction_type": result["contradiction_type"],
                        "confidence": result["confidence"],
                        "explanation": result["explanation"]
                    }

                    # Add classification if available
                    if "classification" in result:
                        contradiction["classification"] = result["classification"]

                    contradictions.append(contradiction)

        logger.info(f"Detected {len(contradictions)} contradictions")
        return contradictions

    async def _detect_direct_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect direct contradiction between two claims using BioMedLM.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Direct contradiction detection result
        """
        logger.debug(f"Detecting direct contradiction between '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.LOW,
            "explanation": ""
        }

        try:
            # Use BioMedLM to detect contradiction
            _, score = self.biomedlm_service.detect_contradiction(claim1, claim2)

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

    async def _detect_negation_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect negation contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Negation contradiction detection result
        """
        logger.debug(f"Detecting negation contradiction between '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.LOW,
            "explanation": ""
        }

        # Check for negation patterns
        negation_patterns = [
            ("not ", ""),
            ("no ", ""),
            ("never ", ""),
            ("doesn't ", "does "),
            ("don't ", "do "),
            ("isn't ", "is "),
            ("aren't ", "are "),
            ("cannot ", "can "),
            ("can't ", "can "),
            ("won't ", "will ")
        ]

        # Normalize claims
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()

        # Check if one claim is the negation of the other
        for pattern, replacement in negation_patterns:
            # Check if claim1 contains negation and claim2 doesn't
            if pattern in claim1_lower and pattern not in claim2_lower:
                # Replace negation in claim1
                modified_claim1 = claim1_lower.replace(pattern, replacement)

                # Calculate similarity between modified claim1 and claim2
                if self.biomedlm_service:
                    similarity = self.biomedlm_service.calculate_similarity(modified_claim1, claim2_lower)

                    if similarity > 0.8:  # High similarity threshold
                        result["is_contradiction"] = True
                        result["score"] = float(similarity)
                        result["confidence"] = ContradictionConfidence.HIGH
                        result["explanation"] = f"Claim 1 is a negation of Claim 2 with similarity {similarity:.2f}."
                        return result

            # Check if claim2 contains negation and claim1 doesn't
            if pattern in claim2_lower and pattern not in claim1_lower:
                # Replace negation in claim2
                modified_claim2 = claim2_lower.replace(pattern, replacement)

                # Calculate similarity between claim1 and modified claim2
                if self.biomedlm_service:
                    similarity = self.biomedlm_service.calculate_similarity(claim1_lower, modified_claim2)

                    if similarity > 0.8:  # High similarity threshold
                        result["is_contradiction"] = True
                        result["score"] = float(similarity)
                        result["confidence"] = ContradictionConfidence.HIGH
                        result["explanation"] = f"Claim 2 is a negation of Claim 1 with similarity {similarity:.2f}."
                        return result

        return result

    async def _detect_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect temporal contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim

        Returns:
            Temporal contradiction detection result
        """
        logger.debug(f"Detecting temporal contradiction between '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.LOW,
            "explanation": ""
        }

        # If temporal service is not available, return default result
        if not self.temporal_service:
            return result

        try:
            # Extract publication dates from metadata
            pub_date1 = metadata1.get("publication_date") if metadata1 else None
            pub_date2 = metadata2.get("publication_date") if metadata2 else None

            # Create sequence for temporal analysis
            sequence = [
                {"claim": claim1, "timestamp": pub_date1 if pub_date1 else 0},
                {"claim": claim2, "timestamp": pub_date2 if pub_date2 else 1}
            ]

            # Analyze temporal sequence
            if self.biomedlm_service:
                analysis = await self.temporal_service.analyze_temporal_sequence(
                    sequence,
                    embedding_fn=self.biomedlm_service.encode
                )

                # Check for temporal contradiction
                if "contradiction_scores" in analysis and len(analysis["contradiction_scores"]) > 1:
                    score = analysis["contradiction_scores"][1]

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

                    # Generate explanation
                    if result["is_contradiction"]:
                        time_diff = ""
                        if pub_date1 and pub_date2:
                            time_diff = f" published {abs(pub_date2 - pub_date1)} days apart"

                        result["explanation"] = f"The claims show temporal contradiction{time_diff} with a score of {score:.2f}."

            return result
        except Exception as e:
            logger.error(f"Error detecting temporal contradiction: {str(e)}")
            return result

    async def _detect_methodological_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect methodological contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim

        Returns:
            Methodological contradiction detection result
        """
        logger.debug(f"Detecting methodological contradiction between '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.LOW,
            "explanation": ""
        }

        try:
            # Check if claims are semantically contradictory
            if self.biomedlm_service:
                _, semantic_score = self.biomedlm_service.detect_contradiction(claim1, claim2)

                # If claims are semantically contradictory, check for methodological differences
                if semantic_score > 0.6:  # Lower threshold for potential contradiction
                    # Extract study design information
                    design1 = metadata1.get("study_design", "").lower()
                    design2 = metadata2.get("study_design", "").lower()

                    # Extract sample size information
                    sample_size1 = metadata1.get("sample_size", 0)
                    sample_size2 = metadata2.get("sample_size", 0)

                    # Calculate methodological score
                    method_score = 0.0
                    explanation_parts = []

                    # Check for different study designs
                    if design1 and design2 and design1 != design2:
                        method_score += 0.3
                        explanation_parts.append(f"different study designs ({design1} vs {design2})")

                    # Check for large sample size difference
                    if sample_size1 and sample_size2:
                        size_ratio = max(sample_size1, sample_size2) / min(sample_size1, sample_size2)
                        if size_ratio > 5:  # Significant difference
                            method_score += 0.3
                            explanation_parts.append(f"large sample size difference ({sample_size1} vs {sample_size2})")

                    # Check for different populations
                    population1 = metadata1.get("population", "").lower()
                    population2 = metadata2.get("population", "").lower()

                    if population1 and population2 and population1 != population2:
                        method_score += 0.2
                        explanation_parts.append(f"different populations ({population1} vs {population2})")

                    # Calculate final score
                    final_score = semantic_score * (0.7 + method_score)

                    # Set result
                    result["is_contradiction"] = final_score > self.thresholds[ContradictionType.METHODOLOGICAL]
                    result["score"] = float(final_score)

                    # Set confidence based on score
                    if final_score > 0.9:
                        result["confidence"] = ContradictionConfidence.HIGH
                    elif final_score > 0.8:
                        result["confidence"] = ContradictionConfidence.MEDIUM
                    else:
                        result["confidence"] = ContradictionConfidence.LOW

                    # Generate explanation
                    if result["is_contradiction"] and explanation_parts:
                        result["explanation"] = f"The claims contradict each other with methodological differences: {', '.join(explanation_parts)}."

            return result
        except Exception as e:
            logger.error(f"Error detecting methodological contradiction: {str(e)}")
            return result

    async def _detect_statistical_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect statistical contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim

        Returns:
            Statistical contradiction detection result
        """
        logger.debug(f"Detecting statistical contradiction between '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.LOW,
            "explanation": ""
        }

        try:
            # Check if claims are semantically contradictory
            if self.biomedlm_service:
                _, semantic_score = self.biomedlm_service.detect_contradiction(claim1, claim2)

                # If claims are semantically contradictory, check for statistical differences
                if semantic_score > 0.6:  # Lower threshold for potential contradiction
                    # Extract p-values
                    p_value1 = metadata1.get("p_value")
                    p_value2 = metadata2.get("p_value")

                    # Extract confidence intervals
                    ci1 = metadata1.get("confidence_interval")
                    ci2 = metadata2.get("confidence_interval")

                    # Calculate statistical score
                    stat_score = 0.0
                    explanation_parts = []

                    # Check for statistical significance difference
                    if p_value1 is not None and p_value2 is not None:
                        if (p_value1 < 0.05 and p_value2 >= 0.05) or (p_value1 >= 0.05 and p_value2 < 0.05):
                            stat_score += 0.4
                            explanation_parts.append(f"different statistical significance (p={p_value1} vs p={p_value2})")

                    # Check for non-overlapping confidence intervals
                    if ci1 and ci2:
                        ci1_lower, ci1_upper = ci1
                        ci2_lower, ci2_upper = ci2

                        if ci1_upper < ci2_lower or ci2_upper < ci1_lower:
                            stat_score += 0.4
                            explanation_parts.append(f"non-overlapping confidence intervals ({ci1} vs {ci2})")

                    # Calculate final score
                    final_score = semantic_score * (0.7 + stat_score)

                    # Set result
                    result["is_contradiction"] = final_score > self.thresholds[ContradictionType.STATISTICAL]
                    result["score"] = float(final_score)

                    # Set confidence based on score
                    if final_score > 0.9:
                        result["confidence"] = ContradictionConfidence.HIGH
                    elif final_score > 0.8:
                        result["confidence"] = ContradictionConfidence.MEDIUM
                    else:
                        result["confidence"] = ContradictionConfidence.LOW

                    # Generate explanation
                    if result["is_contradiction"] and explanation_parts:
                        result["explanation"] = f"The claims contradict each other with statistical differences: {', '.join(explanation_parts)}."

            return result
        except Exception as e:
            logger.error(f"Error detecting statistical contradiction: {str(e)}")
            return result

    async def _generate_explanation(
        self,
        claim1: str,
        claim2: str,
        contradiction_type: ContradictionType
    ) -> str:
        """
        Generate explanation for contradiction.

        Args:
            claim1: First claim
            claim2: Second claim
            contradiction_type: Type of contradiction

        Returns:
            Explanation string
        """
        logger.debug(f"Generating explanation for {contradiction_type} contradiction")

        # If SHAP explainer is available, use it
        if self.shap_explainer:
            try:
                explanation = self.shap_explainer.explain_contradiction(claim1, claim2)
                return explanation.get("explanation", "")
            except Exception as e:
                logger.error(f"Error generating SHAP explanation: {str(e)}")

        # Default explanations based on contradiction type
        if contradiction_type == ContradictionType.DIRECT:
            return "The claims directly contradict each other."
        elif contradiction_type == ContradictionType.NEGATION:
            return "One claim is a negation of the other."
        elif contradiction_type == ContradictionType.TEMPORAL:
            return "The claims show temporal contradiction."
        elif contradiction_type == ContradictionType.HIERARCHICAL:
            return "The claims show hierarchical contradiction."
        elif contradiction_type == ContradictionType.METHODOLOGICAL:
            return "The claims contradict each other due to methodological differences."
        elif contradiction_type == ContradictionType.STATISTICAL:
            return "The claims contradict each other due to statistical differences."
        else:
            return "The claims contradict each other."

    def _extract_main_claim(self, article: Dict[str, Any]) -> str:
        """
        Extract main claim from an article.

        Args:
            article: Article data

        Returns:
            Main claim
        """
        # Check if article has a conclusion
        if "conclusion" in article and article["conclusion"]:
            return article["conclusion"]

        # Check if article has an abstract
        if "abstract" in article and article["abstract"]:
            # Try to extract conclusion from abstract
            abstract = article["abstract"].lower()

            # Look for conclusion section
            conclusion_markers = [
                "conclusion:", "conclusions:", "in conclusion", "we conclude",
                "our results suggest", "our findings suggest", "these results suggest",
                "these findings suggest", "this study suggests", "our study suggests"
            ]

            for marker in conclusion_markers:
                if marker in abstract:
                    # Extract text after marker
                    start_idx = abstract.find(marker) + len(marker)
                    end_idx = abstract.find(".", start_idx)

                    if end_idx != -1:
                        return abstract[start_idx:end_idx + 1].strip()

            # If no conclusion found, return last sentence of abstract
            sentences = abstract.split(".")
            if sentences:
                return sentences[-2].strip() if len(sentences) > 1 else sentences[-1].strip()

        # If no abstract, use title
        if "title" in article and article["title"]:
            return article["title"]

        return ""

    async def _apply_enhanced_classification(self, result: Dict[str, Any], metadata1: Optional[Dict[str, Any]], metadata2: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply enhanced classification to a contradiction result.

        Args:
            result: Contradiction detection result
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim

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

    def _extract_metadata(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from an article.

        Args:
            article: Article data

        Returns:
            Metadata dictionary
        """
        metadata = {}

        # Extract publication date
        if "publication_date" in article:
            metadata["publication_date"] = article["publication_date"]

        # Extract study design
        if "study_design" in article:
            metadata["study_design"] = article["study_design"]

        # Extract sample size
        if "sample_size" in article:
            metadata["sample_size"] = article["sample_size"]

        # Extract population
        if "population" in article:
            metadata["population"] = article["population"]

        # Extract p-value
        if "p_value" in article:
            metadata["p_value"] = article["p_value"]

        # Extract confidence interval
        if "confidence_interval" in article:
            metadata["confidence_interval"] = article["confidence_interval"]

        return metadata
