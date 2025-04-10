"""
Contradiction detection service for the Medical Research Synthesizer.

This module provides a service for detecting contradictions between medical claims.
It implements a simplified MVP version of contradiction detection using rule-based
approaches and text similarity.
"""

import logging
import re
import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.services.temporal_service import TemporalService

# Set up logging
logger = logging.getLogger(__name__)

class ContradictionType(str, Enum):
    """Contradiction type enum."""
    NONE = "none"
    DIRECT = "direct"
    NEGATION = "negation"
    STATISTICAL = "statistical"
    METHODOLOGICAL = "methodological"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"

class ContradictionConfidence(str, Enum):
    """Contradiction confidence enum."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"

class ContradictionService:
    """
    Service for detecting contradictions between medical claims.

    This service provides methods for detecting contradictions between medical claims
    using rule-based approaches and text similarity.
    """

    def __init__(self, biomedlm_service: Optional[BioMedLMService] = None, temporal_service: Optional[TemporalService] = None):
        """Initialize the contradiction service.

        Args:
            biomedlm_service: BioMedLM service for semantic analysis
            temporal_service: Temporal service for temporal analysis
        """
        # Initialize services
        self.biomedlm_service = biomedlm_service
        self.temporal_service = temporal_service

        # Negation patterns for rule-based contradiction detection
        self.negation_patterns = [
            ("not ", ""),
            ("no ", ""),
            ("never ", ""),
            ("doesn't ", "does "),
            ("does not ", "does "),
            ("don't ", "do "),
            ("do not ", "do "),
            ("didn't ", "did "),
            ("did not ", "did "),
            ("isn't ", "is "),
            ("is not ", "is "),
            ("aren't ", "are "),
            ("are not ", "are "),
            ("wasn't ", "was "),
            ("was not ", "was "),
            ("weren't ", "were "),
            ("were not ", "were "),
            ("hasn't ", "has "),
            ("has not ", "has "),
            ("haven't ", "have "),
            ("have not ", "have "),
            ("hadn't ", "had "),
            ("had not ", "had "),
            ("cannot ", "can "),
            ("can't ", "can "),
            ("couldn't ", "could "),
            ("could not ", "could "),
            ("shouldn't ", "should "),
            ("should not ", "should "),
            ("wouldn't ", "would "),
            ("would not ", "would "),
            ("won't ", "will "),
            ("will not ", "will "),
            ("without ", "with "),
            ("absence of ", "presence of "),
            ("lack of ", "presence of "),
            ("failed to ", "succeeded to "),
            ("failure to ", "success to "),
            ("ineffective ", "effective "),
            ("inefficacy ", "efficacy "),
            ("insufficient ", "sufficient "),
            ("inadequate ", "adequate "),
            ("unable to ", "able to "),
            ("inability to ", "ability to "),
        ]

        # Contradiction keywords for rule-based contradiction detection
        self.contradiction_keywords = [
            "contrary",
            "opposite",
            "disagree",
            "conflict",
            "contradict",
            "inconsistent",
            "refute",
            "disprove",
            "rebut",
            "counter",
            "oppose",
            "challenge",
            "dispute",
            "reject",
            "deny",
            "negate",
            "differ",
            "discrepancy",
            "diverge",
            "contrast",
            "versus",
            "vs",
            "against",
            "unlike",
            "whereas",
            "while",
            "but",
            "however",
            "nevertheless",
            "nonetheless",
            "yet",
            "still",
            "though",
            "although",
            "despite",
            "in spite of",
            "notwithstanding",
            "conversely",
            "on the contrary",
            "on the other hand",
            "in contrast",
            "by contrast",
            "instead",
            "rather",
            "alternatively",
        ]

        # Thresholds for contradiction detection
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.NEGATION: 0.8,
            ContradictionType.STATISTICAL: 0.7,
            ContradictionType.METHODOLOGICAL: 0.7,
            ContradictionType.TEMPORAL: 0.7,
        }

        logger.info("Contradiction service initialized")

    async def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None,
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_temporal: bool = True,
        use_tsmixer: bool = True
    ) -> Dict[str, Any]:
        """
        Detect contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim
            threshold: Contradiction detection threshold
            use_biomedlm: Whether to use BioMedLM for semantic analysis

        Returns:
            Contradiction detection result
        """
        logger.info(f"Detecting contradiction between claims: '{claim1}' and '{claim2}'")

        # Initialize result
        result = {
            "is_contradiction": False,
            "contradiction_score": 0.0,
            "contradiction_type": ContradictionType.NONE,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None,
            "methods_used": [],
            "details": {}
        }

        # Detect semantic contradiction using BioMedLM if available and requested
        if use_biomedlm and self.biomedlm_service:
            semantic_result = await self._detect_semantic_contradiction(claim1, claim2)
            result["methods_used"].append("biomedlm")
            result["details"]["semantic"] = semantic_result

            # Update result if semantic contradiction is detected
            if semantic_result["is_contradiction"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = semantic_result["score"]
                result["contradiction_type"] = ContradictionType.DIRECT
                result["confidence"] = semantic_result["confidence"]
                result["explanation"] = semantic_result["explanation"]

        # Detect temporal contradiction if temporal service is available and requested
        if use_temporal and self.temporal_service and metadata1 and metadata2:
            # Pass the use_tsmixer parameter to control whether TSMixer is used for temporal analysis
            temporal_result = await self._detect_temporal_contradiction(
                claim1=claim1,
                claim2=claim2,
                metadata1=metadata1,
                metadata2=metadata2,
                use_tsmixer=use_tsmixer
            )
            result["methods_used"].append("temporal")
            if use_tsmixer and "tsmixer" in temporal_result.get("details", {}):
                result["methods_used"].append("tsmixer")
            result["details"]["temporal"] = temporal_result

            # Update result if temporal contradiction is detected and has higher score
            if temporal_result["is_contradiction"] and temporal_result["score"] > result["contradiction_score"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = temporal_result["score"]
                result["contradiction_type"] = ContradictionType.TEMPORAL
                result["confidence"] = temporal_result["confidence"]
                result["explanation"] = temporal_result["explanation"]

        # Detect negation contradiction
        negation_result = self._detect_negation_contradiction(claim1, claim2)
        result["methods_used"].append("negation")
        result["details"]["negation"] = negation_result

        # Update result if negation contradiction is detected and has higher score
        if negation_result["is_contradiction"] and negation_result["score"] > result["contradiction_score"]:
            result["is_contradiction"] = True
            result["contradiction_score"] = negation_result["score"]
            result["contradiction_type"] = ContradictionType.NEGATION
            result["confidence"] = negation_result["confidence"]
            result["explanation"] = negation_result["explanation"]

        # Detect keyword-based contradiction
        keyword_result = self._detect_keyword_contradiction(claim1, claim2)
        result["methods_used"].append("keyword")
        result["details"]["keyword"] = keyword_result

        # Update result if keyword contradiction is detected and has higher score
        if keyword_result["is_contradiction"] and keyword_result["score"] > result["contradiction_score"]:
            result["is_contradiction"] = True
            result["contradiction_score"] = keyword_result["score"]
            result["contradiction_type"] = ContradictionType.DIRECT
            result["confidence"] = keyword_result["confidence"]
            result["explanation"] = keyword_result["explanation"]

        # Detect statistical contradiction if metadata is available
        if metadata1 and metadata2:
            statistical_result = self._detect_statistical_contradiction(claim1, claim2, metadata1, metadata2)
            result["methods_used"].append("statistical")
            result["details"]["statistical"] = statistical_result

            # Update result if statistical contradiction is detected and has higher score
            if statistical_result["is_contradiction"] and statistical_result["score"] > result["contradiction_score"]:
                result["is_contradiction"] = True
                result["contradiction_score"] = statistical_result["score"]
                result["contradiction_type"] = ContradictionType.STATISTICAL
                result["confidence"] = statistical_result["confidence"]
                result["explanation"] = statistical_result["explanation"]

        logger.info(f"Contradiction detection result: {result['is_contradiction']} (score: {result['contradiction_score']}, type: {result['contradiction_type']})")
        return result

    def _detect_negation_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect negation contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Negation contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }

        # Convert claims to lowercase for case-insensitive matching
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()

        # Check if one claim is the negation of the other
        for pattern, replacement in self.negation_patterns:
            # Check if claim1 contains negation and claim2 doesn't
            if pattern in claim1_lower and pattern not in claim2_lower:
                # Replace negation in claim1
                modified_claim1 = claim1_lower.replace(pattern, replacement)

                # Calculate similarity between modified claim1 and claim2
                if self.biomedlm_service:
                    # Use BioMedLM for semantic similarity if available
                    try:
                        similarity = self.biomedlm_service.calculate_similarity(modified_claim1, claim2_lower)
                    except Exception as e:
                        logger.error(f"Error calculating BioMedLM similarity: {str(e)}")
                        similarity = self._calculate_text_similarity(modified_claim1, claim2_lower)
                else:
                    # Fall back to Jaccard similarity
                    similarity = self._calculate_text_similarity(modified_claim1, claim2_lower)

                if similarity > self.thresholds[ContradictionType.NEGATION]:
                    result["is_contradiction"] = True
                    result["score"] = similarity
                    result["confidence"] = ContradictionConfidence.HIGH
                    result["explanation"] = f"Claim 1 is a negation of Claim 2 with similarity {similarity:.2f}."
                    return result

            # Check if claim2 contains negation and claim1 doesn't
            if pattern in claim2_lower and pattern not in claim1_lower:
                # Replace negation in claim2
                modified_claim2 = claim2_lower.replace(pattern, replacement)

                # Calculate similarity between claim1 and modified claim2
                if self.biomedlm_service:
                    # Use BioMedLM for semantic similarity if available
                    try:
                        similarity = self.biomedlm_service.calculate_similarity(claim1_lower, modified_claim2)
                    except Exception as e:
                        logger.error(f"Error calculating BioMedLM similarity: {str(e)}")
                        similarity = self._calculate_text_similarity(claim1_lower, modified_claim2)
                else:
                    # Fall back to Jaccard similarity
                    similarity = self._calculate_text_similarity(claim1_lower, modified_claim2)

                if similarity > self.thresholds[ContradictionType.NEGATION]:
                    result["is_contradiction"] = True
                    result["score"] = similarity
                    result["confidence"] = ContradictionConfidence.HIGH
                    result["explanation"] = f"Claim 2 is a negation of Claim 1 with similarity {similarity:.2f}."
                    return result

        return result

    def _detect_keyword_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect keyword-based contradiction between two claims.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Keyword contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }

        # Convert claims to lowercase for case-insensitive matching
        claim1_lower = claim1.lower()
        claim2_lower = claim2.lower()

        # Combine claims for keyword search
        combined_text = f"{claim1_lower} {claim2_lower}"

        # Count contradiction keywords
        keyword_count = 0
        found_keywords = []

        for keyword in self.contradiction_keywords:
            if keyword in combined_text:
                keyword_count += 1
                found_keywords.append(keyword)

        # Calculate score based on keyword count
        if keyword_count > 0:
            # Normalize score between 0 and 1, with diminishing returns after 5 keywords
            score = min(1.0, keyword_count / 5.0)

            # Set result
            result["is_contradiction"] = score > self.thresholds[ContradictionType.DIRECT]
            result["score"] = score

            # Set confidence based on score
            if score > 0.9:
                result["confidence"] = ContradictionConfidence.HIGH
            elif score > 0.8:
                result["confidence"] = ContradictionConfidence.MEDIUM
            else:
                result["confidence"] = ContradictionConfidence.LOW

            # Generate explanation
            if result["is_contradiction"]:
                result["explanation"] = f"Found contradiction keywords: {', '.join(found_keywords[:5])}."
                if len(found_keywords) > 5:
                    result["explanation"] += f" and {len(found_keywords) - 5} more."

        return result

    def _detect_statistical_contradiction(
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
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }

        # Check if p-values are available
        p_value1 = metadata1.get("p_value")
        p_value2 = metadata2.get("p_value")

        if p_value1 is not None and p_value2 is not None:
            # Check if one p-value is significant and the other is not
            is_significant1 = p_value1 < 0.05
            is_significant2 = p_value2 < 0.05

            if is_significant1 != is_significant2:
                # Calculate score based on p-value difference
                p_value_diff = abs(p_value1 - p_value2)
                score = min(1.0, p_value_diff)

                # Set result
                result["is_contradiction"] = True
                result["score"] = score

                # Set confidence based on score
                if score > 0.9:
                    result["confidence"] = ContradictionConfidence.HIGH
                elif score > 0.8:
                    result["confidence"] = ContradictionConfidence.MEDIUM
                else:
                    result["confidence"] = ContradictionConfidence.LOW

                # Generate explanation
                result["explanation"] = f"Statistical contradiction: p-value1={p_value1:.3f} (significant={is_significant1}), p-value2={p_value2:.3f} (significant={is_significant2})."

        # Check if effect sizes are available
        effect_size1 = metadata1.get("effect_size")
        effect_size2 = metadata2.get("effect_size")

        if effect_size1 is not None and effect_size2 is not None:
            # Check if effect sizes have opposite signs
            if effect_size1 * effect_size2 < 0:
                # Calculate score based on effect size difference
                effect_size_diff = abs(effect_size1 - effect_size2)
                score = min(1.0, effect_size_diff)

                # Set result if score is higher than current score
                if score > result["score"]:
                    result["is_contradiction"] = True
                    result["score"] = score

                    # Set confidence based on score
                    if score > 0.9:
                        result["confidence"] = ContradictionConfidence.HIGH
                    elif score > 0.8:
                        result["confidence"] = ContradictionConfidence.MEDIUM
                    else:
                        result["confidence"] = ContradictionConfidence.LOW

                    # Generate explanation
                    result["explanation"] = f"Statistical contradiction: effect_size1={effect_size1:.3f}, effect_size2={effect_size2:.3f} (opposite directions)."

        return result

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using a simple Jaccard similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize texts
        tokens1 = set(re.findall(r'\b\w+\b', text1.lower()))
        tokens2 = set(re.findall(r'\b\w+\b', text2.lower()))

        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        if union == 0:
            return 0.0

        return intersection / union

    async def _detect_semantic_contradiction(
        self,
        claim1: str,
        claim2: str
    ) -> Dict[str, Any]:
        """
        Detect semantic contradiction between two claims using BioMedLM.

        Args:
            claim1: First claim
            claim2: Second claim

        Returns:
            Semantic contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }

        try:
            # Use BioMedLM to detect contradiction
            is_contradiction, score = self.biomedlm_service.detect_contradiction(claim1, claim2)

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
                result["explanation"] = f"The claims semantically contradict each other with a score of {score:.2f}."

            return result
        except Exception as e:
            logger.error(f"Error detecting semantic contradiction: {str(e)}")
            return result

    async def _detect_temporal_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any],
        use_tsmixer: bool = True
    ) -> Dict[str, Any]:
        """
        Detect temporal contradiction between two claims based on publication dates and content evolution.

        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for first claim
            metadata2: Metadata for second claim

        Returns:
            Temporal contradiction detection result
        """
        # Initialize result
        result = {
            "is_contradiction": False,
            "score": 0.0,
            "confidence": ContradictionConfidence.UNKNOWN,
            "explanation": None
        }

        try:
            # Extract publication dates
            pub_date1 = self._parse_date(metadata1.get("publication_date"))
            pub_date2 = self._parse_date(metadata2.get("publication_date"))

            # If either date is missing, return default result
            if not pub_date1 or not pub_date2:
                return result

            # Calculate time difference in days
            time_diff = abs((pub_date2 - pub_date1).days)

            # If time difference is less than 30 days, it's unlikely to be a temporal contradiction
            if time_diff < 30:
                return result

            # Calculate semantic similarity between claims
            if self.biomedlm_service:
                similarity = self.biomedlm_service.calculate_similarity(claim1, claim2)
            else:
                similarity = self._calculate_text_similarity(claim1, claim2)

            # Check if one claim is a subset of the other
            is_subset = claim1 in claim2 or claim2 in claim1

            # For temporal contradictions, we need to handle different cases:
            # 1. Identical claims published far apart in time
            # 2. Similar claims with significant semantic overlap published far apart in time
            # 3. Claims where one is an extension of the other (like adding "but resistance is increasing")
            # 4. Claims with different numerical values (like "95%" vs "85%")

            # Check for numerical differences in claims
            import re
            numbers1 = re.findall(r'\d+(?:\.\d+)?%?', claim1)
            numbers2 = re.findall(r'\d+(?:\.\d+)?%?', claim2)
            has_different_numbers = False

            if numbers1 and numbers2 and len(numbers1) == len(numbers2):
                for n1, n2 in zip(numbers1, numbers2):
                    if n1 != n2:
                        has_different_numbers = True
                        break

            # If claims are very similar but published far apart in time, check for temporal factors
            if similarity > 0.7 or is_subset or has_different_numbers:
                # Use TSMixer for temporal sequence analysis if available and requested
                if use_tsmixer and self.temporal_service and self.temporal_service.tsmixer_service:
                    # Create a sequence of claims for TSMixer analysis
                    sequence = [
                        {
                            "text": claim1,
                            "timestamp": str(pub_date1.date()),
                            "domain": metadata1.get("domain", "default"),
                            "publication_date": str(pub_date1.date())
                        },
                        {
                            "text": claim2,
                            "timestamp": str(pub_date2.date()),
                            "domain": metadata2.get("domain", "default"),
                            "publication_date": str(pub_date2.date())
                        }
                    ]

                    # Add additional claims from the same domain if available in metadata
                    if "related_claims" in metadata1:
                        for related_claim in metadata1["related_claims"]:
                            if "text" in related_claim and "timestamp" in related_claim:
                                sequence.append(related_claim)

                    if "related_claims" in metadata2:
                        for related_claim in metadata2["related_claims"]:
                            if "text" in related_claim and "timestamp" in related_claim:
                                sequence.append(related_claim)

                    # Analyze the temporal sequence using TSMixer
                    tsmixer_analysis = await self.temporal_service.analyze_temporal_sequence(sequence)

                    # Extract contradiction scores from TSMixer analysis
                    contradiction_scores = tsmixer_analysis.get("contradiction_scores", [])

                    # If we have contradiction scores, use them to determine if there's a temporal contradiction
                    if contradiction_scores and len(contradiction_scores) >= 2:
                        # The last score represents the contradiction between the two claims
                        tsmixer_contradiction_score = contradiction_scores[-1]

                        # Get trend analysis
                        trend = tsmixer_analysis.get("trend", {})
                        trend_direction = trend.get("direction", "stable")
                        trend_magnitude = trend.get("magnitude", "weak")

                        # Determine if there's a temporal contradiction based on TSMixer analysis
                        if tsmixer_contradiction_score > self.thresholds[ContradictionType.TEMPORAL]:
                            result["is_contradiction"] = True
                            result["score"] = tsmixer_contradiction_score

                            # Set confidence based on trend magnitude
                            if trend_magnitude == "strong":
                                result["confidence"] = ContradictionConfidence.HIGH
                            elif trend_magnitude == "moderate":
                                result["confidence"] = ContradictionConfidence.MEDIUM
                            else:
                                result["confidence"] = ContradictionConfidence.LOW

                            # Generate explanation
                            newer_date = max(pub_date1, pub_date2).strftime("%Y-%m-%d")
                            older_date = min(pub_date1, pub_date2).strftime("%Y-%m-%d")
                            time_diff_years = time_diff / 365.0

                            result["explanation"] = f"Temporal contradiction detected by TSMixer: Claims show a {trend_magnitude} {trend_direction} trend over {time_diff_years:.1f} years ({older_date} vs {newer_date}). The more recent publication likely reflects updated evidence or changing medical knowledge."

                            # Add TSMixer analysis details
                            result["details"] = {
                                "tsmixer_analysis": {
                                    "contradiction_scores": contradiction_scores,
                                    "trend": trend
                                }
                            }

                            return result

                # Fall back to simpler temporal analysis if TSMixer is not available or didn't detect a contradiction
                # Extract study periods if available
                study_period1 = metadata1.get("study_period")
                study_period2 = metadata2.get("study_period")

                # Extract domains if available
                domain1 = metadata1.get("domain", "default")
                domain2 = metadata2.get("domain", "default")

                # Calculate temporal confidence for each claim
                if self.temporal_service:
                    confidence1 = self.temporal_service.calculate_temporal_confidence(
                        str(pub_date1.date()), domain1
                    )
                    confidence2 = self.temporal_service.calculate_temporal_confidence(
                        str(pub_date2.date()), domain2
                    )

                    # Calculate confidence difference
                    confidence_diff = abs(confidence1 - confidence2)
                else:
                    # Simple time-based confidence calculation if temporal service is not available
                    # Older publications have lower confidence
                    max_years = 10  # Maximum years to consider for confidence calculation
                    now = datetime.datetime.now()
                    years_diff1 = min(max_years, (now - pub_date1).days / 365)
                    years_diff2 = min(max_years, (now - pub_date2).days / 365)

                    confidence1 = 1.0 - (years_diff1 / max_years)
                    confidence2 = 1.0 - (years_diff2 / max_years)
                    confidence_diff = abs(confidence1 - confidence2)

                # Calculate contradiction score based on confidence difference and time difference
                # Normalize time difference to a score between 0 and 1
                time_score = min(1.0, time_diff / 365)  # Cap at 1 year

                # For identical claims, the time difference is the primary factor
                if claim1 == claim2:
                    # If claims are identical, we need a significant time difference (> 5 years)
                    # to consider it a temporal contradiction
                    if time_diff > 365 * 5:  # More than 5 years
                        contradiction_score = time_score
                    else:
                        contradiction_score = 0.0
                elif has_different_numbers:
                    # If claims have different numerical values and are published far apart,
                    # it's likely a temporal contradiction with updated information
                    if time_diff > 365 * 5:  # More than 5 years
                        contradiction_score = time_score
                    else:
                        contradiction_score = 0.0
                elif is_subset:
                    # If one claim is a subset of the other and they're published far apart,
                    # it's likely a temporal contradiction with updated information
                    if time_diff > 365 * 5:  # More than 5 years
                        contradiction_score = time_score
                    else:
                        contradiction_score = 0.0
                else:
                    # For similar but not identical claims, combine confidence difference and time score
                    contradiction_score = (confidence_diff * 0.7) + (time_score * 0.3)

                # Check if there are contradictory findings with significant time difference
                if contradiction_score > self.thresholds[ContradictionType.TEMPORAL]:
                    result["is_contradiction"] = True
                    result["score"] = contradiction_score

                    # Set confidence based on score
                    if contradiction_score > 0.9:
                        result["confidence"] = ContradictionConfidence.HIGH
                    elif contradiction_score > 0.8:
                        result["confidence"] = ContradictionConfidence.MEDIUM
                    else:
                        result["confidence"] = ContradictionConfidence.LOW

                    # Generate explanation
                    newer_date = max(pub_date1, pub_date2).strftime("%Y-%m-%d")
                    older_date = min(pub_date1, pub_date2).strftime("%Y-%m-%d")
                    time_diff_years = time_diff / 365.0

                    result["explanation"] = f"Temporal contradiction detected: Claims are similar but published {time_diff_years:.1f} years apart ({older_date} vs {newer_date}). The more recent publication may reflect updated evidence or changing medical knowledge."

            return result
        except Exception as e:
            logger.error(f"Error detecting temporal contradiction: {str(e)}")
            return result

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime.datetime]:
        """
        Parse a date string into a datetime object.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed datetime object or None if parsing fails
        """
        if not date_str:
            return None

        try:
            # Try different date formats
            formats = [
                "%Y-%m-%d",  # 2020-01-01
                "%Y/%m/%d",  # 2020/01/01
                "%d-%m-%Y",  # 01-01-2020
                "%d/%m/%Y",  # 01/01/2020
                "%b %d, %Y",  # Jan 01, 2020
                "%B %d, %Y",  # January 01, 2020
                "%Y"  # 2020 (year only)
            ]

            for fmt in formats:
                try:
                    return datetime.datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # If all formats fail, try to extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
            if year_match:
                year = int(year_match.group(0))
                return datetime.datetime(year, 1, 1)  # Default to January 1st

            return None
        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {str(e)}")
            return None

    async def detect_contradictions_in_articles(
        self,
        articles: List[Dict[str, Any]],
        threshold: float = 0.7,
        use_biomedlm: bool = True,
        use_temporal: bool = True,
        use_tsmixer: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Detect contradictions in a list of articles.

        Args:
            articles: List of articles
            threshold: Contradiction detection threshold

        Returns:
            List of contradictions
        """
        logger.info(f"Detecting contradictions in {len(articles)} articles")

        # Initialize results
        contradictions = []

        # Compare each pair of articles
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                article1 = articles[i]
                article2 = articles[j]

                # Extract claims from articles
                claim1 = article1.get("title", "") + ". " + article1.get("abstract", "")
                claim2 = article2.get("title", "") + ". " + article2.get("abstract", "")

                # Extract metadata from articles
                metadata1 = {
                    "publication_date": article1.get("publication_date"),
                    "study_design": article1.get("study_design"),
                    "sample_size": article1.get("sample_size"),
                    "p_value": article1.get("p_value"),
                    "effect_size": article1.get("effect_size")
                }

                metadata2 = {
                    "publication_date": article2.get("publication_date"),
                    "study_design": article2.get("study_design"),
                    "sample_size": article2.get("sample_size"),
                    "p_value": article2.get("p_value"),
                    "effect_size": article2.get("effect_size")
                }

                # Detect contradiction
                result = await self.detect_contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    metadata1=metadata1,
                    metadata2=metadata2,
                    threshold=threshold,
                    use_biomedlm=use_biomedlm,
                    use_temporal=use_temporal,
                    use_tsmixer=use_tsmixer
                )

                # Add contradiction to results if detected
                if result["is_contradiction"]:
                    contradictions.append({
                        "article1": article1,
                        "article2": article2,
                        "contradiction": result
                    })

        logger.info(f"Found {len(contradictions)} contradictions")
        return contradictions
