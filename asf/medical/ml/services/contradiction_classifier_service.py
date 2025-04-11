Contradiction Classification Service for Medical Research Synthesizer.
This module provides multi-dimensional classification of medical contradictions,
integrating clinical significance assessment, evidence quality assessment,
temporal factor detection, population difference detection, and methodological
difference detection.
import logging
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any
from asf.medical.core.exceptions import OperationError
logger = logging.getLogger(__name__)
class ContradictionType(str, Enum):
    Types of contradictions that can be detected.
        Assess temporal factors for potential contradiction.
        Args:
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
        Returns:
            Dictionary with temporal factor assessment
        """
        # Default values
        result = {
            "detected": False,
            "score": 0.0,
            "publication_date_difference": None,
            "factors": []
        }
        # Check for publication dates
        pub_date1 = metadata1.get("publication_date")
        pub_date2 = metadata2.get("publication_date")
        if pub_date1 and pub_date2:
            try:
                # Parse dates
                if isinstance(pub_date1, str):
                    date1 = datetime.fromisoformat(pub_date1.replace("Z", "+00:00"))
                else:
                    date1 = pub_date1
                if isinstance(pub_date2, str):
                    date2 = datetime.fromisoformat(pub_date2.replace("Z", "+00:00"))
                else:
                    date2 = pub_date2
                # Calculate difference in years
                diff_years = abs((date1.year - date2.year) + 
                                (date1.month - date2.month) / 12)
                result["publication_date_difference"] = round(diff_years, 1)
                # Assess temporal factor
                if diff_years > 10:
                    result["detected"] = True
                    result["score"] = 0.8
                    result["factors"].append(f"Publications are {round(diff_years, 1)} years apart")
                elif diff_years > 5:
                    result["detected"] = True
                    result["score"] = 0.6
                    result["factors"].append(f"Publications are {round(diff_years, 1)} years apart")
                elif diff_years > 2:
                    result["detected"] = True
                    result["score"] = 0.4
                    result["factors"].append(f"Publications are {round(diff_years, 1)} years apart")
            except Exception as e:
                logger.warning(f"Error parsing publication dates: {e}")
                raise OperationError(f"Operation failed: {str(e)}")
                raise OperationError(f"Operation failed: {str(e)}")
        return result
    def _assess_population_difference(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess population differences for potential contradiction.
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
        Returns:
            Dictionary with population difference assessment
        """
        # Default values
        result = {
            "detected": False,
            "score": 0.0,
            "differences": []
        }
        # Check for population information
        population1 = metadata1.get("population", "")
        population2 = metadata2.get("population", "")
        if population1 and population2 and population1 != population2:
            result["detected"] = True
            result["score"] = 0.7
            result["differences"].append(f"Different populations: '{population1}' vs '{population2}'")
        # Check for age differences
        age1 = metadata1.get("age_range", "")
        age2 = metadata2.get("age_range", "")
        if age1 and age2 and age1 != age2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.6)
            result["differences"].append(f"Different age ranges: '{age1}' vs '{age2}'")
        # Check for gender differences
        gender1 = metadata1.get("gender", "")
        gender2 = metadata2.get("gender", "")
        if gender1 and gender2 and gender1 != gender2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.6)
            result["differences"].append(f"Different gender focus: '{gender1}' vs '{gender2}'")
        # Check for ethnicity differences
        ethnicity1 = metadata1.get("ethnicity", "")
        ethnicity2 = metadata2.get("ethnicity", "")
        if ethnicity1 and ethnicity2 and ethnicity1 != ethnicity2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.7)
            result["differences"].append(f"Different ethnicities: '{ethnicity1}' vs '{ethnicity2}'")
        # Check for geographic differences
        location1 = metadata1.get("location", "")
        location2 = metadata2.get("location", "")
        if location1 and location2 and location1 != location2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.5)
            result["differences"].append(f"Different locations: '{location1}' vs '{location2}'")
        return result
    def _assess_methodological_difference(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess methodological differences for potential contradiction.
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
        Returns:
            Dictionary with methodological difference assessment
        """
        # Default values
        result = {
            "detected": False,
            "score": 0.0,
            "differences": []
        }
        # Check for study type differences
        study_type1 = metadata1.get("study_type", "")
        study_type2 = metadata2.get("study_type", "")
        if study_type1 and study_type2 and study_type1 != study_type2:
            result["detected"] = True
            result["score"] = 0.7
            result["differences"].append(f"Different study types: '{study_type1}' vs '{study_type2}'")
        # Check for intervention differences
        intervention1 = metadata1.get("intervention", "")
        intervention2 = metadata2.get("intervention", "")
        if intervention1 and intervention2 and intervention1 != intervention2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.6)
            result["differences"].append(f"Different interventions: '{intervention1}' vs '{intervention2}'")
        # Check for outcome measure differences
        outcome1 = metadata1.get("outcome", "")
        outcome2 = metadata2.get("outcome", "")
        if outcome1 and outcome2 and outcome1 != outcome2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.6)
            result["differences"].append(f"Different outcome measures: '{outcome1}' vs '{outcome2}'")
        # Check for statistical method differences
        stats1 = metadata1.get("statistical_method", "")
        stats2 = metadata2.get("statistical_method", "")
        if stats1 and stats2 and stats1 != stats2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.5)
            result["differences"].append(f"Different statistical methods: '{stats1}' vs '{stats2}'")
        return result
    async def classify_contradiction(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify a potential contradiction between two claims.
        Args:
            claim1: First claim text
            claim2: Second claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
        Returns:
            Dictionary with contradiction classification results
        """
        # Ensure we have valid inputs
        if not claim1 or not claim2:
            return {
                "is_contradiction": False,
                "contradiction_type": ContradictionType.UNKNOWN,
                "confidence": ContradictionConfidence.UNKNOWN,
                "score": 0.0,
                "explanation": "Insufficient input data",
                "evidence": [],
                "analysis": {}
            }
        # Initialize metadata if None
        metadata1 = metadata1 or {}
        metadata2 = metadata2 or {}
        # Perform detailed analysis of the claims
        clinical_significance = await self._assess_clinical_significance(claim1, claim2, metadata1, metadata2)
        evidence_quality1 = self._assess_evidence_quality(metadata1)
        evidence_quality2 = self._assess_evidence_quality(metadata2)
        temporal_factor = self._assess_temporal_factor(metadata1, metadata2)
        population_difference = self._assess_population_difference(claim1, claim2, metadata1, metadata2)
        methodological_difference = self._assess_methodological_difference(claim1, claim2, metadata1, metadata2)
        # Determine if there's a contradiction
        is_contradiction = False
        contradiction_score = 0.0
        contradiction_type = ContradictionType.UNKNOWN
        contradiction_confidence = ContradictionConfidence.UNKNOWN
        explanation = ""
        evidence = []
        # Check for direct contradiction
        if self.biomedlm_service:
            try:
                direct_result = await self.biomedlm_service.detect_contradiction(claim1, claim2)
                is_direct_contradiction, direct_score = direct_result
                if direct_score > self.thresholds[ContradictionType.DIRECT]:
                    is_contradiction = True
                    contradiction_score = direct_score
                    contradiction_type = ContradictionType.DIRECT
                    evidence.append({
                        "type": "direct",
                        "score": direct_score,
                        "source": "BioMedLM"
                    })
            except Exception as e:
                logger.warning(f"Error detecting direct contradiction: {e}")
                raise OperationError(f"Operation failed: {str(e)}")
                raise OperationError(f"Operation failed: {str(e)}")
        # Check for temporal contradiction
        if temporal_factor["detected"] and temporal_factor["score"] > self.thresholds[ContradictionType.TEMPORAL]:
            is_contradiction = True
            if temporal_factor["score"] > contradiction_score:
                contradiction_score = temporal_factor["score"]
                contradiction_type = ContradictionType.TEMPORAL
            evidence.append({
                "type": "temporal",
                "score": temporal_factor["score"],
                "factors": temporal_factor["factors"]
            })
        # Check for population differences
        if population_difference["detected"] and population_difference["score"] > self.thresholds[ContradictionType.POPULATION]:
            # Population differences don't necessarily indicate contradiction, but contribute to explanation
            evidence.append({
                "type": "population",
                "score": population_difference["score"],
                "differences": population_difference["differences"]
            })
        # Check for methodological differences
        if methodological_difference["detected"] and methodological_difference["score"] > self.thresholds[ContradictionType.METHODOLOGICAL]:
            # Methodological differences don't necessarily indicate contradiction, but contribute to explanation
            evidence.append({
                "type": "methodological",
                "score": methodological_difference["score"],
                "differences": methodological_difference["differences"]
            })
        # Determine confidence level
        if contradiction_score >= 0.8:
            contradiction_confidence = ContradictionConfidence.HIGH
        elif contradiction_score >= 0.6:
            contradiction_confidence = ContradictionConfidence.MODERATE
        elif contradiction_score > 0:
            contradiction_confidence = ContradictionConfidence.LOW
        # Generate explanation
        if is_contradiction:
            explanation = f"Detected a {contradiction_confidence.lower()} confidence {contradiction_type.lower()} contradiction between the claims."
            if contradiction_type == ContradictionType.TEMPORAL:
                explanation += f" The claims are separated by {temporal_factor.get('publication_date_difference', 'unknown')} years."
            if clinical_significance["significance"] != ClinicalSignificance.UNKNOWN:
                explanation += f" This contradiction has {clinical_significance['significance'].lower()} clinical significance."
            if evidence_quality1["quality"] != EvidenceQuality.UNKNOWN and evidence_quality2["quality"] != EvidenceQuality.UNKNOWN:
                explanation += f" The evidence quality is {evidence_quality1['quality'].lower()} for the first claim and {evidence_quality2['quality'].lower()} for the second claim."
        else:
            explanation = "No significant contradiction detected between the claims."
        # Compile analysis results
        analysis = {
            "clinical_significance": {
                "significance": clinical_significance["significance"],
                "score": clinical_significance["score"],
                "terms": clinical_significance["terms"]
            },
            "evidence_quality": {
                "claim1": evidence_quality1["quality"],
                "claim1_score": evidence_quality1["score"],
                "claim1_factors": evidence_quality1["factors"],
                "claim2": evidence_quality2["quality"],
                "claim2_score": evidence_quality2["score"],
                "claim2_factors": evidence_quality2["factors"],
                "differential": evidence_quality1["score"] - evidence_quality2["score"]
            },
            "temporal_factor": temporal_factor,
            "population_difference": population_difference,
            "methodological_difference": methodological_difference
        }
        # Return the final result
        return {
            "is_contradiction": is_contradiction,
            "contradiction_type": contradiction_type,
            "confidence": contradiction_confidence,
            "score": contradiction_score,
            "explanation": explanation,
            "evidence": evidence,
            "analysis": analysis
        }