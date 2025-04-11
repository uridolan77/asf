"""Contradiction Classification Service for Medical Research Synthesizer.

This module provides multi-dimensional classification of medical contradictions,
integrating clinical significance assessment, evidence quality assessment,
temporal factor detection, population difference detection, and methodological
difference detection.
"""

import logging
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any, List, Tuple

from asf.medical.core.exceptions import OperationError
from asf.medical.ml.models.biomedlm import BioMedLMService

logger = logging.getLogger(__name__)


class ContradictionType(str, Enum):
    """Types of contradictions that can be detected."""
    
    UNKNOWN = "unknown"
    DIRECT = "direct"
    TEMPORAL = "temporal"
    POPULATION = "population"
    METHODOLOGICAL = "methodological"


class ContradictionConfidence(str, Enum):
    """Confidence levels for contradiction detection."""
    
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ClinicalSignificance(str, Enum):
    """Clinical significance levels for contradictions."""
    
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EvidenceQuality(str, Enum):
    """Evidence quality levels for claims."""
    
    UNKNOWN = "unknown"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class ContradictionClassifierService:
    """
    Service for classifying contradictions between medical claims.
    
    This service provides multi-dimensional classification of contradictions,
    analyzing temporal factors, population differences, methodological differences,
    clinical significance, and evidence quality.
    """
    
    def __init__(self, biomedlm_service: Optional[BioMedLMService] = None):
        """
        Initialize the contradiction classifier service.
        
        Args:
            biomedlm_service: BioMedLM service for semantic analysis
        """
        self.biomedlm_service = biomedlm_service
        
        # Thresholds for different contradiction types
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.TEMPORAL: 0.6,
            ContradictionType.POPULATION: 0.5,
            ContradictionType.METHODOLOGICAL: 0.5
        }
        
        # Clinical significance terms
        self.high_significance_terms = [
            "mortality", "death", "fatal", "life-threatening", "survival",
            "severe", "critical", "emergency", "urgent", "intensive care"
        ]
        
        self.moderate_significance_terms = [
            "hospitalization", "disability", "impairment", "chronic",
            "persistent", "long-term", "significant", "substantial"
        ]
        
        logger.info("Contradiction classifier service initialized")
    
    def _assess_temporal_factor(
        self,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
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
            "factors": []
        }
        
        # Check for publication dates
        pub_date1 = metadata1.get("publication_date")
        pub_date2 = metadata2.get("publication_date")
        
        if pub_date1 and pub_date2:
            try:
                # Parse dates
                date1 = datetime.fromisoformat(pub_date1.replace("Z", "+00:00"))
                date2 = datetime.fromisoformat(pub_date2.replace("Z", "+00:00"))
                
                # Calculate difference in years
                diff_days = abs((date1 - date2).days)
                diff_years = diff_days / 365.0
                
                # Store the difference
                result["publication_date_difference"] = round(diff_years, 1)
                
                # Assess temporal contradiction based on time difference
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
    
    def _assess_evidence_quality(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the quality of evidence for a claim.
        
        Args:
            metadata: Metadata for the claim
            
        Returns:
            Dictionary with evidence quality assessment
        """
        # Default values
        result = {
            "quality": EvidenceQuality.UNKNOWN,
            "score": 0.0,
            "factors": []
        }
        
        # Check for study type
        study_type = metadata.get("study_type", "").lower()
        
        if "meta-analysis" in study_type or "systematic review" in study_type:
            result["quality"] = EvidenceQuality.HIGH
            result["score"] = 0.9
            result["factors"].append(f"High-quality study type: {study_type}")
        elif "randomized" in study_type or "rct" in study_type:
            result["quality"] = EvidenceQuality.HIGH
            result["score"] = 0.8
            result["factors"].append(f"High-quality study type: {study_type}")
        elif "cohort" in study_type or "case-control" in study_type:
            result["quality"] = EvidenceQuality.MODERATE
            result["score"] = 0.6
            result["factors"].append(f"Moderate-quality study type: {study_type}")
        elif "case series" in study_type or "case report" in study_type:
            result["quality"] = EvidenceQuality.LOW
            result["score"] = 0.3
            result["factors"].append(f"Lower-quality study type: {study_type}")
        
        # Check for sample size
        sample_size = metadata.get("sample_size", 0)
        
        if isinstance(sample_size, str):
            try:
                sample_size = int(sample_size)
            except ValueError:
                sample_size = 0
        
        if sample_size > 10000:
            result["score"] += 0.2
            result["factors"].append(f"Large sample size: {sample_size}")
        elif sample_size > 1000:
            result["score"] += 0.1
            result["factors"].append(f"Good sample size: {sample_size}")
        elif sample_size < 30 and sample_size > 0:
            result["score"] -= 0.2
            result["factors"].append(f"Small sample size: {sample_size}")
        
        # Check for journal impact factor
        impact_factor = metadata.get("journal_impact_factor", 0)
        
        if isinstance(impact_factor, str):
            try:
                impact_factor = float(impact_factor)
            except ValueError:
                impact_factor = 0
        
        if impact_factor > 10:
            result["score"] += 0.1
            result["factors"].append(f"High-impact journal: {impact_factor}")
        
        # Determine final quality based on score
        if result["score"] >= 0.7:
            result["quality"] = EvidenceQuality.HIGH
        elif result["score"] >= 0.4:
            result["quality"] = EvidenceQuality.MODERATE
        elif result["score"] > 0:
            result["quality"] = EvidenceQuality.LOW
        
        return result
    
    async def _assess_clinical_significance(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the clinical significance of a potential contradiction.
        
        Args:
            claim1: First claim
            claim2: Second claim
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim
            
        Returns:
            Dictionary with clinical significance assessment
        """
        # Default values
        result = {
            "significance": ClinicalSignificance.UNKNOWN,
            "score": 0.0,
            "terms": []
        }
        
        # Combine claims for analysis
        combined_text = f"{claim1} {claim2}".lower()
        
        # Check for high significance terms
        high_terms = [term for term in self.high_significance_terms if term in combined_text]
        if high_terms:
            result["significance"] = ClinicalSignificance.HIGH
            result["score"] = 0.9
            result["terms"] = high_terms
        
        # Check for moderate significance terms
        moderate_terms = [term for term in self.moderate_significance_terms if term in combined_text]
        if moderate_terms and not high_terms:
            result["significance"] = ClinicalSignificance.MODERATE
            result["score"] = 0.6
            result["terms"] = moderate_terms
        
        # If no significant terms found, check with BioMedLM if available
        if result["significance"] == ClinicalSignificance.UNKNOWN and self.biomedlm_service:
            try:
                significance_score = await self.biomedlm_service.assess_clinical_significance(claim1, claim2)
                
                if significance_score > 0.7:
                    result["significance"] = ClinicalSignificance.HIGH
                    result["score"] = significance_score
                elif significance_score > 0.4:
                    result["significance"] = ClinicalSignificance.MODERATE
                    result["score"] = significance_score
                elif significance_score > 0.1:
                    result["significance"] = ClinicalSignificance.LOW
                    result["score"] = significance_score
            except Exception as e:
                logger.warning(f"Error assessing clinical significance with BioMedLM: {e}")
                # Continue with default values
        
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
    
    def unload_models(self):
        """
        Unload all models to free up memory.
        
        Returns:
            None
        """
        if self.biomedlm_service is not None:
            self.biomedlm_service.unload_model()
        logger.info("All models unloaded")
