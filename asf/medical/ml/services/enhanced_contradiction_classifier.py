"""Enhanced Contradiction Classification Service for Medical Research Synthesizer.

This module provides multi-dimensional classification of medical contradictions,
integrating clinical significance assessment, evidence quality assessment,
temporal factor detection, population difference detection, and methodological
difference detection.
"""

import logging
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Any

from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.core.exceptions import OperationError


logger = logging.getLogger(__name__)

class ContradictionType(str, Enum):
    """Types of contradictions.
    
    This enum defines the different types of contradictions that can be detected.
    """

    def _assess_evidence_quality(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Assess the quality of evidence based on metadata.

        Args:
            metadata: Metadata containing information about the study

        Returns:
            Dictionary containing evidence quality assessment results
        """
        result = {
            "quality": EvidenceQuality.UNKNOWN,
            "score": 0.0,
            "factors": {}
        }

        if not metadata:
            return result

        factors = {
            "study_design": 0.0,
            "sample_size": 0.0,
            "publication_year": 0.0,
            "journal_impact_factor": 0.0,
            "bias_risk": 0.0
        }

        study_design = metadata.get("study_design", "").lower()
        design_score = 0.0

        for design, keywords in self.study_design_keywords.items():
            if any(keyword in study_design for keyword in keywords):
                design_score = self.study_design_hierarchy.get(design, 0) / 7.0  # Normalize to 0-1
                factors["study_design"] = design_score
                break

        sample_size = metadata.get("sample_size", 0)
        if sample_size > 0:
            sample_size_score = min(0.5, max(0.0, 0.1 * np.log10(max(1, sample_size))))
            factors["sample_size"] = sample_size_score

        publication_year = metadata.get("publication_year", 0)
        current_year = datetime.now().year
        if publication_year > 0:
            years_old = max(0, current_year - publication_year)
            publication_year_score = max(0.05, min(0.2, 0.2 - 0.01 * years_old))
            factors["publication_year"] = publication_year_score

        impact_factor = metadata.get("impact_factor", 0.0)
        if impact_factor > 0:
            impact_factor_score = min(0.2, max(0.0, 0.01 * impact_factor))
            factors["journal_impact_factor"] = impact_factor_score

        bias_risk = metadata.get("bias_risk", "").lower()
        if bias_risk:
            if "low" in bias_risk:
                bias_risk_score = 0.1
            elif "moderate" in bias_risk or "medium" in bias_risk:
                bias_risk_score = 0.05
            elif "high" in bias_risk:
                bias_risk_score = 0.0
            else:
                bias_risk_score = 0.0
            factors["bias_risk"] = bias_risk_score

        quality_score = sum(factors.values())

        if quality_score >= 0.7:
            quality = EvidenceQuality.HIGH
        elif quality_score >= 0.4:
            quality = EvidenceQuality.MODERATE
        elif quality_score >= 0.2:
            quality = EvidenceQuality.LOW
        elif quality_score > 0:
            quality = EvidenceQuality.VERY_LOW
        else:
            quality = EvidenceQuality.UNKNOWN

        result["quality"] = quality
        result["score"] = quality_score
        result["factors"] = factors

        return result

    def _assess_temporal_factor(
        self,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess temporal factors that might contribute to contradictions.

        Args:
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing temporal factor assessment results
        """
        result = {
            "detected": False,
            "score": 0.0,
            "publication_date_difference": None,
            "factors": {}
        }

        if not metadata1 or not metadata2:
            return result

        pub_year1 = metadata1.get("publication_year", 0)
        pub_year2 = metadata2.get("publication_year", 0)

        if pub_year1 > 0 and pub_year2 > 0:
            date_diff = abs(pub_year1 - pub_year2)
            result["publication_date_difference"] = date_diff

            if date_diff >= 5:
                result["detected"] = True
                result["score"] = min(1.0, date_diff / 20.0)
                result["factors"]["publication_date_difference"] = date_diff

        temporal_terms = ["follow-up", "follow up", "followup", "long-term", "short-term",
                         "longitudinal", "years later", "months later", "weeks later"]

        temporal_terms1 = []
        temporal_terms2 = []

        abstract1 = metadata1.get("abstract", "").lower()
        abstract2 = metadata2.get("abstract", "").lower()

        for term in temporal_terms:
            if term in abstract1:
                temporal_terms1.append(term)
            if term in abstract2:
                temporal_terms2.append(term)

        if temporal_terms1 and temporal_terms2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.7)  # At least 0.7 if temporal terms found
            result["factors"]["temporal_terms1"] = temporal_terms1
            result["factors"]["temporal_terms2"] = temporal_terms2

        return result

    def _assess_population_difference(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess differences in study populations that might explain contradictions.

        Args:
            claim1: First medical claim text
            claim2: Second medical claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing population difference assessment results
        """
        result = {
            "detected": False,
            "score": 0.0,
            "differences": []
        }

        text1 = f"{claim1} {metadata1.get('abstract', '')}".lower() if metadata1 else claim1.lower()
        text2 = f"{claim2} {metadata2.get('abstract', '')}".lower() if metadata2 else claim2.lower()

        for category, terms in self.population_keywords.items():
            category_terms1 = []
            category_terms2 = []

            for term in terms:
                if term.lower() in text1:
                    category_terms1.append(term)
                if term.lower() in text2:
                    category_terms2.append(term)

            if category_terms1 and category_terms2:
                common_terms = set(category_terms1).intersection(set(category_terms2))
                diff_terms1 = set(category_terms1) - common_terms
                diff_terms2 = set(category_terms2) - common_terms

                if diff_terms1 or diff_terms2:
                    result["detected"] = True
                    result["differences"].append({
                        "category": category,
                        "claim1_terms": list(diff_terms1),
                        "claim2_terms": list(diff_terms2),
                        "common_terms": list(common_terms)
                    })

        if result["differences"]:
            result["score"] = min(1.0, len(result["differences"]) / 4.0)  # Max score at 4+ different categories

        population1 = metadata1.get("population", "") if metadata1 else ""
        population2 = metadata2.get("population", "") if metadata2 else ""

        if population1 and population2 and population1 != population2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.8)  # At least 0.8 if explicit population difference
            result["differences"].append({
                "category": "explicit_population",
                "claim1_population": population1,
                "claim2_population": population2
            })

        return result

    def _assess_methodological_difference(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Assess methodological differences that might explain contradictions.

        Args:
            claim1: First medical claim text
            claim2: Second medical claim text
            metadata1: Metadata for the first claim
            metadata2: Metadata for the second claim

        Returns:
            Dictionary containing methodological difference assessment results
        """
        result = {
            "detected": False,
            "score": 0.0,
            "differences": []
        }

        if not metadata1 or not metadata2:
            return result

        study_design1 = metadata1.get("study_design", "").lower()
        study_design2 = metadata2.get("study_design", "").lower()

        if study_design1 and study_design2 and study_design1 != study_design2:
            design_type1 = StudyDesignHierarchy.UNKNOWN
            design_type2 = StudyDesignHierarchy.UNKNOWN

            for design, keywords in self.study_design_keywords.items():
                if any(keyword in study_design1 for keyword in keywords):
                    design_type1 = design
                if any(keyword in study_design2 for keyword in keywords):
                    design_type2 = design

            design_score1 = self.study_design_hierarchy.get(design_type1, 0)
            design_score2 = self.study_design_hierarchy.get(design_type2, 0)
            design_diff = abs(design_score1 - design_score2) / 7.0  # Normalize to 0-1

            if design_diff > 0:
                result["detected"] = True
                result["score"] = max(result["score"], design_diff)
                result["differences"].append({
                    "category": "study_design",
                    "claim1_design": str(design_type1),
                    "claim2_design": str(design_type2),
                    "design_difference_score": design_diff
                })

        sample_size1 = metadata1.get("sample_size", 0)
        sample_size2 = metadata2.get("sample_size", 0)

        if sample_size1 > 0 and sample_size2 > 0:
            ratio = max(sample_size1, sample_size2) / max(1, min(sample_size1, sample_size2))

            if ratio > 2:
                result["detected"] = True
                sample_size_score = min(1.0, (ratio - 1) / 9.0)
                result["score"] = max(result["score"], sample_size_score)
                result["differences"].append({
                    "category": "sample_size",
                    "claim1_sample_size": sample_size1,
                    "claim2_sample_size": sample_size2,
                    "ratio": ratio,
                    "sample_size_difference_score": sample_size_score
                })

        p_value1 = metadata1.get("p_value")
        p_value2 = metadata2.get("p_value")

        if p_value1 is not None and p_value2 is not None:
            if (p_value1 <= 0.05 and p_value2 > 0.05) or (p_value1 > 0.05 and p_value2 <= 0.05):
                result["detected"] = True
                result["score"] = max(result["score"], 0.9)  # High score for statistical significance difference
                result["differences"].append({
                    "category": "statistical_significance",
                    "claim1_p_value": p_value1,
                    "claim2_p_value": p_value2,
                    "statistical_significance_difference_score": 0.9
                })

        measurement_method1 = metadata1.get("measurement_method", "").lower()
        measurement_method2 = metadata2.get("measurement_method", "").lower()

        if measurement_method1 and measurement_method2 and measurement_method1 != measurement_method2:
            result["detected"] = True
            result["score"] = max(result["score"], 0.7)  # Moderate score for measurement method difference
            result["differences"].append({
                "category": "measurement_method",
                "claim1_method": measurement_method1,
                "claim2_method": measurement_method2,
                "measurement_method_difference_score": 0.7
            })

        return result
