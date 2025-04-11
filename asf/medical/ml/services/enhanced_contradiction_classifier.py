"""
Enhanced Contradiction Classification Service for Medical Research Synthesizer.

This module provides multi-dimensional classification of medical contradictions,
integrating clinical significance assessment, evidence quality assessment,
temporal factor detection, population difference detection, and methodological
difference detection.
"""

import logging
import asyncio
import numpy as np
import re
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple, Set

from asf.medical.ml.models.biomedlm import BioMedLMService
from asf.medical.ml.services.temporal_service import TemporalService
from asf.medical.core.config import settings

logger = logging.getLogger(__name__)

class ContradictionType(str, Enum):
    """Types of contradictions."""
    DIRECT = "direct"
    NEGATION = "negation"
    TEMPORAL = "temporal"
    HIERARCHICAL = "hierarchical"
    METHODOLOGICAL = "methodological"
    STATISTICAL = "statistical"
    POPULATION = "population"
    UNKNOWN = "unknown"

class ContradictionConfidence(str, Enum):
    """Confidence levels for contradiction detection."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ClinicalSignificance(str, Enum):
    """Clinical significance levels."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"

class EvidenceQuality(str, Enum):
    """Evidence quality levels."""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    VERY_LOW = "very_low"
    UNKNOWN = "unknown"

class StudyDesignHierarchy(str, Enum):
    """Study design hierarchy based on evidence-based medicine."""
    SYSTEMATIC_REVIEW_META_ANALYSIS = "systematic_review_meta_analysis"
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    COHORT_STUDY = "cohort_study"
    CASE_CONTROL_STUDY = "case_control_study"
    CASE_SERIES = "case_series"
    CASE_REPORT = "case_report"
    EXPERT_OPINION = "expert_opinion"
    UNKNOWN = "unknown"

class EnhancedContradictionClassifier:
    """
    Enhanced contradiction classifier for medical literature.
    
    This service provides multi-dimensional classification of medical contradictions,
    integrating clinical significance assessment, evidence quality assessment,
    temporal factor detection, population difference detection, and methodological
    difference detection.
        self.biomedlm_service = None
        self.temporal_service = None
        
        try:
            self.biomedlm_service = BioMedLMService()
            logger.info("BioMedLM service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize BioMedLM service: {e}")
        
        try:
            self.temporal_service = TemporalService()
            logger.info("Temporal service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize temporal service: {e}")
        
        self.thresholds = {
            ContradictionType.DIRECT: 0.7,
            ContradictionType.NEGATION: 0.8,
            ContradictionType.TEMPORAL: 0.6,
            ContradictionType.HIERARCHICAL: 0.65,
            ContradictionType.METHODOLOGICAL: 0.75,
            ContradictionType.STATISTICAL: 0.8,
            ContradictionType.POPULATION: 0.7
        }
        
        self.clinical_significance_terms = {
            "high": [
                "mortality", "death", "survival", "fatal", "life-threatening",
                "heart attack", "stroke", "cancer", "malignant", "emergency",
                "intensive care", "icu", "hospitalization", "severe", "critical"
            ],
            "moderate": [
                "morbidity", "complication", "adverse event", "side effect",
                "quality of life", "disability", "impairment", "chronic",
                "pain", "symptom", "treatment", "therapy", "medication"
            ],
            "low": [
                "mild", "minor", "cosmetic", "temporary", "self-limiting",
                "benign", "non-significant", "minimal", "slight"
            ]
        }
        
        self.study_design_hierarchy = {
            StudyDesignHierarchy.SYSTEMATIC_REVIEW_META_ANALYSIS: 7,
            StudyDesignHierarchy.RANDOMIZED_CONTROLLED_TRIAL: 6,
            StudyDesignHierarchy.COHORT_STUDY: 5,
            StudyDesignHierarchy.CASE_CONTROL_STUDY: 4,
            StudyDesignHierarchy.CASE_SERIES: 3,
            StudyDesignHierarchy.CASE_REPORT: 2,
            StudyDesignHierarchy.EXPERT_OPINION: 1,
            StudyDesignHierarchy.UNKNOWN: 0
        }
        
        self.study_design_keywords = {
            StudyDesignHierarchy.SYSTEMATIC_REVIEW_META_ANALYSIS: [
                "systematic review", "meta-analysis", "meta analysis", "metaanalysis"
            ],
            StudyDesignHierarchy.RANDOMIZED_CONTROLLED_TRIAL: [
                "randomized controlled trial", "rct", "randomised controlled trial",
                "randomized clinical trial", "randomised clinical trial"
            ],
            StudyDesignHierarchy.COHORT_STUDY: [
                "cohort study", "cohort analysis", "longitudinal study",
                "prospective study", "retrospective study", "follow-up study"
            ],
            StudyDesignHierarchy.CASE_CONTROL_STUDY: [
                "case-control study", "case control study", "case-control analysis"
            ],
            StudyDesignHierarchy.CASE_SERIES: [
                "case series", "case study series", "clinical series"
            ],
            StudyDesignHierarchy.CASE_REPORT: [
                "case report", "case study", "patient report"
            ],
            StudyDesignHierarchy.EXPERT_OPINION: [
                "expert opinion", "expert consensus", "clinical opinion",
                "narrative review", "commentary", "editorial"
            ]
        }
        
        self.population_keywords = {
            "age": [
                "infant", "child", "children", "adolescent", "teenager",
                "young adult", "adult", "middle-aged", "elderly", "geriatric"
            ],
            "gender": [
                "male", "female", "men", "women", "boy", "girl"
            ],
            "ethnicity": [
                "caucasian", "white", "black", "african", "asian", "hispanic",
                "latino", "native american", "indigenous", "ethnic"
            ],
            "condition": [
                "healthy", "patient", "diabetic", "hypertensive", "obese",
                "overweight", "smoker", "non-smoker", "pregnant"
            ]
        }
        
        logger.info("Enhanced contradiction classifier initialized")
    
    async def classify_contradiction(
        self,
        contradiction: Dict[str, Any]
    ) -> Dict[str, Any]:
        claim1 = contradiction.get("claim1", "")
        claim2 = contradiction.get("claim2", "")
        metadata1 = contradiction.get("metadata1", {})
        metadata2 = contradiction.get("metadata2", {})
        
        classification = {
            "contradiction_type": contradiction.get("contradiction_type", ContradictionType.UNKNOWN),
            "clinical_significance": ClinicalSignificance.UNKNOWN,
            "clinical_significance_score": 0.0,
            "evidence_quality": {
                "claim1": EvidenceQuality.UNKNOWN,
                "claim2": EvidenceQuality.UNKNOWN,
                "differential": 0.0
            },
            "temporal_factor": {
                "detected": False,
                "score": 0.0,
                "publication_date_difference": None
            },
            "population_difference": {
                "detected": False,
                "score": 0.0,
                "differences": []
            },
            "methodological_difference": {
                "detected": False,
                "score": 0.0,
                "differences": []
            }
        }
        
        clinical_significance = await self._assess_clinical_significance(
            claim1, claim2, metadata1, metadata2
        )
        classification["clinical_significance"] = clinical_significance["significance"]
        classification["clinical_significance_score"] = clinical_significance["score"]
        classification["clinical_significance_terms"] = clinical_significance["terms"]
        
        evidence_quality1 = self._assess_evidence_quality(metadata1)
        evidence_quality2 = self._assess_evidence_quality(metadata2)
        classification["evidence_quality"]["claim1"] = evidence_quality1["quality"]
        classification["evidence_quality"]["claim1_score"] = evidence_quality1["score"]
        classification["evidence_quality"]["claim1_factors"] = evidence_quality1["factors"]
        classification["evidence_quality"]["claim2"] = evidence_quality2["quality"]
        classification["evidence_quality"]["claim2_score"] = evidence_quality2["score"]
        classification["evidence_quality"]["claim2_factors"] = evidence_quality2["factors"]
        classification["evidence_quality"]["differential"] = evidence_quality1["score"] - evidence_quality2["score"]
        
        temporal_factor = self._assess_temporal_factor(metadata1, metadata2)
        classification["temporal_factor"] = temporal_factor
        
        population_difference = self._assess_population_difference(
            claim1, claim2, metadata1, metadata2
        )
        classification["population_difference"] = population_difference
        
        methodological_difference = self._assess_methodological_difference(
            claim1, claim2, metadata1, metadata2
        )
        classification["methodological_difference"] = methodological_difference
        
        classified_contradiction = {
            **contradiction,  # Include original contradiction data
            "classification": classification
        }
        
        return classified_contradiction
    
    async def _assess_clinical_significance(
        self,
        claim1: str,
        claim2: str,
        metadata1: Optional[Dict[str, Any]] = None,
        metadata2: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result = {
            "significance": ClinicalSignificance.UNKNOWN,
            "score": 0.0,
            "terms": []
        }
        
        combined_text = f"{claim1} {claim2}".lower()
        
        high_significance_terms = []
        for term in self.clinical_significance_terms["high"]:
            if term.lower() in combined_text:
                high_significance_terms.append(term)
        
        moderate_significance_terms = []
        for term in self.clinical_significance_terms["moderate"]:
            if term.lower() in combined_text:
                moderate_significance_terms.append(term)
        
        low_significance_terms = []
        for term in self.clinical_significance_terms["low"]:
            if term.lower() in combined_text:
                low_significance_terms.append(term)
        
        high_count = len(high_significance_terms)
        moderate_count = len(moderate_significance_terms)
        low_count = len(low_significance_terms)
        
        significance_score = (high_count * 1.0 + moderate_count * 0.5 + low_count * 0.1) / (high_count + moderate_count + low_count) if (high_count + moderate_count + low_count) > 0 else 0.0
        
        if high_count > 0:
            significance = ClinicalSignificance.HIGH
        elif moderate_count > 0:
            significance = ClinicalSignificance.MODERATE
        elif low_count > 0:
            significance = ClinicalSignificance.LOW
        else:
            significance = ClinicalSignificance.UNKNOWN
        
        result["significance"] = significance
        result["score"] = significance_score
        result["terms"] = {
            "high": high_significance_terms,
            "moderate": moderate_significance_terms,
            "low": low_significance_terms
        }
        
        return result
    
    def _assess_evidence_quality(
        self,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
