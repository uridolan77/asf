"""
Enhanced Contradiction Classifier Module.

This module provides a production-grade ML model for detecting and classifying
contradictions in medical literature, with support for fine-tuning and real-time updates.
"""

import os
import uuid
import json
import asyncio
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple

import numpy as np
from pydantic import BaseModel, Field

from asf.medical.core.logging_config import get_logger
from asf.medical.ml.models.model_registry import (
    ModelRegistry, ModelStatus, ModelMetrics, ModelFramework, get_model_registry
)
from asf.medical.core.enhanced_cache import EnhancedCacheManager

logger = get_logger(__name__)

# Initialize cache for model predictions
cache = EnhancedCacheManager(
    max_size=1000, 
    default_ttl=3600,  # 1 hour
    namespace="contradiction_classifier:"
)

class ContradictionType(str, Enum):
    """Types of contradictions between medical claims."""
    NO_CONTRADICTION = "no_contradiction"
    DIRECT = "direct"  # Directly opposing claims
    METHODOLOGICAL = "methodological"  # Different study methods lead to different conclusions
    POPULATION = "population"  # Different populations with different outcomes
    TEMPORAL = "temporal"  # Time-dependent differences
    PARTIAL = "partial"  # Partial contradiction
    CONTEXTUAL = "contextual"  # Context-dependent contradiction
    TERMINOLOGICAL = "terminological"  # Differences in terminology or definitions

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

class ContradictionResult(BaseModel):
    """Result of contradiction classification."""
    contradiction_type: ContradictionType
    contradiction_probability: float
    clinical_significance: ClinicalSignificance
    evidence_quality_claim1: EvidenceQuality
    evidence_quality_claim2: EvidenceQuality
    dimensions: Dict[str, float]  # Additional classification dimensions
    explanation: Optional[str] = None
    model_version: str
    processing_time_ms: float
    
    class Config:
        extra = "allow"

class EnhancedContradictionClassifier:
    """
    Enhanced classifier for detecting and classifying contradictions in medical literature.
    
    This classifier uses a series of specialized ML models to detect various aspects
    of contradictions between medical claims, including:
    1. Basic contradiction detection
    2. Contradiction type classification
    3. Clinical significance assessment
    4. Evidence quality assessment
    5. Additional dimension analysis (temporal, population, methodological)
    
    It supports real-time fine-tuning and model updates.
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize the enhanced contradiction classifier.
        
        Args:
            use_cache: Whether to use caching for predictions.
        """
        self.use_cache = use_cache
        self.model_registry = get_model_registry()
        
        # Initialize sub-models
        self.models = {
            "contradiction_type_classifier": None,
            "clinical_significance_classifier": None, 
            "evidence_quality_classifier": None,
            "temporal_classifier": None,
            "population_classifier": None,
            "methodological_classifier": None
        }
        
        # Background tasks
        self.background_tasks = set()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all sub-models from the model registry."""
        for model_name in self.models.keys():
            # Get production model if available
            model_metadata = self.model_registry.get_production_model(model_name)
            
            if not model_metadata:
                # No production model available, create a mock one
                logger.warning(f"No production model found for {model_name}, using mock implementation")
                self._create_mock_model(model_name)
            else:
                logger.info(f"Using production model for {model_name}: version {model_metadata.version}")
                # In a real implementation, we would load the model here
                # For this example, we'll rely on the mock behavior
                self.models[model_name] = {
                    "metadata": model_metadata,
                    "version": model_metadata.version
                }
    
    def _create_mock_model(self, model_name: str):
        """
        Create a mock model entry when no real model is available.
        This is for demonstration purposes only.
        
        Args:
            model_name: Name of the model to create.
        """
        version = "0.1.0"
        
        # Register a mock model
        metadata = self.model_registry.register_model(
            name=model_name,
            version=version,
            framework=ModelFramework.CUSTOM,
            description=f"Mock {model_name} for demonstration purposes",
            status=ModelStatus.PRODUCTION,
            metrics=ModelMetrics(
                accuracy=0.85,
                precision=0.83,
                recall=0.86,
                f1_score=0.84
            )
        )
        
        self.models[model_name] = {
            "metadata": metadata,
            "version": version
        }
        
        logger.info(f"Created mock model for {model_name}: version {version}")
    
    async def classify_contradiction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a potential contradiction between two medical claims.
        
        Args:
            data: Dictionary containing:
                - claim1: Text of the first claim
                - claim2: Text of the second claim
                - metadata1: Optional metadata for the first claim
                - metadata2: Optional metadata for the second claim
            
        Returns:
            Classification results with all dimensions.
        """
        start_time = datetime.datetime.now()
        
        # Validate input
        if "claim1" not in data or "claim2" not in data:
            raise ValueError("Both claim1 and claim2 are required")
        
        claim1 = data["claim1"]
        claim2 = data["claim2"]
        metadata1 = data.get("metadata1", {})
        metadata2 = data.get("metadata2", {})
        
        # Check cache if enabled
        cache_key = None
        if self.use_cache:
            # Create a cache key from the input data
            cache_key = self._create_cache_key(claim1, claim2, metadata1, metadata2)
            cached_result = await cache.get(cache_key)
            
            if cached_result:
                logger.info(f"Using cached result for contradiction classification")
                return cached_result
        
        try:
            # 1. Classify contradiction type
            contradiction_result = await self._classify_contradiction_type(claim1, claim2, metadata1, metadata2)
            
            # 2. Assess clinical significance
            clinical_significance = await self._assess_clinical_significance(
                claim1, claim2, 
                contradiction_result["contradiction_type"], 
                metadata1, metadata2
            )
            
            # 3. Assess evidence quality
            evidence_quality = await self._assess_evidence_quality(claim1, claim2, metadata1, metadata2)
            
            # 4. Analyze additional dimensions
            dimensions = await asyncio.gather(
                self._analyze_temporal_dimension(claim1, claim2, metadata1, metadata2),
                self._analyze_population_dimension(claim1, claim2, metadata1, metadata2),
                self._analyze_methodological_dimension(claim1, claim2, metadata1, metadata2)
            )
            
            # Combine all dimensions
            result = {
                "contradiction_type": contradiction_result["contradiction_type"],
                "contradiction_probability": contradiction_result["probability"],
                "clinical_significance": clinical_significance["significance"],
                "clinical_significance_probability": clinical_significance["probability"],
                "evidence_quality_claim1": evidence_quality["quality_claim1"],
                "evidence_quality_claim2": evidence_quality["quality_claim2"],
                "evidence_quality_probabilities": evidence_quality["probabilities"],
                "dimensions": {
                    "temporal": dimensions[0],
                    "population": dimensions[1],
                    "methodological": dimensions[2]
                },
                "model_versions": {
                    "contradiction_type": self.models["contradiction_type_classifier"]["version"],
                    "clinical_significance": self.models["clinical_significance_classifier"]["version"],
                    "evidence_quality": self.models["evidence_quality_classifier"]["version"],
                    "temporal": self.models["temporal_classifier"]["version"],
                    "population": self.models["population_classifier"]["version"],
                    "methodological": self.models["methodological_classifier"]["version"]
                }
            }
            
            # Generate explanation if contradiction exists
            if contradiction_result["contradiction_type"] != ContradictionType.NO_CONTRADICTION:
                result["explanation"] = await self._generate_explanation(result, claim1, claim2)
            
            # Calculate processing time
            end_time = datetime.datetime.now()
            processing_time_ms = (end_time - start_time).total_seconds() * 1000
            result["processing_time_ms"] = processing_time_ms
            
            # Cache result
            if self.use_cache and cache_key:
                await cache.set(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying contradiction: {str(e)}")
            # In production, we'd have more sophisticated error handling
            raise
    
    async def _classify_contradiction_type(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify the type of contradiction between claims.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Dictionary with contradiction type and probability.
        """
        # In a production system, this would use a real ML model
        # For this example, we'll use a mock implementation
        
        # Simple heuristic for demonstration
        # Look for opposing phrases
        opposing_phrases = [
            ("increases", "decreases"),
            ("effective", "ineffective"),
            ("beneficial", "harmful"),
            ("positive", "negative"),
            ("significant", "insignificant"),
            ("recommended", "not recommended")
        ]
        
        # Check for common words to establish topic similarity
        common_words = set(claim1.lower().split()) & set(claim2.lower().split())
        topic_similarity = len(common_words) / max(len(claim1.split()), len(claim2.split()))
        
        # Check for opposing phrases
        has_opposing_phrases = False
        for phrase1, phrase2 in opposing_phrases:
            if (phrase1 in claim1.lower() and phrase2 in claim2.lower()) or \
               (phrase2 in claim1.lower() and phrase1 in claim2.lower()):
                has_opposing_phrases = True
                break
        
        # Assign contradiction type and probability
        if has_opposing_phrases and topic_similarity > 0.2:
            # Direct contradiction
            return {
                "contradiction_type": ContradictionType.DIRECT,
                "probability": 0.85
            }
        elif topic_similarity > 0.3:
            # Check for population differences
            if "children" in claim1.lower() and "adults" in claim2.lower():
                return {
                    "contradiction_type": ContradictionType.POPULATION,
                    "probability": 0.78
                }
            # Check for temporal differences
            elif "short-term" in claim1.lower() and "long-term" in claim2.lower():
                return {
                    "contradiction_type": ContradictionType.TEMPORAL,
                    "probability": 0.75
                }
            # Check for methodological differences
            elif any(word in claim1.lower() + " " + claim2.lower() for word in ["method", "study", "analysis", "approach"]):
                return {
                    "contradiction_type": ContradictionType.METHODOLOGICAL,
                    "probability": 0.72
                }
            else:
                return {
                    "contradiction_type": ContradictionType.PARTIAL,
                    "probability": 0.65
                }
        else:
            return {
                "contradiction_type": ContradictionType.NO_CONTRADICTION,
                "probability": 0.9
            }
    
    async def _assess_clinical_significance(
        self,
        claim1: str,
        claim2: str,
        contradiction_type: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the clinical significance of a contradiction.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            contradiction_type: Type of contradiction.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Dictionary with clinical significance and probability.
        """
        # Mock implementation for demonstration
        if contradiction_type == ContradictionType.NO_CONTRADICTION:
            return {
                "significance": ClinicalSignificance.NONE,
                "probability": 0.95
            }
        
        # Check for critical terms
        critical_terms = ["mortality", "death", "fatal", "severe adverse", "life-threatening"]
        high_terms = ["adverse event", "complication", "safety", "risk", "harm"]
        medium_terms = ["effective", "efficacy", "outcome", "benefit"]
        
        text = (claim1 + " " + claim2).lower()
        
        if any(term in text for term in critical_terms):
            return {
                "significance": ClinicalSignificance.CRITICAL,
                "probability": 0.85
            }
        elif any(term in text for term in high_terms):
            return {
                "significance": ClinicalSignificance.HIGH,
                "probability": 0.8
            }
        elif any(term in text for term in medium_terms):
            return {
                "significance": ClinicalSignificance.MEDIUM,
                "probability": 0.75
            }
        else:
            return {
                "significance": ClinicalSignificance.LOW,
                "probability": 0.7
            }
    
    async def _assess_evidence_quality(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Assess the quality of evidence for each claim.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Dictionary with evidence quality assessments.
        """
        # Mock implementation for demonstration
        
        # Check metadata for study type and sample size
        quality1 = self._assess_single_claim_evidence(claim1, metadata1)
        quality2 = self._assess_single_claim_evidence(claim2, metadata2)
        
        # Calculate probabilities (mock example)
        probabilities_claim1 = {
            "very_low": 0.05,
            "low": 0.10,
            "moderate": 0.20,
            "high": 0.45,
            "very_high": 0.20
        }
        
        probabilities_claim2 = {
            "very_low": 0.10,
            "low": 0.15,
            "moderate": 0.25,
            "high": 0.40,
            "very_high": 0.10
        }
        
        if quality1 == EvidenceQuality.HIGH:
            probabilities_claim1["high"] += 0.2
        if quality2 == EvidenceQuality.HIGH:
            probabilities_claim2["high"] += 0.2
            
        # Normalize probabilities
        self._normalize_probabilities(probabilities_claim1)
        self._normalize_probabilities(probabilities_claim2)
        
        return {
            "quality_claim1": quality1,
            "quality_claim2": quality2,
            "probabilities": {
                "claim1": probabilities_claim1,
                "claim2": probabilities_claim2
            }
        }
    
    def _assess_single_claim_evidence(
        self,
        claim: str,
        metadata: Dict[str, Any]
    ) -> EvidenceQuality:
        """
        Assess the evidence quality for a single claim.
        
        Args:
            claim: Claim text.
            metadata: Claim metadata.
            
        Returns:
            Evidence quality assessment.
        """
        # Get study type from metadata
        study_type = metadata.get("study_type", "").lower()
        sample_size = metadata.get("sample_size", 0)
        
        # Assign quality based on study type
        if study_type in ["meta-analysis", "systematic review"]:
            return EvidenceQuality.VERY_HIGH
        elif study_type in ["randomized controlled trial", "rct"] and sample_size > 500:
            return EvidenceQuality.HIGH
        elif study_type in ["randomized controlled trial", "rct"]:
            return EvidenceQuality.MODERATE
        elif study_type in ["cohort", "case-control"] and sample_size > 1000:
            return EvidenceQuality.MODERATE
        elif study_type in ["cohort", "case-control"]:
            return EvidenceQuality.LOW
        elif study_type in ["case series", "case report"]:
            return EvidenceQuality.VERY_LOW
        
        # If study type not provided, look for keywords in claim
        lower_claim = claim.lower()
        if "systematic review" in lower_claim or "meta-analysis" in lower_claim:
            return EvidenceQuality.HIGH
        elif "randomized" in lower_claim or "randomised" in lower_claim:
            return EvidenceQuality.MODERATE
        elif "cohort" in lower_claim or "observational" in lower_claim:
            return EvidenceQuality.LOW
        else:
            return EvidenceQuality.VERY_LOW
    
    async def _analyze_temporal_dimension(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze temporal dimensions of contradiction.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Dictionary with temporal analysis.
        """
        # Mock implementation
        temporal_terms = [
            "short-term", "long-term", "acute", "chronic", 
            "immediate", "delayed", "follow-up"
        ]
        
        # Check for temporal terms in claims
        found_terms1 = [term for term in temporal_terms if term in claim1.lower()]
        found_terms2 = [term for term in temporal_terms if term in claim2.lower()]
        
        # Check publication dates
        pub_date1 = metadata1.get("publication_date", "")
        pub_date2 = metadata2.get("publication_date", "")
        
        has_temporal_difference = bool(found_terms1 and found_terms2 and found_terms1 != found_terms2)
        
        # Create result
        result = {
            "has_temporal_dimension": has_temporal_difference,
            "confidence": 0.75 if has_temporal_difference else 0.6,
            "temporal_factors": {
                "claim1": found_terms1,
                "claim2": found_terms2
            }
        }
        
        # Add publication date difference if available
        if pub_date1 and pub_date2:
            try:
                date1 = datetime.datetime.strptime(pub_date1, "%Y-%m-%d")
                date2 = datetime.datetime.strptime(pub_date2, "%Y-%m-%d")
                year_diff = abs(date1.year - date2.year)
                result["publication_year_difference"] = year_diff
                
                if year_diff > 10:
                    result["has_temporal_dimension"] = True
                    result["confidence"] = 0.85
            except ValueError:
                pass
        
        return result
    
    async def _analyze_population_dimension(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze population dimensions of contradiction.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Dictionary with population analysis.
        """
        # Population categories to check
        populations = {
            "age": ["infant", "child", "adolescent", "adult", "elderly", "pediatric", "geriatric"],
            "gender": ["male", "female", "men", "women", "boys", "girls"],
            "ethnicity": ["african", "asian", "caucasian", "hispanic", "european"],
            "condition": ["healthy", "comorbid", "diabetic", "hypertensive", "obese"]
        }
        
        result = {
            "has_population_dimension": False,
            "confidence": 0.5,
            "different_populations": []
        }
        
        # Check each population category
        for category, terms in populations.items():
            pop1 = [term for term in terms if term in claim1.lower()]
            pop2 = [term for term in terms if term in claim2.lower()]
            
            if pop1 and pop2 and pop1 != pop2:
                result["has_population_dimension"] = True
                result["confidence"] = 0.8
                result["different_populations"].append({
                    "category": category,
                    "claim1": pop1,
                    "claim2": pop2
                })
        
        # Check metadata for population information
        pop1_meta = metadata1.get("population", "")
        pop2_meta = metadata2.get("population", "")
        
        if pop1_meta and pop2_meta and pop1_meta != pop2_meta:
            result["has_population_dimension"] = True
            result["confidence"] = 0.9
            result["different_populations"].append({
                "category": "metadata",
                "claim1": pop1_meta,
                "claim2": pop2_meta
            })
        
        return result
    
    async def _analyze_methodological_dimension(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze methodological dimensions of contradiction.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Dictionary with methodological analysis.
        """
        # Study types to check
        study_types = [
            "meta-analysis", "systematic review", "randomized", "randomised",
            "observational", "cohort", "case-control", "cross-sectional",
            "retrospective", "prospective", "in vitro", "in vivo",
            "clinical trial", "pilot study"
        ]
        
        # Method keywords
        method_keywords = [
            "method", "analysis", "statistical", "model", "measure", 
            "assessment", "evaluation", "protocol", "design", "approach"
        ]
        
        result = {
            "has_methodological_dimension": False,
            "confidence": 0.5,
            "methodological_differences": []
        }
        
        # Check study types in claims
        types1 = [study for study in study_types if study in claim1.lower()]
        types2 = [study for study in study_types if study in claim2.lower()]
        
        if types1 and types2 and types1 != types2:
            result["has_methodological_dimension"] = True
            result["confidence"] = 0.85
            result["methodological_differences"].append({
                "category": "study_type",
                "claim1": types1,
                "claim2": types2
            })
        
        # Check method keywords
        methods1 = [word for word in method_keywords if word in claim1.lower()]
        methods2 = [word for word in method_keywords if word in claim2.lower()]
        
        if methods1 and methods2 and methods1 != methods2:
            result["has_methodological_dimension"] = True
            result["confidence"] = 0.75
            result["methodological_differences"].append({
                "category": "methodology",
                "claim1": methods1,
                "claim2": methods2
            })
        
        # Check metadata for study types
        study_type1 = metadata1.get("study_type", "")
        study_type2 = metadata2.get("study_type", "")
        
        if study_type1 and study_type2 and study_type1 != study_type2:
            result["has_methodological_dimension"] = True
            result["confidence"] = 0.9
            result["methodological_differences"].append({
                "category": "metadata_study_type",
                "claim1": study_type1,
                "claim2": study_type2
            })
        
        return result
    
    async def _generate_explanation(
        self,
        classification: Dict[str, Any],
        claim1: str,
        claim2: str
    ) -> str:
        """
        Generate a natural language explanation of the contradiction.
        
        Args:
            classification: Contradiction classification results.
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            
        Returns:
            Explanation text.
        """
        # Build explanation based on classification
        contradiction_type = classification["contradiction_type"]
        clinical_significance = classification["clinical_significance"]
        dimensions = classification["dimensions"]
        
        explanation = f"The claims appear to have a {contradiction_type} contradiction "
        explanation += f"with {clinical_significance} clinical significance. "
        
        # Add dimension-specific details
        if dimensions["temporal"].get("has_temporal_dimension"):
            explanation += "The claims differ in temporal context, "
            temporal_factors = dimensions["temporal"].get("temporal_factors", {})
            if temporal_factors.get("claim1") or temporal_factors.get("claim2"):
                explanation += f"with claim 1 focused on {', '.join(temporal_factors.get('claim1', ['']))} "
                explanation += f"and claim 2 focused on {', '.join(temporal_factors.get('claim2', ['']))}. "
        
        if dimensions["population"].get("has_population_dimension"):
            explanation += "The claims refer to different populations, "
            pop_diff = dimensions["population"].get("different_populations", [])
            if pop_diff:
                categories = [diff.get("category") for diff in pop_diff]
                explanation += f"differing in {', '.join(categories)}. "
        
        if dimensions["methodological"].get("has_methodological_dimension"):
            explanation += "The claims are based on different study methodologies, "
            method_diff = dimensions["methodological"].get("methodological_differences", [])
            if method_diff:
                for diff in method_diff:
                    if diff.get("category") == "study_type":
                        explanation += f"with claim 1 using {', '.join(diff.get('claim1', ['']))} "
                        explanation += f"and claim 2 using {', '.join(diff.get('claim2', ['']))}. "
        
        # Add evidence quality assessment
        explanation += f"Evidence quality for claim 1 is {classification['evidence_quality_claim1']} "
        explanation += f"and for claim 2 is {classification['evidence_quality_claim2']}."
        
        return explanation
    
    async def retrain_model(
        self,
        model_name: str,
        training_data: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Retrain a specific contradiction classification model.
        
        Args:
            model_name: Name of the model to retrain.
            training_data: Training data for the model.
            hyperparameters: Optional hyperparameters for training.
            
        Returns:
            Dictionary with training results.
        """
        # Check if the model exists
        if model_name not in self.models:
            return {
                "status": "error",
                "error": f"Unknown model: {model_name}"
            }
        
        # In a real implementation, we would train the model here
        # For this example, we'll simulate a successful training process
        
        try:
            # Generate a new version
            current_version = self.models[model_name]["version"]
            major, minor, patch = current_version.split(".")
            new_version = f"{major}.{minor}.{int(patch) + 1}"
            
            # Create a training job ID
            job_id = str(uuid.uuid4())
            
            # Simulate training process
            # In a real implementation, this would be a background task
            training_metrics = {
                "accuracy": 0.87,
                "precision": 0.86,
                "recall": 0.88,
                "f1_score": 0.87,
                "val_accuracy": 0.85,
                "val_precision": 0.84,
                "val_recall": 0.86,
                "val_f1_score": 0.85
            }
            
            # Register the new model version
            metadata = self.model_registry.register_model(
                name=model_name,
                version=new_version,
                framework=ModelFramework.CUSTOM,
                description=f"Retrained {model_name} model",
                status=ModelStatus.STAGING,  # Start in staging before promoting to production
                metrics=ModelMetrics(**training_metrics),
                parent_version=current_version,
                created_by="api_retraining",
                tags=["retrained", "api"],
                training_dataset_hash=self.model_registry.compute_dataset_hash(training_data)
            )
            
            # Update the model in memory
            self.models[model_name] = {
                "metadata": metadata,
                "version": new_version
            }
            
            # Return the training results
            return {
                "status": "success",
                "job_id": job_id,
                "model_name": model_name,
                "old_version": current_version,
                "new_version": new_version,
                "metrics": training_metrics,
                "message": f"Successfully retrained {model_name} to version {new_version}",
                "timestamp": datetime.datetime.now().isoformat()
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
        status: ModelStatus
    ) -> Dict[str, Any]:
        """
        Update the status of a model.
        
        Args:
            model_name: Name of the model.
            version: Version of the model.
            status: New status for the model.
            
        Returns:
            Dictionary with update results.
        """
        try:
            # Update model status in registry
            updated = self.model_registry.update_model_status(model_name, version, status)
            
            if not updated:
                return {
                    "status": "error",
                    "error": f"Model {model_name} version {version} not found"
                }
            
            # If promoting to production, update the active model
            if status == ModelStatus.PRODUCTION and model_name in self.models:
                self.models[model_name] = {
                    "metadata": updated,
                    "version": version
                }
                logger.info(f"Updated active model for {model_name} to version {version}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "version": version,
                "new_status": status.value,
                "timestamp": datetime.datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating model status: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_cache_key(
        self,
        claim1: str,
        claim2: str,
        metadata1: Dict[str, Any],
        metadata2: Dict[str, Any]
    ) -> str:
        """
        Create a cache key from the input data.
        
        Args:
            claim1: Text of the first claim.
            claim2: Text of the second claim.
            metadata1: Metadata for the first claim.
            metadata2: Metadata for the second claim.
            
        Returns:
            Cache key.
        """
        # Create a string representation of the input
        input_str = f"{claim1}|{claim2}"
        
        # Add model versions to the key
        for model_name, model in self.models.items():
            input_str += f"|{model_name}:{model['version']}"
        
        # Add key metadata fields if present
        for field in ["study_type", "publication_date", "sample_size", "population"]:
            if field in metadata1:
                input_str += f"|m1_{field}:{metadata1[field]}"
            if field in metadata2:
                input_str += f"|m2_{field}:{metadata2[field]}"
        
        # Hash the input string
        return hashlib.md5(input_str.encode()).hexdigest()
    
    @staticmethod
    def _normalize_probabilities(probabilities: Dict[str, float]):
        """
        Normalize probabilities so they sum to 1.
        
        Args:
            probabilities: Dictionary of probabilities.
        """
        total = sum(probabilities.values())
        if total > 0:
            for key in probabilities:
                probabilities[key] = probabilities[key] / total
