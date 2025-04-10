"""
Contradiction service for the Medical Research Synthesizer.

This module provides a service for detecting contradictions in medical literature.
"""

import logging
import os
import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple

from asf.medical.ml.models import BioMedLMService, TSMixerService, LorentzEmbeddingService, SHAPExplainer
from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class ContradictionService:
    """
    Service for detecting contradictions in medical literature.
    
    This service provides methods for detecting and explaining contradictions.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the contradiction service.
        
        Returns:
            ContradictionService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(ContradictionService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the contradiction service."""
        self.biomedlm_service = None
        self.tsmixer_service = None
        self.lorentz_service = None
        self.shap_explainer = None
        
        logger.info("Contradiction service initialized")
    
    def _get_biomedlm_service(self) -> BioMedLMService:
        """
        Get the BioMedLM service.
        
        Returns:
            BioMedLMService: The BioMedLM service
        """
        if self.biomedlm_service is None:
            logger.info("Initializing BioMedLM service")
            self.biomedlm_service = BioMedLMService()
        return self.biomedlm_service
    
    def _get_tsmixer_service(self) -> TSMixerService:
        """
        Get the TSMixer service.
        
        Returns:
            TSMixerService: The TSMixer service
        """
        if self.tsmixer_service is None:
            logger.info("Initializing TSMixer service")
            self.tsmixer_service = TSMixerService()
        return self.tsmixer_service
    
    def _get_lorentz_service(self) -> LorentzEmbeddingService:
        """
        Get the Lorentz embedding service.
        
        Returns:
            LorentzEmbeddingService: The Lorentz embedding service
        """
        if self.lorentz_service is None:
            logger.info("Initializing Lorentz embedding service")
            self.lorentz_service = LorentzEmbeddingService()
        return self.lorentz_service
    
    def _get_shap_explainer(self) -> SHAPExplainer:
        """
        Get the SHAP explainer.
        
        Returns:
            SHAPExplainer: The SHAP explainer
        """
        if self.shap_explainer is None:
            logger.info("Initializing SHAP explainer")
            biomedlm_service = self._get_biomedlm_service()
            self.shap_explainer = SHAPExplainer(
                model_fn=lambda x: np.array([biomedlm_service.detect_contradiction(text, text)[1] for text in x]),
                tokenizer=biomedlm_service.tokenizer
            )
        return self.shap_explainer
    
    def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
        use_biomedlm: bool = True,
        use_tsmixer: bool = False,
        use_lorentz: bool = False,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Detect contradiction between two claims.
        
        Args:
            claim1: First claim
            claim2: Second claim
            use_biomedlm: Whether to use BioMedLM
            use_tsmixer: Whether to use TSMixer
            use_lorentz: Whether to use Lorentz embeddings
            threshold: Threshold for contradiction detection
            
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
            "confidence": "low",
            "methods_used": []
        }
        
        # Use BioMedLM
        if use_biomedlm:
            logger.info("Using BioMedLM for contradiction detection")
            biomedlm_service = self._get_biomedlm_service()
            is_contradiction, score = biomedlm_service.detect_contradiction(claim1, claim2)
            
            result["methods_used"].append("biomedlm")
            result["biomedlm_score"] = float(score)
            
            # Update result
            result["contradiction_score"] = float(score)
            result["is_contradiction"] = score > threshold
        
        # Use TSMixer
        if use_tsmixer:
            logger.info("Using TSMixer for contradiction detection")
            tsmixer_service = self._get_tsmixer_service()
            
            # Create a simple temporal sequence
            sequence = [
                {"claim": claim1, "timestamp": 0},
                {"claim": claim2, "timestamp": 1}
            ]
            
            # Get BioMedLM embeddings
            biomedlm_service = self._get_biomedlm_service()
            embedding_fn = biomedlm_service.encode
            
            # Analyze sequence
            analysis = tsmixer_service.analyze_temporal_sequence(sequence, embedding_fn)
            
            result["methods_used"].append("tsmixer")
            result["tsmixer_score"] = float(analysis["contradiction_scores"][1])
            
            # Update result if not using BioMedLM
            if not use_biomedlm:
                result["contradiction_score"] = float(analysis["contradiction_scores"][1])
                result["is_contradiction"] = analysis["contradiction_scores"][1] > threshold
        
        # Use Lorentz embeddings
        if use_lorentz:
            logger.info("Using Lorentz embeddings for contradiction detection")
            lorentz_service = self._get_lorentz_service()
            
            # Initialize with claims if not already initialized
            if lorentz_service._entity_to_idx is None or len(lorentz_service._entity_to_idx) == 0:
                lorentz_service.initialize_model(["claim1", "claim2"])
            
            # Get distance
            distance = lorentz_service.get_distance("claim1", "claim2")
            
            # Normalize distance to [0, 1]
            normalized_distance = 1.0 - np.exp(-distance)
            
            result["methods_used"].append("lorentz")
            result["lorentz_score"] = float(normalized_distance)
            
            # Update result if not using BioMedLM or TSMixer
            if not use_biomedlm and not use_tsmixer:
                result["contradiction_score"] = float(normalized_distance)
                result["is_contradiction"] = normalized_distance > threshold
        
        # Set confidence based on score
        if result["contradiction_score"] > 0.8:
            result["confidence"] = "high"
        elif result["contradiction_score"] > 0.6:
            result["confidence"] = "medium"
        else:
            result["confidence"] = "low"
        
        logger.info(f"Contradiction detection result: {result['is_contradiction']} (score: {result['contradiction_score']}, confidence: {result['confidence']})")
        
        return result
    
    def explain_contradiction(
        self,
        claim1: str,
        claim2: str,
        contradiction_score: float
    ) -> Dict[str, Any]:
        """
        Explain a contradiction.
        
        Args:
            claim1: First claim
            claim2: Second claim
            contradiction_score: Contradiction score
            
        Returns:
            Explanation
        """
        logger.info(f"Explaining contradiction between '{claim1}' and '{claim2}'")
        
        # Get SHAP explainer
        shap_explainer = self._get_shap_explainer()
        
        # Get explanation
        explanation = shap_explainer.explain_contradiction(claim1, claim2)
        
        # Add contradiction score
        explanation["contradiction_score"] = contradiction_score
        
        logger.info("Contradiction explanation generated")
        
        return explanation
    
    def unload_models(self):
        """Unload all models from memory."""
        if self.biomedlm_service is not None:
            self.biomedlm_service.unload_model()
        
        if self.tsmixer_service is not None:
            self.tsmixer_service.unload_model()
        
        if self.lorentz_service is not None:
            self.lorentz_service.unload_model()
        
        logger.info("All models unloaded")
