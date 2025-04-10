"""
TSMixer-based Contradiction Detector for Medical Claims

This module provides a TSMixer-based implementation for detecting temporal contradictions
in medical claims. It integrates with the BioMedLM wrapper to provide enhanced
contradiction detection capabilities.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import re
from dataclasses import dataclass, field

from asf.layer1_knowledge_substrate.temporal.tsmixer import AdaptiveTSMixer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tsmixer-contradiction-detector")

class TSMixerContradictionDetector:
    """
    TSMixer-based contradiction detector for medical claims.
    
    This class uses TSMixer to detect temporal contradictions in medical claims,
    focusing on time series data and temporal patterns.
    """
    
    def __init__(
        self, 
        biomedlm_scorer=None, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the TSMixer-based contradiction detector.
        
        Args:
            biomedlm_scorer: BioMedLMScorer instance for semantic contradiction detection
            device: Device to run the model on
            config: Configuration dictionary
        """
        self.biomedlm_scorer = biomedlm_scorer
        self.device = device
        self.config = config or {}
        
        # Initialize TSMixer model
        self.tsmixer_model = None
        
        # Initialize TSMixer model if requested
        if self.config.get("use_tsmixer", True):
            self._initialize_tsmixer_model()
    
    def _initialize_tsmixer_model(self):
        """Initialize TSMixer model for temporal pattern analysis."""
        try:
            # Get TSMixer configuration
            seq_len = self.config.get("seq_len", 64)
            num_features = self.config.get("num_features", 1)
            num_blocks = self.config.get("num_blocks", 3)
            forecast_horizon = self.config.get("forecast_horizon", 12)
            
            # Create AdaptiveTSMixer model for variable-length sequences
            self.tsmixer_model = AdaptiveTSMixer(
                max_seq_len=seq_len,
                num_features=num_features,
                num_blocks=num_blocks,
                forecast_horizon=forecast_horizon
            ).to(self.device)
            
            # Load pre-trained weights if available
            model_path = self.config.get("tsmixer_model_path")
            if model_path:
                try:
                    self.tsmixer_model.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info(f"Loaded TSMixer model from {model_path}")
                except Exception as e:
                    logger.warning(f"Failed to load TSMixer model from {model_path}: {e}")
            
            logger.info("TSMixer model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TSMixer model: {e}")
            self.tsmixer_model = None
    
    def detect_contradiction(self, claim1: str, claim2: str) -> Dict[str, Any]:
        """
        Detect contradiction between medical claims.
        
        This method combines TSMixer-based temporal analysis with BioMedLM-based
        semantic analysis to detect contradictions in medical claims.
        
        Args:
            claim1: First medical claim
            claim2: Second medical claim
            
        Returns:
            Dictionary with contradiction detection results
        """
        result = {
            "text1": claim1,
            "text2": claim2,
            "has_contradiction": False,
            "contradiction_score": 0.0,
            "method": "tsmixer_biomedlm",
            "temporal_analysis": {}
        }
        
        # Extract temporal data from claims
        temporal_data1 = self._extract_temporal_data(claim1)
        temporal_data2 = self._extract_temporal_data(claim2)
        
        # Check if we have temporal data to analyze
        has_temporal_data = (temporal_data1 is not None and temporal_data2 is not None)
        
        # Get semantic contradiction score from BioMedLM if available
        semantic_score = 0.0
        if self.biomedlm_scorer is not None:
            try:
                biomedlm_result = self.biomedlm_scorer.detect_contradiction(claim1, claim2)
                semantic_score = biomedlm_result.get("contradiction_score", 0.0)
                
                # Add BioMedLM results to our result
                result["biomedlm_result"] = {
                    "contradiction_score": semantic_score,
                    "agreement_score": biomedlm_result.get("agreement_score", 0.0),
                    "confidence": biomedlm_result.get("confidence", 0.0)
                }
            except Exception as e:
                logger.error(f"Error getting BioMedLM contradiction score: {e}")
        
        # If we have temporal data and TSMixer model, analyze temporal patterns
        temporal_score = 0.0
        if has_temporal_data and self.tsmixer_model is not None:
            try:
                # Analyze temporal patterns
                temporal_result = self._analyze_temporal_patterns(temporal_data1, temporal_data2)
                temporal_score = temporal_result.get("contradiction_score", 0.0)
                
                # Add temporal analysis results to our result
                result["temporal_analysis"] = temporal_result
            except Exception as e:
                logger.error(f"Error in temporal pattern analysis: {e}")
        
        # Combine semantic and temporal scores
        if has_temporal_data:
            # If we have temporal data, use a weighted combination
            combined_score = 0.7 * semantic_score + 0.3 * temporal_score
        else:
            # If we don't have temporal data, use only semantic score
            combined_score = semantic_score
        
        # Update result with combined score
        result["contradiction_score"] = combined_score
        result["has_contradiction"] = combined_score > 0.7
        
        return result
    
    def _extract_temporal_data(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract temporal data from text.
        
        This method extracts time series data, trends, and other temporal information
        from medical claims.
        
        Args:
            text: Medical claim text
            
        Returns:
            Dictionary with temporal data or None if no temporal data found
        """
        # Initialize temporal data
        temporal_data = {
            "values": [],
            "timestamps": [],
            "trends": [],
            "has_temporal_data": False
        }
        
        # Extract numeric values with units
        numeric_pattern = r'(\d+\.?\d*)\s*(%|percent|mg/dl|mmol/l|kg|cm|mm|mmHg)'
        numeric_matches = re.finditer(numeric_pattern, text, re.IGNORECASE)
        
        for match in numeric_matches:
            value = float(match.group(1))
            unit = match.group(2)
            temporal_data["values"].append(value)
            temporal_data["units"] = unit
        
        # Extract time-related information
        time_pattern = r'(\d+)\s*(days?|weeks?|months?|years?)'
        time_matches = re.finditer(time_pattern, text, re.IGNORECASE)
        
        for match in time_matches:
            value = int(match.group(1))
            unit = match.group(2)
            temporal_data["timestamps"].append(value)
            temporal_data["time_units"] = unit
        
        # Extract trend information
        trend_keywords = {
            "increase": 1.0,
            "decrease": -1.0,
            "higher": 1.0,
            "lower": -1.0,
            "improved": 1.0,
            "worsened": -1.0,
            "rising": 1.0,
            "falling": -1.0,
            "grew": 1.0,
            "shrank": -1.0
        }
        
        for keyword, trend_value in trend_keywords.items():
            if keyword in text.lower():
                temporal_data["trends"].append(trend_value)
        
        # Check if we have any temporal data
        if temporal_data["values"] or temporal_data["timestamps"] or temporal_data["trends"]:
            temporal_data["has_temporal_data"] = True
            return temporal_data
        
        return None
    
    def _analyze_temporal_patterns(self, temporal_data1: Dict[str, Any], temporal_data2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze temporal patterns in the extracted temporal data.
        
        This method uses TSMixer to analyze temporal patterns and detect contradictions.
        
        Args:
            temporal_data1: Temporal data from first claim
            temporal_data2: Temporal data from second claim
            
        Returns:
            Dictionary with temporal analysis results
        """
        result = {
            "contradiction_score": 0.0,
            "contradiction_type": None,
            "details": {}
        }
        
        # Check for trend contradictions
        trends1 = temporal_data1.get("trends", [])
        trends2 = temporal_data2.get("trends", [])
        
        if trends1 and trends2:
            # Calculate average trend
            avg_trend1 = sum(trends1) / len(trends1)
            avg_trend2 = sum(trends2) / len(trends2)
            
            # Check if trends have opposite signs (contradiction)
            if avg_trend1 * avg_trend2 < 0:
                trend_contradiction_score = min(abs(avg_trend1) + abs(avg_trend2), 1.0)
                result["contradiction_score"] = max(result["contradiction_score"], trend_contradiction_score)
                result["contradiction_type"] = "trend_reversal"
                result["details"]["trend_reversal"] = {
                    "trend1": avg_trend1,
                    "trend2": avg_trend2,
                    "score": trend_contradiction_score
                }
        
        # Check for value contradictions
        values1 = temporal_data1.get("values", [])
        values2 = temporal_data2.get("values", [])
        
        if values1 and values2:
            # Use TSMixer for advanced temporal analysis
            try:
                # Convert to tensors
                if len(values1) == 1:
                    # If we only have one value, repeat it to create a sequence
                    values1 = [values1[0]] * 2
                
                if len(values2) == 1:
                    # If we only have one value, repeat it to create a sequence
                    values2 = [values2[0]] * 2
                
                # Create tensors
                tensor1 = torch.tensor(values1, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
                tensor2 = torch.tensor(values2, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
                
                # Move to device
                tensor1 = tensor1.to(self.device)
                tensor2 = tensor2.to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    pred1 = self.tsmixer_model(tensor1)
                    pred2 = self.tsmixer_model(tensor2)
                
                # Calculate prediction difference
                pred_diff = torch.abs(pred1 - pred2).mean().item()
                
                # Check for significant prediction difference
                if pred_diff > 0.3:  # Threshold for significant difference
                    value_contradiction_score = min(pred_diff, 1.0)
                    result["contradiction_score"] = max(result["contradiction_score"], value_contradiction_score)
                    result["contradiction_type"] = "value_divergence"
                    result["details"]["value_divergence"] = {
                        "prediction_difference": pred_diff,
                        "score": value_contradiction_score
                    }
            except Exception as e:
                logger.error(f"Error in TSMixer analysis: {e}")
        
        return result
