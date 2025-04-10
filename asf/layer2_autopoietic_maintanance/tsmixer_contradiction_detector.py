"""
TSMixer-based Contradiction Detector

This module provides a TSMixer-based implementation for detecting temporal contradictions
in the ASF framework. It extends the base ContradictionDetector with advanced temporal
pattern analysis using TSMixer.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import time

from asf.layer2_autopoietic_maintanance.contradiction_detection import ContradictionDetector
from asf.layer1_knowledge_substrate.temporal.tsmixer import TSMixer, AdaptiveTSMixer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("tsmixer-contradiction-detector")

class TSMixerContradictionDetector(ContradictionDetector):
    """
    TSMixer-based contradiction detector for temporal pattern analysis.
    
    This class extends the base ContradictionDetector with advanced temporal
    pattern analysis using TSMixer, which provides improved detection of
    temporal contradictions in time series data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TSMixer-based contradiction detector.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config or {}
        
        # Initialize TSMixer model
        self.tsmixer_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
    
    def _detect_temporal_pattern_contradictions(self, current_temporal: Dict, 
                                              update_temporal: Dict) -> List[Dict]:
        """
        Detect temporal pattern contradictions using TSMixer.
        
        This method overrides the base implementation to use TSMixer for
        advanced temporal pattern analysis.
        
        Args:
            current_temporal: Current temporal data
            update_temporal: New temporal data
            
        Returns:
            List of temporal pattern contradictions
        """
        # If TSMixer is not available, fall back to base implementation
        if self.tsmixer_model is None:
            return super()._detect_temporal_pattern_contradictions(current_temporal, update_temporal)
        
        contradictions = []
        
        # Extract time series data
        current_series = self._extract_time_series(current_temporal)
        update_series = self._extract_time_series(update_temporal)
        
        # Check if we have valid time series data
        if current_series is None or update_series is None:
            # Fall back to base implementation
            return super()._detect_temporal_pattern_contradictions(current_temporal, update_temporal)
        
        # Analyze temporal patterns using TSMixer
        try:
            # Convert to tensors
            current_tensor = torch.tensor(current_series, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, features]
            update_tensor = torch.tensor(update_series, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, features]
            
            # Move to device
            current_tensor = current_tensor.to(self.device)
            update_tensor = update_tensor.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                current_pred = self.tsmixer_model(current_tensor)
                update_pred = self.tsmixer_model(update_tensor)
            
            # Calculate prediction difference
            pred_diff = torch.abs(current_pred - update_pred).mean().item()
            
            # Check for significant prediction difference
            if pred_diff > self.detection_thresholds["temporal_pattern"]:
                contradictions.append({
                    'type': 'temporal_pattern',
                    'subtype': 'prediction_divergence',
                    'prediction_difference': pred_diff,
                    'severity': min(pred_diff, 1.0)
                })
            
            # Check for trend reversals
            current_trend = self._calculate_trend(current_tensor[0].cpu().numpy())
            update_trend = self._calculate_trend(update_tensor[0].cpu().numpy())
            
            # Check if trends have opposite signs (reversal)
            if (current_trend * update_trend < 0) and (abs(current_trend) > 0.1) and (abs(update_trend) > 0.1):
                contradictions.append({
                    'type': 'temporal_pattern',
                    'subtype': 'trend_reversal',
                    'current_trend': current_trend,
                    'new_trend': update_trend,
                    'severity': min(abs(current_trend) + abs(update_trend), 1.0)
                })
            
            # Check for volatility contradictions
            current_volatility = self._calculate_volatility(current_tensor[0].cpu().numpy())
            update_volatility = self._calculate_volatility(update_tensor[0].cpu().numpy())
            
            # Calculate relative change in volatility
            if current_volatility != 0:
                volatility_change = (update_volatility - current_volatility) / current_volatility
                
                if abs(volatility_change) > 0.5:  # Significant volatility change
                    contradictions.append({
                        'type': 'temporal_pattern',
                        'subtype': 'volatility_change',
                        'current_volatility': current_volatility,
                        'new_volatility': update_volatility,
                        'relative_change': volatility_change,
                        'severity': min(abs(volatility_change), 1.0)
                    })
        
        except Exception as e:
            logger.error(f"Error in TSMixer temporal pattern analysis: {e}")
            # Fall back to base implementation
            return super()._detect_temporal_pattern_contradictions(current_temporal, update_temporal)
        
        return contradictions
    
    def _extract_time_series(self, temporal_data: Dict) -> Optional[np.ndarray]:
        """
        Extract time series data from temporal data dictionary.
        
        Args:
            temporal_data: Temporal data dictionary
            
        Returns:
            Numpy array of time series data or None if not available
        """
        # Check if we have time series data
        if 'values' in temporal_data:
            values = temporal_data['values']
            if isinstance(values, list) and len(values) > 0:
                # Convert to numpy array
                return np.array(values)
        
        # Check if we have time series data in a different format
        if 'time_series' in temporal_data:
            time_series = temporal_data['time_series']
            if isinstance(time_series, list) and len(time_series) > 0:
                # Convert to numpy array
                return np.array(time_series)
        
        return None
    
    def _calculate_trend(self, time_series: np.ndarray) -> float:
        """
        Calculate trend in time series data.
        
        Args:
            time_series: Time series data
            
        Returns:
            Trend value (positive for upward, negative for downward)
        """
        # Check if time series is 1D or 2D
        if len(time_series.shape) == 1:
            # 1D time series
            x = np.arange(len(time_series))
            y = time_series
        else:
            # 2D time series, use first feature
            x = np.arange(time_series.shape[0])
            y = time_series[:, 0]
        
        # Calculate trend using linear regression
        if len(x) > 1:
            # Calculate slope using numpy's polyfit
            slope, _ = np.polyfit(x, y, 1)
            return slope
        
        return 0.0
    
    def _calculate_volatility(self, time_series: np.ndarray) -> float:
        """
        Calculate volatility in time series data.
        
        Args:
            time_series: Time series data
            
        Returns:
            Volatility value
        """
        # Check if time series is 1D or 2D
        if len(time_series.shape) == 1:
            # 1D time series
            y = time_series
        else:
            # 2D time series, use first feature
            y = time_series[:, 0]
        
        # Calculate volatility as standard deviation of differences
        if len(y) > 1:
            # Calculate differences
            diffs = np.diff(y)
            # Calculate standard deviation
            volatility = np.std(diffs)
            return volatility
        
        return 0.0
