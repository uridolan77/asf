"""
CL-PEFT Strategy Manager

This module provides a manager for initializing and managing CL strategies.
"""

import os
import logging
from typing import Dict, Any, Optional, Union

from peft import PeftModel

from asf.medical.core.logging_config import get_logger
from .config.enums import CLStrategy as CLStrategyEnum
from .strategies import (
    ElasticWeightConsolidation,
    ExperienceReplay,
    GenerativeReplay,
    OrthogonalLoRA,
    AdaptiveSVD,
    MaskBasedCL
)

logger = get_logger(__name__)

class StrategyManager:
    """
    Manager for CL strategies.
    
    This class handles the initialization and management of CL strategies.
    """
    
    def __init__(self):
        """Initialize the strategy manager."""
        self.strategy_map = {
            CLStrategyEnum.EWC: ElasticWeightConsolidation,
            CLStrategyEnum.REPLAY: ExperienceReplay,
            CLStrategyEnum.GENERATIVE_REPLAY: GenerativeReplay,
            CLStrategyEnum.ORTHOGONAL_LORA: OrthogonalLoRA,
            CLStrategyEnum.ADAPTIVE_SVD: AdaptiveSVD,
            CLStrategyEnum.MASK_BASED: MaskBasedCL
        }
    
    def get_strategy(
        self,
        strategy_name: Union[str, CLStrategyEnum],
        model: PeftModel,
        strategy_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Get a CL strategy instance.
        
        Args:
            strategy_name: Name of the strategy
            model: The PEFT model to apply the strategy to
            strategy_config: Configuration for the strategy
            **kwargs: Additional arguments for the strategy
            
        Returns:
            CL strategy instance
        """
        # Convert string to enum if needed
        if isinstance(strategy_name, str):
            try:
                strategy_name = CLStrategyEnum(strategy_name)
            except ValueError:
                logger.warning(f"Unknown strategy name: {strategy_name}, falling back to naive")
                return None
        
        # Get strategy class
        strategy_class = self.strategy_map.get(strategy_name)
        if strategy_class is None:
            logger.warning(f"No implementation found for strategy: {strategy_name}")
            return None
        
        # Prepare strategy configuration
        config = strategy_config or {}
        config.update(kwargs)
        
        # Initialize strategy
        try:
            logger.info(f"Initializing {strategy_name} strategy with config: {config}")
            strategy = strategy_class(model=model, **config)
            return strategy
        except Exception as e:
            logger.error(f"Error initializing strategy {strategy_name}: {str(e)}")
            return None

# Singleton instance
strategy_manager = StrategyManager()
