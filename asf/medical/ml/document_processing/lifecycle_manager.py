"""
Medical Research Synthesizer Lifecycle Management

This module provides functionality for integrating the Medical Research Synthesizer
with the model lifecycle management system.
"""

import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LifecycleManager:
    """
    Handles integration with the model lifecycle management system.
    """
    
    @staticmethod
    def register_synthesizer(
        component_names: Dict[str, str],
        version: str = "1.0.0",
        description: str = "Medical Research Synthesizer for processing and analyzing research papers",
        status: str = "active"
    ) -> bool:
        """
        Register the synthesizer with the model lifecycle manager.
        
        Args:
            component_names: Dictionary of component names
            version: Model version
            description: Model description
            status: Model status
            
        Returns:
            True if registration was successful, False otherwise
        """
        try:
            from asf.medical.ml.model_lifecycle_manager import get_model_lifecycle_manager
            from asf.medical.ml.models.model_registry import ModelFramework
            
            # Get lifecycle manager
            lifecycle_manager = get_model_lifecycle_manager()
            
            # Register components
            lifecycle_manager.registry.register_model(
                name="medical_research_synthesizer",
                version=version,
                framework=ModelFramework.CUSTOM,
                description=description,
                status=status,
                path="",
                parameters=component_names
            )
            
            logger.info("Registered Medical Research Synthesizer with model lifecycle manager")
            return True
        except Exception as e:
            logger.warning(f"Could not register with model lifecycle manager: {str(e)}")
            return False
    
    @staticmethod
    def track_usage(document_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Track usage of the synthesizer for a specific document.
        
        Args:
            document_id: Unique identifier for the document
            metrics: Dictionary of usage metrics
            
        Returns:
            True if tracking was successful, False otherwise
        """
        try:
            from asf.medical.ml.model_lifecycle_manager import get_model_lifecycle_manager
            
            # Get lifecycle manager
            lifecycle_manager = get_model_lifecycle_manager()
            
            # Track usage
            lifecycle_manager.tracker.track_usage(
                model_name="medical_research_synthesizer",
                instance_id=document_id,
                metrics=metrics
            )
            
            logger.info(f"Tracked usage for document {document_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not track usage: {str(e)}")
            return False
    
    @staticmethod
    def log_performance(
        document_id: str,
        entity_count: int,
        relation_count: int,
        processing_time: float,
        additional_metrics: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log performance metrics for a document processing run.
        
        Args:
            document_id: Unique identifier for the document
            entity_count: Number of entities extracted
            relation_count: Number of relations extracted
            processing_time: Total processing time in seconds
            additional_metrics: Additional performance metrics
            
        Returns:
            True if logging was successful, False otherwise
        """
        try:
            from asf.medical.ml.model_lifecycle_manager import get_model_lifecycle_manager
            
            # Get lifecycle manager
            lifecycle_manager = get_model_lifecycle_manager()
            
            # Prepare metrics
            metrics = {
                "entity_count": entity_count,
                "relation_count": relation_count,
                "processing_time_seconds": processing_time,
                "entities_per_second": entity_count / processing_time if processing_time > 0 else 0,
                "relations_per_second": relation_count / processing_time if processing_time > 0 else 0
            }
            
            # Add additional metrics if provided
            if additional_metrics:
                metrics.update(additional_metrics)
            
            # Log performance
            lifecycle_manager.monitor.log_performance(
                model_name="medical_research_synthesizer",
                instance_id=document_id,
                metrics=metrics
            )
            
            logger.info(f"Logged performance metrics for document {document_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not log performance: {str(e)}")
            return False
