"""
Factory for LLM service implementations.

This module provides a factory for creating LLM service implementations,
mapping model identifiers to the appropriate service implementation.
"""

import logging
from typing import Dict, Any, Optional

from asf.medical.llm_gateway.interfaces.llm_service import LLMServiceInterface
from asf.medical.llm_gateway.interfaces.exceptions import ModelNotAvailableException
from asf.medical.llm_gateway.services.mcp_service import MCPService
from asf.medical.llm_gateway.observability.metrics import MetricsService

logger = logging.getLogger(__name__)


class ServiceFactory:
    """
    Factory for creating LLM service implementations.
    
    This class is responsible for creating the appropriate LLM service implementation
    based on the requested model, using configuration to map models to services.
    """
    
    def __init__(self, config: Dict[str, Any], metrics_service: Optional[MetricsService] = None):
        """
        Initialize the service factory with configuration.
        
        Args:
            config: Configuration dictionary for the factory
            metrics_service: Optional metrics service for instrumentation
        """
        self.config = config
        self.metrics_service = metrics_service or MetricsService()
        
        # Model mapping configuration
        self.model_to_service_map = config.get('model_to_service_map', {})
        
        # Available services
        self.services: Dict[str, LLMServiceInterface] = {}
        
        # Initialize services
        self._initialize_services()
        
        logger.info(f"ServiceFactory initialized with {len(self.services)} services")
    
    def _initialize_services(self):
        """
        Initialize available LLM services based on configuration.
        """
        # Get service configurations
        service_configs = self.config.get('services', {})
        
        # Initialize MCP service if configured
        if 'mcp' in service_configs:
            try:
                self.services['mcp'] = MCPService(self.config, self.metrics_service)
                logger.info("Initialized MCP service")
            except Exception as e:
                logger.error(f"Failed to initialize MCP service: {str(e)}")
        
        # Additional services can be added here as they are implemented
    
    def get_service_for_model(self, model: str) -> LLMServiceInterface:
        """
        Get the appropriate service for the requested model.
        
        Args:
            model: The model identifier
            
        Returns:
            An instance of LLMServiceInterface for the model
            
        Raises:
            ModelNotAvailableException: If no service is available for the model
        """
        # Check if we have a direct mapping for this model
        service_id = self.model_to_service_map.get(model)
        
        # If not found in the map, try to infer from model name
        if not service_id:
            service_id = self._infer_service_from_model(model)
            
        # Get the service instance
        service = self.services.get(service_id)
        
        if not service:
            self.metrics_service.record_counter_inc(f"service_factory.model_not_available.{model}")
            raise ModelNotAvailableException(f"No service available for model: {model}")
            
        return service
    
    def _infer_service_from_model(self, model: str) -> Optional[str]:
        """
        Infer the service ID from a model identifier.
        
        Args:
            model: The model identifier
            
        Returns:
            The inferred service ID, or None if no service can be inferred
        """
        # Model naming conventions can be used to infer the service
        model_lower = model.lower()
        
        # Check for MCP models
        if model_lower.startswith('mcp-') or model_lower.startswith('mc-'):
            return 'mcp'
        
        # Default to MCP if available
        if 'mcp' in self.services:
            return 'mcp'
        
        # No service could be inferred
        return None
    
    def list_available_models(self) -> Dict[str, str]:
        """
        List all available models and their associated services.
        
        Returns:
            A dictionary of model identifiers to service IDs
        """
        available_models = {}
        
        # Add explicitly mapped models
        available_models.update(self.model_to_service_map)
        
        # Add models from services
        for service_id, service in self.services.items():
            if hasattr(service, 'list_models') and callable(service.list_models):
                for model in service.list_models():
                    available_models[model] = service_id
        
        return available_models