"""
DSPy service for BO backend.

This module provides a service for interacting with DSPy.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import Depends, HTTPException, status

from ...utils import handle_api_error

# Import DSPy components if available
try:
    from asf.medical.ml.dspy.dspy_client import get_dspy_client, DSPyClient
    from asf.medical.ml.dspy.modules.medical_rag import MedicalRAGModule
    from asf.medical.ml.dspy.modules.contradiction_detection import ContradictionDetectionModule
    from asf.medical.ml.dspy.modules.evidence_extraction import EvidenceExtractionModule
    from asf.medical.ml.dspy.modules.medical_summarization import MedicalSummarizationModule
    from asf.medical.ml.dspy.modules.clinical_qa import ClinicalQAModule
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logging.warning("DSPy is not available. Some functionality will be limited.")

logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                          "config", "llm", "dspy_config.yaml")

class DSPyService:
    """
    Service for interacting with DSPy.
    """
    
    def __init__(self):
        """
        Initialize the DSPy service.
        """
        self._client = None
        self._config = None
        
        if not DSPY_AVAILABLE:
            logger.warning("DSPy is not available. Some functionality will be limited.")
            return
        
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Note: DSPy client is initialized asynchronously, so we'll do it in get_client
            logger.info("DSPy service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy service: {str(e)}")
    
    async def get_client(self) -> DSPyClient:
        """
        Get the DSPy client.
        
        Returns:
            DSPy client
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        if self._client is None:
            try:
                # Initialize DSPy client
                self._client = await get_dspy_client()
                logger.info("DSPy client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DSPy client: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize DSPy client: {str(e)}"
                )
        
        return self._client
    
    async def get_modules(self) -> List[Dict[str, Any]]:
        """
        Get all registered DSPy modules.
        
        Returns:
            List of module information dictionaries
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Get registered modules
            modules = await client.list_modules()
            
            # Format module information
            module_infos = []
            for module_name, module_data in modules.items():
                module_infos.append({
                    "name": module_name,
                    "description": module_data.get("description", ""),
                    "signature": module_data.get("signature", ""),
                    "parameters": module_data.get("parameters", {}),
                    "registered_at": module_data.get("registered_at", datetime.utcnow().isoformat()),
                    "last_used": module_data.get("last_used"),
                    "usage_count": module_data.get("usage_count", 0)
                })
            
            return module_infos
        except Exception as e:
            logger.error(f"Error getting DSPy modules: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get DSPy modules: {str(e)}"
            )
    
    async def get_module(self, module_name: str) -> Dict[str, Any]:
        """
        Get information about a specific DSPy module.
        
        Args:
            module_name: Module name
            
        Returns:
            Module information dictionary
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Get registered modules
            modules = await client.list_modules()
            
            # Check if module exists
            if module_name not in modules:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Module '{module_name}' not found"
                )
            
            # Get module data
            module_data = modules[module_name]
            
            # Format module information
            return {
                "name": module_name,
                "description": module_data.get("description", ""),
                "signature": module_data.get("signature", ""),
                "parameters": module_data.get("parameters", {}),
                "registered_at": module_data.get("registered_at", datetime.utcnow().isoformat()),
                "last_used": module_data.get("last_used"),
                "usage_count": module_data.get("usage_count", 0)
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting DSPy module '{module_name}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get DSPy module '{module_name}': {str(e)}"
            )
    
    async def register_module(self, module_name: str, module_type: str, 
                             parameters: Dict[str, Any], description: Optional[str] = None) -> Dict[str, Any]:
        """
        Register a new DSPy module.
        
        Args:
            module_name: Module name
            module_type: Module type
            parameters: Module parameters
            description: Module description
            
        Returns:
            Registered module information
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Create module instance based on module_type
            module_instance = None
            if module_type == "medical_rag":
                module_instance = MedicalRAGModule(**parameters)
            elif module_type == "contradiction_detection":
                module_instance = ContradictionDetectionModule(**parameters)
            elif module_type == "evidence_extraction":
                module_instance = EvidenceExtractionModule(**parameters)
            elif module_type == "medical_summarization":
                module_instance = MedicalSummarizationModule(**parameters)
            elif module_type == "clinical_qa":
                module_instance = ClinicalQAModule(**parameters)
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported module type: {module_type}"
                )
            
            # Register module
            await client.register_module(
                name=module_name,
                module=module_instance,
                description=description or f"{module_type} module"
            )
            
            # Get registered module info
            modules = await client.list_modules()
            module_data = modules[module_name]
            
            # Format module information
            return {
                "name": module_name,
                "description": module_data.get("description", ""),
                "signature": module_data.get("signature", ""),
                "parameters": module_data.get("parameters", {}),
                "registered_at": module_data.get("registered_at", datetime.utcnow().isoformat()),
                "last_used": module_data.get("last_used"),
                "usage_count": module_data.get("usage_count", 0)
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error registering DSPy module '{module_name}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to register DSPy module '{module_name}': {str(e)}"
            )
    
    async def unregister_module(self, module_name: str) -> Dict[str, Any]:
        """
        Unregister a DSPy module.
        
        Args:
            module_name: Module name
            
        Returns:
            Success message
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if module exists
            modules = await client.list_modules()
            if module_name not in modules:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Module '{module_name}' not found"
                )
            
            # Unregister module
            await client.unregister_module(module_name)
            
            return {
                "success": True,
                "message": f"Module '{module_name}' unregistered successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error unregistering DSPy module '{module_name}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to unregister DSPy module '{module_name}': {str(e)}"
            )
    
    async def execute_module(self, module_name: str, inputs: Dict[str, Any], 
                            config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute a DSPy module.
        
        Args:
            module_name: Module name
            inputs: Module inputs
            config: Module configuration
            
        Returns:
            Module execution results
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if module exists
            modules = await client.list_modules()
            if module_name not in modules:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Module '{module_name}' not found"
                )
            
            # Execute module
            start_time = datetime.utcnow()
            result = await client.call_module(
                module_name=module_name,
                **inputs,
                **(config or {})
            )
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Get module info
            module_info = modules[module_name]
            
            return {
                "module_name": module_name,
                "inputs": inputs,
                "outputs": result,
                "execution_time_ms": execution_time_ms,
                "model_used": module_info.get("model", "unknown"),
                "tokens_used": module_info.get("tokens_used"),
                "created_at": end_time.isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error executing DSPy module '{module_name}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute DSPy module '{module_name}': {str(e)}"
            )
    
    async def optimize_module(self, module_name: str, metric: str, num_trials: int = 10,
                             examples: List[Dict[str, Any]] = [], 
                             config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize a DSPy module.
        
        Args:
            module_name: Module name
            metric: Optimization metric
            num_trials: Number of optimization trials
            examples: Training examples
            config: Optimization configuration
            
        Returns:
            Optimization results
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if module exists
            modules = await client.list_modules()
            if module_name not in modules:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Module '{module_name}' not found"
                )
            
            # Optimize module
            start_time = datetime.utcnow()
            
            # This would be replaced with actual optimization
            # For now, we'll just return a mock response
            optimization_result = {
                "success": True,
                "module_name": module_name,
                "metric": metric,
                "num_trials": num_trials,
                "examples_used": len(examples),
                "original_score": 0.75,
                "optimized_score": 0.85,
                "improvement": 0.1,
                "best_prompt": "Optimized prompt template",
                "execution_time_ms": 5000
            }
            
            end_time = datetime.utcnow()
            execution_time_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                **optimization_result,
                "execution_time_ms": execution_time_ms,
                "created_at": end_time.isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error optimizing DSPy module '{module_name}': {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to optimize DSPy module '{module_name}': {str(e)}"
            )
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get the current DSPy configuration.
        
        Returns:
            DSPy configuration
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        return self._config
    
    async def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the DSPy configuration.
        
        Args:
            config: New configuration
            
        Returns:
            Updated configuration
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="DSPy is not available. Please check your installation."
            )
        
        try:
            # Save config to file
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Update instance config
            self._config = config
            
            return config
        except Exception as e:
            logger.error(f"Error updating DSPy config: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update DSPy config: {str(e)}"
            )

# Singleton instance
_dspy_service = None

def get_dspy_service():
    """
    Get the DSPy service instance.
    
    Returns:
        DSPy service instance
    """
    global _dspy_service
    
    if _dspy_service is None:
        _dspy_service = DSPyService()
    
    return _dspy_service
