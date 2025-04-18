"""
DSPy service for BO backend.

This module provides a service for interacting with DSPy.
"""

import os
import sys
import yaml
import logging
import traceback
import importlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import Depends, HTTPException, status

from ...utils import handle_api_error

# Add the project root to the system path to allow importing from medical
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path in dspy_service.py")

# Import DSPy components if available
try:
    # Fix the import paths to correctly point to the DSPy modules
    from medical.ml.dspy.client import get_enhanced_client, EnhancedDSPyClient
    from medical.ml.dspy.modules.medical_rag import MedicalRAGModule
    from medical.ml.dspy.modules.contradiction_detection import ContradictionDetectionModule
    from medical.ml.dspy.modules.evidence_extraction import EvidenceExtractionModule
    from medical.ml.dspy.modules.medical_summarization import MedicalSummarizationModule
    from medical.ml.dspy.modules.clinical_qa import ClinicalQAModule
    DSPY_AVAILABLE = True
    print("Successfully imported DSPy modules!")
except ImportError as e:
    DSPY_AVAILABLE = False
    logging.warning(f"DSPy is not available. Some functionality will be limited. Error: {str(e)}")
    print(f"Failed to import DSPy modules: {str(e)}")

logger = logging.getLogger(__name__)

# Constants
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                          "config", "llm", "dspy_config.yaml")

# Mock modules data for when DSPy is not available
MOCK_MODULES = [
    {
        "name": "medical_rag",
        "description": "Medical Retrieval Augmented Generation module",
        "type": "RAG",
        "registered_at": datetime.now().isoformat()
    },
    {
        "name": "contradiction_detection",
        "description": "Module for detecting contradictions in medical texts",
        "type": "Classification",
        "registered_at": datetime.now().isoformat()
    },
    {
        "name": "evidence_extraction",
        "description": "Module for extracting evidence from medical texts",
        "type": "Extraction",
        "registered_at": datetime.now().isoformat()
    },
    {
        "name": "medical_summarization",
        "description": "Module for summarizing medical texts",
        "type": "Summarization",
        "registered_at": datetime.now().isoformat()
    },
    {
        "name": "clinical_qa",
        "description": "Module for answering clinical questions",
        "type": "QA",
        "registered_at": datetime.now().isoformat()
    }
]

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
        self._modules = {}
        
        if not DSPY_AVAILABLE:
            logger.warning("DSPy is not available. Using mock data.")
            return
        
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
            
            # Note: DSPy client is initialized asynchronously, so we'll do it in get_client
            logger.info("DSPy service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize DSPy service: {str(e)}")
    
    async def get_client(self):
        """
        Get the DSPy client.
        
        Returns:
            DSPy client
        """
        if not DSPY_AVAILABLE:
            # Instead of raising an exception, log a warning and return None
            logger.warning("DSPy is not available. Using mock data.")
            return None
        
        if self._client is None:
            try:
                # Initialize DSPy client using the global enhanced client
                self._client = await get_enhanced_client()
                
                # Register default modules if they don't exist yet
                await self._register_default_modules()
                
                logger.info("DSPy client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DSPy client: {str(e)}")
                logger.error(f"Error traceback: {traceback.format_exc()}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to initialize DSPy client: {str(e)}"
                )
        
        return self._client
    
    async def _register_default_modules(self):
        """Register default DSPy modules with the client."""
        if not self._client or not hasattr(self._client, 'register_module'):
            logger.warning("Cannot register modules: Client not initialized properly")
            return
            
        try:
            # Register Medical RAG module
            try:
                medical_rag = MedicalRAGModule()
                await self._client.register_module(
                    name="medical_rag",
                    module=medical_rag,
                    description="Medical Retrieval Augmented Generation module"
                )
                logger.info("Registered MedicalRAGModule")
            except Exception as e:
                logger.warning(f"Failed to register MedicalRAGModule: {str(e)}")
                
            # Register Contradiction Detection module
            try:
                contradiction_detection = ContradictionDetectionModule()
                await self._client.register_module(
                    name="contradiction_detection",
                    module=contradiction_detection,
                    description="Module for detecting contradictions in medical texts"
                )
                logger.info("Registered ContradictionDetectionModule")
            except Exception as e:
                logger.warning(f"Failed to register ContradictionDetectionModule: {str(e)}")
                
            # Register Evidence Extraction module
            try:
                evidence_extraction = EvidenceExtractionModule()
                await self._client.register_module(
                    name="evidence_extraction",
                    module=evidence_extraction,
                    description="Module for extracting evidence from medical texts"
                )
                logger.info("Registered EvidenceExtractionModule")
            except Exception as e:
                logger.warning(f"Failed to register EvidenceExtractionModule: {str(e)}")
                
            # Register Medical Summarization module
            try:
                medical_summarization = MedicalSummarizationModule()
                await self._client.register_module(
                    name="medical_summarization",
                    module=medical_summarization,
                    description="Module for summarizing medical texts"
                )
                logger.info("Registered MedicalSummarizationModule")
            except Exception as e:
                logger.warning(f"Failed to register MedicalSummarizationModule: {str(e)}")
                
            # Register Clinical QA module
            try:
                clinical_qa = ClinicalQAModule()
                await self._client.register_module(
                    name="clinical_qa",
                    module=clinical_qa,
                    description="Module for answering clinical questions"
                )
                logger.info("Registered ClinicalQAModule")
            except Exception as e:
                logger.warning(f"Failed to register ClinicalQAModule: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error registering default modules: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            
    async def get_modules(self) -> List[Dict[str, Any]]:
        """
        Get all registered DSPy modules.
        
        Returns:
            List of module information dictionaries
        """
        if not DSPY_AVAILABLE:
            # Return mock data instead of raising an exception
            logger.warning("DSPy not available, returning mock modules data")
            return MOCK_MODULES
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if client is properly initialized
            if client is None:
                logger.warning("DSPy client is None, returning mock modules data")
                return MOCK_MODULES
                
            # Log client attributes for debugging
            logger.info(f"DSPy client type: {type(client)}")
            
            try:
                # Get registered modules
                modules = client.list_modules()
                
                # Return the modules data - it should already be in the right format
                if modules:
                    return modules
                
                logger.warning("No modules found in client, returning mock data")
                return MOCK_MODULES
            except AttributeError as attr_err:
                # If client doesn't have list_modules method, return mock data
                logger.warning(f"DSPy client missing required method: {str(attr_err)}. Using mock data.")
                return MOCK_MODULES
        except Exception as e:
            logger.error(f"Error getting DSPy modules: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            # Return mock data instead of raising an exception
            return MOCK_MODULES
    
    async def get_module(self, module_name: str) -> Dict[str, Any]:
        """
        Get information about a specific DSPy module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Dict[str, Any]: Module information
            
        Raises:
            HTTPException: If the module is not found
        """
        if not DSPY_AVAILABLE:
            # Return mock data for the requested module
            mock_module = next((m for m in MOCK_MODULES if m["name"] == module_name), None)
            if mock_module:
                return mock_module
            raise HTTPException(status_code=404, detail=f"Module not found: {module_name}")
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if client is properly initialized
            if client is None:
                raise HTTPException(status_code=500, detail="DSPy client initialization failed")
            
            # Check if module exists using client's get_module method
            module = client.get_module(module_name)
            if not module:
                raise HTTPException(status_code=404, detail=f"Module not found: {module_name}")
            
            # Get module info from client's modules dictionary
            modules = client.list_modules()
            module_info = next((m for m in modules if m["name"] == module_name), None)
            
            if not module_info:
                # If not in the list, create module info from the module itself
                module_info = {
                    "name": module_name,
                    "description": getattr(module, "description", f"{module.__class__.__name__} module"),
                    "type": module.__class__.__name__,
                    "registered_at": datetime.now().isoformat()
                }
                
            return module_info
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Error getting module {module_name}: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error getting module: {str(e)}")
    
    async def register_module(
        self,
        module_name: str,
        module_type: str,
        parameters: Dict[str, Any],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a new DSPy module.
        
        Args:
            module_name: Name for the module
            module_type: Type of module to create
            parameters: Module parameters
            description: Module description
            
        Returns:
            Dict[str, Any]: Registered module information
            
        Raises:
            HTTPException: If registration fails
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(status_code=500, detail="DSPy is not available")
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if client is properly initialized
            if client is None:
                raise HTTPException(status_code=500, detail="DSPy client initialization failed")
            
            # Import the module class based on module_type
            try:
                if '.' not in module_type:
                    # Try different module paths to find the class
                    module_paths = [
                        f"medical.ml.dspy.modules.{module_type}",
                        f"medical.ml.dspy.{module_type}",
                        f"dspy.{module_type}"
                    ]
                    
                    for path in module_paths:
                        try:
                            module_module = importlib.import_module(path)
                            module_class = getattr(module_module, module_type)
                            break
                        except (ImportError, AttributeError):
                            continue
                    else:
                        raise ImportError(f"Could not find module class: {module_type}")
                else:
                    # Module type includes full path
                    module_path, class_name = module_type.rsplit('.', 1)
                    module_module = importlib.import_module(module_path)
                    module_class = getattr(module_module, class_name)
                
                # Create an instance of the module class with the provided parameters
                module_instance = module_class(**parameters)
                
                # Register the module with the client
                await client.register_module(
                    name=module_name,
                    module=module_instance,
                    description=description or f"{module_type} module"
                )
                
                # Return module info
                return {
                    "name": module_name,
                    "description": description or f"{module_type} module",
                    "type": module_type,
                    "parameters": parameters,
                    "registered_at": datetime.now().isoformat()
                }
                
            except ImportError as e:
                logger.error(f"Failed to import module class: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid module type: {module_type}")
                
            except Exception as e:
                logger.error(f"Failed to create module: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Failed to create module: {str(e)}")
                
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Error registering module: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error registering module: {str(e)}")
    
    async def unregister_module(self, module_name: str) -> Dict[str, Any]:
        """
        Unregister a DSPy module.
        
        Args:
            module_name: Name of the module to unregister
            
        Returns:
            Dict[str, Any]: Response information
            
        Raises:
            HTTPException: If unregistration fails
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(status_code=500, detail="DSPy is not available")
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if client is properly initialized
            if client is None:
                raise HTTPException(status_code=500, detail="DSPy client initialization failed")
            
            # Check if module exists
            if not client.get_module(module_name):
                raise HTTPException(status_code=404, detail=f"Module not found: {module_name}")
            
            # Remove the module from client's modules dictionary
            if hasattr(client, 'modules') and module_name in client.modules:
                del client.modules[module_name]
                
                return {
                    "name": module_name,
                    "status": "unregistered",
                    "unregistered_at": datetime.now().isoformat()
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to unregister module")
                
        except HTTPException:
            raise  # Re-raise HTTP exceptions
        except Exception as e:
            logger.error(f"Error unregistering module: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error unregistering module: {str(e)}")
    
    async def execute_module(
        self,
        module_name: str,
        inputs: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a DSPy module.
        
        Args:
            module_name: Name of the module to execute
            inputs: Module inputs
            config: Optional execution configuration
            
        Returns:
            Dict[str, Any]: Module execution results
            
        Raises:
            HTTPException: If execution fails
        """
        if not DSPY_AVAILABLE:
            raise HTTPException(status_code=500, detail="DSPy is not available")
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if client is properly initialized
            if client is None:
                raise HTTPException(status_code=500, detail="DSPy client initialization failed")
            
            # Execute the module
            result = await client.call_module(module_name, **inputs)
            
            # Format the result
            if hasattr(result, "to_dict"):
                # Use to_dict method if available
                formatted_result = result.to_dict()
            elif hasattr(result, "__dict__"):
                # Use __dict__ if to_dict is not available
                formatted_result = {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
            else:
                # Try to convert to dict directly
                formatted_result = dict(result)
            
            return {
                "module_name": module_name,
                "inputs": inputs,
                "config": config,
                "result": formatted_result,
                "execution_time": datetime.now().isoformat()
            }
            
        except ValueError as e:
            # Module not found or invalid inputs
            logger.error(f"Module execution error: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
            
        except Exception as e:
            logger.error(f"Error executing module: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error executing module: {str(e)}")
    
    async def optimize_module(
        self,
        module_name: str,
        metric: str,
        num_trials: int = 10,
        examples: List[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize a DSPy module.
        
        Args:
            module_name: Name of the module to optimize
            metric: Optimization metric
            num_trials: Number of optimization trials
            examples: Examples for optimization
            config: Optional optimization configuration
            
        Returns:
            Dict[str, Any]: Optimization results
            
        Raises:
            HTTPException: If optimization fails
        """
        # This is a placeholder for module optimization
        # In a real implementation, this would use DSPy's optimization capabilities
        
        if not DSPY_AVAILABLE:
            raise HTTPException(status_code=500, detail="DSPy is not available")
        
        try:
            # Get DSPy client
            client = await self.get_client()
            
            # Check if client is properly initialized
            if client is None:
                raise HTTPException(status_code=500, detail="DSPy client initialization failed")
            
            # Check if module exists
            module = client.get_module(module_name)
            if not module:
                raise HTTPException(status_code=404, detail=f"Module not found: {module_name}")
            
            # This is a mock optimization result
            # In a real implementation, you would use DSPy's optimization functionality
            return {
                "module_name": module_name,
                "metric": metric,
                "num_trials": num_trials,
                "examples_count": len(examples) if examples else 0,
                "config": config,
                "optimization_result": {
                    "best_score": 0.85,
                    "trials_completed": num_trials,
                    "best_configuration": {"temperature": 0.2, "top_p": 0.95}
                },
                "optimization_time": datetime.now().isoformat()
            }
            
        except HTTPException:
            raise  # Re-raise HTTP exceptions
            
        except Exception as e:
            logger.error(f"Error optimizing module: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error optimizing module: {str(e)}")
    
    async def get_config(self) -> Dict[str, Any]:
        """
        Get the current DSPy configuration.
        
        Returns:
            Dict[str, Any]: DSPy configuration
            
        Raises:
            HTTPException: If getting config fails
        """
        if self._config:
            return self._config
        
        try:
            # Load config from file
            with open(CONFIG_PATH, 'r') as f:
                self._config = yaml.safe_load(f)
            return self._config
        except Exception as e:
            logger.error(f"Error getting DSPy config: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error getting DSPy config: {str(e)}")
    
    async def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the DSPy configuration.
        
        Args:
            config: New configuration
            
        Returns:
            Dict[str, Any]: Updated configuration
            
        Raises:
            HTTPException: If updating config fails
        """
        try:
            # Update config in memory
            if self._config:
                self._config.update(config)
            else:
                self._config = config
            
            # Save config to file
            with open(CONFIG_PATH, 'w') as f:
                yaml.dump(self._config, f)
            
            return self._config
        except Exception as e:
            logger.error(f"Error updating DSPy config: {str(e)}")
            logger.error(f"Error traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error updating DSPy config: {str(e)}")

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
