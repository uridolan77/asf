"""
API client for LLM Gateway.

This module provides a client for interacting with the LLM Gateway API.
"""

import requests
import logging
from typing import Dict, Any, List, Optional, Union
import json

logger = logging.getLogger(__name__)

class LLMGatewayClient:
    """
    Client for interacting with the LLM Gateway API.
    
    This class provides methods for interacting with the LLM Gateway API,
    including provider management, model management, and LLM interaction.
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL for the LLM Gateway API
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
    
    # Provider methods
    
    def get_providers(self) -> List[Dict[str, Any]]:
        """
        Get all providers.
        
        Returns:
            List of provider dictionaries
        """
        response = requests.get(f"{self.base_url}/providers", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Get a provider by ID.
        
        Args:
            provider_id: ID of the provider to get
            
        Returns:
            Provider dictionary
        """
        response = requests.get(f"{self.base_url}/providers/{provider_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def create_provider(self, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new provider.
        
        Args:
            provider_data: Dictionary containing provider data
            
        Returns:
            Created provider dictionary
        """
        response = requests.post(
            f"{self.base_url}/providers",
            headers=self.headers,
            json=provider_data
        )
        response.raise_for_status()
        return response.json()
    
    def update_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a provider.
        
        Args:
            provider_id: ID of the provider to update
            provider_data: Dictionary containing updated provider data
            
        Returns:
            Updated provider dictionary
        """
        response = requests.put(
            f"{self.base_url}/providers/{provider_id}",
            headers=self.headers,
            json=provider_data
        )
        response.raise_for_status()
        return response.json()
    
    def delete_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Delete a provider.
        
        Args:
            provider_id: ID of the provider to delete
            
        Returns:
            Response dictionary
        """
        response = requests.delete(f"{self.base_url}/providers/{provider_id}", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def test_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Test a provider connection.
        
        Args:
            provider_id: ID of the provider to test
            
        Returns:
            Test results dictionary
        """
        response = requests.post(f"{self.base_url}/providers/{provider_id}/test", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def sync_provider(self, provider_id: str) -> Dict[str, Any]:
        """
        Synchronize a provider with the LLM Gateway configuration.
        
        Args:
            provider_id: ID of the provider to synchronize
            
        Returns:
            Response dictionary
        """
        response = requests.post(f"{self.base_url}/providers/{provider_id}/sync", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    # Model methods
    
    def get_models(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get all models for a provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of model dictionaries
        """
        response = requests.get(f"{self.base_url}/providers/{provider_id}/models", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def get_model(self, provider_id: str, model_id: str) -> Dict[str, Any]:
        """
        Get a model by ID.
        
        Args:
            provider_id: ID of the provider
            model_id: ID of the model to get
            
        Returns:
            Model dictionary
        """
        response = requests.get(
            f"{self.base_url}/providers/{provider_id}/models/{model_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def create_model(self, provider_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new model.
        
        Args:
            provider_id: ID of the provider
            model_data: Dictionary containing model data
            
        Returns:
            Created model dictionary
        """
        # Ensure provider_id in model_data matches provider_id in path
        model_data["provider_id"] = provider_id
        
        response = requests.post(
            f"{self.base_url}/providers/{provider_id}/models",
            headers=self.headers,
            json=model_data
        )
        response.raise_for_status()
        return response.json()
    
    def update_model(self, provider_id: str, model_id: str, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update a model.
        
        Args:
            provider_id: ID of the provider
            model_id: ID of the model to update
            model_data: Dictionary containing updated model data
            
        Returns:
            Updated model dictionary
        """
        response = requests.put(
            f"{self.base_url}/providers/{provider_id}/models/{model_id}",
            headers=self.headers,
            json=model_data
        )
        response.raise_for_status()
        return response.json()
    
    def delete_model(self, provider_id: str, model_id: str) -> Dict[str, Any]:
        """
        Delete a model.
        
        Args:
            provider_id: ID of the provider
            model_id: ID of the model to delete
            
        Returns:
            Response dictionary
        """
        response = requests.delete(
            f"{self.base_url}/providers/{provider_id}/models/{model_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    # API Key methods
    
    def get_api_keys(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get all API keys for a provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of API key dictionaries
        """
        response = requests.get(
            f"{self.base_url}/providers/{provider_id}/api-keys",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def create_api_key(self, provider_id: str, api_key_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new API key.
        
        Args:
            provider_id: ID of the provider
            api_key_data: Dictionary containing API key data
            
        Returns:
            Created API key dictionary
        """
        # Ensure provider_id in api_key_data matches provider_id in path
        api_key_data["provider_id"] = provider_id
        
        response = requests.post(
            f"{self.base_url}/providers/{provider_id}/api-keys",
            headers=self.headers,
            json=api_key_data
        )
        response.raise_for_status()
        return response.json()
    
    def get_api_key_value(self, provider_id: str, key_id: int) -> Dict[str, Any]:
        """
        Get the actual API key value.
        
        Args:
            provider_id: ID of the provider
            key_id: ID of the API key
            
        Returns:
            API key value dictionary
        """
        response = requests.get(
            f"{self.base_url}/providers/{provider_id}/api-keys/{key_id}/value",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    # Connection Parameter methods
    
    def get_connection_params(self, provider_id: str, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Get all connection parameters for a provider.
        
        Args:
            provider_id: ID of the provider
            include_sensitive: Whether to include sensitive parameters
            
        Returns:
            Dictionary of connection parameters
        """
        response = requests.get(
            f"{self.base_url}/providers/{provider_id}/connection-params",
            headers=self.headers,
            params={"include_sensitive": str(include_sensitive).lower()}
        )
        response.raise_for_status()
        return response.json()
    
    def set_connection_param(self, provider_id: str, param_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set a connection parameter.
        
        Args:
            provider_id: ID of the provider
            param_data: Dictionary containing parameter data
            
        Returns:
            Created or updated parameter dictionary
        """
        # Ensure provider_id in param_data matches provider_id in path
        param_data["provider_id"] = provider_id
        
        response = requests.post(
            f"{self.base_url}/providers/{provider_id}/connection-params",
            headers=self.headers,
            json=param_data
        )
        response.raise_for_status()
        return response.json()
    
    # LLM Interaction methods
    
    def generate(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a response from an LLM.
        
        Args:
            request_data: Dictionary containing request data
            
        Returns:
            Response dictionary
        """
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json=request_data
        )
        response.raise_for_status()
        return response.json()
