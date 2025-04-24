"""
Provider Service for LLM Gateway.

This module provides a service for managing LLM providers, including
CRUD operations for providers, models, API keys, and connection parameters.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
import json
from datetime import datetime

from asf.medical.llm_gateway.repositories.provider_repository import ProviderRepository
from asf.medical.llm_gateway.repositories.audit_repository import AuditRepository
from asf.medical.llm_gateway.models.provider import Provider, ProviderModel, ApiKey, ConnectionParameter

logger = logging.getLogger(__name__)

class ProviderService:
    """
    Service for managing LLM providers.
    
    This class provides methods for managing providers, models, API keys,
    and connection parameters, with audit logging.
    """
    
    def __init__(self, db: Session, encryption_key: bytes = None, current_user_id: Optional[int] = None):
        """
        Initialize the service.
        
        Args:
            db: SQLAlchemy database session
            encryption_key: Optional encryption key for sensitive data
            current_user_id: Optional ID of the current user for audit logging
        """
        self.db = db
        self.encryption_key = encryption_key
        self.current_user_id = current_user_id
        self.provider_repo = ProviderRepository(db, encryption_key)
        self.audit_repo = AuditRepository(db)
    
    # Provider methods
    
    def get_all_providers(self) -> List[Dict[str, Any]]:
        """
        Get all providers with their models and connection parameters.
        
        Returns:
            List of provider dictionaries
        """
        providers = self.provider_repo.get_all_providers()
        result = []
        
        for provider in providers:
            provider_dict = {
                "provider_id": provider.provider_id,
                "display_name": provider.display_name,
                "provider_type": provider.provider_type,
                "description": provider.description,
                "enabled": provider.enabled,
                "created_at": provider.created_at,
                "updated_at": provider.updated_at,
                "models": [],
                "connection_parameters": []
            }
            
            # Parse JSON fields
            if provider.connection_params:
                try:
                    provider_dict["connection_params"] = json.loads(provider.connection_params)
                except json.JSONDecodeError:
                    provider_dict["connection_params"] = {}
            
            if provider.request_settings:
                try:
                    provider_dict["request_settings"] = json.loads(provider.request_settings)
                except json.JSONDecodeError:
                    provider_dict["request_settings"] = {}
            
            # Get models
            models = self.provider_repo.get_models_by_provider_id(provider.provider_id)
            for model in models:
                model_dict = {
                    "model_id": model.model_id,
                    "display_name": model.display_name,
                    "model_type": model.model_type,
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                    "enabled": model.enabled
                }
                
                # Parse JSON fields
                if model.capabilities:
                    try:
                        model_dict["capabilities"] = json.loads(model.capabilities)
                    except json.JSONDecodeError:
                        model_dict["capabilities"] = []
                
                if model.parameters:
                    try:
                        model_dict["parameters"] = json.loads(model.parameters)
                    except json.JSONDecodeError:
                        model_dict["parameters"] = {}
                
                provider_dict["models"].append(model_dict)
            
            # Get connection parameters (excluding sensitive ones)
            params = self.provider_repo.get_connection_parameters_by_provider_id(provider.provider_id)
            for param in params:
                if not param.is_sensitive:
                    provider_dict["connection_parameters"].append({
                        "param_name": param.param_name,
                        "param_value": param.param_value,
                        "is_sensitive": param.is_sensitive,
                        "environment": param.environment
                    })
            
            result.append(provider_dict)
        
        return result
    
    def get_provider_by_id(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a provider by ID with its models and connection parameters.
        
        Args:
            provider_id: ID of the provider to get
            
        Returns:
            Provider dictionary or None if not found
        """
        provider = self.provider_repo.get_provider_by_id(provider_id)
        if not provider:
            return None
        
        result = {
            "provider_id": provider.provider_id,
            "display_name": provider.display_name,
            "provider_type": provider.provider_type,
            "description": provider.description,
            "enabled": provider.enabled,
            "created_at": provider.created_at,
            "updated_at": provider.updated_at,
            "models": [],
            "connection_parameters": []
        }
        
        # Parse JSON fields
        if provider.connection_params:
            try:
                result["connection_params"] = json.loads(provider.connection_params)
            except json.JSONDecodeError:
                result["connection_params"] = {}
        
        if provider.request_settings:
            try:
                result["request_settings"] = json.loads(provider.request_settings)
            except json.JSONDecodeError:
                result["request_settings"] = {}
        
        # Get models
        models = self.provider_repo.get_models_by_provider_id(provider.provider_id)
        for model in models:
            model_dict = {
                "model_id": model.model_id,
                "display_name": model.display_name,
                "model_type": model.model_type,
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "enabled": model.enabled
            }
            
            # Parse JSON fields
            if model.capabilities:
                try:
                    model_dict["capabilities"] = json.loads(model.capabilities)
                except json.JSONDecodeError:
                    model_dict["capabilities"] = []
            
            if model.parameters:
                try:
                    model_dict["parameters"] = json.loads(model.parameters)
                except json.JSONDecodeError:
                    model_dict["parameters"] = {}
            
            result["models"].append(model_dict)
        
        # Get connection parameters (excluding sensitive ones)
        params = self.provider_repo.get_connection_parameters_by_provider_id(provider.provider_id)
        for param in params:
            if not param.is_sensitive:
                result["connection_parameters"].append({
                    "param_name": param.param_name,
                    "param_value": param.param_value,
                    "is_sensitive": param.is_sensitive,
                    "environment": param.environment
                })
        
        return result
    
    def create_provider(self, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new provider with its models and connection parameters.
        
        Args:
            provider_data: Dictionary containing provider data
            
        Returns:
            Created provider dictionary
        """
        # Add current user ID if available
        if self.current_user_id and "created_by_user_id" not in provider_data:
            provider_data["created_by_user_id"] = self.current_user_id
        
        # Create provider
        provider = self.provider_repo.create_provider(provider_data)
        
        # Create models if provided
        models_data = provider_data.get("models", [])
        for model_data in models_data:
            model_data["provider_id"] = provider.provider_id
            self.provider_repo.create_model(model_data)
        
        # Create connection parameters if provided
        params_data = provider_data.get("connection_parameters", [])
        for param_data in params_data:
            param_data["provider_id"] = provider.provider_id
            self.provider_repo.set_connection_parameter(param_data)
        
        # Create API key if provided
        api_key_data = provider_data.get("api_key")
        if api_key_data:
            api_key_data["provider_id"] = provider.provider_id
            api_key_data["created_by_user_id"] = provider_data.get("created_by_user_id")
            self.provider_repo.create_api_key(api_key_data)
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_providers",
            "record_id": provider.provider_id,
            "action": "create",
            "changed_by_user_id": self.current_user_id,
            "new_values": {
                "provider_id": provider.provider_id,
                "display_name": provider.display_name,
                "provider_type": provider.provider_type,
                "enabled": provider.enabled
            }
        })
        
        # Return the created provider
        return self.get_provider_by_id(provider.provider_id)
    
    def update_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a provider.
        
        Args:
            provider_id: ID of the provider to update
            provider_data: Dictionary containing updated provider data
            
        Returns:
            Updated provider dictionary or None if not found
        """
        # Get the provider before update for audit
        old_provider = self.provider_repo.get_provider_by_id(provider_id)
        if not old_provider:
            return None
        
        # Update provider
        provider = self.provider_repo.update_provider(provider_id, provider_data)
        if not provider:
            return None
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_providers",
            "record_id": provider.provider_id,
            "action": "update",
            "changed_by_user_id": self.current_user_id,
            "old_values": {
                "display_name": old_provider.display_name,
                "provider_type": old_provider.provider_type,
                "description": old_provider.description,
                "enabled": old_provider.enabled
            },
            "new_values": {
                "display_name": provider.display_name,
                "provider_type": provider.provider_type,
                "description": provider.description,
                "enabled": provider.enabled
            }
        })
        
        # Return the updated provider
        return self.get_provider_by_id(provider.provider_id)
    
    def delete_provider(self, provider_id: str) -> bool:
        """
        Delete a provider.
        
        Args:
            provider_id: ID of the provider to delete
            
        Returns:
            True if deleted, False if not found
        """
        # Get the provider before delete for audit
        old_provider = self.provider_repo.get_provider_by_id(provider_id)
        if not old_provider:
            return False
        
        # Delete provider
        result = self.provider_repo.delete_provider(provider_id)
        if not result:
            return False
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_providers",
            "record_id": provider_id,
            "action": "delete",
            "changed_by_user_id": self.current_user_id,
            "old_values": {
                "provider_id": old_provider.provider_id,
                "display_name": old_provider.display_name,
                "provider_type": old_provider.provider_type
            }
        })
        
        return True
    
    # Model methods
    
    def get_models_by_provider_id(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get all models for a provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of model dictionaries
        """
        models = self.provider_repo.get_models_by_provider_id(provider_id)
        result = []
        
        for model in models:
            model_dict = {
                "model_id": model.model_id,
                "provider_id": model.provider_id,
                "display_name": model.display_name,
                "model_type": model.model_type,
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "enabled": model.enabled
            }
            
            # Parse JSON fields
            if model.capabilities:
                try:
                    model_dict["capabilities"] = json.loads(model.capabilities)
                except json.JSONDecodeError:
                    model_dict["capabilities"] = []
            
            if model.parameters:
                try:
                    model_dict["parameters"] = json.loads(model.parameters)
                except json.JSONDecodeError:
                    model_dict["parameters"] = {}
            
            result.append(model_dict)
        
        return result
    
    def get_model_by_id(self, model_id: str, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by ID.
        
        Args:
            model_id: ID of the model to get
            provider_id: ID of the provider
            
        Returns:
            Model dictionary or None if not found
        """
        model = self.provider_repo.get_model_by_id(model_id, provider_id)
        if not model:
            return None
        
        result = {
            "model_id": model.model_id,
            "provider_id": model.provider_id,
            "display_name": model.display_name,
            "model_type": model.model_type,
            "context_window": model.context_window,
            "max_tokens": model.max_tokens,
            "enabled": model.enabled
        }
        
        # Parse JSON fields
        if model.capabilities:
            try:
                result["capabilities"] = json.loads(model.capabilities)
            except json.JSONDecodeError:
                result["capabilities"] = []
        
        if model.parameters:
            try:
                result["parameters"] = json.loads(model.parameters)
            except json.JSONDecodeError:
                result["parameters"] = {}
        
        return result
    
    def create_model(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new model.
        
        Args:
            model_data: Dictionary containing model data
            
        Returns:
            Created model dictionary
        """
        # Create model
        model = self.provider_repo.create_model(model_data)
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_models",
            "record_id": f"{model.provider_id}:{model.model_id}",
            "action": "create",
            "changed_by_user_id": self.current_user_id,
            "new_values": {
                "model_id": model.model_id,
                "provider_id": model.provider_id,
                "display_name": model.display_name,
                "model_type": model.model_type,
                "enabled": model.enabled
            }
        })
        
        # Return the created model
        return self.get_model_by_id(model.model_id, model.provider_id)
    
    def update_model(self, model_id: str, provider_id: str, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a model.
        
        Args:
            model_id: ID of the model to update
            provider_id: ID of the provider
            model_data: Dictionary containing updated model data
            
        Returns:
            Updated model dictionary or None if not found
        """
        # Get the model before update for audit
        old_model = self.provider_repo.get_model_by_id(model_id, provider_id)
        if not old_model:
            return None
        
        # Update model
        model = self.provider_repo.update_model(model_id, provider_id, model_data)
        if not model:
            return None
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_models",
            "record_id": f"{model.provider_id}:{model.model_id}",
            "action": "update",
            "changed_by_user_id": self.current_user_id,
            "old_values": {
                "display_name": old_model.display_name,
                "model_type": old_model.model_type,
                "context_window": old_model.context_window,
                "max_tokens": old_model.max_tokens,
                "enabled": old_model.enabled
            },
            "new_values": {
                "display_name": model.display_name,
                "model_type": model.model_type,
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "enabled": model.enabled
            }
        })
        
        # Return the updated model
        return self.get_model_by_id(model.model_id, model.provider_id)
    
    def delete_model(self, model_id: str, provider_id: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: ID of the model to delete
            provider_id: ID of the provider
            
        Returns:
            True if deleted, False if not found
        """
        # Get the model before delete for audit
        old_model = self.provider_repo.get_model_by_id(model_id, provider_id)
        if not old_model:
            return False
        
        # Delete model
        result = self.provider_repo.delete_model(model_id, provider_id)
        if not result:
            return False
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_models",
            "record_id": f"{provider_id}:{model_id}",
            "action": "delete",
            "changed_by_user_id": self.current_user_id,
            "old_values": {
                "model_id": old_model.model_id,
                "provider_id": old_model.provider_id,
                "display_name": old_model.display_name,
                "model_type": old_model.model_type
            }
        })
        
        return True
    
    # API Key methods
    
    def get_api_keys_by_provider_id(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get all API keys for a provider (without the actual key values).
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of API key dictionaries
        """
        api_keys = self.provider_repo.get_api_keys_by_provider_id(provider_id)
        result = []
        
        for api_key in api_keys:
            result.append({
                "key_id": api_key.key_id,
                "provider_id": api_key.provider_id,
                "is_encrypted": api_key.is_encrypted,
                "environment": api_key.environment,
                "created_at": api_key.created_at,
                "expires_at": api_key.expires_at
            })
        
        return result
    
    def create_api_key(self, api_key_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new API key.
        
        Args:
            api_key_data: Dictionary containing API key data
            
        Returns:
            Created API key dictionary
        """
        # Add current user ID if available
        if self.current_user_id and "created_by_user_id" not in api_key_data:
            api_key_data["created_by_user_id"] = self.current_user_id
        
        # Create API key
        api_key = self.provider_repo.create_api_key(api_key_data)
        
        # Log audit (without the actual key value)
        self.audit_repo.create_audit_log({
            "table_name": "llm_provider_api_keys",
            "record_id": str(api_key.key_id),
            "action": "create",
            "changed_by_user_id": self.current_user_id,
            "new_values": {
                "key_id": api_key.key_id,
                "provider_id": api_key.provider_id,
                "is_encrypted": api_key.is_encrypted,
                "environment": api_key.environment,
                "expires_at": api_key.expires_at
            }
        })
        
        # Return the created API key (without the actual key value)
        return {
            "key_id": api_key.key_id,
            "provider_id": api_key.provider_id,
            "is_encrypted": api_key.is_encrypted,
            "environment": api_key.environment,
            "created_at": api_key.created_at,
            "expires_at": api_key.expires_at
        }
    
    def get_api_key_value(self, key_id: int) -> Optional[str]:
        """
        Get the actual API key value (decrypted if necessary).
        
        Args:
            key_id: ID of the API key to get
            
        Returns:
            Decrypted API key value or None if not found
        """
        return self.provider_repo.get_decrypted_api_key(key_id)
    
    # Connection Parameter methods
    
    def get_provider_connection_params(self, provider_id: str, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Get all connection parameters for a provider as a dictionary.
        
        Args:
            provider_id: ID of the provider
            include_sensitive: Whether to include sensitive parameters
            
        Returns:
            Dictionary of connection parameters
        """
        params = self.provider_repo.get_connection_parameters_by_provider_id(provider_id)
        result = {}
        
        for param in params:
            if include_sensitive or not param.is_sensitive:
                result[param.param_name] = param.param_value
        
        return result
    
    def set_connection_parameter(self, param_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set a connection parameter.
        
        Args:
            param_data: Dictionary containing parameter data
            
        Returns:
            Created or updated parameter dictionary
        """
        # Set parameter
        param = self.provider_repo.set_connection_parameter(param_data)
        
        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "llm_provider_connection_parameters",
            "record_id": str(param.param_id),
            "action": "set",
            "changed_by_user_id": self.current_user_id,
            "new_values": {
                "provider_id": param.provider_id,
                "param_name": param.param_name,
                "is_sensitive": param.is_sensitive,
                "environment": param.environment
            }
        })
        
        # Return the parameter (without value if sensitive)
        return {
            "param_id": param.param_id,
            "provider_id": param.provider_id,
            "param_name": param.param_name,
            "param_value": None if param.is_sensitive else param.param_value,
            "is_sensitive": param.is_sensitive,
            "environment": param.environment
        }
    
    # Integration with LLM Gateway
    
    def sync_provider_with_gateway_config(self, provider_id: str) -> bool:
        """
        Synchronize a provider with the LLM Gateway configuration.
        
        Args:
            provider_id: ID of the provider to synchronize
            
        Returns:
            True if synchronized, False if not found
        """
        provider = self.get_provider_by_id(provider_id)
        if not provider:
            return False
        
        try:
            from asf.medical.llm_gateway.core.config_loader import ConfigLoader
            
            # Get the current gateway configuration
            config_loader = ConfigLoader(self.db)
            gateway_config = config_loader.load_config()
            
            # Update the provider configuration
            if "additional_config" not in gateway_config:
                gateway_config["additional_config"] = {}
            
            if "providers" not in gateway_config["additional_config"]:
                gateway_config["additional_config"]["providers"] = {}
            
            # Create provider config
            provider_config = {
                "provider_type": provider["provider_type"],
                "display_name": provider["display_name"],
                "models": {}
            }
            
            # Add connection parameters
            if "connection_params" in provider:
                provider_config.update(provider["connection_params"])
            
            # Add models
            for model in provider["models"]:
                if model["enabled"]:
                    model_config = {
                        "display_name": model["display_name"],
                        "context_window": model["context_window"],
                        "max_tokens": model["max_tokens"]
                    }
                    
                    # Add model parameters
                    if "parameters" in model:
                        model_config.update(model["parameters"])
                    
                    provider_config["models"][model["model_id"]] = model_config
            
            # Update the gateway configuration
            gateway_config["additional_config"]["providers"][provider_id] = provider_config
            
            # Add to allowed providers if not already there
            if "allowed_providers" not in gateway_config:
                gateway_config["allowed_providers"] = []
            
            if provider_id not in gateway_config["allowed_providers"] and provider["enabled"]:
                gateway_config["allowed_providers"].append(provider_id)
            elif provider_id in gateway_config["allowed_providers"] and not provider["enabled"]:
                gateway_config["allowed_providers"].remove(provider_id)
            
            # Save the updated configuration
            config_loader.save_config(gateway_config)
            
            return True
        except Exception as e:
            logger.error(f"Error synchronizing provider with gateway config: {e}")
            return False
    
    async def test_provider_connection(self, provider_id: str) -> Dict[str, Any]:
        """
        Test the connection to a provider.
        
        Args:
            provider_id: ID of the provider to test
            
        Returns:
            Dictionary containing test results
        """
        provider = self.get_provider_by_id(provider_id)
        if not provider:
            return {
                "success": False,
                "message": f"Provider '{provider_id}' not found"
            }
        
        try:
            # Synchronize with gateway config
            sync_result = self.sync_provider_with_gateway_config(provider_id)
            if not sync_result:
                return {
                    "success": False,
                    "message": f"Failed to synchronize provider '{provider_id}' with gateway config"
                }
            
            # Get a model to test
            if not provider["models"]:
                return {
                    "success": False,
                    "message": f"Provider '{provider_id}' has no models configured"
                }
            
            test_model = provider["models"][0]["model_id"]
            
            # Create a test request
            from asf.medical.llm_gateway.core.models import LLMRequest, LLMConfig, InterventionContext
            from asf.medical.llm_gateway.core.client import LLMGatewayClient
            from asf.medical.llm_gateway.core.factory import ProviderFactory
            
            # Create a gateway client
            provider_factory = ProviderFactory()
            from asf.medical.llm_gateway.core.config_loader import ConfigLoader
            config_loader = ConfigLoader(self.db)
            gateway_config = config_loader.load_config_as_object()
            
            client = LLMGatewayClient(gateway_config, provider_factory, self.db)
            
            # Create a test request
            llm_config = LLMConfig(model_identifier=test_model)
            context = InterventionContext(session_id=f"test-{datetime.utcnow().timestamp()}")
            
            llm_req = LLMRequest(
                prompt_content="Hello, this is a test request. Please respond with 'Test successful'.",
                config=llm_config,
                initial_context=context
            )
            
            # Send the test request
            start_time = datetime.utcnow()
            response = await client.generate(llm_req)
            end_time = datetime.utcnow()
            
            # Calculate duration
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            # Check for errors
            if response.error_details:
                return {
                    "success": False,
                    "message": f"Provider test failed: {response.error_details.message}",
                    "provider_id": provider_id,
                    "model_tested": test_model,
                    "duration_ms": duration_ms,
                    "error": response.error_details.message
                }
            
            # Return success
            return {
                "success": True,
                "message": "Provider connection test successful",
                "provider_id": provider_id,
                "model_tested": test_model,
                "response": response.generated_content,
                "duration_ms": duration_ms
            }
        except Exception as e:
            logger.error(f"Error testing provider connection: {e}")
            return {
                "success": False,
                "message": f"Error testing provider connection: {str(e)}",
                "provider_id": provider_id
            }
