from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
import logging
# Use absolute imports
from asf.bollm.backend.repositories.provider_repository import ProviderRepository
from asf.bollm.backend.repositories.audit_repository import AuditRepository
from asf.bollm.backend.models.provider import Provider, ApiKey, ConnectionParameter
from asf.bollm.backend.models.llm_model import LLMModel
from asf.bollm.backend.utils.crypto import encrypt_value, decrypt_value

logger = logging.getLogger(__name__)

class ProviderService:
    def __init__(self, db: Session, encryption_key: bytes = None, current_user_id: Optional[int] = None):
        self.db = db
        self.encryption_key = encryption_key
        self.current_user_id = current_user_id
        self.provider_repo = ProviderRepository(db, encryption_key)
        self.audit_repo = AuditRepository(db)

    # Provider methods

    def get_all_providers(self) -> List[Dict[str, Any]]:
        """Get all providers with their models and connection parameters."""
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

            # Get models
            models = self.provider_repo.get_models_by_provider_id(provider.provider_id)
            for model in models:
                provider_dict["models"].append({
                    "model_id": model.model_id,
                    "display_name": model.display_name,
                    "model_type": model.model_type,
                    "context_window": model.context_window,
                    "max_tokens": model.max_tokens,
                    "enabled": model.enabled
                })

            # Get connection parameters (excluding sensitive ones)
            params = self.provider_repo.get_connection_parameters_by_provider_id(provider.provider_id)
            for param in params:
                if not param.is_sensitive:
                    provider_dict["connection_parameters"].append({
                        "param_name": param.param_name,
                        "param_value": param.param_value,
                        "is_sensitive": param.is_sensitive
                    })

            result.append(provider_dict)

        return result

    def get_provider_by_id(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """Get a provider by ID with its models and connection parameters."""
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

        # Get models
        models = self.provider_repo.get_models_by_provider_id(provider.provider_id)
        for model in models:
            result["models"].append({
                "model_id": model.model_id,
                "display_name": model.display_name,
                "model_type": model.model_type,
                "context_window": model.context_window,
                "max_tokens": model.max_tokens,
                "enabled": model.enabled
            })

        # Get connection parameters (excluding sensitive ones)
        params = self.provider_repo.get_connection_parameters_by_provider_id(provider.provider_id)
        for param in params:
            if not param.is_sensitive:
                result["connection_parameters"].append({
                    "param_name": param.param_name,
                    "param_value": param.param_value,
                    "is_sensitive": param.is_sensitive
                })

        return result

    def create_provider(self, provider_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new provider with its models and connection parameters."""
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
            "table_name": "providers",
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
        """Update a provider."""
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
            "table_name": "providers",
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
        """Delete a provider."""
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
            "table_name": "providers",
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

    # API Key methods

    def get_api_keys_by_provider_id(self, provider_id: str) -> List[Dict[str, Any]]:
        """Get all API keys for a provider (without the actual key values)."""
        api_keys = self.provider_repo.get_api_keys_by_provider_id(provider_id)
        result = []

        for api_key in api_keys:
            result.append({
                "key_id": api_key.key_id,
                "provider_id": api_key.provider_id,
                "is_encrypted": api_key.is_encrypted,
                "environment": api_key.environment,
                "created_at": api_key.created_at,
                "updated_at": api_key.updated_at,
                "expires_at": api_key.expires_at
            })

        return result

    def create_api_key(self, api_key_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new API key."""
        # Add current user ID if available
        if self.current_user_id and "created_by_user_id" not in api_key_data:
            api_key_data["created_by_user_id"] = self.current_user_id

        # Create API key
        api_key = self.provider_repo.create_api_key(api_key_data)

        # Log audit (without the actual key value)
        self.audit_repo.create_audit_log({
            "table_name": "api_keys",
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
            "updated_at": api_key.updated_at,
            "expires_at": api_key.expires_at
        }

    def get_api_key_value(self, key_id: int) -> Optional[str]:
        """Get the actual API key value (decrypted if necessary)."""
        return self.provider_repo.get_decrypted_api_key(key_id)

    # Connection Parameter methods

    def get_provider_connection_params(self, provider_id: str, include_sensitive: bool = False) -> Dict[str, Any]:
        """Get all connection parameters for a provider as a dictionary."""
        params = self.provider_repo.get_connection_parameters_by_provider_id(provider_id)
        result = {}

        for param in params:
            if include_sensitive or not param.is_sensitive:
                result[param.param_name] = param.param_value

        return result

    def set_connection_parameter(self, param_data: Dict[str, Any]) -> Dict[str, Any]:
        """Set a connection parameter."""
        # Set parameter
        param = self.provider_repo.set_connection_parameter(param_data)

        # Log audit
        self.audit_repo.create_audit_log({
            "table_name": "connection_parameters",
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
