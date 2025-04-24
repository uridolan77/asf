"""
Provider Repository for LLM Gateway.

This module provides a repository for LLM providers that directly connects to the database
and performs CRUD operations on the provider-related tables.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import json
from datetime import datetime

from asf.medical.llm_gateway.models.provider import Provider, ProviderModel, ApiKey, ConnectionParameter

logger = logging.getLogger(__name__)

class ProviderRepository:
    """
    Repository for LLM providers.
    
    This class provides methods for database operations on provider-related tables.
    """
    
    def __init__(self, db: Session, encryption_key: bytes = None):
        """
        Initialize the repository with a database session.
        
        Args:
            db: SQLAlchemy database session
            encryption_key: Optional encryption key for sensitive data
        """
        self.db = db
        self.encryption_key = encryption_key
    
    # Provider methods
    
    def get_all_providers(self) -> List[Provider]:
        """
        Get all providers.
        
        Returns:
            List of Provider objects
        """
        return self.db.query(Provider).all()
    
    def get_provider_by_id(self, provider_id: str) -> Optional[Provider]:
        """
        Get a provider by ID.
        
        Args:
            provider_id: ID of the provider to get
            
        Returns:
            Provider object or None if not found
        """
        return self.db.query(Provider).filter(Provider.provider_id == provider_id).first()
    
    def create_provider(self, provider_data: Dict[str, Any]) -> Provider:
        """
        Create a new provider.
        
        Args:
            provider_data: Dictionary containing provider data
            
        Returns:
            Created Provider object
            
        Raises:
            SQLAlchemyError: If there's an error creating the provider
        """
        try:
            # Convert JSON fields to strings if they're dictionaries
            if "connection_params" in provider_data and isinstance(provider_data["connection_params"], dict):
                provider_data["connection_params"] = json.dumps(provider_data["connection_params"])
            
            if "request_settings" in provider_data and isinstance(provider_data["request_settings"], dict):
                provider_data["request_settings"] = json.dumps(provider_data["request_settings"])
            
            provider = Provider(
                provider_id=provider_data["provider_id"],
                provider_type=provider_data["provider_type"],
                display_name=provider_data.get("display_name", provider_data["provider_id"]),
                description=provider_data.get("description"),
                enabled=provider_data.get("enabled", True),
                connection_params=provider_data.get("connection_params"),
                request_settings=provider_data.get("request_settings"),
                created_by_user_id=provider_data.get("created_by_user_id")
            )
            self.db.add(provider)
            self.db.commit()
            self.db.refresh(provider)
            return provider
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating provider: {e}")
            raise
    
    def update_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> Optional[Provider]:
        """
        Update a provider.
        
        Args:
            provider_id: ID of the provider to update
            provider_data: Dictionary containing updated provider data
            
        Returns:
            Updated Provider object or None if not found
            
        Raises:
            SQLAlchemyError: If there's an error updating the provider
        """
        try:
            provider = self.get_provider_by_id(provider_id)
            if not provider:
                return None
            
            # Convert JSON fields to strings if they're dictionaries
            if "connection_params" in provider_data and isinstance(provider_data["connection_params"], dict):
                provider_data["connection_params"] = json.dumps(provider_data["connection_params"])
            
            if "request_settings" in provider_data and isinstance(provider_data["request_settings"], dict):
                provider_data["request_settings"] = json.dumps(provider_data["request_settings"])
            
            for key, value in provider_data.items():
                if hasattr(provider, key):
                    setattr(provider, key, value)
            
            self.db.commit()
            self.db.refresh(provider)
            return provider
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating provider: {e}")
            raise
    
    def delete_provider(self, provider_id: str) -> bool:
        """
        Delete a provider.
        
        Args:
            provider_id: ID of the provider to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            SQLAlchemyError: If there's an error deleting the provider
        """
        try:
            provider = self.get_provider_by_id(provider_id)
            if not provider:
                return False
            
            self.db.delete(provider)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting provider: {e}")
            raise
    
    # Provider Model methods
    
    def get_models_by_provider_id(self, provider_id: str) -> List[ProviderModel]:
        """
        Get all models for a provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of ProviderModel objects
        """
        return self.db.query(ProviderModel).filter(ProviderModel.provider_id == provider_id).all()
    
    def get_model_by_id(self, model_id: str, provider_id: str) -> Optional[ProviderModel]:
        """
        Get a model by ID.
        
        Args:
            model_id: ID of the model to get
            provider_id: ID of the provider
            
        Returns:
            ProviderModel object or None if not found
        """
        return self.db.query(ProviderModel).filter(
            ProviderModel.model_id == model_id,
            ProviderModel.provider_id == provider_id
        ).first()
    
    def create_model(self, model_data: Dict[str, Any]) -> ProviderModel:
        """
        Create a new model.
        
        Args:
            model_data: Dictionary containing model data
            
        Returns:
            Created ProviderModel object
            
        Raises:
            SQLAlchemyError: If there's an error creating the model
        """
        try:
            # Convert JSON fields to strings if they're dictionaries or lists
            if "capabilities" in model_data and (isinstance(model_data["capabilities"], dict) or isinstance(model_data["capabilities"], list)):
                model_data["capabilities"] = json.dumps(model_data["capabilities"])
            
            if "parameters" in model_data and isinstance(model_data["parameters"], dict):
                model_data["parameters"] = json.dumps(model_data["parameters"])
            
            model = ProviderModel(
                model_id=model_data["model_id"],
                provider_id=model_data["provider_id"],
                display_name=model_data.get("display_name", model_data["model_id"]),
                model_type=model_data.get("model_type", "chat"),
                context_window=model_data.get("context_window"),
                max_tokens=model_data.get("max_tokens"),
                enabled=model_data.get("enabled", True),
                capabilities=model_data.get("capabilities"),
                parameters=model_data.get("parameters")
            )
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
            return model
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating model: {e}")
            raise
    
    def update_model(self, model_id: str, provider_id: str, model_data: Dict[str, Any]) -> Optional[ProviderModel]:
        """
        Update a model.
        
        Args:
            model_id: ID of the model to update
            provider_id: ID of the provider
            model_data: Dictionary containing updated model data
            
        Returns:
            Updated ProviderModel object or None if not found
            
        Raises:
            SQLAlchemyError: If there's an error updating the model
        """
        try:
            model = self.get_model_by_id(model_id, provider_id)
            if not model:
                return None
            
            # Convert JSON fields to strings if they're dictionaries or lists
            if "capabilities" in model_data and (isinstance(model_data["capabilities"], dict) or isinstance(model_data["capabilities"], list)):
                model_data["capabilities"] = json.dumps(model_data["capabilities"])
            
            if "parameters" in model_data and isinstance(model_data["parameters"], dict):
                model_data["parameters"] = json.dumps(model_data["parameters"])
            
            for key, value in model_data.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            
            self.db.commit()
            self.db.refresh(model)
            return model
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating model: {e}")
            raise
    
    def delete_model(self, model_id: str, provider_id: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_id: ID of the model to delete
            provider_id: ID of the provider
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            SQLAlchemyError: If there's an error deleting the model
        """
        try:
            model = self.get_model_by_id(model_id, provider_id)
            if not model:
                return False
            
            self.db.delete(model)
            self.db.commit()
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting model: {e}")
            raise
    
    # API Key methods
    
    def get_api_keys_by_provider_id(self, provider_id: str) -> List[ApiKey]:
        """
        Get all API keys for a provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of ApiKey objects
        """
        return self.db.query(ApiKey).filter(ApiKey.provider_id == provider_id).all()
    
    def get_api_key_by_id(self, key_id: int) -> Optional[ApiKey]:
        """
        Get an API key by ID.
        
        Args:
            key_id: ID of the API key to get
            
        Returns:
            ApiKey object or None if not found
        """
        return self.db.query(ApiKey).filter(ApiKey.key_id == key_id).first()
    
    def create_api_key(self, api_key_data: Dict[str, Any]) -> ApiKey:
        """
        Create a new API key.
        
        Args:
            api_key_data: Dictionary containing API key data
            
        Returns:
            Created ApiKey object
            
        Raises:
            SQLAlchemyError: If there's an error creating the API key
        """
        try:
            # Encrypt the key value if encryption key is provided
            key_value = api_key_data["key_value"]
            if self.encryption_key and api_key_data.get("is_encrypted", True):
                from asf.medical.llm_gateway.utils.crypto import encrypt_value
                key_value = encrypt_value(key_value, self.encryption_key)
            
            api_key = ApiKey(
                provider_id=api_key_data["provider_id"],
                key_value=key_value,
                is_encrypted=api_key_data.get("is_encrypted", True),
                environment=api_key_data.get("environment", "development"),
                expires_at=api_key_data.get("expires_at"),
                created_by_user_id=api_key_data.get("created_by_user_id")
            )
            self.db.add(api_key)
            self.db.commit()
            self.db.refresh(api_key)
            return api_key
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating API key: {e}")
            raise
    
    def get_decrypted_api_key(self, key_id: int) -> Optional[str]:
        """
        Get a decrypted API key.
        
        Args:
            key_id: ID of the API key to get
            
        Returns:
            Decrypted API key value or None if not found or decryption fails
        """
        api_key = self.get_api_key_by_id(key_id)
        if not api_key:
            return None
        
        if api_key.is_encrypted and self.encryption_key:
            try:
                from asf.medical.llm_gateway.utils.crypto import decrypt_value
                return decrypt_value(api_key.key_value, self.encryption_key)
            except Exception as e:
                logger.error(f"Error decrypting API key: {e}")
                return None
        
        return api_key.key_value
    
    # Connection Parameter methods
    
    def get_connection_parameters_by_provider_id(self, provider_id: str, environment: str = "development") -> List[ConnectionParameter]:
        """
        Get all connection parameters for a provider.
        
        Args:
            provider_id: ID of the provider
            environment: Environment (development, staging, production)
            
        Returns:
            List of ConnectionParameter objects
        """
        return self.db.query(ConnectionParameter).filter(
            ConnectionParameter.provider_id == provider_id,
            ConnectionParameter.environment == environment
        ).all()
    
    def get_connection_parameter(self, provider_id: str, param_name: str, environment: str = "development") -> Optional[ConnectionParameter]:
        """
        Get a connection parameter.
        
        Args:
            provider_id: ID of the provider
            param_name: Name of the parameter
            environment: Environment (development, staging, production)
            
        Returns:
            ConnectionParameter object or None if not found
        """
        return self.db.query(ConnectionParameter).filter(
            ConnectionParameter.provider_id == provider_id,
            ConnectionParameter.param_name == param_name,
            ConnectionParameter.environment == environment
        ).first()
    
    def set_connection_parameter(self, param_data: Dict[str, Any]) -> ConnectionParameter:
        """
        Set a connection parameter (create or update).
        
        Args:
            param_data: Dictionary containing parameter data
            
        Returns:
            Created or updated ConnectionParameter object
            
        Raises:
            SQLAlchemyError: If there's an error setting the parameter
        """
        try:
            # Check if parameter already exists
            param = self.get_connection_parameter(
                param_data["provider_id"],
                param_data["param_name"],
                param_data.get("environment", "development")
            )
            
            if param:
                # Update existing parameter
                param.param_value = param_data["param_value"]
                param.is_sensitive = param_data.get("is_sensitive", False)
            else:
                # Create new parameter
                param = ConnectionParameter(
                    provider_id=param_data["provider_id"],
                    param_name=param_data["param_name"],
                    param_value=param_data["param_value"],
                    is_sensitive=param_data.get("is_sensitive", False),
                    environment=param_data.get("environment", "development")
                )
                self.db.add(param)
            
            self.db.commit()
            self.db.refresh(param)
            return param
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error setting connection parameter: {e}")
            raise
