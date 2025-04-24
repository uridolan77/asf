from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
# Use absolute imports
from asf.bollm.backend.models.provider import Provider, ApiKey, ConnectionParameter
from asf.bollm.backend.models.llm_model import LLMModel
from asf.bollm.backend.utils.crypto import encrypt_value, decrypt_value

logger = logging.getLogger(__name__)

class ProviderRepository:
    def __init__(self, db: Session, encryption_key: bytes = None):
        self.db = db
        self.encryption_key = encryption_key

    def get_all_providers(self) -> List[Provider]:
        """Get all providers."""
        return self.db.query(Provider).all()

    def get_provider_by_id(self, provider_id: str) -> Optional[Provider]:
        """Get a provider by ID."""
        return self.db.query(Provider).filter(Provider.provider_id == provider_id).first()

    def create_provider(self, provider_data: Dict[str, Any]) -> Provider:
        """Create a new provider."""
        try:
            provider = Provider(
                provider_id=provider_data["provider_id"],
                display_name=provider_data["display_name"],
                provider_type=provider_data["provider_type"],
                description=provider_data.get("description"),
                enabled=provider_data.get("enabled", True),
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
        """Update a provider."""
        try:
            provider = self.get_provider_by_id(provider_id)
            if not provider:
                return None

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
        """Delete a provider."""
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

    def get_models_by_provider_id(self, provider_id: str) -> List[LLMModel]:
        """Get all models for a provider."""
        return self.db.query(LLMModel).filter(LLMModel.provider_id == provider_id).all()

    def get_model_by_id(self, model_id: str) -> Optional[LLMModel]:
        """Get a model by ID."""
        return self.db.query(LLMModel).filter(LLMModel.model_id == model_id).first()

    def create_model(self, model_data: Dict[str, Any]) -> LLMModel:
        """Create a new model."""
        try:
            model = LLMModel(
                model_id=model_data["model_id"],
                provider_id=model_data["provider_id"],
                display_name=model_data["display_name"],
                model_type=model_data.get("model_type"),
                context_window=model_data.get("context_window"),
                max_tokens=model_data.get("max_tokens"),
                enabled=model_data.get("enabled", True)
            )
            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)
            return model
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating model: {e}")
            raise

    # API Key methods

    def get_api_keys_by_provider_id(self, provider_id: str) -> List[ApiKey]:
        """Get all API keys for a provider."""
        return self.db.query(ApiKey).filter(ApiKey.provider_id == provider_id).all()

    def get_api_key_by_id(self, key_id: int) -> Optional[ApiKey]:
        """Get an API key by ID."""
        return self.db.query(ApiKey).filter(ApiKey.key_id == key_id).first()

    def create_api_key(self, api_key_data: Dict[str, Any]) -> ApiKey:
        """Create a new API key."""
        try:
            # Encrypt the key value if encryption key is provided
            key_value = api_key_data["key_value"]
            if self.encryption_key and api_key_data.get("is_encrypted", True):
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
        """Get a decrypted API key."""
        api_key = self.get_api_key_by_id(key_id)
        if not api_key:
            return None

        if api_key.is_encrypted and self.encryption_key:
            try:
                return decrypt_value(api_key.key_value, self.encryption_key)
            except Exception as e:
                logger.error(f"Error decrypting API key: {e}")
                return None

        return api_key.key_value

    # Connection Parameter methods

    def get_connection_parameters_by_provider_id(self, provider_id: str, environment: str = "development") -> List[ConnectionParameter]:
        """Get all connection parameters for a provider."""
        return self.db.query(ConnectionParameter).filter(
            ConnectionParameter.provider_id == provider_id,
            ConnectionParameter.environment == environment
        ).all()

    def get_connection_parameter(self, provider_id: str, param_name: str, environment: str = "development") -> Optional[ConnectionParameter]:
        """Get a connection parameter."""
        return self.db.query(ConnectionParameter).filter(
            ConnectionParameter.provider_id == provider_id,
            ConnectionParameter.param_name == param_name,
            ConnectionParameter.environment == environment
        ).first()

    def set_connection_parameter(self, param_data: Dict[str, Any]) -> ConnectionParameter:
        """Set a connection parameter (create or update)."""
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
