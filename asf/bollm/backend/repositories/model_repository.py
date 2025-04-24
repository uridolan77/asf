from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
from models.llm_model import LLMModel

logger = logging.getLogger(__name__)

class ModelRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_models(self, provider_id: Optional[str] = None,
                      model_type: Optional[str] = None,
                      skip: int = 0, limit: int = 100) -> List[LLMModel]:
        """
        Get all LLM models with optional filtering.

        Args:
            provider_id: Filter by provider ID (optional)
            model_type: Filter by model type (optional)
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of LLM models
        """
        try:
            # Build query
            query = self.db.query(LLMModel)
            logger.info(f"ModelRepository.get_all_models: Building query with provider_id={provider_id}, model_type={model_type}")

            # Apply filters
            if provider_id:
                query = query.filter(LLMModel.provider_id == provider_id)

            if model_type:
                query = query.filter(LLMModel.model_type == model_type)

            # Apply pagination
            query = query.offset(skip).limit(limit)

            # Execute query
            result = query.all()
            logger.info(f"ModelRepository.get_all_models: Found {len(result)} models")
            for model in result:
                logger.info(f"  - Model: {model.model_id}, Provider: {model.provider_id}")

            return result
        except SQLAlchemyError as e:
            logger.error(f"Error getting LLM models: {e}")
            return []

    def get_model_by_id(self, model_id: str, provider_id: str) -> Optional[LLMModel]:
        """
        Get an LLM model by ID and provider ID.

        Args:
            model_id: ID of the model to get
            provider_id: ID of the provider

        Returns:
            LLM model or None if not found
        """
        try:
            return self.db.query(LLMModel).filter(
                LLMModel.model_id == model_id,
                LLMModel.provider_id == provider_id
            ).first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting LLM model: {e}")
            return None

    def create_model(self, model_data: Dict[str, Any]) -> Optional[LLMModel]:
        """
        Create a new LLM model.

        Args:
            model_data: Dictionary containing model data

        Returns:
            Created LLM model or None if error
        """
        try:
            # Create model
            model = LLMModel(
                model_id=model_data["model_id"],
                provider_id=model_data["provider_id"],
                display_name=model_data.get("display_name", model_data["model_id"]),
                model_type=model_data.get("model_type", "chat"),
                context_window=model_data.get("context_window"),
                max_output_tokens=model_data.get("max_output_tokens"),
                capabilities=model_data.get("capabilities", []),
                parameters=model_data.get("parameters", {})
            )

            self.db.add(model)
            self.db.commit()
            self.db.refresh(model)

            return model
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating LLM model: {e}")
            return None

    def update_model(self, model_id: str, provider_id: str, model_data: Dict[str, Any]) -> Optional[LLMModel]:
        """
        Update an LLM model.

        Args:
            model_id: ID of the model to update
            provider_id: ID of the provider
            model_data: Dictionary containing updated model data

        Returns:
            Updated LLM model or None if error
        """
        try:
            # Get model
            model = self.get_model_by_id(model_id, provider_id)

            if not model:
                return None

            # Update model
            if "display_name" in model_data:
                model.display_name = model_data["display_name"]
            if "model_type" in model_data:
                model.model_type = model_data["model_type"]
            if "context_window" in model_data:
                model.context_window = model_data["context_window"]
            if "max_output_tokens" in model_data:
                model.max_output_tokens = model_data["max_output_tokens"]
            if "capabilities" in model_data:
                model.capabilities = model_data["capabilities"]
            if "parameters" in model_data:
                model.parameters = model_data["parameters"]

            self.db.commit()
            self.db.refresh(model)

            return model
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating LLM model: {e}")
            return None

    def delete_model(self, model_id: str, provider_id: str) -> bool:
        """
        Delete an LLM model.

        Args:
            model_id: ID of the model to delete
            provider_id: ID of the provider

        Returns:
            True if deleted, False if not found or error
        """
        try:
            # Get model
            model = self.get_model_by_id(model_id, provider_id)

            if not model:
                return False

            # Delete model
            self.db.delete(model)
            self.db.commit()

            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting LLM model: {e}")
            return False
