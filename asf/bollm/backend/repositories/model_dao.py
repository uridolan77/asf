"""
Model Data Access Object (DAO) for direct database operations.

This module provides a DAO for LLM models that directly connects to the database
and performs CRUD operations on the llm_models table.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import logging
import json
from datetime import datetime

from asf.bollm.backend.config.config import SQLALCHEMY_DATABASE_URL
from asf.bollm.backend.models.llm_model import LLMModel
from asf.bollm.backend.models.base import Base

logger = logging.getLogger(__name__)

class ModelDAO:
    """
    Data Access Object for LLM models.
    
    This class provides methods for direct database operations on the llm_models table.
    It establishes its own database connection and session.
    """
    
    def __init__(self):
        """Initialize the DAO with a database connection."""
        self.engine = create_engine(SQLALCHEMY_DATABASE_URL)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def get_db_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def ensure_table_exists(self):
        """Ensure the llm_models table exists."""
        try:
            # Check if table exists
            with self.get_db_session() as db:
                from sqlalchemy import inspect
                inspector = inspect(db.get_bind())
                if not inspector.has_table("llm_models"):
                    logger.info("Creating llm_models table...")
                    Base.metadata.create_all(bind=self.engine)
                    logger.info("llm_models table created successfully")
                else:
                    logger.info("llm_models table already exists")
        except Exception as e:
            logger.error(f"Error ensuring table exists: {e}")
            raise
    
    # No mapping needed - use data directly from the database
    
    def get_all_models(self, provider_id: Optional[str] = None, 
                      model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all LLM models with optional filtering.
        
        Args:
            provider_id: Filter by provider ID (optional)
            model_type: Filter by model type (optional)
            
        Returns:
            List of LLM models as dictionaries
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Build query
            query = "SELECT * FROM llm_models WHERE 1=1"
            params = {}
            
            if provider_id:
                query += " AND provider_id = :provider_id"
                params["provider_id"] = provider_id
            
            if model_type:
                query += " AND model_type = :model_type"
                params["model_type"] = model_type
            
            # Execute query
            with self.get_db_session() as db:
                result = db.execute(text(query), params).fetchall()
                
                # Convert to dictionaries
                models = []
                for row in result:
                    model_dict = {
                        "model_id": row.model_id,
                        "provider_id": row.provider_id,
                        "display_name": row.display_name,
                        "model_type": row.model_type or "chat",
                        "context_window": row.context_window,
                        "max_output_tokens": row.max_output_tokens,
                        "capabilities": json.loads(row.capabilities) if row.capabilities else [],
                        "parameters": json.loads(row.parameters) if row.parameters else {},
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None
                    }
                    models.append(model_dict)
                
                logger.info(f"Found {len(models)} models in the database")
                return models
        except Exception as e:
            logger.error(f"Error getting LLM models: {e}")
            return []
    
    def get_model_by_id(self, model_id: str, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an LLM model by ID and provider ID.
        
        Args:
            model_id: ID of the model to get
            provider_id: ID of the provider
            
        Returns:
            LLM model as a dictionary or None if not found
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Build query
            query = "SELECT * FROM llm_models WHERE model_id = :model_id AND provider_id = :provider_id"
            params = {"model_id": model_id, "provider_id": provider_id}
            
            # Execute query
            with self.get_db_session() as db:
                row = db.execute(text(query), params).fetchone()
                
                if not row:
                    return None
                
                # Convert to dictionary
                model_dict = {
                    "model_id": row.model_id,
                    "provider_id": row.provider_id,
                    "display_name": row.display_name,
                    "model_type": row.model_type or "chat",
                    "context_window": row.context_window,
                    "max_output_tokens": row.max_output_tokens,
                    "capabilities": json.loads(row.capabilities) if row.capabilities else [],
                    "parameters": json.loads(row.parameters) if row.parameters else {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None
                }
                
                return model_dict
        except Exception as e:
            logger.error(f"Error getting LLM model: {e}")
            return None
    
    def create_model(self, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new LLM model.
        
        Args:
            model_data: Dictionary containing model data
            
        Returns:
            Created LLM model as a dictionary or None if error
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Extract required fields
            model_id = model_data.get("model_id")
            provider_id = model_data.get("provider_id")
            
            display_name = model_data.get("display_name", model_id)
            model_type = model_data.get("model_type", "chat")
            context_window = model_data.get("context_window")
            max_output_tokens = model_data.get("max_output_tokens")
            capabilities = json.dumps(model_data.get("capabilities", []))
            parameters = json.dumps(model_data.get("parameters", {}))
            
            # Build query
            query = """
            INSERT INTO llm_models 
            (model_id, provider_id, display_name, model_type, context_window, max_output_tokens, capabilities, parameters)
            VALUES 
            (:model_id, :provider_id, :display_name, :model_type, :context_window, :max_output_tokens, :capabilities, :parameters)
            """
            params = {
                "model_id": model_id,
                "provider_id": provider_id,
                "display_name": display_name,
                "model_type": model_type,
                "context_window": context_window,
                "max_output_tokens": max_output_tokens,
                "capabilities": capabilities,
                "parameters": parameters
            }
            
            # Execute query
            with self.get_db_session() as db:
                db.execute(text(query), params)
                db.commit()
                
                # Get created model
                return self.get_model_by_id(model_id, provider_id)
        except Exception as e:
            logger.error(f"Error creating LLM model: {e}")
            return None
    
    def update_model(self, model_id: str, provider_id: str, model_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an LLM model.
        
        Args:
            model_id: ID of the model to update
            provider_id: ID of the provider
            model_data: Dictionary containing updated model data
            
        Returns:
            Updated LLM model as a dictionary or None if error
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Get model
            model = self.get_model_by_id(model_id, provider_id)
            
            if not model:
                return None
            
            # Build query
            query = "UPDATE llm_models SET "
            params = {"model_id": model_id, "provider_id": provider_id}
            
            # Add fields to update
            update_parts = []
            
            if "display_name" in model_data:
                update_parts.append("display_name = :display_name")
                params["display_name"] = model_data["display_name"]
            
            if "model_type" in model_data:
                update_parts.append("model_type = :model_type")
                params["model_type"] = model_data["model_type"]
            
            if "context_window" in model_data:
                update_parts.append("context_window = :context_window")
                params["context_window"] = model_data["context_window"]
            
            if "max_output_tokens" in model_data:
                update_parts.append("max_output_tokens = :max_output_tokens")
                params["max_output_tokens"] = model_data["max_output_tokens"]
            
            if "capabilities" in model_data:
                update_parts.append("capabilities = :capabilities")
                params["capabilities"] = json.dumps(model_data["capabilities"])
            
            if "parameters" in model_data:
                update_parts.append("parameters = :parameters")
                params["parameters"] = json.dumps(model_data["parameters"])
            
            # Add updated_at
            update_parts.append("updated_at = :updated_at")
            params["updated_at"] = datetime.utcnow()
            
            # Complete query
            query += ", ".join(update_parts)
            query += " WHERE model_id = :model_id AND provider_id = :provider_id"
            
            # Execute query
            with self.get_db_session() as db:
                db.execute(text(query), params)
                db.commit()
                
                # Get updated model
                return self.get_model_by_id(model_id, provider_id)
        except Exception as e:
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
            # Ensure table exists
            self.ensure_table_exists()
            
            # Get model
            model = self.get_model_by_id(model_id, provider_id)
            
            if not model:
                return False
            
            # Build query
            query = "DELETE FROM llm_models WHERE model_id = :model_id AND provider_id = :provider_id"
            params = {"model_id": model_id, "provider_id": provider_id}
            
            # Execute query
            with self.get_db_session() as db:
                db.execute(text(query), params)
                db.commit()
                
                return True
        except Exception as e:
            logger.error(f"Error deleting LLM model: {e}")
            return False
