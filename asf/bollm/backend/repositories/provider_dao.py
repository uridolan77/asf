"""
Provider Data Access Object (DAO) for direct database operations.

This module provides a DAO for LLM providers that directly connects to the database
and performs CRUD operations on the llm_providers table.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
import logging
import json
from datetime import datetime

from asf.bollm.backend.config.config import SQLALCHEMY_DATABASE_URL
from asf.bollm.backend.models.provider import Provider
from asf.bollm.backend.models.base import Base

logger = logging.getLogger(__name__)

class ProviderDAO:
    """
    Data Access Object for LLM providers.
    
    This class provides methods for direct database operations on the llm_providers table.
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
        """Ensure the llm_providers table exists."""
        try:
            # Check if table exists
            with self.get_db_session() as db:
                from sqlalchemy import inspect
                inspector = inspect(db.get_bind())
                if not inspector.has_table("llm_providers"):
                    logger.info("Creating llm_providers table...")
                    Base.metadata.create_all(bind=self.engine)
                    logger.info("llm_providers table created successfully")
                else:
                    logger.info("llm_providers table already exists")
        except Exception as e:
            logger.error(f"Error ensuring table exists: {e}")
            raise
    
    def get_all_providers(self) -> List[Dict[str, Any]]:
        """
        Get all LLM providers.
            
        Returns:
            List of LLM providers as dictionaries
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Build query
            query = "SELECT * FROM llm_providers"
            
            # Execute query
            with self.get_db_session() as db:
                result = db.execute(text(query)).fetchall()
                
                # Convert to dictionaries
                providers = []
                for row in result:
                    provider_dict = {
                        "provider_id": row.provider_id,
                        "provider_type": row.provider_type,
                        "display_name": row.display_name,
                        "description": row.description,
                        "enabled": row.enabled,
                        "connection_params": json.loads(row.connection_params) if row.connection_params else {},
                        "request_settings": json.loads(row.request_settings) if row.request_settings else {},
                        "created_at": row.created_at.isoformat() if row.created_at else None,
                        "updated_at": row.updated_at.isoformat() if row.updated_at else None
                    }
                    providers.append(provider_dict)
                
                logger.info(f"Found {len(providers)} providers in the database")
                return providers
        except Exception as e:
            logger.error(f"Error getting LLM providers: {e}")
            return []
    
    def get_provider_by_id(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """
        Get an LLM provider by ID.
        
        Args:
            provider_id: ID of the provider to get
            
        Returns:
            LLM provider as a dictionary or None if not found
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Build query
            query = "SELECT * FROM llm_providers WHERE provider_id = :provider_id"
            params = {"provider_id": provider_id}
            
            # Execute query
            with self.get_db_session() as db:
                row = db.execute(text(query), params).fetchone()
                
                if not row:
                    return None
                
                # Convert to dictionary
                provider_dict = {
                    "provider_id": row.provider_id,
                    "provider_type": row.provider_type,
                    "display_name": row.display_name,
                    "description": row.description,
                    "enabled": row.enabled,
                    "connection_params": json.loads(row.connection_params) if row.connection_params else {},
                    "request_settings": json.loads(row.request_settings) if row.request_settings else {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "updated_at": row.updated_at.isoformat() if row.updated_at else None
                }
                
                return provider_dict
        except Exception as e:
            logger.error(f"Error getting LLM provider: {e}")
            return None
    
    def create_provider(self, provider_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a new LLM provider.
        
        Args:
            provider_data: Dictionary containing provider data
            
        Returns:
            Created LLM provider as a dictionary or None if error
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Extract required fields
            provider_id = provider_data.get("provider_id")
            provider_type = provider_data.get("provider_type")
            display_name = provider_data.get("display_name", provider_id)
            description = provider_data.get("description", "")
            enabled = provider_data.get("enabled", True)
            
            # JSON fields
            connection_params = json.dumps(provider_data.get("connection_params", {}))
            request_settings = json.dumps(provider_data.get("request_settings", {}))
            
            # Build query
            query = """
            INSERT INTO llm_providers 
            (provider_id, provider_type, display_name, description, enabled, connection_params, request_settings)
            VALUES 
            (:provider_id, :provider_type, :display_name, :description, :enabled, :connection_params, :request_settings)
            """
            params = {
                "provider_id": provider_id,
                "provider_type": provider_type,
                "display_name": display_name,
                "description": description,
                "enabled": enabled,
                "connection_params": connection_params,
                "request_settings": request_settings
            }
            
            # Execute query
            with self.get_db_session() as db:
                db.execute(text(query), params)
                db.commit()
                
                # Get created provider
                return self.get_provider_by_id(provider_id)
        except Exception as e:
            logger.error(f"Error creating LLM provider: {e}")
            return None
    
    def update_provider(self, provider_id: str, provider_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update an LLM provider.
        
        Args:
            provider_id: ID of the provider to update
            provider_data: Dictionary containing updated provider data
            
        Returns:
            Updated LLM provider as a dictionary or None if error
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Get provider
            provider = self.get_provider_by_id(provider_id)
            
            if not provider:
                return None
            
            # Build query
            query = "UPDATE llm_providers SET "
            params = {"provider_id": provider_id}
            
            # Add fields to update
            update_parts = []
            
            if "provider_type" in provider_data:
                update_parts.append("provider_type = :provider_type")
                params["provider_type"] = provider_data["provider_type"]
            
            if "display_name" in provider_data:
                update_parts.append("display_name = :display_name")
                params["display_name"] = provider_data["display_name"]
            
            if "description" in provider_data:
                update_parts.append("description = :description")
                params["description"] = provider_data["description"]
            
            if "enabled" in provider_data:
                update_parts.append("enabled = :enabled")
                params["enabled"] = provider_data["enabled"]
            
            if "connection_params" in provider_data:
                update_parts.append("connection_params = :connection_params")
                params["connection_params"] = json.dumps(provider_data["connection_params"])
            
            if "request_settings" in provider_data:
                update_parts.append("request_settings = :request_settings")
                params["request_settings"] = json.dumps(provider_data["request_settings"])
            
            # Add updated_at
            update_parts.append("updated_at = :updated_at")
            params["updated_at"] = datetime.utcnow()
            
            # Complete query
            query += ", ".join(update_parts)
            query += " WHERE provider_id = :provider_id"
            
            # Execute query
            with self.get_db_session() as db:
                db.execute(text(query), params)
                db.commit()
                
                # Get updated provider
                return self.get_provider_by_id(provider_id)
        except Exception as e:
            logger.error(f"Error updating LLM provider: {e}")
            return None
    
    def delete_provider(self, provider_id: str) -> bool:
        """
        Delete an LLM provider.
        
        Args:
            provider_id: ID of the provider to delete
            
        Returns:
            True if deleted, False if not found or error
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Get provider
            provider = self.get_provider_by_id(provider_id)
            
            if not provider:
                return False
            
            # Build query
            query = "DELETE FROM llm_providers WHERE provider_id = :provider_id"
            params = {"provider_id": provider_id}
            
            # Execute query
            with self.get_db_session() as db:
                db.execute(text(query), params)
                db.commit()
                
                return True
        except Exception as e:
            logger.error(f"Error deleting LLM provider: {e}")
            return False
    
    def get_models_by_provider_id(self, provider_id: str) -> List[Dict[str, Any]]:
        """
        Get all models for a provider.
        
        Args:
            provider_id: ID of the provider
            
        Returns:
            List of models as dictionaries
        """
        try:
            # Ensure table exists
            self.ensure_table_exists()
            
            # Build query
            query = "SELECT * FROM llm_models WHERE provider_id = :provider_id"
            params = {"provider_id": provider_id}
            
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
                
                logger.info(f"Found {len(models)} models for provider {provider_id}")
                return models
        except Exception as e:
            logger.error(f"Error getting models for provider {provider_id}: {e}")
            return []
