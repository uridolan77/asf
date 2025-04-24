"""
Provider models for LLM Gateway.

This module defines the database models for LLM providers, including
Provider, ProviderModel, ApiKey, and ConnectionParameter.
"""

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, Text, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Provider(Base):
    """
    Provider model for LLM providers.
    
    This model represents an LLM provider, such as OpenAI, Anthropic, etc.
    """
    
    __tablename__ = "llm_providers"
    
    provider_id = Column(String(255), primary_key=True)
    provider_type = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    enabled = Column(Boolean, default=True)
    connection_params = Column(Text)  # JSON string
    request_settings = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_by_user_id = Column(Integer)
    
    # Relationships
    models = relationship("ProviderModel", back_populates="provider", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="provider", cascade="all, delete-orphan")
    connection_parameters = relationship("ConnectionParameter", back_populates="provider", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Provider(provider_id='{self.provider_id}', display_name='{self.display_name}')>"

class ProviderModel(Base):
    """
    Model for LLM models associated with a provider.
    
    This model represents an LLM model, such as GPT-4, Claude, etc.
    """
    
    __tablename__ = "llm_models"
    
    model_id = Column(String(255), primary_key=True)
    provider_id = Column(String(255), ForeignKey("llm_providers.provider_id"), primary_key=True)
    display_name = Column(String(255), nullable=False)
    model_type = Column(String(50), default="chat")  # chat, completion, embedding, etc.
    context_window = Column(Integer)
    max_tokens = Column(Integer)
    enabled = Column(Boolean, default=True)
    capabilities = Column(Text)  # JSON string
    parameters = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    provider = relationship("Provider", back_populates="models")
    
    def __repr__(self):
        return f"<ProviderModel(model_id='{self.model_id}', provider_id='{self.provider_id}')>"

class ApiKey(Base):
    """
    Model for API keys associated with a provider.
    
    This model represents an API key for a provider.
    """
    
    __tablename__ = "llm_provider_api_keys"
    
    key_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(255), ForeignKey("llm_providers.provider_id"), nullable=False)
    key_value = Column(Text, nullable=False)
    is_encrypted = Column(Boolean, default=True)
    environment = Column(String(50), default="development")  # development, staging, production
    expires_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    created_by_user_id = Column(Integer)
    
    # Relationships
    provider = relationship("Provider", back_populates="api_keys")
    
    def __repr__(self):
        return f"<ApiKey(key_id={self.key_id}, provider_id='{self.provider_id}')>"

class ConnectionParameter(Base):
    """
    Model for connection parameters associated with a provider.
    
    This model represents a connection parameter for a provider.
    """
    
    __tablename__ = "llm_provider_connection_parameters"
    
    param_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(255), ForeignKey("llm_providers.provider_id"), nullable=False)
    param_name = Column(String(255), nullable=False)
    param_value = Column(Text, nullable=False)
    is_sensitive = Column(Boolean, default=False)
    environment = Column(String(50), default="development")  # development, staging, production
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    provider = relationship("Provider", back_populates="connection_parameters")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('provider_id', 'param_name', 'environment', name='uix_connection_param'),
    )
    
    def __repr__(self):
        return f"<ConnectionParameter(param_id={self.param_id}, provider_id='{self.provider_id}', param_name='{self.param_name}')>"

class AuditLog(Base):
    """
    Model for audit logs.
    
    This model represents an audit log entry for tracking changes to providers and related entities.
    """
    
    __tablename__ = "llm_provider_audit_logs"
    
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(255), nullable=False)
    record_id = Column(String(255), nullable=False)
    action = Column(String(50), nullable=False)  # create, update, delete
    changed_by_user_id = Column(Integer)
    old_values = Column(Text)  # JSON string
    new_values = Column(Text)  # JSON string
    timestamp = Column(DateTime, server_default=func.now())
    
    def __repr__(self):
        return f"<AuditLog(log_id={self.log_id}, table_name='{self.table_name}', action='{self.action}')>"
