"""
Provider model for LLM providers.

This module defines the Provider model for LLM providers.
"""

from sqlalchemy import Column, String, Boolean, Text, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship, foreign, remote
from sqlalchemy.sql import func

from asf.bollm.backend.models.base import Base
from asf.bollm.backend.models.association import users_providers  # Import the association table
from asf.bollm.backend.models.llm_model import LLMModel  # Import the LLMModel class
from asf.bollm.backend.models.user import BOLLMUser  # Import the renamed BOLLMUser class

class Provider(Base):
    """
    Provider model for LLM providers.

    This model represents an LLM provider, such as OpenAI, Anthropic, etc.
    """

    __tablename__ = "llm_providers"

    # Define table arguments with extend_existing=True
    __table_args__ = {'extend_existing': True}

    provider_id = Column(String(255), primary_key=True)
    provider_type = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=False)
    description = Column(Text)
    enabled = Column(Boolean, default=True)
    connection_params = Column(Text)  # JSON string
    request_settings = Column(Text)  # JSON string
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    models = relationship(
        "asf.bollm.backend.models.llm_model.LLMModel",
        back_populates="provider",
        cascade="all, delete-orphan"
    )
    api_keys = relationship("asf.bollm.backend.models.provider.ApiKey", back_populates="provider", cascade="all, delete-orphan")
    connection_parameters = relationship("asf.bollm.backend.models.provider.ConnectionParameter", back_populates="provider", cascade="all, delete-orphan")
    # Update the relationship with fully qualified path
    users = relationship("asf.bollm.backend.models.user.BOLLMUser", secondary=users_providers, back_populates="providers")

    def __repr__(self):
        return f"<Provider(provider_id='{self.provider_id}', display_name='{self.display_name}')>"

class ApiKey(Base):
    """
    Model for API keys associated with a provider.

    This model represents an API key for a provider.
    """

    __tablename__ = "llm_provider_api_keys"

    # Define table arguments with extend_existing=True
    __table_args__ = {'extend_existing': True}

    key_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(255), ForeignKey("llm_providers.provider_id"), nullable=False)
    key_value = Column(Text, nullable=False)
    is_encrypted = Column(Boolean, default=True)
    environment = Column(String(50), default="development")  # development, staging, production
    expires_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
    created_by_user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    provider = relationship("asf.bollm.backend.models.provider.Provider", back_populates="api_keys")
    created_by = relationship("asf.bollm.backend.models.user.BOLLMUser", foreign_keys=[created_by_user_id], back_populates="api_keys")  # Use fully qualified path
    usage_records = relationship("asf.bollm.backend.models.audit.ApiKeyUsage", back_populates="api_key")  # Use fully qualified path

    def __repr__(self):
        return f"<ApiKey(key_id={self.key_id}, provider_id='{self.provider_id}')>"

class ConnectionParameter(Base):
    """
    Model for connection parameters associated with a provider.

    This model represents a connection parameter for a provider.
    """

    __tablename__ = "llm_provider_connection_parameters"

    # Define table arguments with extend_existing=True
    __table_args__ = {'extend_existing': True}

    param_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(255), ForeignKey("llm_providers.provider_id"), nullable=False)
    param_name = Column(String(255), nullable=False)
    param_value = Column(Text, nullable=False)
    is_sensitive = Column(Boolean, default=False)
    environment = Column(String(50), default="development")  # development, staging, production
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    provider = relationship("asf.bollm.backend.models.provider.Provider", back_populates="connection_parameters")

    def __repr__(self):
        return f"<ConnectionParameter(param_id={self.param_id}, provider_id='{self.provider_id}', param_name='{self.param_name}')>"
