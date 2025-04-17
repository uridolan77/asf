from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Integer, Text, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base
from .association import users_providers  # Import the association table

class Provider(Base):
    __tablename__ = "providers"

    provider_id = Column(String(50), primary_key=True)
    display_name = Column(String(100), nullable=False)
    provider_type = Column(String(50), nullable=False)
    description = Column(Text)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    created_by_user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    # Many-to-many relationship with users
    users = relationship("User", secondary=users_providers, back_populates="providers")
    models = relationship("ProviderModel", back_populates="provider", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="provider", cascade="all, delete-orphan")
    connection_parameters = relationship("ConnectionParameter", back_populates="provider", cascade="all, delete-orphan")

class ProviderModel(Base):
    __tablename__ = "provider_models"

    model_id = Column(String(50), primary_key=True)
    provider_id = Column(String(50), ForeignKey("providers.provider_id"), nullable=False)
    display_name = Column(String(100), nullable=False)
    model_type = Column(String(50))
    context_window = Column(Integer)
    max_tokens = Column(Integer)
    enabled = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    provider = relationship("Provider", back_populates="models")

class ApiKey(Base):
    __tablename__ = "api_keys"

    key_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(50), ForeignKey("providers.provider_id"), nullable=False)
    key_value = Column(Text, nullable=False)
    is_encrypted = Column(Boolean, default=True)
    environment = Column(String(20), default="development")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime)
    created_by_user_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    provider = relationship("Provider", back_populates="api_keys")
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    usage_records = relationship("ApiKeyUsage", back_populates="api_key", cascade="all, delete-orphan")

class ConnectionParameter(Base):
    __tablename__ = "connection_parameters"

    param_id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(String(50), ForeignKey("providers.provider_id"), nullable=False)
    param_name = Column(String(50), nullable=False)
    param_value = Column(Text)
    is_sensitive = Column(Boolean, default=False)
    environment = Column(String(20), default="development")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    provider = relationship("Provider", back_populates="connection_parameters")

    # Constraints
    __table_args__ = (
        UniqueConstraint('provider_id', 'param_name', 'environment', name='uix_connection_param'),
    )
