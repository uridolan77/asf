"""
Medical Client Models

This module defines the database models for medical clients, their configurations,
status, and usage statistics.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, Text, ForeignKey, DateTime, Date, Enum, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from datetime import datetime

Base = declarative_base()

class ClientStatus(enum.Enum):
    """Enum for client connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    UNKNOWN = "unknown"

class MedicalClient(Base):
    """Medical client model"""
    __tablename__ = "medical_clients"

    client_id = Column(String(50), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    base_url = Column(String(255))
    api_version = Column(String(50))
    logo_url = Column(String(255))
    documentation_url = Column(String(255))
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    configs = relationship("MedicalClientConfig", back_populates="client", cascade="all, delete-orphan")
    status = relationship("MedicalClientStatus", back_populates="client", uselist=False, cascade="all, delete-orphan")
    status_logs = relationship("MedicalClientStatusLog", back_populates="client", cascade="all, delete-orphan")
    usage_stats = relationship("MedicalClientUsageStat", back_populates="client", cascade="all, delete-orphan")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "client_id": self.client_id,
            "name": self.name,
            "description": self.description,
            "base_url": self.base_url,
            "api_version": self.api_version,
            "logo_url": self.logo_url,
            "documentation_url": self.documentation_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status.status.value if self.status else "unknown",
            "response_time": self.status.response_time if self.status else None,
            "last_checked": self.status.last_checked.isoformat() if self.status and self.status.last_checked else None
        }

class MedicalClientConfig(Base):
    """Medical client configuration model"""
    __tablename__ = "medical_client_configs"

    config_id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), ForeignKey("medical_clients.client_id", ondelete="CASCADE"), nullable=False)
    api_key = Column(String(255))
    email = Column(String(255))
    username = Column(String(100))
    password = Column(String(255))
    token = Column(String(255))
    token_expiry = Column(DateTime)
    rate_limit = Column(Integer)
    rate_limit_period = Column(String(20))
    timeout = Column(Integer, default=30)
    retry_count = Column(Integer, default=3)
    use_cache = Column(Boolean, default=True)
    cache_ttl = Column(Integer, default=3600)
    additional_config = Column(JSON)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    client = relationship("MedicalClient", back_populates="configs")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "config_id": self.config_id,
            "client_id": self.client_id,
            "api_key": "********" if self.api_key else None,  # Mask sensitive data
            "email": self.email,
            "username": self.username,
            "password": "********" if self.password else None,  # Mask sensitive data
            "token": "********" if self.token else None,  # Mask sensitive data
            "token_expiry": self.token_expiry.isoformat() if self.token_expiry else None,
            "rate_limit": self.rate_limit,
            "rate_limit_period": self.rate_limit_period,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "use_cache": self.use_cache,
            "cache_ttl": self.cache_ttl,
            "additional_config": self.additional_config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class MedicalClientStatus(Base):
    """Medical client status model"""
    __tablename__ = "medical_client_status"

    status_id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), ForeignKey("medical_clients.client_id", ondelete="CASCADE"), nullable=False)
    status = Column(Enum(ClientStatus), default=ClientStatus.UNKNOWN)
    response_time = Column(Float)
    last_checked = Column(DateTime, default=func.now())
    error_message = Column(Text)

    # Relationships
    client = relationship("MedicalClient", back_populates="status")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "status_id": self.status_id,
            "client_id": self.client_id,
            "status": self.status.value,
            "response_time": self.response_time,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "error_message": self.error_message
        }

class MedicalClientStatusLog(Base):
    """Medical client status log model"""
    __tablename__ = "medical_client_status_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), ForeignKey("medical_clients.client_id", ondelete="CASCADE"), nullable=False)
    status = Column(Enum(ClientStatus), nullable=False)
    response_time = Column(Float)
    checked_at = Column(DateTime, default=func.now())
    error_message = Column(Text)

    # Relationships
    client = relationship("MedicalClient", back_populates="status_logs")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "log_id": self.log_id,
            "client_id": self.client_id,
            "status": self.status.value,
            "response_time": self.response_time,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None,
            "error_message": self.error_message
        }

class MedicalClientUsageStat(Base):
    """Medical client usage statistics model"""
    __tablename__ = "medical_client_usage_stats"

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    client_id = Column(String(50), ForeignKey("medical_clients.client_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    requests_count = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    cached_requests = Column(Integer, default=0)
    total_response_time = Column(Float, default=0)
    average_response_time = Column(Float, default=0)

    # Relationships
    client = relationship("MedicalClient", back_populates="usage_stats")

    def to_dict(self):
        """Convert model to dictionary"""
        return {
            "stat_id": self.stat_id,
            "client_id": self.client_id,
            "date": self.date.isoformat() if self.date else None,
            "requests_count": self.requests_count,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cached_requests": self.cached_requests,
            "total_response_time": self.total_response_time,
            "average_response_time": self.average_response_time
        }
