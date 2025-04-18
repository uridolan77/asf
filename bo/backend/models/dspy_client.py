"""
DSPy Client Models

This module provides database models for DSPy clients, modules, and related entities.
"""

import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Date, ForeignKey, JSON, Enum, Text
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, date
import enum
from typing import Dict, Any, List, Optional

from ..db.database import Base


class DSPyClientStatus(str, enum.Enum):
    """Enumeration of DSPy client status values"""
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"
    DISCONNECTED = "DISCONNECTED"


class DSPyCircuitState(str, enum.Enum):
    """Enumeration of circuit breaker states"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class DSPyClient(Base):
    """DSPy client model"""
    __tablename__ = "dspy_clients"

    client_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    base_url = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    config = relationship("DSPyClientConfig", back_populates="client", uselist=False)
    status = relationship("DSPyClientStatus", back_populates="client", uselist=False)
    modules = relationship("DSPyModule", back_populates="client")
    usage_stats = relationship("DSPyClientUsageStat", back_populates="client")
    status_logs = relationship("DSPyClientStatusLog", back_populates="client")
    circuit_breakers = relationship("DSPyCircuitBreakerStatus", back_populates="client")
    audit_logs = relationship("DSPyAuditLog", back_populates="client")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "client_id": self.client_id,
            "name": self.name,
            "description": self.description,
            "base_url": self.base_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status.to_dict() if self.status else None
        }


class DSPyClientConfig(Base):
    """DSPy client configuration model"""
    __tablename__ = "dspy_client_configs"

    config_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    
    # LLM Provider Settings
    provider = Column(String, default="openai")
    api_key = Column(String, nullable=True)
    organization_id = Column(String, nullable=True)
    default_model = Column(String, default="gpt-4")
    
    # Azure-specific Settings
    azure_endpoint = Column(String, nullable=True)
    azure_deployment_name = Column(String, nullable=True)
    azure_api_version = Column(String, default="2023-05-15")
    
    # Model Parameters
    max_tokens = Column(Integer, default=1000)
    temperature = Column(Float, default=0.7)
    top_p = Column(Float, default=1.0)
    
    # Caching Settings
    cache_backend = Column(String, default="disk")
    cache_directory = Column(String, default=".dspy_cache")
    cache_ttl = Column(Integer, default=3600)  # seconds
    redis_url = Column(String, nullable=True)
    redis_password = Column(String, nullable=True)
    redis_pool_size = Column(Integer, default=10)
    
    # Thread Settings
    thread_limit = Column(Integer, default=4)
    
    # Circuit Breaker Settings
    circuit_breaker_failure_threshold = Column(Integer, default=5)
    circuit_breaker_reset_timeout = Column(Float, default=30.0)  # seconds
    circuit_breaker_success_threshold = Column(Integer, default=2)
    
    # Audit Logging Settings
    enable_audit_logging = Column(Boolean, default=True)
    audit_log_path = Column(String, default="audit_logs")
    enable_phi_detection = Column(Boolean, default=True)
    
    # Security Settings
    enable_input_validation = Column(Boolean, default=True)
    enable_output_filtering = Column(Boolean, default=True)
    max_prompt_length = Column(Integer, default=16000)
    
    # Retry Settings
    max_retries = Column(Integer, default=3)
    retry_min_wait = Column(Float, default=1.0)  # seconds
    retry_max_wait = Column(Float, default=10.0)  # seconds
    
    # Connection Settings
    timeout = Column(Integer, default=30)  # seconds
    
    # Additional configuration (stored as JSON)
    additional_config = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    client = relationship("DSPyClient", back_populates="config")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "config_id": self.config_id,
            "client_id": self.client_id,
            "provider": self.provider,
            "api_key": "********" if self.api_key else None,  # Hide actual API key
            "organization_id": self.organization_id,
            "default_model": self.default_model,
            "azure_endpoint": self.azure_endpoint,
            "azure_deployment_name": self.azure_deployment_name,
            "azure_api_version": self.azure_api_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "cache_backend": self.cache_backend,
            "cache_directory": self.cache_directory,
            "cache_ttl": self.cache_ttl,
            "redis_url": self.redis_url,
            "redis_password": "********" if self.redis_password else None,  # Hide actual password
            "redis_pool_size": self.redis_pool_size,
            "thread_limit": self.thread_limit,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_reset_timeout": self.circuit_breaker_reset_timeout,
            "circuit_breaker_success_threshold": self.circuit_breaker_success_threshold,
            "enable_audit_logging": self.enable_audit_logging,
            "audit_log_path": self.audit_log_path,
            "enable_phi_detection": self.enable_phi_detection,
            "enable_input_validation": self.enable_input_validation,
            "enable_output_filtering": self.enable_output_filtering,
            "max_prompt_length": self.max_prompt_length,
            "max_retries": self.max_retries,
            "retry_min_wait": self.retry_min_wait,
            "retry_max_wait": self.retry_max_wait,
            "timeout": self.timeout,
            "additional_config": self.additional_config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DSPyClientStatus(Base):
    """DSPy client status model"""
    __tablename__ = "dspy_client_status"

    status_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    status = Column(String, default=DSPyClientStatus.DISCONNECTED)
    response_time = Column(Float, nullable=True)
    last_checked = Column(DateTime, default=datetime.now)
    error_message = Column(String, nullable=True)

    # Relationships
    client = relationship("DSPyClient", back_populates="status")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "status_id": self.status_id,
            "client_id": self.client_id,
            "status": self.status,
            "response_time": self.response_time,
            "last_checked": self.last_checked.isoformat() if self.last_checked else None,
            "error_message": self.error_message
        }


class DSPyClientStatusLog(Base):
    """DSPy client status log model"""
    __tablename__ = "dspy_client_status_logs"

    log_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    status = Column(String, nullable=True)
    response_time = Column(Float, nullable=True)
    error_message = Column(String, nullable=True)
    checked_at = Column(DateTime, default=datetime.now)

    # Relationships
    client = relationship("DSPyClient", back_populates="status_logs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "log_id": self.log_id,
            "client_id": self.client_id,
            "status": self.status,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "checked_at": self.checked_at.isoformat() if self.checked_at else None
        }


class DSPyClientUsageStat(Base):
    """DSPy client usage statistics model"""
    __tablename__ = "dspy_client_usage_stats"

    stat_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    date = Column(Date, nullable=False)
    requests_count = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)
    cached_requests = Column(Integer, default=0)
    total_response_time = Column(Float, default=0.0)
    average_response_time = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)  # Track token usage for cost estimation

    # Relationships
    client = relationship("DSPyClient", back_populates="usage_stats")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "stat_id": self.stat_id,
            "client_id": self.client_id,
            "date": self.date.isoformat() if self.date else None,
            "requests_count": self.requests_count,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "cached_requests": self.cached_requests,
            "total_response_time": self.total_response_time,
            "average_response_time": self.average_response_time,
            "tokens_used": self.tokens_used
        }


class DSPyModule(Base):
    """DSPy module model"""
    __tablename__ = "dspy_modules"

    module_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    module_type = Column(String, nullable=True)
    class_name = Column(String, nullable=True)
    registered_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    client = relationship("DSPyClient", back_populates="modules")
    metrics = relationship("DSPyModuleMetric", back_populates="module")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "module_id": self.module_id,
            "client_id": self.client_id,
            "name": self.name,
            "description": self.description,
            "module_type": self.module_type,
            "class_name": self.class_name,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DSPyModuleMetric(Base):
    """DSPy module metric model"""
    __tablename__ = "dspy_module_metrics"

    metric_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    module_id = Column(String, ForeignKey("dspy_modules.module_id", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)
    response_time = Column(Float, nullable=True)
    success = Column(Boolean, default=True)
    cached = Column(Boolean, default=False)
    tokens_used = Column(Integer, default=0)

    # Relationships
    module = relationship("DSPyModule", back_populates="metrics")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "metric_id": self.metric_id,
            "module_id": self.module_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "response_time": self.response_time,
            "success": self.success,
            "cached": self.cached,
            "tokens_used": self.tokens_used
        }


class DSPyAuditLog(Base):
    """DSPy audit log model"""
    __tablename__ = "dspy_audit_logs"

    log_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    module_id = Column(String, ForeignKey("dspy_modules.module_id", ondelete="SET NULL"), nullable=True)
    event_type = Column(String, nullable=False)  # LLM_CALL, MODULE_CALL, ERROR, etc.
    event_id = Column(String, nullable=False)    # Unique ID for the event
    component = Column(String, nullable=True)    # Component that generated the event
    timestamp = Column(DateTime, default=datetime.now)
    user_id = Column(String, nullable=True)      # User who initiated the request (if applicable)
    session_id = Column(String, nullable=True)   # Session ID (if applicable)
    correlation_id = Column(String, nullable=True)  # Correlation ID for tracing requests
    inputs = Column(JSON, nullable=True)         # Sanitized inputs
    outputs = Column(JSON, nullable=True)        # Sanitized outputs
    error = Column(String, nullable=True)        # Error message (if applicable)
    metadata = Column(JSON, nullable=True)       # Additional metadata

    # Relationships
    client = relationship("DSPyClient", back_populates="audit_logs")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "log_id": self.log_id,
            "client_id": self.client_id,
            "module_id": self.module_id,
            "event_type": self.event_type,
            "event_id": self.event_id,
            "component": self.component,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "error": self.error,
            "metadata": self.metadata
        }


class DSPyCircuitBreakerStatus(Base):
    """DSPy circuit breaker status model"""
    __tablename__ = "dspy_circuit_breaker_status"

    status_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = Column(String, ForeignKey("dspy_clients.client_id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)  # Name of the circuit breaker (usually the model name)
    state = Column(String, default=DSPyCircuitState.CLOSED)
    failure_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    last_failure = Column(DateTime, nullable=True)
    last_success = Column(DateTime, nullable=True)
    last_state_change = Column(DateTime, nullable=True)
    next_attempt = Column(DateTime, nullable=True)  # When to attempt recovery in OPEN state

    # Relationships
    client = relationship("DSPyClient", back_populates="circuit_breakers")

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "status_id": self.status_id,
            "client_id": self.client_id,
            "name": self.name,
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_state_change": self.last_state_change.isoformat() if self.last_state_change else None,
            "next_attempt": self.next_attempt.isoformat() if self.next_attempt else None
        }