from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, Boolean, UniqueConstraint, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .base import Base

class ServiceConfiguration(Base):
    """
    Service Configuration model for storing LLM service configurations.
    
    This model stores the main configuration for the LLM service abstraction layer,
    including feature toggles and general settings.
    """
    __tablename__ = "service_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_id = Column(String(100), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    
    # Feature toggles
    enable_caching = Column(Boolean, default=True)
    enable_resilience = Column(Boolean, default=True)
    enable_observability = Column(Boolean, default=True)
    enable_events = Column(Boolean, default=True)
    enable_progress_tracking = Column(Boolean, default=True)
    
    # User ownership and sharing
    created_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    is_public = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    caching_config = relationship("CachingConfiguration", back_populates="service_config", uselist=False, cascade="all, delete-orphan")
    resilience_config = relationship("ResilienceConfiguration", back_populates="service_config", uselist=False, cascade="all, delete-orphan")
    observability_config = relationship("ObservabilityConfiguration", back_populates="service_config", uselist=False, cascade="all, delete-orphan")
    events_config = relationship("EventsConfiguration", back_populates="service_config", uselist=False, cascade="all, delete-orphan")
    progress_tracking_config = relationship("ProgressTrackingConfiguration", back_populates="service_config", uselist=False, cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('service_id', 'name', 'created_by_user_id', name='uix_service_config'),
    )
    
    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "id": self.id,
            "service_id": self.service_id,
            "name": self.name,
            "description": self.description,
            "enable_caching": self.enable_caching,
            "enable_resilience": self.enable_resilience,
            "enable_observability": self.enable_observability,
            "enable_events": self.enable_events,
            "enable_progress_tracking": self.enable_progress_tracking,
            "created_by_user_id": self.created_by_user_id,
            "is_public": self.is_public,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": {
                "caching": self.caching_config.to_dict() if self.caching_config else {},
                "resilience": self.resilience_config.to_dict() if self.resilience_config else {},
                "observability": self.observability_config.to_dict() if self.observability_config else {},
                "events": self.events_config.to_dict() if self.events_config else {},
                "progress_tracking": self.progress_tracking_config.to_dict() if self.progress_tracking_config else {}
            }
        }


class CachingConfiguration(Base):
    """
    Caching Configuration model for storing LLM service caching settings.
    """
    __tablename__ = "caching_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_config_id = Column(Integer, ForeignKey("service_configurations.id"), nullable=False)
    
    # Caching settings
    similarity_threshold = Column(Float, default=0.92)
    max_entries = Column(Integer, default=10000)
    ttl_seconds = Column(Integer, default=3600)
    persistence_type = Column(String(20), default="disk")
    persistence_config = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    service_config = relationship("ServiceConfiguration", back_populates="caching_config")
    
    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "id": self.id,
            "similarity_threshold": self.similarity_threshold,
            "max_entries": self.max_entries,
            "ttl_seconds": self.ttl_seconds,
            "persistence_type": self.persistence_type,
            "persistence_config": self.persistence_config
        }


class ResilienceConfiguration(Base):
    """
    Resilience Configuration model for storing LLM service resilience settings.
    """
    __tablename__ = "resilience_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_config_id = Column(Integer, ForeignKey("service_configurations.id"), nullable=False)
    
    # Resilience settings
    max_retries = Column(Integer, default=3)
    retry_delay = Column(Float, default=1.0)
    backoff_factor = Column(Float, default=2.0)
    circuit_breaker_failure_threshold = Column(Integer, default=5)
    circuit_breaker_reset_timeout = Column(Integer, default=30)
    timeout_seconds = Column(Float, default=30.0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    service_config = relationship("ServiceConfiguration", back_populates="resilience_config")
    
    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "id": self.id,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "backoff_factor": self.backoff_factor,
            "circuit_breaker_failure_threshold": self.circuit_breaker_failure_threshold,
            "circuit_breaker_reset_timeout": self.circuit_breaker_reset_timeout,
            "timeout_seconds": self.timeout_seconds
        }


class ObservabilityConfiguration(Base):
    """
    Observability Configuration model for storing LLM service observability settings.
    """
    __tablename__ = "observability_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_config_id = Column(Integer, ForeignKey("service_configurations.id"), nullable=False)
    
    # Observability settings
    metrics_enabled = Column(Boolean, default=True)
    tracing_enabled = Column(Boolean, default=True)
    logging_level = Column(String(10), default="INFO")
    export_metrics = Column(Boolean, default=False)
    metrics_export_url = Column(String(255))
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    service_config = relationship("ServiceConfiguration", back_populates="observability_config")
    
    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "id": self.id,
            "metrics_enabled": self.metrics_enabled,
            "tracing_enabled": self.tracing_enabled,
            "logging_level": self.logging_level,
            "export_metrics": self.export_metrics,
            "metrics_export_url": self.metrics_export_url
        }


class EventsConfiguration(Base):
    """
    Events Configuration model for storing LLM service events settings.
    """
    __tablename__ = "events_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_config_id = Column(Integer, ForeignKey("service_configurations.id"), nullable=False)
    
    # Events settings
    max_event_history = Column(Integer, default=100)
    publish_to_external = Column(Boolean, default=False)
    external_event_url = Column(String(255))
    event_types_filter = Column(JSON)  # List of event types to publish
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    service_config = relationship("ServiceConfiguration", back_populates="events_config")
    
    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "id": self.id,
            "max_event_history": self.max_event_history,
            "publish_to_external": self.publish_to_external,
            "external_event_url": self.external_event_url,
            "event_types_filter": self.event_types_filter
        }


class ProgressTrackingConfiguration(Base):
    """
    Progress Tracking Configuration model for storing LLM service progress tracking settings.
    """
    __tablename__ = "progress_tracking_configurations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    service_config_id = Column(Integer, ForeignKey("service_configurations.id"), nullable=False)
    
    # Progress tracking settings
    max_active_operations = Column(Integer, default=100)
    operation_ttl_seconds = Column(Integer, default=3600)
    publish_updates = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    service_config = relationship("ServiceConfiguration", back_populates="progress_tracking_config")
    
    def to_dict(self):
        """Convert the model to a dictionary for API responses."""
        return {
            "id": self.id,
            "max_active_operations": self.max_active_operations,
            "operation_ttl_seconds": self.operation_ttl_seconds,
            "publish_updates": self.publish_updates
        }
