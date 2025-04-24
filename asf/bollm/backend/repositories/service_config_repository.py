from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
from models.service_config import (
    ServiceConfiguration, CachingConfiguration, ResilienceConfiguration,
    ObservabilityConfiguration, EventsConfiguration, ProgressTrackingConfiguration
)

logger = logging.getLogger(__name__)

class ServiceConfigRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def get_all_configurations(self, user_id: Optional[int] = None, 
                              include_public: bool = True,
                              skip: int = 0, limit: int = 100) -> List[ServiceConfiguration]:
        """
        Get all service configurations with optional filtering.
        
        Args:
            user_id: Filter by user ID (optional)
            include_public: Include public configurations
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of service configurations
        """
        try:
            # Build query
            query = self.db.query(ServiceConfiguration)
            
            # Filter by ownership and public access
            if user_id is not None:
                if include_public:
                    query = query.filter(
                        (ServiceConfiguration.created_by_user_id == user_id) | 
                        (ServiceConfiguration.is_public == True)
                    )
                else:
                    query = query.filter(ServiceConfiguration.created_by_user_id == user_id)
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            # Execute query
            return query.all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting service configurations: {e}")
            return []
    
    def get_configuration_by_id(self, config_id: int, user_id: Optional[int] = None) -> Optional[ServiceConfiguration]:
        """
        Get a service configuration by ID.
        
        Args:
            config_id: ID of the configuration to get
            user_id: User ID for access control (optional)
            
        Returns:
            Service configuration or None if not found
        """
        try:
            query = self.db.query(ServiceConfiguration).filter(ServiceConfiguration.id == config_id)
            
            # Apply access control if user_id is provided
            if user_id is not None:
                query = query.filter(
                    (ServiceConfiguration.created_by_user_id == user_id) | 
                    (ServiceConfiguration.is_public == True)
                )
            
            return query.first()
        except SQLAlchemyError as e:
            logger.error(f"Error getting service configuration: {e}")
            return None
    
    def create_configuration(self, config_data: Dict[str, Any]) -> Optional[ServiceConfiguration]:
        """
        Create a new service configuration.
        
        Args:
            config_data: Dictionary containing configuration data
            
        Returns:
            Created service configuration or None if error
        """
        try:
            # Create main configuration
            service_config = ServiceConfiguration(
                service_id=config_data["service_id"],
                name=config_data["name"],
                description=config_data.get("description"),
                enable_caching=config_data.get("enable_caching", True),
                enable_resilience=config_data.get("enable_resilience", True),
                enable_observability=config_data.get("enable_observability", True),
                enable_events=config_data.get("enable_events", True),
                enable_progress_tracking=config_data.get("enable_progress_tracking", True),
                is_public=config_data.get("is_public", False),
                created_by_user_id=config_data["created_by_user_id"]
            )
            
            self.db.add(service_config)
            self.db.flush()  # Flush to get the ID
            
            # Create component configurations if provided
            caching_data = config_data.get("caching")
            if caching_data:
                caching_config = CachingConfiguration(
                    service_config_id=service_config.id,
                    similarity_threshold=caching_data.get("similarity_threshold", 0.92),
                    max_entries=caching_data.get("max_entries", 10000),
                    ttl_seconds=caching_data.get("ttl_seconds", 3600),
                    persistence_type=caching_data.get("persistence_type", "disk"),
                    persistence_config=caching_data.get("persistence_config")
                )
                self.db.add(caching_config)
            
            resilience_data = config_data.get("resilience")
            if resilience_data:
                resilience_config = ResilienceConfiguration(
                    service_config_id=service_config.id,
                    max_retries=resilience_data.get("max_retries", 3),
                    retry_delay=resilience_data.get("retry_delay", 1.0),
                    backoff_factor=resilience_data.get("backoff_factor", 2.0),
                    circuit_breaker_failure_threshold=resilience_data.get("circuit_breaker_failure_threshold", 5),
                    circuit_breaker_reset_timeout=resilience_data.get("circuit_breaker_reset_timeout", 30),
                    timeout_seconds=resilience_data.get("timeout_seconds", 30.0)
                )
                self.db.add(resilience_config)
            
            observability_data = config_data.get("observability")
            if observability_data:
                observability_config = ObservabilityConfiguration(
                    service_config_id=service_config.id,
                    metrics_enabled=observability_data.get("metrics_enabled", True),
                    tracing_enabled=observability_data.get("tracing_enabled", True),
                    logging_level=observability_data.get("logging_level", "INFO"),
                    export_metrics=observability_data.get("export_metrics", False),
                    metrics_export_url=observability_data.get("metrics_export_url")
                )
                self.db.add(observability_config)
            
            events_data = config_data.get("events")
            if events_data:
                events_config = EventsConfiguration(
                    service_config_id=service_config.id,
                    max_event_history=events_data.get("max_event_history", 100),
                    publish_to_external=events_data.get("publish_to_external", False),
                    external_event_url=events_data.get("external_event_url"),
                    event_types_filter=events_data.get("event_types_filter")
                )
                self.db.add(events_config)
            
            progress_tracking_data = config_data.get("progress_tracking")
            if progress_tracking_data:
                progress_tracking_config = ProgressTrackingConfiguration(
                    service_config_id=service_config.id,
                    max_active_operations=progress_tracking_data.get("max_active_operations", 100),
                    operation_ttl_seconds=progress_tracking_data.get("operation_ttl_seconds", 3600),
                    publish_updates=progress_tracking_data.get("publish_updates", True)
                )
                self.db.add(progress_tracking_config)
            
            self.db.commit()
            self.db.refresh(service_config)
            
            return service_config
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error creating service configuration: {e}")
            return None
    
    def update_configuration(self, config_id: int, config_data: Dict[str, Any]) -> Optional[ServiceConfiguration]:
        """
        Update a service configuration.
        
        Args:
            config_id: ID of the configuration to update
            config_data: Dictionary containing updated configuration data
            
        Returns:
            Updated service configuration or None if error
        """
        try:
            # Get configuration
            config = self.get_configuration_by_id(config_id)
            
            if not config:
                return None
            
            # Update main configuration
            if "name" in config_data:
                config.name = config_data["name"]
            if "description" in config_data:
                config.description = config_data["description"]
            if "enable_caching" in config_data:
                config.enable_caching = config_data["enable_caching"]
            if "enable_resilience" in config_data:
                config.enable_resilience = config_data["enable_resilience"]
            if "enable_observability" in config_data:
                config.enable_observability = config_data["enable_observability"]
            if "enable_events" in config_data:
                config.enable_events = config_data["enable_events"]
            if "enable_progress_tracking" in config_data:
                config.enable_progress_tracking = config_data["enable_progress_tracking"]
            if "is_public" in config_data:
                config.is_public = config_data["is_public"]
            
            # Update caching configuration
            caching_data = config_data.get("caching")
            if caching_data:
                if not config.caching_config:
                    config.caching_config = CachingConfiguration(service_config_id=config.id)
                
                caching = config.caching_config
                if "similarity_threshold" in caching_data:
                    caching.similarity_threshold = caching_data["similarity_threshold"]
                if "max_entries" in caching_data:
                    caching.max_entries = caching_data["max_entries"]
                if "ttl_seconds" in caching_data:
                    caching.ttl_seconds = caching_data["ttl_seconds"]
                if "persistence_type" in caching_data:
                    caching.persistence_type = caching_data["persistence_type"]
                if "persistence_config" in caching_data:
                    caching.persistence_config = caching_data["persistence_config"]
            
            # Update resilience configuration
            resilience_data = config_data.get("resilience")
            if resilience_data:
                if not config.resilience_config:
                    config.resilience_config = ResilienceConfiguration(service_config_id=config.id)
                
                resilience = config.resilience_config
                if "max_retries" in resilience_data:
                    resilience.max_retries = resilience_data["max_retries"]
                if "retry_delay" in resilience_data:
                    resilience.retry_delay = resilience_data["retry_delay"]
                if "backoff_factor" in resilience_data:
                    resilience.backoff_factor = resilience_data["backoff_factor"]
                if "circuit_breaker_failure_threshold" in resilience_data:
                    resilience.circuit_breaker_failure_threshold = resilience_data["circuit_breaker_failure_threshold"]
                if "circuit_breaker_reset_timeout" in resilience_data:
                    resilience.circuit_breaker_reset_timeout = resilience_data["circuit_breaker_reset_timeout"]
                if "timeout_seconds" in resilience_data:
                    resilience.timeout_seconds = resilience_data["timeout_seconds"]
            
            # Update observability configuration
            observability_data = config_data.get("observability")
            if observability_data:
                if not config.observability_config:
                    config.observability_config = ObservabilityConfiguration(service_config_id=config.id)
                
                observability = config.observability_config
                if "metrics_enabled" in observability_data:
                    observability.metrics_enabled = observability_data["metrics_enabled"]
                if "tracing_enabled" in observability_data:
                    observability.tracing_enabled = observability_data["tracing_enabled"]
                if "logging_level" in observability_data:
                    observability.logging_level = observability_data["logging_level"]
                if "export_metrics" in observability_data:
                    observability.export_metrics = observability_data["export_metrics"]
                if "metrics_export_url" in observability_data:
                    observability.metrics_export_url = observability_data["metrics_export_url"]
            
            # Update events configuration
            events_data = config_data.get("events")
            if events_data:
                if not config.events_config:
                    config.events_config = EventsConfiguration(service_config_id=config.id)
                
                events = config.events_config
                if "max_event_history" in events_data:
                    events.max_event_history = events_data["max_event_history"]
                if "publish_to_external" in events_data:
                    events.publish_to_external = events_data["publish_to_external"]
                if "external_event_url" in events_data:
                    events.external_event_url = events_data["external_event_url"]
                if "event_types_filter" in events_data:
                    events.event_types_filter = events_data["event_types_filter"]
            
            # Update progress tracking configuration
            progress_tracking_data = config_data.get("progress_tracking")
            if progress_tracking_data:
                if not config.progress_tracking_config:
                    config.progress_tracking_config = ProgressTrackingConfiguration(service_config_id=config.id)
                
                progress = config.progress_tracking_config
                if "max_active_operations" in progress_tracking_data:
                    progress.max_active_operations = progress_tracking_data["max_active_operations"]
                if "operation_ttl_seconds" in progress_tracking_data:
                    progress.operation_ttl_seconds = progress_tracking_data["operation_ttl_seconds"]
                if "publish_updates" in progress_tracking_data:
                    progress.publish_updates = progress_tracking_data["publish_updates"]
            
            self.db.commit()
            self.db.refresh(config)
            
            return config
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error updating service configuration: {e}")
            return None
    
    def delete_configuration(self, config_id: int) -> bool:
        """
        Delete a service configuration.
        
        Args:
            config_id: ID of the configuration to delete
            
        Returns:
            True if deleted, False if not found or error
        """
        try:
            # Get configuration
            config = self.get_configuration_by_id(config_id)
            
            if not config:
                return False
            
            # Delete configuration
            self.db.delete(config)
            self.db.commit()
            
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Error deleting service configuration: {e}")
            return False
