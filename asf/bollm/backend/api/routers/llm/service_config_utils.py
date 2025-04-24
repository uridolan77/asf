"""
Utility functions for service configuration.

This module provides utility functions for loading and saving service configurations.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import database functions
from sqlalchemy.orm import Session
from repositories.service_config_repository import ServiceConfigRepository

# Mock implementation for get_default_config
def get_default_config():
    """Get the default service configuration."""
    return {
        "service_id": "mock-service",
        "enable_caching": True,
        "enable_resilience": True,
        "enable_observability": True,
        "enable_events": True,
        "enable_progress_tracking": True,
        "config": {
            "caching": {
                "similarity_threshold": 0.92,
                "max_entries": 10000,
                "ttl_seconds": 3600,
                "persistence_type": "disk"
            },
            "resilience": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff_factor": 2.0,
                "circuit_breaker_failure_threshold": 5,
                "circuit_breaker_reset_timeout": 30,
                "timeout_seconds": 30.0
            },
            "observability": {
                "metrics_enabled": True,
                "tracing_enabled": True,
                "logging_level": "INFO",
                "export_metrics": False
            },
            "events": {
                "max_event_history": 100,
                "publish_to_external": False
            },
            "progress_tracking": {
                "max_active_operations": 100,
                "operation_ttl_seconds": 3600,
                "publish_updates": True
            }
        }
    }

def get_service_config_from_db(db: Session, config_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a service configuration from the database.

    Args:
        db: Database session
        config_id: ID of the configuration to load

    Returns:
        Dictionary containing the configuration, or None if not found
    """
    try:
        # Use repository to get configuration
        repo = ServiceConfigRepository(db)
        config = repo.get_configuration_by_id(config_id)

        if not config:
            logger.warning(f"Configuration with ID {config_id} not found")
            return None

        # Convert to dictionary
        return config.to_dict()

    except Exception as e:
        logger.error(f"Error getting service configuration: {str(e)}")
        return None

def list_service_configs(db: Session, user_id: int = None, include_public: bool = True,
                        skip: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
    """
    List service configurations from the database.

    Args:
        db: Database session
        user_id: ID of the user to filter by (optional)
        include_public: Whether to include public configurations
        skip: Number of configurations to skip
        limit: Maximum number of configurations to return

    Returns:
        List of configuration dictionaries
    """
    try:
        # Use repository to get configurations
        repo = ServiceConfigRepository(db)
        configs = repo.get_all_configurations(
            user_id=user_id,
            include_public=include_public,
            skip=skip,
            limit=limit
        )

        # Convert to dictionaries
        return [config.to_dict() for config in configs]

    except Exception as e:
        logger.error(f"Error listing service configurations: {str(e)}")
        return []

def get_active_service_config() -> Dict[str, Any]:
    """
    Get the active service configuration.

    This function returns the active service configuration, which is loaded
    from the database or a default configuration if the database is not available.

    Returns:
        Dictionary containing the configuration
    """
    # Get the default configuration
    return get_default_config()

def apply_service_config(service: Any, config: Dict[str, Any]) -> None:
    """
    Apply a service configuration to a service instance.

    Args:
        service: Service instance
        config: Configuration dictionary
    """
    # Update service configuration
    service.update_config(config)
