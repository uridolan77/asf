"""
Service configuration loader.

This module provides functions for loading service configurations from the database.
"""

import os
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the function to load configuration from database
from asf.bollm.backend.scripts.load_service_config_from_db import load_configuration

# Get database URL from environment variables or use defaults
# Try multiple password options to handle different environments
DB_USER = os.getenv('BO_DB_USER', 'root')
DB_PASSWORD = os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I')  # Use the correct password
DB_HOST = os.getenv('BO_DB_HOST', 'localhost')
DB_PORT = os.getenv('BO_DB_PORT', '3306')
DB_NAME = os.getenv('BO_DB_NAME', 'bo_admin')

# Alternative passwords to try if the first one fails
ALTERNATIVE_PASSWORDS = [
    '',               # Try empty password
    'password',       # Try simple password
    'root'            # Try username as password
]

def get_service_config() -> Dict[str, Any]:
    """
    Load service configuration from the database.
    
    Returns:
        Dictionary containing the configuration
    """
    config = load_configuration(
        DB_HOST,
        int(DB_PORT),
        DB_USER,
        DB_PASSWORD,
        DB_NAME,
        alternative_passwords=ALTERNATIVE_PASSWORDS
    )
    
    if config:
        logger.info(f"Loaded configuration '{config['name']}' (ID: {config['id']}) from database")
        return config
    else:
        logger.warning("Failed to load configuration from database, using default configuration")
        return {
            "service_id": "enhanced_llm_service",
            "name": "Default Configuration",
            "description": "Default configuration for the LLM service",
            "enable_caching": True,
            "enable_resilience": True,
            "enable_observability": True,
            "enable_events": True,
            "enable_progress_tracking": True,
            "config": {
                "cache": {
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
