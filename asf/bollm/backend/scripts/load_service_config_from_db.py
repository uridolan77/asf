"""
Load service configuration from the database.

This script retrieves the service configuration from the database and prints it.
It can be used to verify that the configuration was properly created and to load
it into the service.
"""

import os
import sys
import json
import logging
import argparse
import mysql.connector
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_configuration(host, port, user, password, database, config_id=None, alternative_passwords=None):
    """
    Load service configuration from the database.
    
    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        database: Database name
        config_id: ID of the configuration to load (optional)
        alternative_passwords: List of alternative passwords to try if the first one fails (optional)
    
    Returns:
        Dictionary containing the configuration
    """
    # Try with the provided password first
    passwords_to_try = [password]
    
    # Add alternative passwords if provided
    if alternative_passwords:
        passwords_to_try.extend(alternative_passwords)
    
    # Try each password
    conn = None
    cursor = None
    last_error = None
    
    for pwd in passwords_to_try:
        try:
            # Connect to the database
            logger.info(f"Connecting to MySQL database {database} on {host}:{port} as {user}")
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=pwd,
                database=database
            )
            cursor = conn.cursor(dictionary=True)
            
            # If we get here, the connection was successful
            if pwd != password:
                logger.info(f"Connected successfully using alternative password")
            
            # Connection successful, break the loop
            break
        
        except Exception as e:
            logger.warning(f"Failed to connect with password: {e}")
            last_error = e
            conn = None
            continue
    
    # If all passwords failed, return None
    if conn is None:
        logger.error(f"All connection attempts failed. Last error: {last_error}")
        return None
    
    try:
        # Query to get the main configuration
        if config_id:
            logger.info(f"Loading configuration with ID {config_id}")
            query = """
                SELECT * FROM service_configurations
                WHERE id = %s
            """
            cursor.execute(query, (config_id,))
        else:
            logger.info("Loading the default configuration")
            query = """
                SELECT * FROM service_configurations
                WHERE is_public = TRUE
                ORDER BY id ASC
                LIMIT 1
            """
            cursor.execute(query)
        
        # Get the main configuration
        config = cursor.fetchone()
        if not config:
            logger.error("No configuration found")
            return None
        
        logger.info(f"Found configuration: {config['name']} (ID: {config['id']})")
        
        # Get the configuration ID
        config_id = config['id']
        
        # Get the caching configuration
        query = """
            SELECT * FROM caching_configurations
            WHERE service_config_id = %s
        """
        cursor.execute(query, (config_id,))
        caching_config = cursor.fetchone()
        
        # Get the resilience configuration
        query = """
            SELECT * FROM resilience_configurations
            WHERE service_config_id = %s
        """
        cursor.execute(query, (config_id,))
        resilience_config = cursor.fetchone()
        
        # Get the observability configuration
        query = """
            SELECT * FROM observability_configurations
            WHERE service_config_id = %s
        """
        cursor.execute(query, (config_id,))
        observability_config = cursor.fetchone()
        
        # Get the events configuration
        query = """
            SELECT * FROM events_configurations
            WHERE service_config_id = %s
        """
        cursor.execute(query, (config_id,))
        events_config = cursor.fetchone()
        
        # Get the progress tracking configuration
        query = """
            SELECT * FROM progress_tracking_configurations
            WHERE service_config_id = %s
        """
        cursor.execute(query, (config_id,))
        progress_tracking_config = cursor.fetchone()
        
        # Combine all configurations
        full_config = {
            "id": config['id'],
            "service_id": config['service_id'],
            "name": config['name'],
            "description": config['description'],
            "enable_caching": bool(config['enable_caching']),
            "enable_resilience": bool(config['enable_resilience']),
            "enable_observability": bool(config['enable_observability']),
            "enable_events": bool(config['enable_events']),
            "enable_progress_tracking": bool(config['enable_progress_tracking']),
            "is_public": bool(config['is_public']),
            "created_by_user_id": config['created_by_user_id'],
            "created_at": config['created_at'].isoformat() if config['created_at'] else None,
            "updated_at": config['updated_at'].isoformat() if config['updated_at'] else None,
            "config": {
                "cache": {
                    "similarity_threshold": float(caching_config['similarity_threshold']) if caching_config else 0.92,
                    "max_entries": int(caching_config['max_entries']) if caching_config else 10000,
                    "ttl_seconds": int(caching_config['ttl_seconds']) if caching_config else 3600,
                    "persistence_type": caching_config['persistence_type'] if caching_config else "disk",
                    "persistence_config": json.loads(caching_config['persistence_config']) if caching_config and caching_config['persistence_config'] else None
                },
                "resilience": {
                    "max_retries": int(resilience_config['max_retries']) if resilience_config else 3,
                    "retry_delay": float(resilience_config['retry_delay']) if resilience_config else 1.0,
                    "backoff_factor": float(resilience_config['backoff_factor']) if resilience_config else 2.0,
                    "circuit_breaker_failure_threshold": int(resilience_config['circuit_breaker_failure_threshold']) if resilience_config else 5,
                    "circuit_breaker_reset_timeout": int(resilience_config['circuit_breaker_reset_timeout']) if resilience_config else 30,
                    "timeout_seconds": float(resilience_config['timeout_seconds']) if resilience_config else 30.0
                },
                "observability": {
                    "metrics_enabled": bool(observability_config['metrics_enabled']) if observability_config else True,
                    "tracing_enabled": bool(observability_config['tracing_enabled']) if observability_config else True,
                    "logging_level": observability_config['logging_level'] if observability_config else "INFO",
                    "export_metrics": bool(observability_config['export_metrics']) if observability_config else False,
                    "metrics_export_url": observability_config['metrics_export_url'] if observability_config else None
                },
                "events": {
                    "max_event_history": int(events_config['max_event_history']) if events_config else 100,
                    "publish_to_external": bool(events_config['publish_to_external']) if events_config else False,
                    "external_event_url": events_config['external_event_url'] if events_config else None,
                    "event_types_filter": json.loads(events_config['event_types_filter']) if events_config and events_config['event_types_filter'] else None
                },
                "progress_tracking": {
                    "max_active_operations": int(progress_tracking_config['max_active_operations']) if progress_tracking_config else 100,
                    "operation_ttl_seconds": int(progress_tracking_config['operation_ttl_seconds']) if progress_tracking_config else 3600,
                    "publish_updates": bool(progress_tracking_config['publish_updates']) if progress_tracking_config else True
                }
            }
        }
        
        return full_config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None
    
    finally:
        if conn is not None and conn.is_connected():
            if cursor is not None:
                cursor.close()
            conn.close()

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load service configuration from the database')
    parser.add_argument('--host', default=os.getenv('BO_DB_HOST', 'localhost'), help='Database host')
    parser.add_argument('--port', type=int, default=int(os.getenv('BO_DB_PORT', '3306')), help='Database port')
    parser.add_argument('--user', default=os.getenv('BO_DB_USER', 'root'), help='Database user')
    parser.add_argument('--password', default=os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I'), help='Database password')
    parser.add_argument('--database', default=os.getenv('BO_DB_NAME', 'bo_admin'), help='Database name')
    parser.add_argument('--config-id', type=int, help='ID of the configuration to load')
    parser.add_argument('--output', help='Output file path (optional)')
    parser.add_argument('--try-passwords', nargs='+', help='Additional passwords to try')
    
    args = parser.parse_args()
    
    # Get alternative passwords
    alternative_passwords = args.try_passwords or [
        '',               # Try empty password
        'Dt%g_9W3z0*!I',  # Try the complex password
        'root'            # Try username as password
    ]
    
    # Load the configuration
    config = load_configuration(
        args.host,
        args.port,
        args.user,
        args.password,
        args.database,
        args.config_id,
        alternative_passwords
    )
    
    if config:
        # Print the configuration
        if args.output:
            # Write to file
            with open(args.output, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Configuration written to {args.output}")
        else:
            # Print to console
            print(json.dumps(config, indent=2))
        
        logger.info("Configuration loaded successfully")
        return 0
    else:
        logger.error("Failed to load configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())
