#!/usr/bin/env python
# migrate_config_to_db.py

"""
Migration script to move configuration from YAML files to the database.
"""

import os
import sys
import yaml
import mysql.connector
from mysql.connector import Error
import argparse
from pathlib import Path
import logging
from cryptography.fernet import Fernet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         "config", "llm")
GATEWAY_CONFIG_PATH = os.path.join(CONFIG_DIR, "llm_gateway_config.yaml")
LOCAL_CONFIG_PATH = os.path.join(CONFIG_DIR, "llm_gateway_config.local.yaml")

# Generate a key for encryption (in production, this should be stored securely)
def generate_key():
    return Fernet.generate_key()

# Encrypt sensitive data
def encrypt_value(value, key):
    f = Fernet(key)
    return f.encrypt(value.encode()).decode()

# Decrypt sensitive data
def decrypt_value(encrypted_value, key):
    f = Fernet(key)
    return f.decrypt(encrypted_value.encode()).decode()

# Load configuration from YAML files
def load_config():
    # Load base configuration
    with open(GATEWAY_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if local configuration exists
    if os.path.exists(LOCAL_CONFIG_PATH):
        try:
            with open(LOCAL_CONFIG_PATH, 'r') as f:
                local_config = yaml.safe_load(f)
            
            # Merge configurations
            if local_config:
                logger.info(f"Merging local configuration from {LOCAL_CONFIG_PATH}")
                config = deep_merge(config, local_config)
        except Exception as e:
            logger.warning(f"Error loading local configuration: {str(e)}")
    
    return config

# Deep merge two dictionaries
def deep_merge(base, override):
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # If both values are dictionaries, merge them recursively
            result[key] = deep_merge(result[key], value)
        else:
            # Otherwise, use the value from the override dictionary
            result[key] = value
    
    return result

# Connect to the database
def connect_to_database(host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        if connection.is_connected():
            logger.info(f"Connected to MySQL database: {database}")
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        return None

# Execute SQL script
def execute_sql_script(connection, script_path):
    try:
        cursor = connection.cursor()
        
        # Read the SQL script
        with open(script_path, 'r') as f:
            sql_script = f.read()
        
        # Split the script into individual statements
        statements = sql_script.split(';')
        
        # Execute each statement
        for statement in statements:
            if statement.strip():
                cursor.execute(statement)
        
        connection.commit()
        logger.info(f"Executed SQL script: {script_path}")
        return True
    except Error as e:
        logger.error(f"Error executing SQL script: {e}")
        return False
    finally:
        if cursor:
            cursor.close()

# Migrate provider configuration
def migrate_provider_config(connection, config, encryption_key):
    try:
        cursor = connection.cursor()
        
        # Get providers from configuration
        providers = config.get("additional_config", {}).get("providers", {})
        
        for provider_id, provider_config in providers.items():
            # Check if provider already exists
            cursor.execute("SELECT provider_id FROM providers WHERE provider_id = %s", (provider_id,))
            if cursor.fetchone():
                logger.info(f"Provider {provider_id} already exists, skipping...")
                continue
            
            # Insert provider
            display_name = provider_config.get("display_name", provider_id)
            provider_type = provider_config.get("provider_type", "unknown")
            
            cursor.execute(
                "INSERT INTO providers (provider_id, display_name, provider_type, enabled) VALUES (%s, %s, %s, %s)",
                (provider_id, display_name, provider_type, True)
            )
            
            # Insert models
            models = provider_config.get("models", {})
            for model_id, model_config in models.items():
                cursor.execute(
                    "INSERT INTO provider_models (model_id, provider_id, display_name, enabled) VALUES (%s, %s, %s, %s)",
                    (model_id, provider_id, model_id, True)
                )
            
            # Insert connection parameters
            connection_params = provider_config.get("connection_params", {})
            for param_name, param_value in connection_params.items():
                # Skip API key, we'll handle it separately
                if param_name == "api_key":
                    continue
                
                is_sensitive = param_name.endswith("_secret") or "password" in param_name or "token" in param_name
                
                cursor.execute(
                    "INSERT INTO connection_parameters (provider_id, param_name, param_value, is_sensitive) VALUES (%s, %s, %s, %s)",
                    (provider_id, param_name, str(param_value), is_sensitive)
                )
            
            # Handle API key
            api_key = connection_params.get("api_key")
            if api_key:
                # Encrypt the API key
                encrypted_key = encrypt_value(api_key, encryption_key)
                
                cursor.execute(
                    "INSERT INTO api_keys (provider_id, key_value, is_encrypted) VALUES (%s, %s, %s)",
                    (provider_id, encrypted_key, True)
                )
        
        connection.commit()
        logger.info("Provider configuration migrated successfully")
        return True
    except Error as e:
        logger.error(f"Error migrating provider configuration: {e}")
        connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Migrate configuration from YAML to database")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--user", required=True, help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--create-tables", action="store_true", help="Create tables")
    parser.add_argument("--drop-tables", action="store_true", help="Drop tables")
    parser.add_argument("--migrate-data", action="store_true", help="Migrate data")
    parser.add_argument("--encryption-key", help="Encryption key for sensitive data")
    
    args = parser.parse_args()
    
    # Connect to the database
    connection = connect_to_database(args.host, args.user, args.password, args.database)
    if not connection:
        sys.exit(1)
    
    try:
        # Generate or use provided encryption key
        encryption_key = args.encryption_key
        if not encryption_key:
            encryption_key = generate_key()
            logger.info(f"Generated encryption key: {encryption_key.decode()}")
            logger.warning("Store this key securely! You'll need it to decrypt sensitive data.")
        else:
            encryption_key = encryption_key.encode()
        
        # Drop tables if requested
        if args.drop_tables:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drop_tables.sql")
            if not execute_sql_script(connection, script_path):
                sys.exit(1)
        
        # Create tables if requested
        if args.create_tables:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "create_tables.sql")
            if not execute_sql_script(connection, script_path):
                sys.exit(1)
        
        # Migrate data if requested
        if args.migrate_data:
            # Load configuration
            config = load_config()
            
            # Migrate provider configuration
            if not migrate_provider_config(connection, config, encryption_key):
                sys.exit(1)
            
            # Execute seed script
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_initial_data.sql")
            if not execute_sql_script(connection, script_path):
                sys.exit(1)
        
        logger.info("Migration completed successfully")
    
    except Exception as e:
        logger.error(f"Error during migration: {e}")
        sys.exit(1)
    
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
