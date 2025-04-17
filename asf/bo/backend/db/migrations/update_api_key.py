#!/usr/bin/env python
# update_api_key.py

"""
Script to update the API key for a provider in the database.
"""

import os
import sys
import logging
import mysql.connector
from mysql.connector import Error
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def connect_to_database(host, user, password, database):
    """Connect to the database."""
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

def update_api_key(connection, provider_id, api_key, encryption_key=None):
    """Update the API key for a provider."""
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Check if the provider exists
        cursor.execute("SELECT * FROM providers WHERE provider_id = %s", (provider_id,))
        provider = cursor.fetchone()
        if not provider:
            logger.error(f"Provider {provider_id} not found")
            return False
        
        # Check if the API key exists
        cursor.execute("SELECT * FROM api_keys WHERE provider_id = %s", (provider_id,))
        existing_key = cursor.fetchone()
        
        # Get current timestamp
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        if existing_key:
            # Update the API key
            cursor.execute(
                "UPDATE api_keys SET key_value = %s, is_encrypted = %s, updated_at = %s WHERE provider_id = %s",
                (api_key, 0, now, provider_id)
            )
            logger.info(f"Updated API key for provider {provider_id}")
        else:
            # Create a new API key
            cursor.execute(
                "INSERT INTO api_keys (provider_id, key_value, is_encrypted, environment, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (provider_id, api_key, 0, "development", now, now)
            )
            logger.info(f"Created new API key for provider {provider_id}")
        
        connection.commit()
        return True
    except Error as e:
        logger.error(f"Error updating API key: {e}")
        connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Update API key for a provider")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--user", required=True, help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--database", required=True, help="Database name")
    parser.add_argument("--provider-id", required=True, help="Provider ID")
    parser.add_argument("--api-key", required=True, help="API key")
    parser.add_argument("--encryption-key", help="Encryption key")
    
    args = parser.parse_args()
    
    # Connect to the database
    connection = connect_to_database(args.host, args.user, args.password, args.database)
    if not connection:
        sys.exit(1)
    
    try:
        # Update the API key
        if not update_api_key(connection, args.provider_id, args.api_key, args.encryption_key):
            sys.exit(1)
        
        logger.info("API key updated successfully")
    
    except Exception as e:
        logger.error(f"Error during update: {e}")
        sys.exit(1)
    
    finally:
        if connection and connection.is_connected():
            connection.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
