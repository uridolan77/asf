#!/usr/bin/env python
# add_users_providers.py

"""
Migration script to add the users_providers association table.
"""

import os
import sys
import logging
import mysql.connector
from mysql.connector import Error
import argparse

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

def create_association_table(connection):
    """Create the users_providers association table."""
    try:
        cursor = connection.cursor()
        
        # Create the association table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users_providers (
            user_id INT NOT NULL,
            provider_id VARCHAR(50) NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, provider_id),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE
        );
        """)
        
        # Modify the providers table to make created_by_user_id nullable
        cursor.execute("""
        ALTER TABLE providers 
        MODIFY COLUMN created_by_user_id INT NULL;
        """)
        
        connection.commit()
        logger.info("Association table created successfully")
        return True
    except Error as e:
        logger.error(f"Error creating association table: {e}")
        connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Add users_providers association table")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--user", required=True, help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--database", required=True, help="Database name")
    
    args = parser.parse_args()
    
    # Connect to the database
    connection = connect_to_database(args.host, args.user, args.password, args.database)
    if not connection:
        sys.exit(1)
    
    try:
        # Create the association table
        if not create_association_table(connection):
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
