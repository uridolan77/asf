#!/usr/bin/env python
# populate_users_providers.py

"""
Migration script to populate the users_providers association table.
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

def populate_users_providers(connection):
    """Populate the users_providers table."""
    try:
        cursor = connection.cursor()
        
        # Get all users
        cursor.execute("SELECT id FROM users")
        users = cursor.fetchall()
        
        # Get all providers
        cursor.execute("SELECT provider_id FROM providers")
        providers = cursor.fetchall()
        
        if not users:
            logger.warning("No users found in the database")
            return False
        
        if not providers:
            logger.warning("No providers found in the database")
            return False
        
        logger.info(f"Found {len(users)} users and {len(providers)} providers")
        
        # Assign each user to each provider with a role
        for user in users:
            user_id = user[0]
            for provider in providers:
                provider_id = provider[0]
                
                # Check if the assignment already exists
                cursor.execute(
                    "SELECT * FROM users_providers WHERE user_id = %s AND provider_id = %s",
                    (user_id, provider_id)
                )
                if not cursor.fetchone():
                    # Assign the user to the provider
                    cursor.execute(
                        "INSERT INTO users_providers (user_id, provider_id, role) VALUES (%s, %s, %s)",
                        (user_id, provider_id, "user")
                    )
                    logger.info(f"Assigned user {user_id} to provider {provider_id} with role 'user'")
        
        # Make the first user an admin for all providers
        if users:
            admin_user_id = users[0][0]
            for provider in providers:
                provider_id = provider[0]
                cursor.execute(
                    "UPDATE users_providers SET role = 'admin' WHERE user_id = %s AND provider_id = %s",
                    (admin_user_id, provider_id)
                )
                logger.info(f"Updated user {admin_user_id} to role 'admin' for provider {provider_id}")
        
        connection.commit()
        logger.info("Successfully populated users_providers table")
        return True
    except Exception as e:
        logger.error(f"Error populating users_providers table: {e}")
        connection.rollback()
        return False
    finally:
        if cursor:
            cursor.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Populate users_providers table")
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
        # Populate the users_providers table
        if not populate_users_providers(connection):
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
