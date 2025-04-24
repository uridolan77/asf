"""
Check and initialize the database for the LLM Gateway.

This script checks if the database tables exist and if the openai_gpt4_default provider
is properly configured. If not, it creates the tables and seeds the database with
the necessary data.
"""

import os
import sys
import logging
import argparse
import mysql.connector
from pathlib import Path

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
        logger.info(f"Connected to database {database} on {host}")
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Error connecting to database: {e}")
        return None

def check_tables_exist(connection):
    """Check if the necessary tables exist."""
    cursor = connection.cursor()
    try:
        cursor.execute("SHOW TABLES")
        tables = [table[0] for table in cursor.fetchall()]
        
        required_tables = [
            "providers",
            "provider_models",
            "connection_parameters",
            "configurations"
        ]
        
        missing_tables = [table for table in required_tables if table not in tables]
        
        if missing_tables:
            logger.warning(f"Missing tables: {', '.join(missing_tables)}")
            return False
        
        logger.info("All required tables exist")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Error checking tables: {e}")
        return False
    finally:
        cursor.close()

def check_provider_exists(connection, provider_id):
    """Check if a provider exists in the database."""
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT provider_id FROM providers WHERE provider_id = %s", (provider_id,))
        result = cursor.fetchone()
        
        if result:
            logger.info(f"Provider {provider_id} exists in the database")
            return True
        
        logger.warning(f"Provider {provider_id} does not exist in the database")
        return False
    except mysql.connector.Error as e:
        logger.error(f"Error checking provider: {e}")
        return False
    finally:
        cursor.close()

def execute_sql_script(connection, script_path):
    """Execute an SQL script."""
    cursor = connection.cursor()
    try:
        logger.info(f"Executing SQL script: {script_path}")
        
        with open(script_path, 'r') as f:
            script = f.read()
        
        # Split the script into individual statements
        statements = script.split(';')
        
        for statement in statements:
            statement = statement.strip()
            if statement:
                cursor.execute(statement)
        
        connection.commit()
        logger.info(f"SQL script executed successfully: {script_path}")
        return True
    except mysql.connector.Error as e:
        logger.error(f"Error executing SQL script: {e}")
        connection.rollback()
        return False
    except Exception as e:
        logger.error(f"Error reading or parsing SQL script: {e}")
        return False
    finally:
        cursor.close()

def create_tables(connection):
    """Create the necessary tables."""
    script_path = os.path.join("asf", "bo", "backend", "db", "migrations", "create_tables.sql")
    return execute_sql_script(connection, script_path)

def seed_database(connection):
    """Seed the database with initial data."""
    script_path = os.path.join("asf", "bo", "backend", "db", "migrations", "seed_initial_data.sql")
    return execute_sql_script(connection, script_path)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Check and initialize the database for the LLM Gateway")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--user", default="root", help="Database user")
    parser.add_argument("--password", default="Dt%g_9W3z0*!I", help="Database password")
    parser.add_argument("--database", default="bo_admin", help="Database name")
    parser.add_argument("--force", action="store_true", help="Force initialization even if tables exist")
    
    args = parser.parse_args()
    
    # Connect to the database
    connection = connect_to_database(args.host, args.user, args.password, args.database)
    if not connection:
        sys.exit(1)
    
    try:
        # Check if tables exist
        tables_exist = check_tables_exist(connection)
        
        # Check if provider exists
        provider_exists = check_provider_exists(connection, "openai_gpt4_default")
        
        # Create tables and seed database if necessary
        if args.force or not tables_exist:
            logger.info("Creating tables...")
            if not create_tables(connection):
                sys.exit(1)
        
        if args.force or not provider_exists:
            logger.info("Seeding database...")
            if not seed_database(connection):
                sys.exit(1)
        
        logger.info("Database check and initialization completed successfully")
    
    except Exception as e:
        logger.error(f"Error during database check and initialization: {e}")
        sys.exit(1)
    
    finally:
        if connection:
            connection.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()
