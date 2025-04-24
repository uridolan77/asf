"""
Setup script for the LLM Service Configuration database.

This script creates all the necessary tables for the LLM Service Configuration system
by executing the SQL script in create_service_config_tables.sql.
"""

import os
import sys
import logging
import argparse
import mysql.connector
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database(host, port, user, password, database, sql_file):
    """
    Set up the database by executing the SQL script.
    
    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        database: Database name
        sql_file: Path to the SQL script file
    """
    # Check if SQL file exists
    if not os.path.exists(sql_file):
        logger.error(f"SQL file not found: {sql_file}")
        return False
    
    # Read SQL script
    with open(sql_file, 'r') as f:
        sql_script = f.read()
    
    # Split script into individual statements
    statements = sql_script.split(';')
    
    try:
        # Connect to the database
        logger.info(f"Connecting to MySQL database {database} on {host}:{port} as {user}")
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        logger.info(f"Creating database {database} if it doesn't exist")
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        
        # Use the database
        cursor.execute(f"USE {database}")
        
        # Execute each statement
        for statement in statements:
            # Skip empty statements
            if statement.strip():
                try:
                    cursor.execute(statement)
                    conn.commit()
                except Exception as e:
                    logger.error(f"Error executing statement: {e}")
                    logger.error(f"Statement: {statement}")
        
        logger.info("Database setup completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False
    
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Set up the LLM Service Configuration database')
    parser.add_argument('--host', default=os.getenv('BO_DB_HOST', 'localhost'), help='Database host')
    parser.add_argument('--port', type=int, default=int(os.getenv('BO_DB_PORT', '3306')), help='Database port')
    parser.add_argument('--user', default=os.getenv('BO_DB_USER', 'root'), help='Database user')
    parser.add_argument('--password', default=os.getenv('BO_DB_PASSWORD', 'Dt%g_9W3z0*!I'), help='Database password')
    parser.add_argument('--database', default=os.getenv('BO_DB_NAME', 'bo_admin'), help='Database name')
    parser.add_argument('--sql-file', default=os.path.join(os.path.dirname(__file__), 'create_service_config_tables.sql'), help='Path to SQL script file')
    
    args = parser.parse_args()
    
    # Set up the database
    success = setup_database(
        args.host,
        args.port,
        args.user,
        args.password,
        args.database,
        args.sql_file
    )
    
    if success:
        logger.info("Database setup completed successfully")
        sys.exit(0)
    else:
        logger.error("Database setup failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
