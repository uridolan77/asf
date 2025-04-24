#!/usr/bin/env python
"""
Setup script for initializing the Conexus database.

This script connects to MySQL, creates the necessary database and tables for Conexus,
and sets up initial data if needed.
"""

import mysql.connector
import os
import sys
from mysql.connector import Error

def connect_to_mysql(host='localhost', user='root', password=None):
    """Connect to the MySQL server."""
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        print(f"Successfully connected to MySQL Server as {user}")
        return connection
    except Error as e:
        print(f"Error connecting to MySQL Server: {e}")
        return None

def create_database(connection, db_name='conexus_db'):
    """Create a database if it doesn't exist."""
    try:
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
        print(f"Database '{db_name}' created successfully")
        cursor.close()
        return True
    except Error as e:
        print(f"Error creating database: {e}")
        return False

def create_tables(connection, db_name='conexus_db'):
    """Create necessary tables for the application."""
    try:
        cursor = connection.cursor()
        
        # Select the database
        cursor.execute(f"USE {db_name}")
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role_id INT NOT NULL DEFAULT 2,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """)
        
        # Create roles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS roles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50) UNIQUE NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create providers table (for LLM providers)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS providers (
            provider_id VARCHAR(50) PRIMARY KEY,
            display_name VARCHAR(100) NOT NULL,
            provider_type VARCHAR(50) NOT NULL,
            description TEXT,
            enabled BOOLEAN DEFAULT TRUE,
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            created_by_user_id INT,
            FOREIGN KEY (created_by_user_id) REFERENCES users(id)
        )
        """)
        
        # Create provider models table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS provider_models (
            model_id VARCHAR(50) PRIMARY KEY,
            provider_id VARCHAR(50) NOT NULL,
            display_name VARCHAR(100) NOT NULL,
            model_type VARCHAR(50),
            context_window INT,
            max_tokens INT,
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            FOREIGN KEY (provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE
        )
        """)
        
        # Create API keys table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_id INT AUTO_INCREMENT PRIMARY KEY,
            provider_id VARCHAR(50) NOT NULL,
            key_value TEXT NOT NULL,
            is_encrypted BOOLEAN DEFAULT TRUE,
            environment VARCHAR(20) DEFAULT 'development',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NULL,
            created_by_user_id INT,
            FOREIGN KEY (provider_id) REFERENCES providers(provider_id) ON DELETE CASCADE,
            FOREIGN KEY (created_by_user_id) REFERENCES users(id)
        )
        """)
        
        # Create configurations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS configurations (
            config_id INT AUTO_INCREMENT PRIMARY KEY,
            config_key VARCHAR(100) NOT NULL,
            config_value TEXT,
            config_type VARCHAR(20) DEFAULT 'string',
            description TEXT,
            environment VARCHAR(20) DEFAULT 'development',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            created_by_user_id INT,
            FOREIGN KEY (created_by_user_id) REFERENCES users(id),
            UNIQUE (config_key, environment)
        )
        """)
        
        # Insert default roles
        cursor.execute("INSERT IGNORE INTO roles (id, name, description) VALUES (1, 'admin', 'Administrator with full access')")
        cursor.execute("INSERT IGNORE INTO roles (id, name, description) VALUES (2, 'user', 'Regular user with limited access')")
        
        # Create default admin user with password 'admin123'
        cursor.execute("""
        INSERT IGNORE INTO users (username, email, password_hash, role_id)
        VALUES ('admin', 'admin@example.com', '$2b$12$gPaJXmU/Dfv8dg0wSt0wS.MVdxIJ0/at9lajJvIqmUPJGJQlSLGGG', 1)
        """)
        
        connection.commit()
        print("Tables created successfully")
        cursor.close()
        return True
    except Error as e:
        print(f"Error creating tables: {e}")
        connection.rollback()
        return False

def setup_conexus_database():
    """Main function to set up the Conexus database."""
    print("Setting up Conexus database...")
    
    try:
        password = getpass.getpass("Enter MySQL root password: ")
        
        # Connect to MySQL
        connection = connect_to_mysql(password=password)
        if not connection:
            print("Failed to connect to MySQL. Exiting...")
            return False
        
        # Create database
        db_name = input("Enter database name (default: conexus_db): ") or "conexus_db"
        if not create_database(connection, db_name):
            print("Failed to create database. Exiting...")
            connection.close()
            return False
        
        # Create tables
        if not create_tables(connection, db_name):
            print("Failed to create tables. Exiting...")
            connection.close()
            return False
        
        # Create database user
        db_user = input("Enter new database user (default: conexus_user): ") or "conexus_user"
        db_user_password = getpass.getpass("Enter password for new database user: ")
        
        cursor = connection.cursor()
        
        # Create user if not exists and grant permissions
        cursor.execute(f"CREATE USER IF NOT EXISTS '{db_user}'@'localhost' IDENTIFIED BY '{db_user_password}'")
        cursor.execute(f"GRANT ALL PRIVILEGES ON {db_name}.* TO '{db_user}'@'localhost'")
        cursor.execute("FLUSH PRIVILEGES")
        
        connection.commit()
        print(f"User '{db_user}' created with access to database '{db_name}'")
        
        # Close the connection
        cursor.close()
        connection.close()
        
        print("\nDatabase setup completed successfully!")
        print(f"Database: {db_name}")
        print(f"User: {db_user}")
        print("You can now configure your application to use these credentials.")
        print("\nDefault admin user created:")
        print("Username: admin")
        print("Password: admin123")
        
        return True
    
    except Exception as e:
        print(f"An error occurred during database setup: {e}")
        return False

if __name__ == "__main__":
    setup_conexus_database()