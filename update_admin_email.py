#!/usr/bin/env python
"""
Script to update the admin username to admin@conexus.ai in the database.
"""

import os
import sys
import argparse
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def update_admin_username(db_url=None, username=None, password=None, host=None, port=None, db_name=None):
    """Update the admin username to admin@conexus.ai."""
    try:
        # If a complete database URL is provided, use it
        if db_url:
            database_url = db_url
        # Otherwise construct it from the individual parameters
        elif all([username, host, db_name]):
            port_str = f":{port}" if port else ""
            password_str = f":{password}" if password else ""
            database_url = f"mysql+pymysql://{username}{password_str}@{host}{port_str}/{db_name}"
        else:
            # Try to import from the application configuration
            try:
                # First attempt to use the bollm backend configuration
                sys.path.append(os.path.join(os.path.dirname(__file__), 'asf', 'bollm', 'backend'))
                from config.config import SQLALCHEMY_DATABASE_URI as database_url
            except ImportError:
                try:
                    # Second attempt to use the bo backend configuration
                    sys.path.append(os.path.join(os.path.dirname(__file__), 'asf', 'bo', 'backend'))
                    from config.database import SQLALCHEMY_DATABASE_URL as database_url
                except ImportError:
                    print("Error: Could not determine database URL from configuration files.")
                    print("Please provide database connection parameters using the command line arguments.")
                    return False
        
        print(f"Connecting to database: {database_url.split('@')[-1] if '@' in database_url else database_url}")
        
        # Create a database engine and session - use raw SQL queries
        # to avoid ORM mapper conflicts between different User classes
        engine = create_engine(database_url)
        connection = engine.connect()
        
        # Use a transaction to ensure all changes are atomic
        with connection.begin():
            # Check if the admin user exists
            result = connection.execute(text("SELECT id, username, email FROM users WHERE username = 'admin'"))
            admin_user = result.fetchone()

            if admin_user:
                user_id = admin_user[0]
                old_username = admin_user[1]
                old_email = admin_user[2]
                
                print(f"Found admin user with ID {user_id}")
                print(f"Current username: {old_username}")
                print(f"Current email: {old_email}")

                # Update both username and email for consistency
                connection.execute(
                    text("UPDATE users SET username = :new_username, email = :new_email WHERE id = :user_id"),
                    {"new_username": "admin@conexus.ai", "new_email": "admin@conexus.ai", "user_id": user_id}
                )
                
                print("Successfully updated admin user:")
                print(f"Username changed from '{old_username}' to 'admin@conexus.ai'")
                print(f"Email changed from '{old_email}' to 'admin@conexus.ai'")
            else:
                # If we can't find a user with username 'admin', try to find one with role_id=2 (admin role)
                result = connection.execute(text("SELECT id, username, email FROM users WHERE role_id = 2 LIMIT 1"))
                admin_by_role = result.fetchone()
                
                if admin_by_role:
                    user_id = admin_by_role[0]
                    old_username = admin_by_role[1]
                    old_email = admin_by_role[2]
                    
                    print(f"Found admin user by role with ID {user_id}")
                    print(f"Current username: {old_username}")
                    print(f"Current email: {old_email}")

                    # Update both username and email for consistency
                    connection.execute(
                        text("UPDATE users SET username = :new_username, email = :new_email WHERE id = :user_id"),
                        {"new_username": "admin@conexus.ai", "new_email": "admin@conexus.ai", "user_id": user_id}
                    )
                    
                    print("Successfully updated admin user:")
                    print(f"Username changed from '{old_username}' to 'admin@conexus.ai'")
                    print(f"Email changed from '{old_email}' to 'admin@conexus.ai'")
                else:
                    print("No admin user found. Make sure the database is properly set up.")

        # Close the connection
        connection.close()

    except Exception as e:
        print(f"Error updating admin username: {e}")
        return False

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the admin username to admin@conexus.ai")
    
    # Database connection options
    parser.add_argument("--db-url", help="Complete database URL (e.g., mysql+pymysql://user:pass@host/dbname)")
    parser.add_argument("--username", help="Database username")
    parser.add_argument("--password", help="Database password")
    parser.add_argument("--host", help="Database host address")
    parser.add_argument("--port", help="Database port", type=int)
    parser.add_argument("--db-name", help="Database name")
    
    args = parser.parse_args()
    
    print("Updating admin username to admin@conexus.ai...")
    success = update_admin_username(
        db_url=args.db_url,
        username=args.username,
        password=args.password,
        host=args.host,
        port=args.port,
        db_name=args.db_name
    )
    
    if success:
        print("Admin username update completed successfully.")
    else:
        print("Failed to update admin username.")