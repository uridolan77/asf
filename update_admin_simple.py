#!/usr/bin/env python
"""
Simple script to update the admin username to admin@conexus.ai in the database.
This script uses direct MySQL connection instead of SQLAlchemy to avoid mapper conflicts.
"""

import argparse
import pymysql
import os
import sys

def update_admin_username(host='localhost', port=3306, user='root', password='', db_name='conexus'):
    """Update the admin username to admin@conexus.ai."""
    try:
        print(f"Connecting to MySQL database {db_name} on {host}:{port} as {user}")
        
        # Connect directly to MySQL without SQLAlchemy
        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=db_name
        )
        
        with connection.cursor() as cursor:
            # Check if the admin user exists
            cursor.execute("SELECT id, username, email FROM users WHERE username = 'admin'")
            admin_user = cursor.fetchone()
            
            if admin_user:
                user_id, old_username, old_email = admin_user
                print(f"Found admin user with ID {user_id}")
                print(f"Current username: {old_username}")
                print(f"Current email: {old_email}")
                
                # Update both username and email for consistency
                cursor.execute(
                    "UPDATE users SET username = %s, email = %s WHERE id = %s",
                    ("admin@conexus.ai", "admin@conexus.ai", user_id)
                )
                connection.commit()
                
                print("Successfully updated admin user:")
                print(f"Username changed from '{old_username}' to 'admin@conexus.ai'")
                print(f"Email changed from '{old_email}' to 'admin@conexus.ai'")
            else:
                # If we can't find a user with username 'admin', try to find one with role_id=2 (admin role)
                cursor.execute("SELECT id, username, email FROM users WHERE role_id = 2 LIMIT 1")
                admin_by_role = cursor.fetchone()
                
                if admin_by_role:
                    user_id, old_username, old_email = admin_by_role
                    print(f"Found admin user by role with ID {user_id}")
                    print(f"Current username: {old_username}")
                    print(f"Current email: {old_email}")
                    
                    # Update both username and email for consistency
                    cursor.execute(
                        "UPDATE users SET username = %s, email = %s WHERE id = %s",
                        ("admin@conexus.ai", "admin@conexus.ai", user_id)
                    )
                    connection.commit()
                    
                    print("Successfully updated admin user:")
                    print(f"Username changed from '{old_username}' to 'admin@conexus.ai'")
                    print(f"Email changed from '{old_email}' to 'admin@conexus.ai'")
                else:
                    print("No admin user found. Make sure the database is properly set up.")
        
        # Close the connection
        connection.close()
        return True

    except Exception as e:
        print(f"Error updating admin username: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the admin username to admin@conexus.ai")
    
    # Database connection options
    parser.add_argument("--host", default="localhost", help="Database host address")
    parser.add_argument("--port", type=int, default=3306, help="Database port")
    parser.add_argument("--user", default="root", help="Database username")
    parser.add_argument("--password", default="", help="Database password")
    parser.add_argument("--db", default="conexus", help="Database name")
    
    args = parser.parse_args()
    
    print("Updating admin username to admin@conexus.ai...")
    success = update_admin_username(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        db_name=args.db
    )
    
    if success:
        print("Admin username update completed successfully.")
    else:
        print("Failed to update admin username.")