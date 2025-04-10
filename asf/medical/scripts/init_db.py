"""
Script to initialize the database.

This script creates the database tables and adds initial data.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.storage.database import init_db, get_db
from asf.medical.core.security import get_password_hash
from asf.medical.storage.models import User

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """Initialize the database."""
    # Create tables
    init_db()
    logger.info("Database tables created")
    
    # Add initial users
    with get_db() as db:
        # Check if admin user exists
        admin = db.query(User).filter(User.email == "admin@example.com").first()
        if not admin:
            admin = User(
                email="admin@example.com",
                hashed_password=get_password_hash("admin_password"),
                role="admin",
                is_active=True
            )
            db.add(admin)
            logger.info("Admin user created")
        
        # Check if regular user exists
        user = db.query(User).filter(User.email == "user@example.com").first()
        if not user:
            user = User(
                email="user@example.com",
                hashed_password=get_password_hash("user_password"),
                role="user",
                is_active=True
            )
            db.add(user)
            logger.info("Regular user created")
        
        db.commit()
    
    logger.info("Database initialization complete")

if __name__ == "__main__":
    init()
