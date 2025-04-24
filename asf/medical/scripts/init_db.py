"""
Database Initialization Script.

This script initializes the database by creating all required tables and adding
initial data such as admin and regular user accounts. It uses SQLAlchemy's
declarative base metadata to create the tables based on the defined models.

Usage:
    python -m asf.medical.scripts.init_db

Note:
    This script should be run once before starting the application for the first time.
    Running it multiple times will not create duplicate data as it checks for existing
    records before creating new ones.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.storage.database import init_db, get_db
from asf.medical.core.security import get_password_hash
from asf.medical.storage.models import MedicalUser
from asf.medical.storage.models.user import Role

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """Initialize the database with tables and initial data.

    This function performs the following operations:
    1. Creates all database tables based on SQLAlchemy models
    2. Creates an admin user if one doesn't already exist
    3. Creates a regular user if one doesn't already exist

    The function checks for existing users to avoid creating duplicates
    when run multiple times.

    Returns:
        None
    """
    init_db()
    logger.info("Database tables created")

    with get_db() as db:
        admin = db.query(MedicalUser).filter(MedicalUser.email == "admin@example.com").first()
        if not admin:
            admin = MedicalUser(
                email="admin@example.com",
                hashed_password=get_password_hash("admin_password"),
                role=Role.ADMIN,
                is_active=True
            )
            db.add(admin)
            logger.info("Admin user created")

        user = db.query(MedicalUser).filter(MedicalUser.email == "user@example.com").first()
        if not user:
            user = MedicalUser(
                email="user@example.com",
                hashed_password=get_password_hash("user_password"),
                role=Role.USER,
                is_active=True
            )
            db.add(user)
            logger.info("Regular user created")

        db.commit()

    logger.info("Database initialization complete")

if __name__ == "__main__":
    init()
