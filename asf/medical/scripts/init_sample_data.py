"""
Sample Data Initialization Script.

This script initializes the database with sample data for development and testing purposes.
It creates sample users, knowledge bases, and search results to provide a starting point
for working with the application. The script checks if data already exists and provides
an option to force initialization even if data is present.

Usage:
    python -m asf.medical.scripts.init_sample_data [--force]

Options:
    --force    Force initialization even if data already exists
"""
import sys
import argparse
import logging
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from asf.medical.core.security import get_password_hash
from asf.medical.core.db import get_db
from asf.medical.storage.models.user import MedicalUser
from asf.medical.models.knowledge_base import KnowledgeBase
from asf.medical.models.search import SearchResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

SAMPLE_USERS = [
    {
        "email": "admin@example.com",
        "password": "admin123",
        "full_name": "Admin User",
        "role": "admin",
        "is_active": True
    },
    {
        "email": "user@example.com",
        "password": "user123",
        "full_name": "Regular User",
        "role": "user",
        "is_active": True
    },
    {
        "email": "researcher@example.com",
        "password": "researcher123",
        "full_name": "Medical Researcher",
        "role": "researcher",
        "is_active": True
    }
]

SAMPLE_KNOWLEDGE_BASES = [
    {
        "name": "CAP Treatment",
        "query": "community acquired pneumonia treatment",
        "update_schedule": "weekly",
        "user_id": 1  # Admin user
    },
    {
        "name": "Antibiotic Resistance",
        "query": "antibiotic resistance pneumonia",
        "update_schedule": "monthly",
        "user_id": 3  # Researcher user
    },
    {
        "name": "COVID-19 Pneumonia",
        "query": "covid-19 pneumonia treatment",
        "update_schedule": "daily",
        "user_id": 2  # Regular user
    }
]

SAMPLE_SEARCHES = [
    {
        "query": "community acquired pneumonia treatment antibiotics",
        "max_results": 10,
        "user_id": 2  # Regular user
    },
    {
        "query": "pneumonia corticosteroids efficacy",
        "max_results": 20,
        "user_id": 3  # Researcher user
    },
    {
        "query": "pneumonia diagnosis imaging",
        "max_results": 15,
        "user_id": 1  # Admin user
    }
]

async def create_sample_users(db):
    """Create sample user accounts in the database.

    This function creates predefined sample users with different roles (admin, user, researcher)
    and credentials. It checks if users already exist to avoid duplicates.

    Args:
        db: Database session for database operations

    Returns:
        None
    """
    logger.info("Creating sample users...")

    for user_data in SAMPLE_USERS:
        existing_user = db.query(MedicalUser).filter(MedicalUser.email == user_data["email"]).first()
        if existing_user:
            logger.info(f"User {user_data['email']} already exists, skipping")
            continue

        user = MedicalUser(
            email=user_data["email"],
            hashed_password=get_password_hash(user_data["password"]),
            full_name=user_data["full_name"],
            role=user_data["role"],
            is_active=user_data["is_active"]
        )
        db.add(user)

    db.commit()
    logger.info("Sample users created")

async def create_sample_knowledge_bases(kb_service):
    """Create sample knowledge bases in the system.

    This function creates predefined sample knowledge bases with different queries
    and update schedules. Each knowledge base is associated with a specific user.

    Args:
        kb_service: Knowledge base service for creating knowledge bases

    Returns:
        None
    """
    logger.info("Creating sample knowledge bases...")

    for kb_data in SAMPLE_KNOWLEDGE_BASES:
        try:
            result = await kb_service.create_knowledge_base(
                name=kb_data["name"],
                query=kb_data["query"],
                update_schedule=kb_data["update_schedule"],
                user_id=kb_data["user_id"]
            )
            logger.info(f"Created knowledge base: {result['name']} (ID: {result['kb_id']})")
        except Exception as e:
            logger.error(f"Error creating knowledge base {kb_data['name']}: {str(e)}")

    logger.info("Sample knowledge bases created")

async def create_sample_searches(search_service):
    """Execute sample searches and store the results.

    This function executes predefined sample searches with different queries
    and maximum result counts. Each search is associated with a specific user.

    Args:
        search_service: Search service for executing searches

    Returns:
        None
    """
    logger.info("Creating sample searches...")

    for search_data in SAMPLE_SEARCHES:
        try:
            result = await search_service.search(
                query=search_data["query"],
                max_results=search_data["max_results"],
                user_id=search_data["user_id"]
            )
            logger.info(f"Executed search: {search_data['query']} ({len(result.get('results', []))} results)")
        except Exception as e:
            logger.error(f"Error executing search {search_data['query']}: {str(e)}")

    logger.info("Sample searches created")

async def init_sample_data():
    """Initialize the database with sample data.

    This function is the main entry point for initializing the database with sample data.
    It parses command-line arguments, checks if data already exists, and calls the
    appropriate functions to create sample users, knowledge bases, and searches.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Initialize the database with sample data")
    parser.add_argument("--force", action="store_true", help="Force initialization even if data already exists")
    args = parser.parse_args()

    with get_db() as db:
        user_count = db.query(MedicalUser).count()
        kb_count = db.query(KnowledgeBase).count()
        search_count = db.query(SearchResult).count()

        if user_count > 0 or kb_count > 0 or search_count > 0:
            if not args.force:
                logger.warning("Database already contains data. Use --force to override.")
                return
            logger.warning("Forcing initialization with sample data...")

    # Create services
    from asf.medical.services.knowledge_base import KnowledgeBaseService
    from asf.medical.services.search import SearchService

    with get_db() as db:
        # Create sample users first
        asyncio.run(create_sample_users(db))

        # Create services with the database session
        kb_service = KnowledgeBaseService(db)
        search_service = SearchService(db)

        # Create sample knowledge bases and searches
        asyncio.run(create_sample_knowledge_bases(kb_service))
        asyncio.run(create_sample_searches(search_service))

def main():
    """Main entry point for the script.

    This function calls the init_sample_data function to initialize the database
    with sample data.

    Returns:
        None
    """
    asyncio.run(init_sample_data())

if __name__ == "__main__":
    main()
