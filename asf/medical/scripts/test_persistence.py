"""
Test script for the enhanced persistence layer.

This script tests the enhanced persistence layer with repository pattern.
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, List

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from asf.medical.storage.database import init_db, get_db_session
from asf.medical.storage.repositories.user_repository import UserRepository
from asf.medical.storage.repositories.query_repository import QueryRepository
from asf.medical.storage.repositories.result_repository import ResultRepository
from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository
from asf.medical.core.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test data
TEST_USER = {
    "username": f"test_user_{uuid.uuid4().hex[:8]}",
    "email": f"test_{uuid.uuid4().hex[:8]}@example.com",
    "hashed_password": "hashed_password",
    "is_active": True,
    "is_superuser": False
}

TEST_QUERY = {
    "query_text": "test query",
    "query_type": "text",
    "parameters": {"max_results": 10}
}

TEST_RESULT = {
    "result_id": str(uuid.uuid4()),
    "result_type": "search",
    "result_data": {
        "articles": [
            {"pmid": "12345", "title": "Test Article 1", "abstract": "This is a test abstract 1"},
            {"pmid": "67890", "title": "Test Article 2", "abstract": "This is a test abstract 2"}
        ]
    }
}

TEST_KB = {
    "name": f"test_kb_{uuid.uuid4().hex[:8]}",
    "query": "test query",
    "file_path": f"test_kb_{uuid.uuid4().hex[:8]}.json",
    "update_schedule": "weekly",
    "initial_results": 10
}

async def test_user_repository():
    """Test the user repository."""
    logger.info("Testing user repository...")
    
    # Create repository
    user_repo = UserRepository()
    
    # Create a user
    user = await user_repo.create_async(
        db=None,
        obj_in=TEST_USER
    )
    logger.info(f"Created user: {user.username} (ID: {user.id})")
    
    # Get the user by ID
    user_by_id = await user_repo.get_async(db=None, id=user.id)
    logger.info(f"Got user by ID: {user_by_id.username} (ID: {user_by_id.id})")
    
    # Get the user by username
    user_by_username = await user_repo.get_by_username_async(db=None, username=user.username)
    logger.info(f"Got user by username: {user_by_username.username} (ID: {user_by_username.id})")
    
    # Update the user
    updated_user = await user_repo.update_async(
        db=None,
        id=user.id,
        obj_in={"email": f"updated_{uuid.uuid4().hex[:8]}@example.com"}
    )
    logger.info(f"Updated user: {updated_user.username} (Email: {updated_user.email})")
    
    # List users
    users = await user_repo.list_async(db=None)
    logger.info(f"Listed {len(users)} users")
    
    # Delete the user
    await user_repo.delete_async(db=None, id=user.id)
    logger.info(f"Deleted user: {user.username} (ID: {user.id})")
    
    # Try to get the deleted user
    deleted_user = await user_repo.get_async(db=None, id=user.id)
    logger.info(f"Got deleted user: {deleted_user}")
    
    return user

async def test_query_repository(user_id: int):
    """Test the query repository."""
    logger.info("Testing query repository...")
    
    # Create repository
    query_repo = QueryRepository()
    
    # Create a query
    query_data = TEST_QUERY.copy()
    query_data["user_id"] = user_id
    
    query = await query_repo.create_async(
        db=None,
        obj_in=query_data
    )
    logger.info(f"Created query: {query.query_text} (ID: {query.id})")
    
    # Get the query by ID
    query_by_id = await query_repo.get_async(db=None, id=query.id)
    logger.info(f"Got query by ID: {query_by_id.query_text} (ID: {query_by_id.id})")
    
    # List queries
    queries = await query_repo.list_async(db=None)
    logger.info(f"Listed {len(queries)} queries")
    
    # List queries by user
    user_queries = await query_repo.list_by_user_async(db=None, user_id=user_id)
    logger.info(f"Listed {len(user_queries)} queries for user {user_id}")
    
    # Delete the query
    await query_repo.delete_async(db=None, id=query.id)
    logger.info(f"Deleted query: {query.query_text} (ID: {query.id})")
    
    return query

async def test_result_repository(user_id: int, query_id: int):
    """Test the result repository."""
    logger.info("Testing result repository...")
    
    # Create repository
    result_repo = ResultRepository()
    
    # Create a result
    result_data = TEST_RESULT.copy()
    result_data["user_id"] = user_id
    result_data["query_id"] = query_id
    
    result = await result_repo.create_async(
        db=None,
        obj_in=result_data
    )
    logger.info(f"Created result: {result.result_id} (ID: {result.id})")
    
    # Get the result by ID
    result_by_id = await result_repo.get_async(db=None, id=result.id)
    logger.info(f"Got result by ID: {result_by_id.result_id} (ID: {result_by_id.id})")
    
    # Get the result by result_id
    result_by_result_id = await result_repo.get_by_result_id_async(db=None, result_id=result.result_id)
    logger.info(f"Got result by result_id: {result_by_result_id.result_id} (ID: {result_by_result_id.id})")
    
    # List results
    results = await result_repo.list_async(db=None)
    logger.info(f"Listed {len(results)} results")
    
    # List results by user
    user_results = await result_repo.list_by_user_async(db=None, user_id=user_id)
    logger.info(f"Listed {len(user_results)} results for user {user_id}")
    
    # Delete the result
    await result_repo.delete_async(db=None, id=result.id)
    logger.info(f"Deleted result: {result.result_id} (ID: {result.id})")
    
    return result

async def test_kb_repository(user_id: int):
    """Test the knowledge base repository."""
    logger.info("Testing knowledge base repository...")
    
    # Create repository
    kb_repo = KnowledgeBaseRepository()
    
    # Create a knowledge base
    kb_data = TEST_KB.copy()
    kb_data["user_id"] = user_id
    
    kb = await kb_repo.create_knowledge_base_async(
        db=None,
        **kb_data
    )
    logger.info(f"Created knowledge base: {kb.name} (ID: {kb.id})")
    
    # Get the knowledge base by ID
    kb_by_id = await kb_repo.get_async(db=None, id=kb.id)
    logger.info(f"Got knowledge base by ID: {kb_by_id.name} (ID: {kb_by_id.id})")
    
    # Get the knowledge base by name
    kb_by_name = await kb_repo.get_by_name_async(db=None, name=kb.name)
    logger.info(f"Got knowledge base by name: {kb_by_name.name} (ID: {kb_by_name.id})")
    
    # Get the knowledge base by kb_id
    kb_by_kb_id = await kb_repo.get_by_kb_id_async(db=None, kb_id=kb.kb_id)
    logger.info(f"Got knowledge base by kb_id: {kb_by_kb_id.name} (ID: {kb_by_kb_id.id})")
    
    # List knowledge bases
    kbs = await kb_repo.list_async(db=None)
    logger.info(f"Listed {len(kbs)} knowledge bases")
    
    # List knowledge bases by user
    user_kbs = await kb_repo.list_by_user_async(db=None, user_id=user_id)
    logger.info(f"Listed {len(user_kbs)} knowledge bases for user {user_id}")
    
    # Update the knowledge base
    updated_kb = await kb_repo.update_async(
        db=None,
        kb_id=kb.kb_id,
        obj_in={"update_schedule": "daily"}
    )
    logger.info(f"Updated knowledge base: {updated_kb.name} (Schedule: {updated_kb.update_schedule})")
    
    # Delete the knowledge base
    await kb_repo.delete_async(db=None, kb_id=kb.kb_id)
    logger.info(f"Deleted knowledge base: {kb.name} (ID: {kb.id})")
    
    return kb

async def main():
    """Main function."""
    logger.info("Starting persistence test...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Create a user for testing
    user = await test_user_repository()
    
    # Create a new user for the remaining tests
    user_repo = UserRepository()
    test_user = await user_repo.create_async(
        db=None,
        obj_in=TEST_USER
    )
    logger.info(f"Created test user: {test_user.username} (ID: {test_user.id})")
    
    # Test query repository
    query = await test_query_repository(test_user.id)
    
    # Create a new query for the result test
    query_repo = QueryRepository()
    query_data = TEST_QUERY.copy()
    query_data["user_id"] = test_user.id
    test_query = await query_repo.create_async(
        db=None,
        obj_in=query_data
    )
    logger.info(f"Created test query: {test_query.query_text} (ID: {test_query.id})")
    
    # Test result repository
    result = await test_result_repository(test_user.id, test_query.id)
    
    # Test knowledge base repository
    kb = await test_kb_repository(test_user.id)
    
    # Clean up
    await user_repo.delete_async(db=None, id=test_user.id)
    logger.info(f"Deleted test user: {test_user.username} (ID: {test_user.id})")
    
    logger.info("Persistence test completed")

if __name__ == "__main__":
    asyncio.run(main())
