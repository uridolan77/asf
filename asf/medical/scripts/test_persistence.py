"""
Test script for the enhanced persistence layer.
This script tests the enhanced persistence layer with repository pattern.
"""
import logging
import os
import sys
import uuid
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from asf.medical.storage.repositories.query_repository import QueryRepository
from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
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
    logger.info("Testing query repository...")
    query_repo = QueryRepository()
    query_data = TEST_QUERY.copy()
    query_data["user_id"] = user_id
    query = await query_repo.create_async(
        db,
        obj_in=query_data
    )
    logger.info(f"Created query: {query.query_text} (ID: {query.id})")
    query_by_id = await await await query_repo.get_async(db, id=query.id)
    logger.info(f"Got query by ID: {query_by_id.query_text} (ID: {query_by_id.id})")
    queries = await await await query_repo.list_async(db)
    logger.info(f"Listed {len(queries)} queries")
    user_queries = await query_repo.list_by_user_async(db, user_id=user_id)
    logger.info(f"Listed {len(user_queries)} queries for user {user_id}")
    await await await query_repo.delete_async(db, id=query.id)
    logger.info(f"Deleted query: {query.query_text} (ID: {query.id})")
    return query
async def test_result_repository(user_id: int, query_id: int):
    logger.info("Testing knowledge base repository...")
    kb_repo = KnowledgeBaseRepository()
    kb_data = TEST_KB.copy()
    kb_data["user_id"] = user_id
    kb = await kb_repo.create_knowledge_base_async(
        db,
        **kb_data
    )
    logger.info(f"Created knowledge base: {kb.name} (ID: {kb.id})")
    kb_by_id = await await await kb_repo.get_async(db, id=kb.id)
    logger.info(f"Got knowledge base by ID: {kb_by_id.name} (ID: {kb_by_id.id})")
    kb_by_name = await await await kb_repo.get_by_name_async(db, name=kb.name)
    logger.info(f"Got knowledge base by name: {kb_by_name.name} (ID: {kb_by_name.id})")
    kb_by_kb_id = await await await kb_repo.get_by_kb_id_async(db, kb_id=kb.kb_id)
    logger.info(f"Got knowledge base by kb_id: {kb_by_kb_id.name} (ID: {kb_by_kb_id.id})")
    kbs = await await await kb_repo.list_async(db)
    logger.info(f"Listed {len(kbs)} knowledge bases")
    user_kbs = await kb_repo.list_by_user_async(db, user_id=user_id)
    logger.info(f"Listed {len(user_kbs)} knowledge bases for user {user_id}")
    updated_kb = await kb_repo.update_async(
        db,
        kb_id=kb.kb_id,
        obj_in={"update_schedule": "daily"}
    )
    logger.info(f"Updated knowledge base: {updated_kb.name} (Schedule: {updated_kb.update_schedule})")
    await await await kb_repo.delete_async(db, kb_id=kb.kb_id)
    logger.info(f"Deleted knowledge base: {kb.name} (ID: {kb.id})")
    return kb
async def main():