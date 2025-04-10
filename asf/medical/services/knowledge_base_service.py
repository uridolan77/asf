"""
Knowledge base service for the Medical Research Synthesizer.

This module provides a service for creating and managing knowledge bases.
"""

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from asf.medical.core.cache import cache_manager
from asf.medical.core.exceptions import (
    ResourceNotFoundError, ValidationError,
    ExternalServiceError, DatabaseError, FileError
)

from asf.medical.services.search_service import SearchService
from asf.medical.storage.repositories.kb_repository import KnowledgeBaseRepository

# Set up logging
logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    """
    Service for creating and managing knowledge bases.
    """

    def __init__(
        self,
        search_service: SearchService,
        kb_repository: KnowledgeBaseRepository,
        kb_dir: str = "knowledge_bases"
    ):
        """
        Initialize the knowledge base service.

        Args:
            search_service: Search service
            kb_repository: Knowledge base repository
            kb_dir: Directory for storing knowledge bases
        """
        self.search_service = search_service
        self.kb_repository = kb_repository
        self.kb_dir = kb_dir

        # Create the knowledge base directory if it doesn't exist
        os.makedirs(self.kb_dir, exist_ok=True)

        # Knowledge base cache namespace
        self.kb_namespace = "kb:"

    async def create_knowledge_base(
        self,
        name: str,
        query: str,
        update_schedule: str = "weekly",
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a new knowledge base.

        Args:
            name: Name of the knowledge base
            query: Query to build the knowledge base
            update_schedule: Update schedule (daily, weekly, monthly)
            user_id: User ID for storing the knowledge base

        Returns:
            Knowledge base information

        Raises:
            ValidationError: If the inputs are invalid
            ResourceNotFoundError: If no results are found for the query
            ExternalServiceError: If the search service fails
            FileError: If there's an error creating the knowledge base file
            DatabaseError: If there's an error storing the knowledge base
        """
        # Validate inputs
        if not name or not name.strip():
            raise ValidationError("Knowledge base name cannot be empty")

        if not query or not query.strip():
            raise ValidationError("Query cannot be empty")

        if update_schedule not in ["daily", "weekly", "monthly"]:
            raise ValidationError("Update schedule must be 'daily', 'weekly', or 'monthly'")

        logger.info(f"Creating knowledge base: {name} (query={query})")

        # Check if a knowledge base with this name already exists
        existing_kb = await self.get_knowledge_base(name)
        if existing_kb:
            raise ValidationError(f"Knowledge base with name '{name}' already exists")

        try:
            # Search for articles
            search_result = await self.search_service.search(query, max_results=100, user_id=user_id)

            if not search_result or not search_result.get('results'):
                logger.warning(f"No results found for query: {query}")
                raise ResourceNotFoundError("Search Results", query)
        except ValidationError:
            # Re-raise validation errors
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except Exception as e:
            logger.error(f"Error searching for articles: {str(e)}")
            raise ExternalServiceError("Search Service", f"Failed to search for articles: {str(e)}")

        # Extract articles
        articles = search_result['results']

        # Generate a unique ID for the knowledge base
        kb_id = str(uuid.uuid4())

        # Create a file path for the knowledge base
        file_path = os.path.join(self.kb_dir, f"{kb_id}.json")

        try:
            # Save the knowledge base to a file
            with open(file_path, 'w') as f:
                json.dump(articles, f)
            logger.info(f"Knowledge base file created: {file_path}")
        except Exception as e:
            logger.error(f"Error creating knowledge base file: {str(e)}")
            raise FileError(file_path, f"Failed to create knowledge base file: {str(e)}")

        # Calculate the next update time
        now = datetime.now()
        try:
            if update_schedule == "daily":
                next_update = now + timedelta(days=1)
            elif update_schedule == "weekly":
                next_update = now + timedelta(weeks=1)
            elif update_schedule == "monthly":
                next_update = now + timedelta(days=30)
            else:
                next_update = now + timedelta(weeks=1)  # Default to weekly
        except Exception as e:
            logger.error(f"Error calculating next update time: {str(e)}")
            # Use a default next update time (1 week from now)
            next_update = now + timedelta(weeks=1)

        # Create the knowledge base object
        kb = {
            'kb_id': kb_id,
            'name': name,
            'query': query,
            'file_path': file_path,
            'update_schedule': update_schedule,
            'last_updated': now.isoformat(),
            'next_update': next_update.isoformat(),
            'initial_results': len(articles),
            'user_id': user_id,
            'created_at': now.isoformat()
        }

        # Store the knowledge base in cache
        cache_key = f"{self.kb_namespace}{kb_id}"
        await cache_manager.set(cache_key, kb, data_type="knowledge_base")

        # Also cache by name for quick lookup
        name_cache_key = f"{self.kb_namespace}name:{name}"
        await cache_manager.set(name_cache_key, kb, data_type="knowledge_base")

        # Store in database
        try:
            await self.kb_repository.create_knowledge_base_async(
                db=None,  # This will be handled by the repository
                name=name,
                query=query,
                file_path=file_path,
                update_schedule=update_schedule,
                initial_results=len(articles),
                user_id=user_id
            )
            logger.info(f"Knowledge base stored in database: {kb_id}")
        except Exception as e:
            logger.error(f"Error storing knowledge base: {str(e)}")
            # Delete the file if database storage fails
            try:
                os.remove(file_path)
                logger.info(f"Deleted knowledge base file due to database error: {file_path}")
            except Exception as file_e:
                logger.error(f"Error deleting knowledge base file: {str(file_e)}")
            # Raise the original error
            raise DatabaseError(f"Failed to store knowledge base: {str(e)}")

        logger.info(f"Knowledge base created: {kb_id}")

        return kb

    async def get_knowledge_base(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a knowledge base by name.

        Args:
            name: Name of the knowledge base

        Returns:
            Knowledge base information or None if not found

        Raises:
            ValidationError: If the name is invalid
            DatabaseError: If there's an error retrieving the knowledge base
        """
        if not name or not name.strip():
            raise ValidationError("Knowledge base name cannot be empty")

        logger.info(f"Getting knowledge base by name: {name}")

        # Try to get from cache by name
        name_cache_key = f"{self.kb_namespace}name:{name}"
        cached_kb = await cache_manager.get(name_cache_key, data_type="knowledge_base")
        if cached_kb is not None:
            logger.debug(f"Knowledge base found in cache: {name}")
            return cached_kb

        # If not found in cache, search in the database
        try:
            kb = await self.kb_repository.get_by_name_async(db=None, name=name)
            if kb:
                logger.debug(f"Knowledge base found in database: {name}")
                # Convert to dictionary
                kb_dict = {
                    'kb_id': kb.kb_id,
                    'name': kb.name,
                    'query': kb.query,
                    'file_path': kb.file_path,
                    'update_schedule': kb.update_schedule,
                    'last_updated': kb.last_updated.isoformat() if kb.last_updated else None,
                    'next_update': kb.next_update.isoformat() if kb.next_update else None,
                    'initial_results': kb.initial_results,
                    'user_id': kb.user_id,
                    'created_at': kb.created_at.isoformat()
                }

                # Store in cache for future use
                cache_key = f"{self.kb_namespace}{kb.kb_id}"
                await cache_manager.set(cache_key, kb_dict, data_type="knowledge_base")

                # Also cache by name
                await cache_manager.set(name_cache_key, kb_dict, data_type="knowledge_base")

                return kb_dict
            else:
                logger.debug(f"Knowledge base not found: {name}")
                return None
        except Exception as e:
            logger.error(f"Error getting knowledge base by name: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")

    async def get_knowledge_base_by_id(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a knowledge base by ID.

        Args:
            kb_id: Knowledge base ID

        Returns:
            Knowledge base information or None if not found

        Raises:
            ValidationError: If the kb_id is invalid
            ResourceNotFoundError: If the knowledge base is not found
            DatabaseError: If there's an error retrieving the knowledge base
        """
        if not kb_id or not kb_id.strip():
            raise ValidationError("Knowledge base ID cannot be empty")

        logger.info(f"Getting knowledge base by ID: {kb_id}")

        # Try to get from cache by ID
        cache_key = f"{self.kb_namespace}{kb_id}"
        cached_kb = await cache_manager.get(cache_key, data_type="knowledge_base")
        if cached_kb is not None:
            logger.debug(f"Knowledge base found in cache: {kb_id}")
            return cached_kb

        # If not found in cache, search in the database
        try:
            kb = await self.kb_repository.get_by_kb_id_async(db=None, kb_id=kb_id)
            if kb:
                logger.debug(f"Knowledge base found in database: {kb_id}")
                # Convert to dictionary
                kb_dict = {
                    'kb_id': kb.kb_id,
                    'name': kb.name,
                    'query': kb.query,
                    'file_path': kb.file_path,
                    'update_schedule': kb.update_schedule,
                    'last_updated': kb.last_updated.isoformat() if kb.last_updated else None,
                    'next_update': kb.next_update.isoformat() if kb.next_update else None,
                    'initial_results': kb.initial_results,
                    'user_id': kb.user_id,
                    'created_at': kb.created_at.isoformat()
                }

                # Store in cache for future use
                await cache_manager.set(cache_key, kb_dict, data_type="knowledge_base")

                # Also cache by name
                name_cache_key = f"{self.kb_namespace}name:{kb.name}"
                await cache_manager.set(name_cache_key, kb_dict, data_type="knowledge_base")

                return kb_dict
            else:
                logger.warning(f"Knowledge base not found: {kb_id}")
                return None
        except Exception as e:
            logger.error(f"Error getting knowledge base by ID: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")

    async def list_knowledge_bases(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        List all knowledge bases.

        Args:
            user_id: User ID to filter by

        Returns:
            List of knowledge bases

        Raises:
            DatabaseError: If there's an error retrieving the knowledge bases
        """
        logger.info(f"Listing knowledge bases: user_id={user_id if user_id else 'all'}")

        # Try to get from cache first
        cache_key = f"{self.kb_namespace}list:{user_id if user_id else 'all'}"
        cached_list = await cache_manager.get(cache_key, data_type="knowledge_base")
        if cached_list is not None:
            logger.debug(f"Knowledge base list found in cache: {len(cached_list)} items")
            return cached_list

        try:
            # Get all knowledge bases from the database
            kbs = await self.kb_repository.list_async(db=None, user_id=user_id)

            # Convert to dictionaries
            kb_dicts = []
            for kb in kbs:
                kb_dict = {
                    'kb_id': kb.kb_id,
                    'name': kb.name,
                    'query': kb.query,
                    'file_path': kb.file_path,
                    'update_schedule': kb.update_schedule,
                    'last_updated': kb.last_updated.isoformat() if kb.last_updated else None,
                    'next_update': kb.next_update.isoformat() if kb.next_update else None,
                    'initial_results': kb.initial_results,
                    'user_id': kb.user_id,
                    'created_at': kb.created_at.isoformat()
                }

                # Store individual KB in cache
                kb_cache_key = f"{self.kb_namespace}{kb.kb_id}"
                await cache_manager.set(kb_cache_key, kb_dict, data_type="knowledge_base")

                # Also cache by name
                name_cache_key = f"{self.kb_namespace}name:{kb.name}"
                await cache_manager.set(name_cache_key, kb_dict, data_type="knowledge_base")

                kb_dicts.append(kb_dict)

            # Cache the list
            await cache_manager.set(cache_key, kb_dicts, data_type="knowledge_base")

            logger.info(f"Found {len(kb_dicts)} knowledge bases")
            return kb_dicts
        except Exception as e:
            logger.error(f"Error listing knowledge bases: {str(e)}")
            raise DatabaseError(f"Failed to list knowledge bases: {str(e)}")

    async def update_knowledge_base(self, kb_id: str) -> Dict[str, Any]:
        """
        Update a knowledge base.

        Args:
            kb_id: Knowledge base ID

        Returns:
            Updated knowledge base information

        Raises:
            ValidationError: If the kb_id is invalid
            ResourceNotFoundError: If the knowledge base is not found
            ExternalServiceError: If the search service fails
            FileError: If there's an error updating the knowledge base file
            DatabaseError: If there's an error updating the knowledge base in the database
        """
        if not kb_id or not kb_id.strip():
            raise ValidationError("Knowledge base ID cannot be empty")

        logger.info(f"Updating knowledge base: {kb_id}")

        try:
            # Get the knowledge base
            kb = await self.get_knowledge_base_by_id(kb_id)

            if not kb:
                logger.warning(f"Knowledge base not found: {kb_id}")
                raise ResourceNotFoundError("Knowledge Base", kb_id)
        except ValidationError:
            # Re-raise validation errors
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except Exception as e:
            logger.error(f"Error getting knowledge base: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")

        logger.info(f"Updating knowledge base: {kb['name']} (kb_id={kb_id})")

        try:
            # Search for new articles
            search_result = await self.search_service.search(
                kb['query'],
                max_results=100,
                user_id=kb['user_id']
            )

            if not search_result or not search_result.get('results'):
                logger.warning(f"No results found for query: {kb['query']}")
                raise ResourceNotFoundError("Search Results", kb['query'])
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except Exception as e:
            logger.error(f"Error searching for articles: {str(e)}")
            raise ExternalServiceError("Search Service", f"Failed to search for articles: {str(e)}")

        # Extract articles
        articles = search_result['results']

        # Load existing articles
        existing_articles = []
        try:
            with open(kb['file_path'], 'r') as f:
                existing_articles = json.load(f)
            logger.debug(f"Loaded {len(existing_articles)} existing articles from {kb['file_path']}")
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {kb['file_path']}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in knowledge base file: {str(e)}")
            raise FileError(kb['file_path'], f"Invalid JSON in knowledge base file: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load existing articles: {str(e)}")
            raise FileError(kb['file_path'], f"Failed to load existing articles: {str(e)}")

        # Merge articles (placeholder - actual implementation would be more sophisticated)
        merged_articles = existing_articles + articles
        logger.info(f"Merged {len(existing_articles)} existing articles with {len(articles)} new articles")

        try:
            # Save the updated knowledge base to a file
            with open(kb['file_path'], 'w') as f:
                json.dump(merged_articles, f)
            logger.info(f"Updated knowledge base file: {kb['file_path']} with {len(merged_articles)} articles")
        except Exception as e:
            logger.error(f"Failed to save updated knowledge base: {str(e)}")
            raise FileError(kb['file_path'], f"Failed to save updated knowledge base: {str(e)}")

        # Update the knowledge base object
        now = datetime.now()
        try:
            if kb['update_schedule'] == "daily":
                next_update = now + timedelta(days=1)
            elif kb['update_schedule'] == "weekly":
                next_update = now + timedelta(weeks=1)
            elif kb['update_schedule'] == "monthly":
                next_update = now + timedelta(days=30)
            else:
                next_update = now + timedelta(weeks=1)  # Default to weekly
        except Exception as e:
            logger.error(f"Error calculating next update time: {str(e)}")
            # Use a default next update time (1 week from now)
            next_update = now + timedelta(weeks=1)

        kb['last_updated'] = now.isoformat()
        kb['next_update'] = next_update.isoformat()

        # Store the updated knowledge base in cache
        cache_key = f"{self.kb_namespace}{kb_id}"
        await cache_manager.set(cache_key, kb, data_type="knowledge_base")

        # Also update the name cache
        name_cache_key = f"{self.kb_namespace}name:{kb['name']}"
        await cache_manager.set(name_cache_key, kb, data_type="knowledge_base")

        # Invalidate the list cache
        list_cache_key = f"{self.kb_namespace}list:all"
        await cache_manager.delete(list_cache_key)
        if kb['user_id']:
            user_list_cache_key = f"{self.kb_namespace}list:{kb['user_id']}"
            await cache_manager.delete(user_list_cache_key)

        # Update in database
        try:
            await self.kb_repository.update_async(
                db=None,  # This will be handled by the repository
                kb_id=kb_id,
                obj_in={
                    'last_updated': now,
                    'next_update': next_update
                }
            )
            logger.info(f"Knowledge base updated in database: {kb_id}")
        except Exception as e:
            logger.error(f"Error updating knowledge base in database: {str(e)}")
            raise DatabaseError(f"Failed to update knowledge base in database: {str(e)}")
            # Continue even if update fails

        logger.info(f"Knowledge base updated: {kb_id}")

        return kb

    async def delete_knowledge_base(self, kb_id: str) -> bool:
        """
        Delete a knowledge base.

        Args:
            kb_id: Knowledge base ID

        Returns:
            True if the knowledge base was deleted, False otherwise

        Raises:
            ValidationError: If the kb_id is invalid
            ResourceNotFoundError: If the knowledge base is not found
            FileError: If there's an error deleting the knowledge base file
            DatabaseError: If there's an error deleting the knowledge base from the database
        """
        if not kb_id or not kb_id.strip():
            raise ValidationError("Knowledge base ID cannot be empty")

        logger.info(f"Deleting knowledge base: {kb_id}")

        try:
            # Get the knowledge base
            kb = await self.get_knowledge_base_by_id(kb_id)

            if not kb:
                logger.warning(f"Knowledge base not found: {kb_id}")
                raise ResourceNotFoundError("Knowledge Base", kb_id)
        except ValidationError:
            # Re-raise validation errors
            raise
        except ResourceNotFoundError:
            # Re-raise resource not found errors
            raise
        except Exception as e:
            logger.error(f"Error getting knowledge base: {str(e)}")
            raise DatabaseError(f"Failed to retrieve knowledge base: {str(e)}")

        logger.info(f"Deleting knowledge base: {kb['name']} (kb_id={kb_id})")

        # Delete the knowledge base file
        try:
            os.remove(kb['file_path'])
            logger.info(f"Deleted knowledge base file: {kb['file_path']}")
        except FileNotFoundError:
            logger.warning(f"Knowledge base file not found: {kb['file_path']}")
        except Exception as e:
            logger.error(f"Failed to delete knowledge base file: {str(e)}")
            raise FileError(kb['file_path'], f"Failed to delete knowledge base file: {str(e)}")

        # Remove from cache
        cache_key = f"{self.kb_namespace}{kb_id}"
        await cache_manager.delete(cache_key)

        # Also remove from name cache
        name_cache_key = f"{self.kb_namespace}name:{kb['name']}"
        await cache_manager.delete(name_cache_key)

        # Invalidate the list cache
        list_cache_key = f"{self.kb_namespace}list:all"
        await cache_manager.delete(list_cache_key)
        if kb['user_id']:
            user_list_cache_key = f"{self.kb_namespace}list:{kb['user_id']}"
            await cache_manager.delete(user_list_cache_key)

        logger.debug(f"Removed knowledge base from cache: {kb_id}")

        # Delete from database
        try:
            await self.kb_repository.delete_async(db=None, kb_id=kb_id)
            logger.info(f"Deleted knowledge base from database: {kb_id}")
        except Exception as e:
            logger.error(f"Error deleting knowledge base from database: {str(e)}")
            raise DatabaseError(f"Failed to delete knowledge base from database: {str(e)}")

        logger.info(f"Knowledge base deleted: {kb_id}")

        return True
