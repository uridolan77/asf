"""
OpenAI Vector Store Client

This module provides a client for interacting with OpenAI's Vector Store API.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

import httpx

logger = logging.getLogger(__name__)

class OpenAIVectorStoreClient:
    """
    Client for interacting with OpenAI's Vector Store API.
    
    This client provides methods for creating and managing vector stores,
    which are used for semantic search and retrieval operations.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize the OpenAI Vector Store Client.
        
        Args:
            api_key: OpenAI API key. If not provided, will be read from OPENAI_API_KEY environment variable.
            base_url: Base URL for the OpenAI API.
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set the OPENAI_API_KEY environment variable or pass it to the constructor.")
        
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "OpenAI-Beta": "assistants=v2"  # Required for vector store operations
            }
        )
        
        logger.info("Initialized OpenAI Vector Store Client")
        
    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # Vector Store operations
    
    async def create_vector_store(self, name: Optional[str] = None) -> Dict:
        """
        Create a new vector store.
        
        Args:
            name: Optional name for the vector store.
            
        Returns:
            JSON response containing the created vector store details.
        """
        data = {}
        if name:
            data["name"] = name
        
        response = await self.client.post("/vector_stores", json=data)
        response.raise_for_status()
        return response.json()
    
    async def list_vector_stores(
        self,
        limit: int = 20,
        after: Optional[str] = None,
        order: str = "desc"
    ) -> Dict:
        """
        List all vector stores.
        
        Args:
            limit: Maximum number of vector stores to return. Default is 20.
            after: Pagination cursor for fetching next page of results.
            order: Sort order, "asc" or "desc". Default is "desc".
            
        Returns:
            JSON response containing the list of vector stores.
        """
        params = {"limit": limit, "order": order}
        if after:
            params["after"] = after
            
        response = await self.client.get("/vector_stores", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_vector_store(self, vector_store_id: str) -> Dict:
        """
        Retrieve a vector store by ID.
        
        Args:
            vector_store_id: The ID of the vector store to retrieve.
            
        Returns:
            JSON response containing the vector store details.
        """
        response = await self.client.get(f"/vector_stores/{vector_store_id}")
        response.raise_for_status()
        return response.json()
    
    async def update_vector_store(
        self,
        vector_store_id: str,
        name: str
    ) -> Dict:
        """
        Update a vector store.
        
        Args:
            vector_store_id: The ID of the vector store to update.
            name: New name for the vector store.
            
        Returns:
            JSON response containing the updated vector store details.
        """
        data = {"name": name}
        response = await self.client.post(f"/vector_stores/{vector_store_id}", json=data)
        response.raise_for_status()
        return response.json()
    
    async def delete_vector_store(self, vector_store_id: str) -> Dict:
        """
        Delete a vector store.
        
        Args:
            vector_store_id: The ID of the vector store to delete.
            
        Returns:
            JSON response confirming deletion.
        """
        response = await self.client.delete(f"/vector_stores/{vector_store_id}")
        response.raise_for_status()
        return response.json()
    
    # Vector Store File operations
    
    async def create_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str
    ) -> Dict:
        """
        Add a file to a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            file_id: The ID of the file to add to the vector store.
            
        Returns:
            JSON response containing the created vector store file details.
        """
        data = {"file_id": file_id}
        response = await self.client.post(f"/vector_stores/{vector_store_id}/files", json=data)
        response.raise_for_status()
        return response.json()
    
    async def list_vector_store_files(
        self,
        vector_store_id: str,
        limit: int = 20,
        after: Optional[str] = None,
        order: str = "desc"
    ) -> Dict:
        """
        List all files in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            limit: Maximum number of files to return. Default is 20.
            after: Pagination cursor for fetching next page of results.
            order: Sort order, "asc" or "desc". Default is "desc".
            
        Returns:
            JSON response containing the list of files in the vector store.
        """
        params = {"limit": limit, "order": order}
        if after:
            params["after"] = after
            
        response = await self.client.get(f"/vector_stores/{vector_store_id}/files", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str
    ) -> Dict:
        """
        Retrieve a file from a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            file_id: The ID of the file to retrieve.
            
        Returns:
            JSON response containing the file details.
        """
        response = await self.client.get(f"/vector_stores/{vector_store_id}/files/{file_id}")
        response.raise_for_status()
        return response.json()
    
    async def delete_vector_store_file(
        self,
        vector_store_id: str,
        file_id: str
    ) -> Dict:
        """
        Delete a file from a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            file_id: The ID of the file to delete.
            
        Returns:
            JSON response confirming deletion.
        """
        response = await self.client.delete(f"/vector_stores/{vector_store_id}/files/{file_id}")
        response.raise_for_status()
        return response.json()
    
    async def update_vector_store_file_attributes(
        self,
        vector_store_id: str,
        file_id: str,
        attributes: Dict[str, Any]
    ) -> Dict:
        """
        Update attributes on a vector store file.
        
        Args:
            vector_store_id: The ID of the vector store.
            file_id: The ID of the file to update.
            attributes: Dictionary of attributes to associate with the file.
            
        Returns:
            JSON response containing the updated file details.
        """
        data = {"attributes": attributes}
        response = await self.client.post(f"/vector_stores/{vector_store_id}/files/{file_id}", json=data)
        response.raise_for_status()
        return response.json()
    
    async def get_vector_store_file_content(
        self,
        vector_store_id: str,
        file_id: str
    ) -> Dict:
        """
        Retrieve the content of a file in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            file_id: The ID of the file to retrieve content for.
            
        Returns:
            JSON response containing the file content.
        """
        response = await self.client.get(f"/vector_stores/{vector_store_id}/files/{file_id}/content")
        response.raise_for_status()
        return response.json()
    
    # Vector Store Batch operations
    
    async def create_vector_store_file_batch(
        self,
        vector_store_id: str,
        file_ids: List[str]
    ) -> Dict:
        """
        Create a batch of files in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            file_ids: List of file IDs to add to the vector store.
            
        Returns:
            JSON response containing the created batch details.
        """
        data = {"file_ids": file_ids}
        response = await self.client.post(f"/vector_stores/{vector_store_id}/file_batches", json=data)
        response.raise_for_status()
        return response.json()
    
    async def get_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str
    ) -> Dict:
        """
        Retrieve a file batch from a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            batch_id: The ID of the batch to retrieve.
            
        Returns:
            JSON response containing the batch details.
        """
        response = await self.client.get(f"/vector_stores/{vector_store_id}/file_batches/{batch_id}")
        response.raise_for_status()
        return response.json()
    
    async def cancel_vector_store_file_batch(
        self,
        vector_store_id: str,
        batch_id: str
    ) -> Dict:
        """
        Cancel a file batch in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            batch_id: The ID of the batch to cancel.
            
        Returns:
            JSON response containing the updated batch details.
        """
        response = await self.client.post(f"/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel")
        response.raise_for_status()
        return response.json()
    
    async def list_files_in_vector_store_batch(
        self,
        vector_store_id: str,
        batch_id: str,
        limit: int = 20,
        after: Optional[str] = None,
        order: str = "desc"
    ) -> Dict:
        """
        List all files in a batch in a vector store.
        
        Args:
            vector_store_id: The ID of the vector store.
            batch_id: The ID of the batch.
            limit: Maximum number of files to return. Default is 20.
            after: Pagination cursor for fetching next page of results.
            order: Sort order, "asc" or "desc". Default is "desc".
            
        Returns:
            JSON response containing the list of files in the batch.
        """
        params = {"limit": limit, "order": order}
        if after:
            params["after"] = after
            
        response = await self.client.get(
            f"/vector_stores/{vector_store_id}/file_batches/{batch_id}/files", 
            params=params
        )
        response.raise_for_status()
        return response.json()
    
    # Vector Store Search operations
    
    async def search_vector_store(
        self,
        vector_store_id: str,
        query: Union[str, List[str]],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        return_attributes: Optional[List[str]] = None
    ) -> Dict:
        """
        Search a vector store for relevant chunks based on a query.
        
        Args:
            vector_store_id: The ID of the vector store to search.
            query: The search query, either a string or a list of strings.
            limit: Maximum number of results to return. Default is 10.
            filters: Optional dictionary of attributes to filter results by.
            return_attributes: Optional list of specific attributes to return.
            
        Returns:
            JSON response containing the search results.
        """
        data = {
            "query": query,
            "limit": limit
        }
        
        if filters:
            data["filters"] = filters
            
        if return_attributes:
            data["return_attributes"] = return_attributes
        
        response = await self.client.post(f"/vector_stores/{vector_store_id}/search", json=data)
        response.raise_for_status()
        return response.json()