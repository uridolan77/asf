"""
OpenAI Fine-Tuning Client

This module provides a client for interacting with OpenAI's Fine-Tuning API.
"""
import os
import logging
from typing import Dict, List, Optional, Union, Any

import httpx

logger = logging.getLogger(__name__)

class OpenAIFineTuningClient:
    """
    Client for interacting with OpenAI's Fine-Tuning API.
    
    This client provides methods for creating and managing fine-tuning jobs
    to customize models for specific use cases.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 180,  # Longer timeout for fine-tuning operations
        max_retries: int = 3
    ):
        """
        Initialize the OpenAI Fine-Tuning Client.
        
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
                "Content-Type": "application/json"
            }
        )
        
        logger.info("Initialized OpenAI Fine-Tuning Client")
        
    async def close(self):
        """Close the HTTP client session."""
        await self.client.aclose()
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    # Fine-tuning job operations
    
    async def create_fine_tuning_job(
        self,
        training_file: str,
        model: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        method: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Create a fine-tuning job.
        
        Args:
            training_file: The ID of the file to use for training.
            model: The base model to fine-tune (e.g., "gpt-4o-mini-2024-07-18").
            validation_file: Optional ID of the file to use for validation.
            hyperparameters: Optional dictionary of hyperparameters.
            method: Optional dictionary specifying the fine-tuning method (e.g., supervised, DPO).
            suffix: Optional custom suffix for the fine-tuned model name.
            metadata: Optional metadata for the fine-tuning job.
            
        Returns:
            JSON response containing the created fine-tuning job details.
        """
        data = {
            "training_file": training_file,
            "model": model
        }
        
        if validation_file:
            data["validation_file"] = validation_file
        
        if hyperparameters:
            data["hyperparameters"] = hyperparameters
            
        if method:
            data["method"] = method
            
        if suffix:
            data["suffix"] = suffix
            
        if metadata:
            data["metadata"] = metadata
        
        response = await self.client.post("/fine_tuning/jobs", json=data)
        response.raise_for_status()
        return response.json()
    
    async def create_supervised_fine_tuning_job(
        self,
        training_file: str,
        model: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Create a supervised fine-tuning job.
        This is a convenience method for the most common type of fine-tuning.
        
        Args:
            training_file: The ID of the file to use for training.
            model: The base model to fine-tune (e.g., "gpt-4o-mini-2024-07-18").
            validation_file: Optional ID of the file to use for validation.
            hyperparameters: Optional dictionary of hyperparameters like n_epochs, batch_size, learning_rate_multiplier.
            suffix: Optional custom suffix for the fine-tuned model name.
            metadata: Optional metadata for the fine-tuning job.
            
        Returns:
            JSON response containing the created fine-tuning job details.
        """
        supervised_method = {
            "type": "supervised",
            "supervised": {
                "hyperparameters": hyperparameters or {
                    "n_epochs": 3,
                    "batch_size": "auto",
                    "learning_rate_multiplier": "auto"
                }
            }
        }
        
        return await self.create_fine_tuning_job(
            training_file=training_file,
            model=model,
            validation_file=validation_file,
            method=supervised_method,
            suffix=suffix,
            metadata=metadata
        )
    
    async def create_dpo_fine_tuning_job(
        self,
        training_file: str,
        model: str,
        validation_file: Optional[str] = None,
        beta: Union[float, str] = "auto",
        batch_size: Union[int, str] = "auto",
        suffix: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict:
        """
        Create a DPO (Direct Preference Optimization) fine-tuning job.
        
        Args:
            training_file: The ID of the file to use for training.
            model: The base model to fine-tune (e.g., "gpt-4o-mini-2024-07-18").
            validation_file: Optional ID of the file to use for validation.
            beta: The beta value for DPO (between 0 and 2, or "auto").
            batch_size: Batch size for training (between 1 and 256, or "auto").
            suffix: Optional custom suffix for the fine-tuned model name.
            metadata: Optional metadata for the fine-tuning job.
            
        Returns:
            JSON response containing the created fine-tuning job details.
        """
        dpo_method = {
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": beta,
                    "batch_size": batch_size
                }
            }
        }
        
        return await self.create_fine_tuning_job(
            training_file=training_file,
            model=model,
            validation_file=validation_file,
            method=dpo_method,
            suffix=suffix,
            metadata=metadata
        )
    
    async def list_fine_tuning_jobs(
        self,
        limit: int = 20,
        after: Optional[str] = None
    ) -> Dict:
        """
        List all fine-tuning jobs.
        
        Args:
            limit: Maximum number of fine-tuning jobs to return. Default is 20.
            after: Pagination cursor for fetching next page of results.
            
        Returns:
            JSON response containing the list of fine-tuning jobs.
        """
        params = {"limit": limit}
        if after:
            params["after"] = after
            
        response = await self.client.get("/fine_tuning/jobs", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_fine_tuning_job(self, job_id: str) -> Dict:
        """
        Retrieve a fine-tuning job by ID.
        
        Args:
            job_id: The ID of the fine-tuning job to retrieve.
            
        Returns:
            JSON response containing the fine-tuning job details.
        """
        response = await self.client.get(f"/fine_tuning/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    
    async def cancel_fine_tuning_job(self, job_id: str) -> Dict:
        """
        Cancel an in-progress fine-tuning job.
        
        Args:
            job_id: The ID of the fine-tuning job to cancel.
            
        Returns:
            JSON response containing the updated fine-tuning job details.
        """
        response = await self.client.post(f"/fine_tuning/jobs/{job_id}/cancel")
        response.raise_for_status()
        return response.json()
    
    # Fine-tuning job events and checkpoints
    
    async def list_fine_tuning_events(
        self,
        job_id: str,
        limit: int = 20,
        after: Optional[str] = None
    ) -> Dict:
        """
        List events for a fine-tuning job.
        
        Args:
            job_id: The ID of the fine-tuning job.
            limit: Maximum number of events to return. Default is 20.
            after: Pagination cursor for fetching next page of results.
            
        Returns:
            JSON response containing the list of events.
        """
        params = {"limit": limit}
        if after:
            params["after"] = after
            
        response = await self.client.get(f"/fine_tuning/jobs/{job_id}/events", params=params)
        response.raise_for_status()
        return response.json()
    
    async def list_fine_tuning_checkpoints(self, job_id: str) -> Dict:
        """
        List checkpoints for a fine-tuning job.
        
        Args:
            job_id: The ID of the fine-tuning job.
            
        Returns:
            JSON response containing the list of checkpoints.
        """
        response = await self.client.get(f"/fine_tuning/jobs/{job_id}/checkpoints")
        response.raise_for_status()
        return response.json()
    
    # Fine-tuning checkpoint permissions (requires admin API key)
    
    async def list_checkpoint_permissions(self, permission_id: str) -> Dict:
        """
        List permissions for a fine-tuned model checkpoint.
        Note: This requires an admin API key.
        
        Args:
            permission_id: The ID of the permission to list.
            
        Returns:
            JSON response containing the list of permissions.
        """
        response = await self.client.get(f"/fine_tuning/checkpoints/{permission_id}/permissions")
        response.raise_for_status()
        return response.json()
    
    async def create_checkpoint_permission(
        self,
        permission_id: str,
        organization_id: str
    ) -> Dict:
        """
        Create a permission for a fine-tuned model checkpoint.
        Note: This requires an admin API key.
        
        Args:
            permission_id: The ID of the permission to create.
            organization_id: The ID of the organization to grant permission to.
            
        Returns:
            JSON response containing the created permission details.
        """
        data = {"organization_id": organization_id}
        response = await self.client.post(f"/fine_tuning/checkpoints/{permission_id}/permissions", json=data)
        response.raise_for_status()
        return response.json()
    
    async def delete_checkpoint_permission(self, permission_id: str) -> Dict:
        """
        Delete a permission for a fine-tuned model checkpoint.
        Note: This requires an admin API key.
        
        Args:
            permission_id: The ID of the permission to delete.
            
        Returns:
            JSON response confirming deletion.
        """
        response = await self.client.delete(f"/fine_tuning/checkpoints/{permission_id}/permissions")
        response.raise_for_status()
        return response.json()