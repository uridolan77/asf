"""
Vault integration for secrets management.

This module provides integration with HashiCorp Vault for secrets management,
including API key storage, retrieval, and rotation.
"""

import logging
import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
import httpx
from fastapi import HTTPException, status

from api.config import settings

logger = logging.getLogger(__name__)


# Vault configuration
VAULT_ENABLED = settings.VAULT_ENABLED
VAULT_URL = settings.VAULT_URL
VAULT_TOKEN = settings.VAULT_TOKEN
VAULT_MOUNT_POINT = settings.VAULT_MOUNT_POINT
VAULT_PATH_PREFIX = settings.VAULT_PATH_PREFIX


class VaultClient:
    """
    Client for HashiCorp Vault.
    
    This class provides methods for interacting with HashiCorp Vault,
    including secret storage, retrieval, and rotation.
    """
    
    def __init__(
        self,
        url: str = VAULT_URL,
        token: str = VAULT_TOKEN,
        mount_point: str = VAULT_MOUNT_POINT,
        path_prefix: str = VAULT_PATH_PREFIX
    ):
        """
        Initialize the Vault client.
        
        Args:
            url: Vault URL
            token: Vault token
            mount_point: Vault mount point
            path_prefix: Vault path prefix
        """
        self.url = url
        self.token = token
        self.mount_point = mount_point
        self.path_prefix = path_prefix
        self.enabled = VAULT_ENABLED
        
        # Cache for secrets
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def _request(
        self,
        method: str,
        path: str,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Send a request to Vault.
        
        Args:
            method: HTTP method
            path: API path
            json_data: JSON data
            
        Returns:
            Response data
        """
        if not self.enabled:
            raise ValueError("Vault integration is not enabled")
        
        url = f"{self.url}/v1/{path}"
        headers = {"X-Vault-Token": self.token}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.request(
                    method,
                    url,
                    headers=headers,
                    json=json_data,
                    timeout=10.0
                )
                
                if response.status_code == 404:
                    return {}
                
                response.raise_for_status()
                
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Vault HTTP error: {e.response.status_code} - {e.response.text}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vault error: {e.response.status_code}"
            )
        except Exception as e:
            logger.error(f"Vault error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Vault error: {str(e)}"
            )
    
    def _get_secret_path(self, path: str) -> str:
        """
        Get the full path for a secret.
        
        Args:
            path: Secret path
            
        Returns:
            Full secret path
        """
        return f"{self.mount_point}/data/{self.path_prefix}/{path}"
    
    async def get_secret(self, path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get a secret from Vault.
        
        Args:
            path: Secret path
            use_cache: Whether to use cache
            
        Returns:
            Secret data
        """
        if not self.enabled:
            # Return empty dict if Vault is not enabled
            return {}
        
        # Check cache
        cache_key = f"secret:{path}"
        if use_cache and cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self._cache_ttl:
                return cache_entry["data"]
        
        # Get secret from Vault
        secret_path = self._get_secret_path(path)
        response = await self._request("GET", secret_path)
        
        # Extract data
        data = response.get("data", {}).get("data", {})
        
        # Update cache
        if use_cache:
            self._cache[cache_key] = {
                "data": data,
                "timestamp": time.time()
            }
        
        return data
    
    async def set_secret(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set a secret in Vault.
        
        Args:
            path: Secret path
            data: Secret data
            
        Returns:
            Response data
        """
        if not self.enabled:
            # Return empty dict if Vault is not enabled
            return {}
        
        # Set secret in Vault
        secret_path = self._get_secret_path(path)
        response = await self._request(
            "POST",
            secret_path,
            json_data={"data": data}
        )
        
        # Clear cache
        cache_key = f"secret:{path}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return response
    
    async def delete_secret(self, path: str) -> Dict[str, Any]:
        """
        Delete a secret from Vault.
        
        Args:
            path: Secret path
            
        Returns:
            Response data
        """
        if not self.enabled:
            # Return empty dict if Vault is not enabled
            return {}
        
        # Delete secret from Vault
        secret_path = self._get_secret_path(path)
        response = await self._request("DELETE", secret_path)
        
        # Clear cache
        cache_key = f"secret:{path}"
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        return response
    
    async def list_secrets(self, path: str = "") -> List[str]:
        """
        List secrets in Vault.
        
        Args:
            path: Secret path
            
        Returns:
            List of secret names
        """
        if not self.enabled:
            # Return empty list if Vault is not enabled
            return []
        
        # List secrets in Vault
        list_path = f"{self.mount_point}/metadata/{self.path_prefix}/{path}"
        response = await self._request("LIST", list_path)
        
        # Extract keys
        keys = response.get("data", {}).get("keys", [])
        
        return keys
    
    async def get_api_key(self, provider: str, key_name: str = "api_key") -> Optional[str]:
        """
        Get an API key from Vault.
        
        Args:
            provider: Provider name
            key_name: Key name
            
        Returns:
            API key or None if not found
        """
        # Get secret
        secret = await self.get_secret(f"api_keys/{provider}")
        
        # Return API key
        return secret.get(key_name)
    
    async def set_api_key(self, provider: str, api_key: str, key_name: str = "api_key") -> Dict[str, Any]:
        """
        Set an API key in Vault.
        
        Args:
            provider: Provider name
            api_key: API key
            key_name: Key name
            
        Returns:
            Response data
        """
        # Get existing secret
        secret = await self.get_secret(f"api_keys/{provider}", use_cache=False)
        
        # Update API key
        secret[key_name] = api_key
        
        # Set secret
        return await self.set_secret(f"api_keys/{provider}", secret)
    
    async def rotate_api_key(
        self,
        provider: str,
        new_api_key: str,
        key_name: str = "api_key"
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Rotate an API key in Vault.
        
        Args:
            provider: Provider name
            new_api_key: New API key
            key_name: Key name
            
        Returns:
            Tuple of old API key and response data
        """
        # Get existing API key
        old_api_key = await self.get_api_key(provider, key_name)
        
        # Set new API key
        response = await self.set_api_key(provider, new_api_key, key_name)
        
        return old_api_key, response


# Global instance
_vault_client = None


def get_vault_client() -> VaultClient:
    """
    Get the global Vault client instance.
    
    Returns:
        VaultClient instance
    """
    global _vault_client
    
    if _vault_client is None:
        _vault_client = VaultClient()
    
    return _vault_client
