"""
Registry for CL-PEFT adapters.

This module provides a registry for managing CL-PEFT adapters, including:
- Adapter registration
- Adapter retrieval
- Adapter metadata management
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime
import logging

from asf.medical.core.logging_config import get_logger

logger = get_logger(__name__)

# Singleton instance
_cl_peft_registry = None

class CLPEFTAdapterStatus(str, Enum):
    """Status of a CL-PEFT adapter."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"
    DEPRECATED = "deprecated"

class CLPEFTAdapterRegistry:
    """Registry for CL-PEFT adapters."""
    
    def __init__(self, storage_dir: str = None):
        """
        Initialize the CL-PEFT adapter registry.
        
        Args:
            storage_dir: Directory for storing adapter metadata
        """
        self.storage_dir = storage_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..", "adapters"
        )
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # In-memory cache of adapter metadata
        self.adapters: Dict[str, Dict[str, Any]] = {}
        
        # Load existing adapters
        self._load_adapters()
        
        logger.info(f"Initialized CL-PEFT adapter registry with {len(self.adapters)} adapters")
    
    def _load_adapters(self):
        """Load adapter metadata from storage."""
        metadata_file = os.path.join(self.storage_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    self.adapters = json.load(f)
                logger.info(f"Loaded {len(self.adapters)} adapters from {metadata_file}")
            except Exception as e:
                logger.error(f"Error loading adapter metadata: {str(e)}")
                self.adapters = {}
    
    def _save_adapters(self):
        """Save adapter metadata to storage."""
        metadata_file = os.path.join(self.storage_dir, "metadata.json")
        try:
            with open(metadata_file, "w") as f:
                json.dump(self.adapters, f, indent=2)
            logger.info(f"Saved {len(self.adapters)} adapters to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving adapter metadata: {str(e)}")
    
    def register_adapter(self, adapter_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Register a new CL-PEFT adapter.
        
        Args:
            adapter_id: Unique identifier for the adapter
            metadata: Metadata for the adapter
            
        Returns:
            Success flag
        """
        if adapter_id in self.adapters:
            logger.warning(f"Adapter {adapter_id} already exists, updating metadata")
        
        # Add timestamp if not present
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.utcnow().isoformat()
        
        # Add status if not present
        if "status" not in metadata:
            metadata["status"] = CLPEFTAdapterStatus.INITIALIZING.value
        
        self.adapters[adapter_id] = metadata
        self._save_adapters()
        
        logger.info(f"Registered adapter {adapter_id}")
        return True
    
    def get_adapter(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Get adapter metadata.
        
        Args:
            adapter_id: Unique identifier for the adapter
            
        Returns:
            Adapter metadata or None if not found
        """
        return self.adapters.get(adapter_id)
    
    def list_adapters(self, filter_by: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List all adapters, optionally filtered.
        
        Args:
            filter_by: Filter criteria
            
        Returns:
            List of adapter metadata
        """
        adapters = list(self.adapters.values())
        
        if filter_by:
            for key, value in filter_by.items():
                adapters = [a for a in adapters if a.get(key) == value]
        
        return adapters
    
    def update_adapter_status(self, adapter_id: str, status: CLPEFTAdapterStatus) -> bool:
        """
        Update adapter status.
        
        Args:
            adapter_id: Unique identifier for the adapter
            status: New status
            
        Returns:
            Success flag
        """
        if adapter_id not in self.adapters:
            logger.error(f"Adapter {adapter_id} not found")
            return False
        
        self.adapters[adapter_id]["status"] = status.value
        self.adapters[adapter_id]["updated_at"] = datetime.utcnow().isoformat()
        self._save_adapters()
        
        logger.info(f"Updated adapter {adapter_id} status to {status.value}")
        return True
    
    def update_adapter_metadata(self, adapter_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update adapter metadata.
        
        Args:
            adapter_id: Unique identifier for the adapter
            metadata: New metadata (partial update)
            
        Returns:
            Success flag
        """
        if adapter_id not in self.adapters:
            logger.error(f"Adapter {adapter_id} not found")
            return False
        
        self.adapters[adapter_id].update(metadata)
        self.adapters[adapter_id]["updated_at"] = datetime.utcnow().isoformat()
        self._save_adapters()
        
        logger.info(f"Updated adapter {adapter_id} metadata")
        return True
    
    def delete_adapter(self, adapter_id: str) -> bool:
        """
        Delete an adapter.
        
        Args:
            adapter_id: Unique identifier for the adapter
            
        Returns:
            Success flag
        """
        if adapter_id not in self.adapters:
            logger.error(f"Adapter {adapter_id} not found")
            return False
        
        del self.adapters[adapter_id]
        self._save_adapters()
        
        logger.info(f"Deleted adapter {adapter_id}")
        return True
    
    def get_adapter_path(self, adapter_id: str) -> str:
        """
        Get the path to the adapter directory.
        
        Args:
            adapter_id: Unique identifier for the adapter
            
        Returns:
            Path to the adapter directory
        """
        return os.path.join(self.storage_dir, adapter_id)

def get_cl_peft_registry() -> CLPEFTAdapterRegistry:
    """
    Get the singleton instance of the CL-PEFT adapter registry.
    
    Returns:
        CLPEFTAdapterRegistry instance
    """
    global _cl_peft_registry
    if _cl_peft_registry is None:
        _cl_peft_registry = CLPEFTAdapterRegistry()
    return _cl_peft_registry
