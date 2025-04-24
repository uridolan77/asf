"""
Model Registry for ML Model Lifecycle Management

This module provides functionality for registering, versioning, and
managing machine learning models throughout their lifecycle.
"""

import os
import json
import hashlib
import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, asdict

import numpy as np

from asf.medical.core.logging_config import get_logger
from asf.medical.core.enhanced_cache import EnhancedCacheManager
from asf.medical.core.singleton import Singleton

logger = get_logger(__name__)

# Model frameworks
class ModelFramework(str, Enum):
    """Supported model frameworks."""
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    HUGGINGFACE = "huggingface"
    SPACY = "spacy"
    ONNX = "onnx"
    CUSTOM = "custom"

# Model status
class ModelStatus(str, Enum):
    """Model status in the lifecycle."""
    DEVELOPMENT = "development"  # Under development
    TESTING = "testing"         # In testing
    STAGING = "staging"         # In staging environment
    PRODUCTION = "production"   # In production
    DEPRECATED = "deprecated"   # Deprecated but still available
    ARCHIVED = "archived"       # Archived, not active

@dataclass
class ModelMetrics:
    """Metrics for evaluating model performance."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    custom_metrics: Optional[Dict[str, float]] = None

@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    name: str
    version: str
    framework: ModelFramework
    description: str
    status: ModelStatus
    metrics: ModelMetrics
    created_at: str
    updated_at: str
    path: Optional[str] = None
    parent_version: Optional[str] = None
    training_dataset_hash: Optional[str] = None
    created_by: Optional[str] = None
    tags: Optional[List[str]] = None
    parameters: Optional[Dict[str, Any]] = None
    drift_metrics: Optional[Dict[str, Any]] = None
    monitoring_config: Optional[Dict[str, Any]] = None
    deployment_config: Optional[Dict[str, Any]] = None

class ModelRegistry(metaclass=Singleton):
    """
    Registry for ML models with versioning and lifecycle management.
    
    This class implements the singleton pattern to ensure a single instance
    is used throughout the application.
    """
    
    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            storage_dir: Directory for storing model registry data.
                If None, uses the default location.
        """
        if storage_dir is None:
            # Use default storage location relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            storage_dir = os.path.join(base_dir, "models", "registry")
        
        self.storage_dir = storage_dir
        self.registry_file = os.path.join(storage_dir, "registry.json")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize registry
        self.registry: Dict[str, Dict[str, ModelMetadata]] = {}
        self._load_registry()
        
        logger.info(f"Model registry initialized at {storage_dir}")
    
    def register_model(
        self,
        name: str,
        version: str,
        framework: ModelFramework,
        description: str,
        status: ModelStatus = ModelStatus.DEVELOPMENT,
        metrics: Optional[ModelMetrics] = None,
        path: Optional[str] = None,
        parent_version: Optional[str] = None,
        training_dataset_hash: Optional[str] = None,
        created_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ModelMetadata:
        """
        Register a new model version.
        
        Args:
            name: Name of the model.
            version: Version of the model (semantic versioning recommended).
            framework: Framework used to build the model.
            description: Description of the model.
            status: Status of the model in the lifecycle.
            metrics: Performance metrics for the model.
            path: Path to the model files.
            parent_version: Previous version of the model if applicable.
            training_dataset_hash: Hash of the training dataset.
            created_by: User or service that created the model.
            tags: Tags for organizing models.
            parameters: Model parameters and hyperparameters.
            
        Returns:
            Metadata for the registered model.
        """
        # Ensure metrics is not None
        if metrics is None:
            metrics = ModelMetrics()
        
        # Check if model with this name and version already exists
        if name in self.registry and version in self.registry[name]:
            raise ValueError(f"Model {name} version {version} already exists")
        
        # Initialize model namespace if it doesn't exist
        if name not in self.registry:
            self.registry[name] = {}
        
        # Create timestamp
        now = datetime.datetime.now().isoformat()
        
        # Create model metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            framework=framework,
            description=description,
            status=status,
            metrics=metrics,
            created_at=now,
            updated_at=now,
            path=path,
            parent_version=parent_version,
            training_dataset_hash=training_dataset_hash,
            created_by=created_by,
            tags=tags,
            parameters=parameters,
            drift_metrics={},
            monitoring_config={},
            deployment_config={}
        )
        
        # Register the model
        self.registry[name][version] = metadata
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Registered model {name} version {version} with status {status.value}")
        return metadata
    
    def update_model_metadata(
        self,
        name: str,
        version: str,
        **kwargs
    ) -> Optional[ModelMetadata]:
        """
        Update metadata for a model version.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            **kwargs: Fields to update.
            
        Returns:
            Updated metadata or None if model not found.
        """
        # Check if model exists
        if name not in self.registry or version not in self.registry[name]:
            logger.warning(f"Model {name} version {version} not found")
            return None
        
        # Get current metadata
        metadata = self.registry[name][version]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        # Update timestamp
        metadata.updated_at = datetime.datetime.now().isoformat()
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Updated metadata for model {name} version {version}")
        return metadata
    
    def update_model_status(
        self,
        name: str,
        version: str,
        status: ModelStatus
    ) -> Optional[ModelMetadata]:
        """
        Update the status of a model.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            status: New status for the model.
            
        Returns:
            Updated metadata or None if model not found.
        """
        return self.update_model_metadata(name, version, status=status)
    
    def get_model(self, name: str, version: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a specific model version.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            
        Returns:
            Model metadata or None if not found.
        """
        if name not in self.registry or version not in self.registry[name]:
            return None
        
        return self.registry[name][version]
    
    def get_model_versions(self, name: str) -> Dict[str, ModelMetadata]:
        """
        Get all versions of a model.
        
        Args:
            name: Name of the model.
            
        Returns:
            Dictionary of model versions and their metadata.
        """
        if name not in self.registry:
            return {}
        
        return self.registry[name]
    
    def get_latest_version(self, name: str, status: Optional[Set[ModelStatus]] = None) -> Optional[ModelMetadata]:
        """
        Get the latest version of a model.
        
        Args:
            name: Name of the model.
            status: Optional set of statuses to filter by.
            
        Returns:
            Metadata for the latest version or None if not found.
        """
        if name not in self.registry:
            return None
        
        versions = self.registry[name]
        if not versions:
            return None
        
        # Filter by status if provided
        if status is not None:
            versions = {k: v for k, v in versions.items() if v.status in status}
            if not versions:
                return None
        
        # Find the latest version using semantic versioning
        try:
            latest_version = max(versions.keys(), key=lambda v: [int(x) for x in v.split(".")])
            return versions[latest_version]
        except (ValueError, TypeError):
            # Fall back to alphabetical sorting if semantic versioning fails
            latest_version = max(versions.keys())
            return versions[latest_version]
    
    def get_production_model(self, name: str) -> Optional[ModelMetadata]:
        """
        Get the production version of a model.
        
        Args:
            name: Name of the model.
            
        Returns:
            Metadata for the production version or None if not found.
        """
        if name not in self.registry:
            return None
        
        # Find a version with production status
        for version, metadata in self.registry[name].items():
            if metadata.status == ModelStatus.PRODUCTION:
                return metadata
        
        return None
    
    def list_models(self) -> List[str]:
        """
        List all registered models.
        
        Returns:
            List of model names.
        """
        return list(self.registry.keys())
    
    def delete_model_version(self, name: str, version: str) -> bool:
        """
        Delete a specific model version.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            
        Returns:
            True if deleted, False if not found.
        """
        if name not in self.registry or version not in self.registry[name]:
            return False
        
        # Delete the model version
        del self.registry[name][version]
        
        # Remove the model if no versions remain
        if not self.registry[name]:
            del self.registry[name]
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Deleted model {name} version {version}")
        return True
    
    def compute_dataset_hash(self, data: Any) -> str:
        """
        Compute a hash of a dataset for tracking.
        
        Args:
            data: Dataset to hash.
            
        Returns:
            Hash string representing the dataset.
        """
        # Convert to JSON and hash
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def record_drift_metrics(
        self,
        name: str,
        version: str,
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Record model drift metrics.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            metrics: Drift metrics to record.
            
        Returns:
            True if successful, False if model not found.
        """
        if name not in self.registry or version not in self.registry[name]:
            return False
        
        metadata = self.registry[name][version]
        
        # Add timestamp
        metrics["timestamp"] = datetime.datetime.now().isoformat()
        
        # Initialize drift_metrics if not present
        if metadata.drift_metrics is None:
            metadata.drift_metrics = {}
        
        # Add metrics to history
        if "history" not in metadata.drift_metrics:
            metadata.drift_metrics["history"] = []
        
        metadata.drift_metrics["history"].append(metrics)
        metadata.drift_metrics["latest"] = metrics
        metadata.updated_at = datetime.datetime.now().isoformat()
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Recorded drift metrics for model {name} version {version}")
        return True
    
    def configure_monitoring(
        self,
        name: str,
        version: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Configure monitoring for a model.
        
        Args:
            name: Name of the model.
            version: Version of the model.
            config: Monitoring configuration.
            
        Returns:
            True if successful, False if model not found.
        """
        if name not in self.registry or version not in self.registry[name]:
            return False
        
        metadata = self.registry[name][version]
        metadata.monitoring_config = config
        metadata.updated_at = datetime.datetime.now().isoformat()
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Configured monitoring for model {name} version {version}")
        return True
    
    def _load_registry(self):
        """Load the registry from file."""
        if not os.path.exists(self.registry_file):
            return
        
        try:
            with open(self.registry_file, "r") as f:
                registry_data = json.load(f)
            
            # Convert registry data to ModelMetadata objects
            for name, versions in registry_data.items():
                self.registry[name] = {}
                for version, data in versions.items():
                    # Convert metrics to ModelMetrics
                    metrics_data = data.pop("metrics", {})
                    metrics = ModelMetrics(**metrics_data)
                    
                    # Create ModelMetadata
                    metadata = ModelMetadata(metrics=metrics, **data)
                    self.registry[name][version] = metadata
            
            logger.info(f"Loaded {len(self.registry)} models from registry")
        except Exception as e:
            logger.error(f"Error loading registry: {str(e)}")
    
    def _save_registry(self):
        """Save the registry to file."""
        try:
            # Convert registry to serializable form
            registry_data = {}
            for name, versions in self.registry.items():
                registry_data[name] = {}
                for version, metadata in versions.items():
                    # Convert to dictionary
                    metadata_dict = asdict(metadata)
                    registry_data[name][version] = metadata_dict
            
            # Save to file
            with open(self.registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info(f"Saved registry with {len(self.registry)} models")
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")

# Global singleton instance
_registry_instance = None

def get_model_registry(storage_dir: Optional[str] = None) -> ModelRegistry:
    """
    Get or create the model registry instance.
    
    Args:
        storage_dir: Optional storage directory.
        
    Returns:
        Model registry instance.
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ModelRegistry(storage_dir)
    return _registry_instance