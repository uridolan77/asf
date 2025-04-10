"""
Model versioning module for the Medical Research Synthesizer.

This module provides utilities for versioning ML models and tracking experiments.
"""

import os
import json
import logging
import hashlib
import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
from dataclasses import dataclass, field, asdict

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Run

from asf.medical.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MODEL_REGISTRY_DIR = Path(os.environ.get("MODEL_REGISTRY_DIR", "model_registry"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")


@dataclass
class ModelMetadata:
    """Metadata for a model version."""
    
    model_name: str
    version: str
    description: str
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    framework: str = "pytorch"
    tags: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    dependencies: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Model registry for the Medical Research Synthesizer.
    
    This class provides utilities for versioning ML models and tracking experiments.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the model registry."""
        if self._initialized:
            return
        
        self._initialized = True
        self._registry_dir = MODEL_REGISTRY_DIR
        self._registry_dir.mkdir(exist_ok=True)
        
        # Initialize MLflow if tracking URI is provided
        self._use_mlflow = bool(MLFLOW_TRACKING_URI)
        if self._use_mlflow:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self._mlflow_client = MlflowClient()
            logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
        else:
            logger.info("MLflow tracking URI not provided, using local model registry")
    
    def register_model(
        self,
        model_name: str,
        model_path: Union[str, Path],
        description: str,
        version: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        dependencies: Optional[Dict[str, str]] = None,
    ) -> ModelMetadata:
        """
        Register a model in the registry.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            description: Description of the model
            version: Version of the model (default: auto-generated)
            tags: Tags for the model
            metrics: Metrics for the model
            parameters: Parameters for the model
            artifacts: Artifacts for the model
            dependencies: Dependencies for the model
            
        Returns:
            ModelMetadata: Metadata for the registered model
        """
        # Convert model_path to Path
        model_path = Path(model_path)
        
        # Generate version if not provided
        if version is None:
            version = self._generate_version(model_name, model_path)
        
        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            description=description,
            tags=tags or {},
            metrics=metrics or {},
            parameters=parameters or {},
            artifacts=artifacts or {},
            dependencies=dependencies or {},
        )
        
        # Register with MLflow if available
        if self._use_mlflow:
            self._register_with_mlflow(metadata, model_path)
        
        # Register locally
        self._register_locally(metadata, model_path)
        
        return metadata
    
    def _generate_version(self, model_name: str, model_path: Path) -> str:
        """
        Generate a version for the model.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            
        Returns:
            str: Generated version
        """
        # Generate a hash of the model file
        with open(model_path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Combine hash and timestamp
        return f"{timestamp}-{file_hash[:8]}"
    
    def _register_with_mlflow(self, metadata: ModelMetadata, model_path: Path) -> None:
        """
        Register a model with MLflow.
        
        Args:
            metadata: Metadata for the model
            model_path: Path to the model file
        """
        try:
            # Start a new run
            with mlflow.start_run() as run:
                # Log parameters
                for key, value in metadata.parameters.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                for key, value in metadata.metrics.items():
                    mlflow.log_metric(key, value)
                
                # Log tags
                for key, value in metadata.tags.items():
                    mlflow.set_tag(key, value)
                
                # Log artifacts
                for key, value in metadata.artifacts.items():
                    mlflow.log_artifact(value, key)
                
                # Log model
                if metadata.framework == "pytorch":
                    import torch
                    mlflow.pytorch.log_model(torch.load(model_path), metadata.model_name)
                elif metadata.framework == "tensorflow":
                    import tensorflow as tf
                    mlflow.tensorflow.log_model(tf.keras.models.load_model(model_path), metadata.model_name)
                else:
                    mlflow.pyfunc.log_model(model_path, metadata.model_name)
                
                # Log metadata
                mlflow.log_dict(metadata.to_dict(), "metadata.json")
                
                # Register model
                mlflow.register_model(f"runs:/{run.info.run_id}/{metadata.model_name}", metadata.model_name)
                
                logger.info(f"Registered model {metadata.model_name} version {metadata.version} with MLflow")
        except Exception as e:
            logger.error(f"Error registering model with MLflow: {e}")
            raise
    
    def _register_locally(self, metadata: ModelMetadata, model_path: Path) -> None:
        """
        Register a model locally.
        
        Args:
            metadata: Metadata for the model
            model_path: Path to the model file
        """
        try:
            # Create model directory
            model_dir = self._registry_dir / metadata.model_name / metadata.version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model file
            import shutil
            shutil.copy2(model_path, model_dir / model_path.name)
            
            # Save metadata
            with open(model_dir / "metadata.json", "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Update latest version
            with open(self._registry_dir / metadata.model_name / "latest_version.txt", "w") as f:
                f.write(metadata.version)
            
            logger.info(f"Registered model {metadata.model_name} version {metadata.version} locally")
        except Exception as e:
            logger.error(f"Error registering model locally: {e}")
            raise
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> ModelMetadata:
        """
        Get metadata for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: latest)
            
        Returns:
            ModelMetadata: Metadata for the model
            
        Raises:
            FileNotFoundError: If the model is not found
        """
        # Get version if not provided
        if version is None:
            version = self.get_latest_version(model_name)
        
        # Check if model exists locally
        model_dir = self._registry_dir / model_name / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} version {version} not found")
        
        # Load metadata
        with open(model_dir / "metadata.json", "r") as f:
            metadata = ModelMetadata.from_dict(json.load(f))
        
        return metadata
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Path:
        """
        Get the path to a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model (default: latest)
            
        Returns:
            Path: Path to the model
            
        Raises:
            FileNotFoundError: If the model is not found
        """
        # Get version if not provided
        if version is None:
            version = self.get_latest_version(model_name)
        
        # Check if model exists locally
        model_dir = self._registry_dir / model_name / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} version {version} not found")
        
        # Find model file
        model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.pth")) + list(model_dir.glob("*.bin"))
        if not model_files:
            raise FileNotFoundError(f"Model file not found for {model_name} version {version}")
        
        return model_files[0]
    
    def get_latest_version(self, model_name: str) -> str:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            str: Latest version
            
        Raises:
            FileNotFoundError: If the model is not found
        """
        # Check if model exists locally
        model_dir = self._registry_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Check if latest version file exists
        latest_version_file = model_dir / "latest_version.txt"
        if latest_version_file.exists():
            with open(latest_version_file, "r") as f:
                return f.read().strip()
        
        # Find latest version by directory name
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No versions found for model {model_name}")
        
        return sorted(versions)[-1]
    
    def list_models(self) -> List[str]:
        """
        List all models in the registry.
        
        Returns:
            List[str]: List of model names
        """
        return [d.name for d in self._registry_dir.iterdir() if d.is_dir()]
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List[str]: List of versions
            
        Raises:
            FileNotFoundError: If the model is not found
        """
        # Check if model exists locally
        model_dir = self._registry_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Find versions by directory name
        versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
        if not versions:
            raise FileNotFoundError(f"No versions found for model {model_name}")
        
        return sorted(versions)
    
    def delete_model(self, model_name: str) -> None:
        """
        Delete a model from the registry.
        
        Args:
            model_name: Name of the model
            
        Raises:
            FileNotFoundError: If the model is not found
        """
        # Check if model exists locally
        model_dir = self._registry_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Delete model directory
        import shutil
        shutil.rmtree(model_dir)
        
        logger.info(f"Deleted model {model_name}")
    
    def delete_version(self, model_name: str, version: str) -> None:
        """
        Delete a version of a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Raises:
            FileNotFoundError: If the model or version is not found
        """
        # Check if model exists locally
        model_dir = self._registry_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Check if version exists
        version_dir = model_dir / version
        if not version_dir.exists():
            raise FileNotFoundError(f"Version {version} of model {model_name} not found")
        
        # Delete version directory
        import shutil
        shutil.rmtree(version_dir)
        
        # Update latest version if needed
        latest_version_file = model_dir / "latest_version.txt"
        if latest_version_file.exists():
            with open(latest_version_file, "r") as f:
                latest_version = f.read().strip()
            
            if latest_version == version:
                # Find new latest version
                versions = [d.name for d in model_dir.iterdir() if d.is_dir()]
                if versions:
                    with open(latest_version_file, "w") as f:
                        f.write(sorted(versions)[-1])
                else:
                    # No versions left, delete latest version file
                    latest_version_file.unlink()
        
        logger.info(f"Deleted version {version} of model {model_name}")


# Create a singleton instance
model_registry = ModelRegistry()

# Export the instance
__all__ = ["model_registry", "ModelMetadata"]
