"""
Model versioning module for the Medical Research Synthesizer.
This module provides utilities for versioning ML models and tracking experiments.
"""
import os
import logging
import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
import mlflow
from mlflow.tracking import MlflowClient
logger = logging.getLogger(__name__)
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
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self):
        """Initialize the model registry.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
        if self._initialized:
            return
        self._initialized = True
        self._registry_dir = MODEL_REGISTRY_DIR
        self._registry_dir.mkdir(exist_ok=True)
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