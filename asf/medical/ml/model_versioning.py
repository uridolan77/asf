"""Model versioning module for the Medical Research Synthesizer.

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
    """Metadata for a model version.
    
    This class holds metadata information for a specific model version
    including its name, version, description, and performance metrics.
    """
        Register a model in the registry.

        Args:
            model_name: Name of the model
            model_path: Path to the model file or directory
            description: Description of the model
            version: Version of the model (optional)
            tags: Tags for the model (optional)
            metrics: Metrics for the model (optional)
            parameters: Parameters for the model (optional)
            artifacts: Artifacts for the model (optional)
            dependencies: Dependencies for the model (optional)

        Returns:
            ModelMetadata: Metadata for the registered model
        """
        # Generate version if not provided
        if version is None:
            version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        # Create metadata
        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            description=description,
            tags=tags or {},
            metrics=metrics or {},
            parameters=parameters or {},
            artifacts=artifacts or {},
            dependencies=dependencies or {}
        )

        # Save to MLflow if enabled
        if self._use_mlflow:
            try:
                # Start a new MLflow run
                with mlflow.start_run():
                    # Log parameters
                    for key, value in metadata.parameters.items():
                        mlflow.log_param(key, value)

                    # Log metrics
                    for key, value in metadata.metrics.items():
                        mlflow.log_metric(key, value)

                    # Log tags
                    for key, value in metadata.tags.items():
                        mlflow.set_tag(key, value)

                    # Log model
                    mlflow.log_artifact(str(model_path))

                    # Register model
                    mlflow.register_model(f"runs:/{{mlflow.active_run().info.run_id}}/artifacts/{model_path}", model_name)

                logger.info(f"Registered model {model_name} version {version} in MLflow")
            except Exception as e:
                logger.error(f"Failed to register model in MLflow: {e}")

        # Save locally
        model_dir = self._registry_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy model files
        import shutil
        model_path = Path(model_path)
        if model_path.is_dir():
            shutil.copytree(model_path, model_dir / "model", dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, model_dir / model_path.name)

        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            import json
            json.dump(metadata.to_dict(), f, indent=2)

        logger.info(f"Registered model {model_name} version {version} in local registry")

        return metadata