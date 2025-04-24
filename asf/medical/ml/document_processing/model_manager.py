"""
Model Manager Module

This module provides functionality for managing model lifecycle, including
versioning, monitoring, and automated retraining.
"""

import os
import json
import logging
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manager for model lifecycle.
    
    This class provides functionality for managing model lifecycle, including
    versioning, monitoring, and automated retraining.
    """
    
    def __init__(
        self,
        model_dir: str,
        metadata_file: str = "model_metadata.json",
        performance_threshold: float = 0.7,
        drift_threshold: float = 0.1,
        auto_retrain: bool = False
    ):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory for model storage
            metadata_file: Metadata file name
            performance_threshold: Performance threshold for model acceptance
            drift_threshold: Drift threshold for model retraining
            auto_retrain: Whether to automatically retrain models
        """
        self.model_dir = model_dir
        self.metadata_file = os.path.join(model_dir, metadata_file)
        self.performance_threshold = performance_threshold
        self.drift_threshold = drift_threshold
        self.auto_retrain = auto_retrain
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Load metadata if it exists
        self.metadata = self._load_metadata()
        
        # Initialize locks for thread safety
        self.metadata_lock = threading.Lock()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load model metadata from file.
        
        Returns:
            Model metadata
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading model metadata: {str(e)}")
        
        # Initialize empty metadata
        return {
            "models": {},
            "versions": {},
            "performance": {},
            "drift": {},
            "last_updated": datetime.datetime.now().isoformat()
        }
    
    def _save_metadata(self) -> None:
        """
        Save model metadata to file.
        """
        with self.metadata_lock:
            try:
                # Update last updated timestamp
                self.metadata["last_updated"] = datetime.datetime.now().isoformat()
                
                with open(self.metadata_file, 'w') as f:
                    json.dump(self.metadata, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving model metadata: {str(e)}")
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        version: str,
        model_path: str,
        performance_metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a model with the model manager.
        
        Args:
            model_name: Model name
            model_type: Model type (e.g., entity_extractor, relation_extractor)
            version: Model version
            model_path: Path to model file
            performance_metrics: Performance metrics
            metadata: Additional metadata
            
        Returns:
            True if registration was successful, False otherwise
        """
        with self.metadata_lock:
            try:
                # Create model entry if it doesn't exist
                if model_name not in self.metadata["models"]:
                    self.metadata["models"][model_name] = {
                        "type": model_type,
                        "versions": [],
                        "current_version": None
                    }
                
                # Check if version already exists
                if version in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} already exists")
                    return False
                
                # Add version
                self.metadata["models"][model_name]["versions"].append(version)
                
                # Create version entry
                self.metadata["versions"][f"{model_name}_{version}"] = {
                    "model_name": model_name,
                    "version": version,
                    "model_path": model_path,
                    "created_at": datetime.datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                
                # Add performance metrics
                self.metadata["performance"][f"{model_name}_{version}"] = performance_metrics
                
                # Check if this is the first version
                if self.metadata["models"][model_name]["current_version"] is None:
                    self.metadata["models"][model_name]["current_version"] = version
                else:
                    # Check if this version performs better than the current version
                    current_version = self.metadata["models"][model_name]["current_version"]
                    current_performance = self.metadata["performance"][f"{model_name}_{current_version}"]
                    
                    # Compare primary metric (first metric in the dictionary)
                    primary_metric = list(performance_metrics.keys())[0]
                    if performance_metrics[primary_metric] > current_performance.get(primary_metric, 0):
                        self.metadata["models"][model_name]["current_version"] = version
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Registered model {model_name} version {version}")
                return True
            
            except Exception as e:
                logger.error(f"Error registering model: {str(e)}")
                return False
    
    def get_model_path(self, model_name: str, version: Optional[str] = None) -> Optional[str]:
        """
        Get the path to a model.
        
        Args:
            model_name: Model name
            version: Model version (if None, use current version)
            
        Returns:
            Path to model file or None if not found
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return None
                
                # Get version
                if version is None:
                    version = self.metadata["models"][model_name]["current_version"]
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return None
                
                # Get model path
                return self.metadata["versions"][f"{model_name}_{version}"]["model_path"]
            
            except Exception as e:
                logger.error(f"Error getting model path: {str(e)}")
                return None
    
    def get_model_performance(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, float]]:
        """
        Get performance metrics for a model.
        
        Args:
            model_name: Model name
            version: Model version (if None, use current version)
            
        Returns:
            Performance metrics or None if not found
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return None
                
                # Get version
                if version is None:
                    version = self.metadata["models"][model_name]["current_version"]
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return None
                
                # Get performance metrics
                return self.metadata["performance"].get(f"{model_name}_{version}")
            
            except Exception as e:
                logger.error(f"Error getting model performance: {str(e)}")
                return None
    
    def update_model_performance(
        self,
        model_name: str,
        version: str,
        performance_metrics: Dict[str, float]
    ) -> bool:
        """
        Update performance metrics for a model.
        
        Args:
            model_name: Model name
            version: Model version
            performance_metrics: Performance metrics
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return False
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return False
                
                # Update performance metrics
                self.metadata["performance"][f"{model_name}_{version}"] = performance_metrics
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Updated performance metrics for model {model_name} version {version}")
                return True
            
            except Exception as e:
                logger.error(f"Error updating model performance: {str(e)}")
                return False
    
    def record_drift(
        self,
        model_name: str,
        version: str,
        drift_metrics: Dict[str, float]
    ) -> bool:
        """
        Record drift metrics for a model.
        
        Args:
            model_name: Model name
            version: Model version
            drift_metrics: Drift metrics
            
        Returns:
            True if recording was successful, False otherwise
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return False
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return False
                
                # Add drift metrics
                if f"{model_name}_{version}" not in self.metadata["drift"]:
                    self.metadata["drift"][f"{model_name}_{version}"] = []
                
                self.metadata["drift"][f"{model_name}_{version}"].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "metrics": drift_metrics
                })
                
                # Save metadata
                self._save_metadata()
                
                # Check if drift exceeds threshold
                primary_metric = list(drift_metrics.keys())[0]
                if drift_metrics[primary_metric] > self.drift_threshold:
                    logger.warning(f"Drift detected for model {model_name} version {version}")
                    
                    # Trigger retraining if auto_retrain is enabled
                    if self.auto_retrain:
                        logger.info(f"Auto-retraining model {model_name}")
                        # TODO: Implement auto-retraining
                
                logger.info(f"Recorded drift metrics for model {model_name} version {version}")
                return True
            
            except Exception as e:
                logger.error(f"Error recording drift: {str(e)}")
                return False
    
    def set_current_version(self, model_name: str, version: str) -> bool:
        """
        Set the current version for a model.
        
        Args:
            model_name: Model name
            version: Model version
            
        Returns:
            True if update was successful, False otherwise
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return False
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return False
                
                # Update current version
                self.metadata["models"][model_name]["current_version"] = version
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Set current version for model {model_name} to {version}")
                return True
            
            except Exception as e:
                logger.error(f"Error setting current version: {str(e)}")
                return False
    
    def get_model_versions(self, model_name: str) -> Optional[List[str]]:
        """
        Get all versions for a model.
        
        Args:
            model_name: Model name
            
        Returns:
            List of versions or None if model not found
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return None
                
                # Get versions
                return self.metadata["models"][model_name]["versions"]
            
            except Exception as e:
                logger.error(f"Error getting model versions: {str(e)}")
                return None
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a model.
        
        Args:
            model_name: Model name
            version: Model version (if None, use current version)
            
        Returns:
            Model metadata or None if not found
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return None
                
                # Get version
                if version is None:
                    version = self.metadata["models"][model_name]["current_version"]
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return None
                
                # Get metadata
                return self.metadata["versions"][f"{model_name}_{version}"]["metadata"]
            
            except Exception as e:
                logger.error(f"Error getting model metadata: {str(e)}")
                return None
    
    def delete_model_version(self, model_name: str, version: str, delete_files: bool = False) -> bool:
        """
        Delete a model version.
        
        Args:
            model_name: Model name
            version: Model version
            delete_files: Whether to delete model files
            
        Returns:
            True if deletion was successful, False otherwise
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return False
                
                # Check if version exists
                if version not in self.metadata["models"][model_name]["versions"]:
                    logger.warning(f"Model {model_name} version {version} not found")
                    return False
                
                # Get model path
                model_path = self.metadata["versions"][f"{model_name}_{version}"]["model_path"]
                
                # Delete model files if requested
                if delete_files and model_path and os.path.exists(model_path):
                    if os.path.isdir(model_path):
                        shutil.rmtree(model_path)
                    else:
                        os.remove(model_path)
                
                # Remove version from model
                self.metadata["models"][model_name]["versions"].remove(version)
                
                # Update current version if needed
                if self.metadata["models"][model_name]["current_version"] == version:
                    if self.metadata["models"][model_name]["versions"]:
                        # Set current version to latest version
                        self.metadata["models"][model_name]["current_version"] = self.metadata["models"][model_name]["versions"][-1]
                    else:
                        # No versions left
                        self.metadata["models"][model_name]["current_version"] = None
                
                # Remove version entry
                if f"{model_name}_{version}" in self.metadata["versions"]:
                    del self.metadata["versions"][f"{model_name}_{version}"]
                
                # Remove performance metrics
                if f"{model_name}_{version}" in self.metadata["performance"]:
                    del self.metadata["performance"][f"{model_name}_{version}"]
                
                # Remove drift metrics
                if f"{model_name}_{version}" in self.metadata["drift"]:
                    del self.metadata["drift"][f"{model_name}_{version}"]
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Deleted model {model_name} version {version}")
                return True
            
            except Exception as e:
                logger.error(f"Error deleting model version: {str(e)}")
                return False
    
    def delete_model(self, model_name: str, delete_files: bool = False) -> bool:
        """
        Delete a model and all its versions.
        
        Args:
            model_name: Model name
            delete_files: Whether to delete model files
            
        Returns:
            True if deletion was successful, False otherwise
        """
        with self.metadata_lock:
            try:
                # Check if model exists
                if model_name not in self.metadata["models"]:
                    logger.warning(f"Model {model_name} not found")
                    return False
                
                # Get versions
                versions = self.metadata["models"][model_name]["versions"].copy()
                
                # Delete each version
                for version in versions:
                    self.delete_model_version(model_name, version, delete_files)
                
                # Remove model entry
                del self.metadata["models"][model_name]
                
                # Save metadata
                self._save_metadata()
                
                logger.info(f"Deleted model {model_name}")
                return True
            
            except Exception as e:
                logger.error(f"Error deleting model: {str(e)}")
                return False
    
    def export_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        export_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Export a model to a directory.
        
        Args:
            model_name: Model name
            version: Model version (if None, use current version)
            export_dir: Export directory (if None, use model_dir/exports)
            
        Returns:
            Path to exported model or None if export failed
        """
        try:
            # Get model path
            model_path = self.get_model_path(model_name, version)
            if not model_path:
                return None
            
            # Get version
            if version is None:
                version = self.metadata["models"][model_name]["current_version"]
            
            # Create export directory
            if export_dir is None:
                export_dir = os.path.join(self.model_dir, "exports")
            
            os.makedirs(export_dir, exist_ok=True)
            
            # Create export path
            export_path = os.path.join(export_dir, f"{model_name}_{version}")
            os.makedirs(export_path, exist_ok=True)
            
            # Copy model files
            if os.path.isdir(model_path):
                # Copy directory contents
                for item in os.listdir(model_path):
                    s = os.path.join(model_path, item)
                    d = os.path.join(export_path, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
            else:
                # Copy file
                shutil.copy2(model_path, export_path)
            
            # Export metadata
            metadata = {
                "model_name": model_name,
                "version": version,
                "exported_at": datetime.datetime.now().isoformat(),
                "metadata": self.get_model_metadata(model_name, version),
                "performance": self.get_model_performance(model_name, version)
            }
            
            with open(os.path.join(export_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Exported model {model_name} version {version} to {export_path}")
            return export_path
        
        except Exception as e:
            logger.error(f"Error exporting model: {str(e)}")
            return None
    
    def import_model(
        self,
        import_path: str,
        overwrite: bool = False
    ) -> Optional[Tuple[str, str]]:
        """
        Import a model from a directory.
        
        Args:
            import_path: Path to imported model
            overwrite: Whether to overwrite existing model
            
        Returns:
            Tuple of (model_name, version) or None if import failed
        """
        try:
            # Check if import path exists
            if not os.path.exists(import_path):
                logger.warning(f"Import path {import_path} does not exist")
                return None
            
            # Load metadata
            metadata_path = os.path.join(import_path, "metadata.json")
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found at {metadata_path}")
                return None
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get model name and version
            model_name = metadata["model_name"]
            version = metadata["version"]
            
            # Check if model already exists
            if model_name in self.metadata["models"] and version in self.metadata["models"][model_name]["versions"]:
                if not overwrite:
                    logger.warning(f"Model {model_name} version {version} already exists")
                    return None
                
                # Delete existing model version
                self.delete_model_version(model_name, version, delete_files=True)
            
            # Create model directory
            model_dir = os.path.join(self.model_dir, model_name, version)
            os.makedirs(model_dir, exist_ok=True)
            
            # Copy model files
            for item in os.listdir(import_path):
                if item == "metadata.json":
                    continue
                
                s = os.path.join(import_path, item)
                d = os.path.join(model_dir, item)
                
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            
            # Register model
            self.register_model(
                model_name=model_name,
                model_type=metadata.get("model_type", "unknown"),
                version=version,
                model_path=model_dir,
                performance_metrics=metadata.get("performance", {}),
                metadata=metadata.get("metadata", {})
            )
            
            logger.info(f"Imported model {model_name} version {version} from {import_path}")
            return (model_name, version)
        
        except Exception as e:
            logger.error(f"Error importing model: {str(e)}")
            return None
