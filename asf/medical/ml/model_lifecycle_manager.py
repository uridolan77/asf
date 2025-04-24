"""
Model Lifecycle Management System

This module provides robust model lifecycle management capabilities including:
1. Versioning: Track model lineage and development history
2. Monitoring: Track model drift and performance degradation
3. Retraining: Automated retraining pipelines with A/B testing
4. Deployment: Manage model deployment across environments

Builds upon the existing ModelRegistry to provide enhanced lifecycle management.
"""

import os
import uuid
import json
import time
import asyncio
import datetime
import hashlib
from enum import Enum
from typing import Dict, Any, List, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd

from asf.medical.core.logging_config import get_logger
from asf.medical.core.enhanced_cache import EnhancedCacheManager
from asf.medical.ml.models.model_registry import (
    ModelRegistry, ModelStatus, ModelMetrics, ModelFramework, get_model_registry
)
from asf.medical.core.events import publish_event, EventType
from asf.medical.core.task_queue import TaskQueue, Task, TaskStatus, TaskPriority
from asf.medical.core.config import settings

logger = get_logger(__name__)

# Initialize cache for drift metrics
drift_metrics_cache = EnhancedCacheManager(
    max_size=1000,
    default_ttl=86400,  # 1 day
    namespace="model_drift:"
)

class ModelDriftSeverity(str, Enum):
    """Severity levels for model drift."""
    NONE = "none"  # No significant drift detected
    LOW = "low"  # Minor drift, monitoring recommended
    MEDIUM = "medium"  # Moderate drift, investigation recommended
    HIGH = "high"  # Significant drift, retraining recommended
    CRITICAL = "critical"  # Severe drift, immediate action required

class TrainingStatus(str, Enum):
    """Status of a model training job."""
    QUEUED = "queued"  # Job is queued for training
    PREPARING = "preparing"  # Preparing data for training
    TRAINING = "training"  # Model is training
    VALIDATING = "validating"  # Model is being validated
    COMPLETED = "completed"  # Training completed successfully
    FAILED = "failed"  # Training failed
    CANCELLED = "cancelled"  # Training was cancelled

class DeploymentEnvironment(str, Enum):
    """Environments for model deployment."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SHADOW = "shadow"  # Shadow deployment for A/B testing

@dataclass
class DriftMetrics:
    """Metrics for model drift detection."""
    timestamp: str
    feature_drift: Dict[str, float]  # Drift scores for each feature
    prediction_drift: float  # Overall prediction distribution drift
    performance_metrics: Dict[str, float]  # Current performance metrics
    baseline_metrics: Dict[str, float]  # Baseline performance metrics
    sample_size: int  # Number of samples used for drift detection
    severity: ModelDriftSeverity  # Overall drift severity
    actions: List[str]  # Recommended actions

@dataclass
class TrainingJob:
    """Representation of a model training job."""
    job_id: str
    model_name: str
    version: str
    status: TrainingStatus
    created_at: str
    updated_at: str
    created_by: str
    hyperparameters: Dict[str, Any]
    training_dataset_hash: str
    validation_dataset_hash: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    artifacts_path: Optional[str] = None
    parent_version: Optional[str] = None
    training_duration_seconds: Optional[float] = None
    events: Optional[List[Dict[str, Any]]] = None

@dataclass
class ABTestResult:
    """Results from A/B testing two model versions."""
    test_id: str
    model_a_name: str
    model_a_version: str
    model_b_name: str
    model_b_version: str
    start_time: str
    end_time: Optional[str] = None
    traffic_split: Dict[str, float] = None  # e.g. {"a": 0.5, "b": 0.5}
    metrics: Dict[str, Dict[str, float]] = None  # Metrics for each model
    winner: Optional[str] = None  # "a", "b", or None if inconclusive
    confidence: Optional[float] = None
    sample_size: Optional[int] = None
    is_active: bool = True

class ModelLifecycleManager:
    """
    Comprehensive system for managing the lifecycle of ML models.
    
    This class builds upon the ModelRegistry to provide enhanced lifecycle
    management capabilities, including drift detection, automated retraining,
    and A/B testing.
    """
    
    def __init__(self, storage_dir: Optional[str] = None, enable_auto_retraining: bool = True):
        """
        Initialize the model lifecycle manager.
        
        Args:
            storage_dir: Directory for storing lifecycle data.
                If None, uses the default location.
            enable_auto_retraining: Whether to enable automated retraining.
        """
        self.registry = get_model_registry()
        
        if storage_dir is None:
            # Use default storage location relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_dir = os.path.join(base_dir, "models", "lifecycle")
        
        self.storage_dir = storage_dir
        self.enable_auto_retraining = enable_auto_retraining
        
        # Create storage directories
        os.makedirs(os.path.join(storage_dir, "drift"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "training"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "ab_testing"), exist_ok=True)
        
        # Training job data
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.training_jobs_file = os.path.join(storage_dir, "training_jobs.json")
        self._load_training_jobs()
        
        # A/B testing data
        self.ab_tests: Dict[str, ABTestResult] = {}
        self.ab_tests_file = os.path.join(storage_dir, "ab_tests.json")
        self._load_ab_tests()
        
        # Task queue for background jobs
        self.task_queue = TaskQueue("model_lifecycle")
        
        # Start background workers
        self.should_stop = False
        self.workers = []
        self._start_workers()
        
        logger.info(f"Model lifecycle manager initialized at {storage_dir}")
    
    async def close(self):
        """
        Close the manager and stop all background workers.
        """
        self.should_stop = True
        
        # Wait for workers to stop
        if self.workers:
            await asyncio.gather(*[worker for worker in self.workers if not worker.done()])
        
        # Save data
        self._save_training_jobs()
        self._save_ab_tests()
        
        logger.info("Model lifecycle manager closed")
    
    async def detect_model_drift(
        self,
        model_name: str,
        version: str,
        input_data: Union[Dict[str, Any], List[Dict[str, Any]]],
        actual_outcomes: Optional[List[Any]] = None,
        predictions: Optional[List[Any]] = None
    ) -> DriftMetrics:
        """
        Detect drift in a model based on new data.
        
        Args:
            model_name: Name of the model.
            version: Version of the model.
            input_data: New input data to check for drift.
            actual_outcomes: Actual outcomes for performance evaluation.
            predictions: Model predictions for direct comparison.
            
        Returns:
            DriftMetrics with drift detection results.
        """
        # Get model metadata
        model_metadata = self.registry.get_model(model_name, version)
        if not model_metadata:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Convert single input to list
        if isinstance(input_data, dict):
            input_data = [input_data]
        
        # Calculate feature statistics
        feature_drift = await self._calculate_feature_drift(model_name, version, input_data)
        
        # Calculate prediction drift if predictions are provided
        prediction_drift = 0.0
        if predictions:
            prediction_drift = await self._calculate_prediction_drift(model_name, version, predictions)
        
        # Calculate performance metrics if actual outcomes are provided
        performance_metrics = {}
        if actual_outcomes and predictions and len(actual_outcomes) == len(predictions):
            performance_metrics = await self._calculate_performance_metrics(actual_outcomes, predictions)
        
        # Get baseline metrics from model registry
        baseline_metrics = {
            "accuracy": model_metadata.metrics.accuracy or 0.0,
            "precision": model_metadata.metrics.precision or 0.0,
            "recall": model_metadata.metrics.recall or 0.0,
            "f1_score": model_metadata.metrics.f1_score or 0.0
        }
        
        # Determine drift severity
        severity = await self._determine_drift_severity(
            feature_drift,
            prediction_drift,
            performance_metrics,
            baseline_metrics
        )
        
        # Create recommended actions
        actions = await self._create_drift_actions(model_name, version, severity)
        
        # Create drift metrics
        drift_metrics = DriftMetrics(
            timestamp=datetime.datetime.now().isoformat(),
            feature_drift=feature_drift,
            prediction_drift=prediction_drift,
            performance_metrics=performance_metrics,
            baseline_metrics=baseline_metrics,
            sample_size=len(input_data),
            severity=severity,
            actions=actions
        )
        
        # Record drift metrics in model registry
        self.registry.record_drift_metrics(
            model_name,
            version,
            {
                "timestamp": drift_metrics.timestamp,
                "feature_drift": drift_metrics.feature_drift,
                "prediction_drift": drift_metrics.prediction_drift,
                "performance_metrics": drift_metrics.performance_metrics,
                "baseline_metrics": drift_metrics.baseline_metrics,
                "sample_size": drift_metrics.sample_size,
                "severity": drift_metrics.severity.value,
                "actions": drift_metrics.actions
            }
        )
        
        # Cache drift metrics
        key = f"{model_name}:{version}:{drift_metrics.timestamp}"
        await drift_metrics_cache.set(key, drift_metrics.__dict__)
        
        # If severity is high and auto-retraining is enabled, queue a training job
        if (severity in (ModelDriftSeverity.HIGH, ModelDriftSeverity.CRITICAL) 
                and self.enable_auto_retraining):
            await self.schedule_retraining(model_name, version)
            logger.info(f"Auto-retraining scheduled for {model_name} v{version} due to {severity.value} drift")
        
        # Publish drift detected event
        await publish_event(
            event_type=EventType.MODEL_DRIFT_DETECTED,
            data={
                "model_name": model_name,
                "version": version,
                "severity": severity.value,
                "timestamp": drift_metrics.timestamp,
                "actions": drift_metrics.actions
            }
        )
        
        return drift_metrics
    
    async def _calculate_feature_drift(
        self,
        model_name: str,
        version: str,
        input_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate drift in input features.
        
        Args:
            model_name: Name of the model.
            version: Version of the model.
            input_data: New input data to check for drift.
            
        Returns:
            Dictionary mapping feature names to drift scores.
        """
        # In a real implementation, this would compare feature distributions
        # between training data and new data using statistical tests
        
        # Example implementation using a simple statistical distance measure
        feature_drift = {}
        
        # Get reference data statistics if available
        model_metadata = self.registry.get_model(model_name, version)
        parameters = model_metadata.parameters or {}
        baseline_stats = parameters.get("feature_statistics", {})
        
        if not baseline_stats and len(input_data) < 10:
            # Not enough data for meaningful comparison
            return {"insufficient_data": 0.0}
        
        # Extract features
        features = {}
        for record in input_data:
            for key, value in record.items():
                if key not in features:
                    features[key] = []
                features[key].append(value)
        
        # Calculate drift for each feature
        for feature_name, values in features.items():
            # Check if feature values are numeric
            try:
                numeric_values = [float(v) for v in values if v is not None]
                if not numeric_values:
                    feature_drift[feature_name] = 0.0
                    continue
                
                # Calculate basic statistics
                mean = sum(numeric_values) / len(numeric_values)
                variance = sum((x - mean) ** 2 for x in numeric_values) / len(numeric_values)
                
                # Compare with baseline if available
                if feature_name in baseline_stats:
                    baseline_mean = baseline_stats[feature_name].get("mean", mean)
                    baseline_variance = baseline_stats[feature_name].get("variance", variance)
                    
                    # Calculate normalized distance
                    mean_diff = abs(mean - baseline_mean)
                    mean_norm = mean_diff / (abs(baseline_mean) + 1e-10)
                    
                    var_diff = abs(variance - baseline_variance)
                    var_norm = var_diff / (abs(baseline_variance) + 1e-10)
                    
                    # Combine distances
                    feature_drift[feature_name] = (mean_norm + var_norm) / 2
                else:
                    # No baseline, use default low drift
                    feature_drift[feature_name] = 0.1
            except (ValueError, TypeError):
                # Non-numeric feature, use simple ratio of unique values
                unique_values = set(str(v) for v in values if v is not None)
                if not unique_values:
                    feature_drift[feature_name] = 0.0
                    continue
                
                ratio = len(unique_values) / len(values)
                
                # Compare with baseline if available
                if feature_name in baseline_stats:
                    baseline_ratio = baseline_stats[feature_name].get("unique_ratio", ratio)
                    feature_drift[feature_name] = abs(ratio - baseline_ratio) / (baseline_ratio + 1e-10)
                else:
                    # No baseline, use default low drift
                    feature_drift[feature_name] = 0.1
        
        return feature_drift
    
    async def _calculate_prediction_drift(
        self,
        model_name: str,
        version: str,
        predictions: List[Any]
    ) -> float:
        """
        Calculate drift in model predictions.
        
        Args:
            model_name: Name of the model.
            version: Version of the model.
            predictions: New model predictions to check for drift.
            
        Returns:
            Drift score for predictions.
        """
        # In a real implementation, this would compare prediction distributions
        # between expected and actual predictions using statistical tests
        
        # Get reference data statistics if available
        model_metadata = self.registry.get_model(model_name, version)
        parameters = model_metadata.parameters or {}
        baseline_stats = parameters.get("prediction_statistics", {})
        
        if not baseline_stats or len(predictions) < 10:
            # Not enough data for meaningful comparison
            return 0.1
        
        # Check if predictions are numeric
        try:
            numeric_predictions = [float(p) for p in predictions if p is not None]
            if not numeric_predictions:
                return 0.0
            
            # Calculate basic statistics
            mean = sum(numeric_predictions) / len(numeric_predictions)
            variance = sum((x - mean) ** 2 for x in numeric_predictions) / len(numeric_predictions)
            
            # Compare with baseline
            baseline_mean = baseline_stats.get("mean", mean)
            baseline_variance = baseline_stats.get("variance", variance)
            
            # Calculate normalized distances
            mean_diff = abs(mean - baseline_mean)
            mean_norm = mean_diff / (abs(baseline_mean) + 1e-10)
            
            var_diff = abs(variance - baseline_variance)
            var_norm = var_diff / (abs(baseline_variance) + 1e-10)
            
            # Combine distances
            return (mean_norm + var_norm) / 2
        except (ValueError, TypeError):
            # Non-numeric predictions, use distribution of unique values
            unique_preds = {}
            for p in predictions:
                p_str = str(p)
                if p_str not in unique_preds:
                    unique_preds[p_str] = 0
                unique_preds[p_str] += 1
            
            # Convert to distribution
            total = len(predictions)
            distribution = {k: v / total for k, v in unique_preds.items()}
            
            # Compare with baseline distribution
            baseline_dist = baseline_stats.get("distribution", {})
            if not baseline_dist:
                return 0.1
            
            # Calculate JS divergence (simplified)
            drift_score = 0.0
            all_keys = set(distribution.keys()) | set(baseline_dist.keys())
            for key in all_keys:
                p = distribution.get(key, 0)
                q = baseline_dist.get(key, 0)
                
                # Add small epsilon to avoid division by zero
                p = max(p, 1e-10)
                q = max(q, 1e-10)
                
                drift_score += abs(p - q)
            
            # Normalize
            return min(1.0, drift_score / 2)
    
    async def _calculate_performance_metrics(
        self,
        actual: List[Any],
        predictions: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate performance metrics from actual outcomes and predictions.
        
        Args:
            actual: Actual outcome values.
            predictions: Predicted outcome values.
            
        Returns:
            Dictionary of performance metrics.
        """
        # Basic implementation for binary classification
        # In a real scenario, this would be more sophisticated and handle different ML tasks
        
        try:
            # Check if binary classification
            unique_actuals = set(actual)
            if len(unique_actuals) == 2:
                # Assume binary classification
                # Convert to 0/1
                pos_label = max(unique_actuals)
                neg_label = min(unique_actuals)
                
                # Calculate metrics
                tp = sum(1 for a, p in zip(actual, predictions) if a == pos_label and p == pos_label)
                fp = sum(1 for a, p in zip(actual, predictions) if a == neg_label and p == pos_label)
                tn = sum(1 for a, p in zip(actual, predictions) if a == neg_label and p == neg_label)
                fn = sum(1 for a, p in zip(actual, predictions) if a == pos_label and p == neg_label)
                
                # Avoid division by zero
                accuracy = (tp + tn) / len(actual) if len(actual) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                }
            else:
                # For regression or multi-class, just compute basic metrics
                # Mean squared error for numeric predictions
                try:
                    mse = sum((float(a) - float(p)) ** 2 for a, p in zip(actual, predictions)) / len(actual)
                    mae = sum(abs(float(a) - float(p)) for a, p in zip(actual, predictions)) / len(actual)
                    return {
                        "mse": mse,
                        "mae": mae,
                        "rmse": mse ** 0.5
                    }
                except (ValueError, TypeError):
                    # Not numeric, calculate accuracy
                    correct = sum(1 for a, p in zip(actual, predictions) if a == p)
                    accuracy = correct / len(actual) if len(actual) > 0 else 0
                    return {"accuracy": accuracy}
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {"error": 1.0}
    
    async def _determine_drift_severity(
        self,
        feature_drift: Dict[str, float],
        prediction_drift: float,
        performance_metrics: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> ModelDriftSeverity:
        """
        Determine the severity of model drift.
        
        Args:
            feature_drift: Drift scores for features.
            prediction_drift: Drift score for predictions.
            performance_metrics: Current performance metrics.
            baseline_metrics: Baseline performance metrics.
            
        Returns:
            ModelDriftSeverity with the determined severity level.
        """
        # Calculate average feature drift
        avg_feature_drift = sum(feature_drift.values()) / len(feature_drift) if feature_drift else 0
        
        # Calculate performance degradation
        perf_degradation = 0.0
        if performance_metrics and baseline_metrics:
            # Compare key metrics if available
            for metric in ["accuracy", "f1_score"]:
                if metric in performance_metrics and metric in baseline_metrics:
                    if baseline_metrics[metric] > 0:
                        degradation = (baseline_metrics[metric] - performance_metrics[metric]) / baseline_metrics[metric]
                        perf_degradation = max(perf_degradation, degradation)
        
        # Determine severity based on combined signals
        if perf_degradation > 0.2 or avg_feature_drift > 0.5 or prediction_drift > 0.5:
            return ModelDriftSeverity.CRITICAL
        elif perf_degradation > 0.15 or avg_feature_drift > 0.3 or prediction_drift > 0.3:
            return ModelDriftSeverity.HIGH
        elif perf_degradation > 0.1 or avg_feature_drift > 0.2 or prediction_drift > 0.2:
            return ModelDriftSeverity.MEDIUM
        elif perf_degradation > 0.05 or avg_feature_drift > 0.1 or prediction_drift > 0.1:
            return ModelDriftSeverity.LOW
        else:
            return ModelDriftSeverity.NONE
    
    async def _create_drift_actions(
        self,
        model_name: str,
        version: str,
        severity: ModelDriftSeverity
    ) -> List[str]:
        """
        Create recommended actions based on drift severity.
        
        Args:
            model_name: Name of the model.
            version: Version of the model.
            severity: Drift severity level.
            
        Returns:
            List of recommended actions.
        """
        actions = []
        
        if severity == ModelDriftSeverity.NONE:
            actions.append("No action needed. Model is performing as expected.")
        elif severity == ModelDriftSeverity.LOW:
            actions.append("Monitor model performance closely.")
            actions.append("Schedule validation with new data within 30 days.")
        elif severity == ModelDriftSeverity.MEDIUM:
            actions.append("Validate model with larger dataset.")
            actions.append("Consider retraining within 14 days.")
            actions.append("Review feature distributions for significant changes.")
        elif severity == ModelDriftSeverity.HIGH:
            actions.append("Retrain model with recent data.")
            actions.append("Review feature engineering process.")
            actions.append("Consider A/B testing with updated model.")
        elif severity == ModelDriftSeverity.CRITICAL:
            actions.append("Immediately retrain model with recent data.")
            actions.append("Consider reverting to previous stable version.")
            actions.append("Perform root cause analysis of performance degradation.")
            actions.append("Implement shadow deployment of new model version.")
        
        return actions
    
    async def schedule_retraining(
        self,
        model_name: str,
        version: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_data: Optional[Dict[str, Any]] = None,
        created_by: str = "system"
    ) -> str:
        """
        Schedule a model retraining job.
        
        Args:
            model_name: Name of the model to retrain.
            version: Version to base the retraining on.
            hyperparameters: Optional hyperparameters for training.
            training_data: Optional training data.
            created_by: Who or what system created this training job.
            
        Returns:
            Job ID of the scheduled training job.
        """
        # Get parent model
        model = self.registry.get_model(model_name, version)
        if not model:
            raise ValueError(f"Model {model_name} version {version} not found")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Generate new version (increment patch version)
        major, minor, patch = version.split(".")
        new_version = f"{major}.{minor}.{int(patch) + 1}"
        
        # Create training job
        now = datetime.datetime.now().isoformat()
        
        # Compute dataset hash if training data provided
        training_dataset_hash = None
        if training_data:
            training_dataset_hash = self.registry.compute_dataset_hash(training_data)
        else:
            # Use previous training dataset if available
            training_dataset_hash = model.training_dataset_hash
        
        # Create job
        job = TrainingJob(
            job_id=job_id,
            model_name=model_name,
            version=new_version,
            status=TrainingStatus.QUEUED,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            hyperparameters=hyperparameters or {},
            training_dataset_hash=training_dataset_hash,
            parent_version=version,
            events=[{"timestamp": now, "status": "queued", "message": "Training job created"}]
        )
        
        # Store job
        self.training_jobs[job_id] = job
        self._save_training_jobs()
        
        # Create training task
        task_data = {
            "job_id": job_id,
            "model_name": model_name,
            "version": new_version,
            "parent_version": version,
            "hyperparameters": hyperparameters or {},
            "training_dataset_hash": training_dataset_hash
        }
        
        # Add training data if provided (for small datasets only)
        if training_data and not isinstance(training_data, str):
            # Check if data is too large
            data_str = json.dumps(training_data)
            if len(data_str) < 10000:  # Only include small datasets in task
                task_data["training_data"] = training_data
        
        # Queue task
        task = Task(
            task_id=job_id,
            task_type="model_training",
            data=task_data,
            priority=TaskPriority.MEDIUM
        )
        await self.task_queue.enqueue_task(task)
        
        # Publish event
        await publish_event(
            event_type=EventType.MODEL_TRAINING_SCHEDULED,
            data={
                "job_id": job_id,
                "model_name": model_name,
                "version": new_version,
                "parent_version": version,
                "created_by": created_by,
                "timestamp": now
            }
        )
        
        logger.info(f"Scheduled training job {job_id} for {model_name} v{new_version}")
        return job_id
    
    async def get_training_job(self, job_id: str) -> Optional[TrainingJob]:
        """
        Get information about a training job.
        
        Args:
            job_id: ID of the training job.
            
        Returns:
            TrainingJob if found, None otherwise.
        """
        return self.training_jobs.get(job_id)
    
    async def list_training_jobs(
        self,
        model_name: Optional[str] = None,
        status: Optional[Set[TrainingStatus]] = None,
        limit: int = 100
    ) -> List[TrainingJob]:
        """
        List training jobs with optional filters.
        
        Args:
            model_name: Filter by model name.
            status: Filter by status.
            limit: Maximum number of jobs to return.
            
        Returns:
            List of matching training jobs.
        """
        # Apply filters
        filtered_jobs = []
        for job in self.training_jobs.values():
            if model_name and job.model_name != model_name:
                continue
                
            if status and job.status not in status:
                continue
                
            filtered_jobs.append(job)
            
            if len(filtered_jobs) >= limit:
                break
        
        # Sort by created_at (newest first)
        filtered_jobs.sort(key=lambda j: j.created_at, reverse=True)
        return filtered_jobs
    
    async def cancel_training_job(self, job_id: str) -> bool:
        """
        Cancel a training job.
        
        Args:
            job_id: ID of the training job.
            
        Returns:
            True if job was cancelled, False otherwise.
        """
        if job_id not in self.training_jobs:
            return False
            
        job = self.training_jobs[job_id]
        if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
            return False
        
        # Update job status
        now = datetime.datetime.now().isoformat()
        job.status = TrainingStatus.CANCELLED
        job.updated_at = now
        
        if not job.events:
            job.events = []
        job.events.append({"timestamp": now, "status": "cancelled", "message": "Training job cancelled"})
        
        # Save jobs
        self._save_training_jobs()
        
        # Cancel task
        await self.task_queue.cancel_task(job_id)
        
        # Publish event
        await publish_event(
            event_type=EventType.MODEL_TRAINING_CANCELLED,
            data={
                "job_id": job_id,
                "model_name": job.model_name,
                "version": job.version,
                "timestamp": now
            }
        )
        
        logger.info(f"Cancelled training job {job_id} for {job.model_name} v{job.version}")
        return True
    
    async def create_ab_test(
        self,
        model_a_name: str,
        model_a_version: str,
        model_b_name: str,
        model_b_version: str,
        traffic_split: Optional[Dict[str, float]] = None,
        metrics_to_compare: Optional[List[str]] = None
    ) -> str:
        """
        Create an A/B test between two model versions.
        
        Args:
            model_a_name: Name of model A.
            model_a_version: Version of model A.
            model_b_name: Name of model B.
            model_b_version: Version of model B.
            traffic_split: Traffic split between models.
                Default is {"a": 0.5, "b": 0.5}.
            metrics_to_compare: Metrics to compare for determining winner.
                Default is ["accuracy", "f1_score"].
            
        Returns:
            ID of the A/B test.
        """
        # Validate models exist
        model_a = self.registry.get_model(model_a_name, model_a_version)
        if not model_a:
            raise ValueError(f"Model {model_a_name} version {model_a_version} not found")
            
        model_b = self.registry.get_model(model_b_name, model_b_version)
        if not model_b:
            raise ValueError(f"Model {model_b_name} version {model_b_version} not found")
        
        # Generate test ID
        test_id = str(uuid.uuid4())
        
        # Set defaults
        if traffic_split is None:
            traffic_split = {"a": 0.5, "b": 0.5}
            
        if metrics_to_compare is None:
            metrics_to_compare = ["accuracy", "f1_score"]
        
        # Create A/B test
        now = datetime.datetime.now().isoformat()
        ab_test = ABTestResult(
            test_id=test_id,
            model_a_name=model_a_name,
            model_a_version=model_a_version,
            model_b_name=model_b_name,
            model_b_version=model_b_version,
            start_time=now,
            traffic_split=traffic_split,
            metrics={m: {"a": 0.0, "b": 0.0} for m in metrics_to_compare},
            is_active=True
        )
        
        # Store test
        self.ab_tests[test_id] = ab_test
        self._save_ab_tests()
        
        # Publish event
        await publish_event(
            event_type=EventType.AB_TEST_STARTED,
            data={
                "test_id": test_id,
                "model_a": f"{model_a_name}:{model_a_version}",
                "model_b": f"{model_b_name}:{model_b_version}",
                "traffic_split": traffic_split,
                "timestamp": now
            }
        )
        
        logger.info(f"Created A/B test {test_id} between {model_a_name}:{model_a_version} and {model_b_name}:{model_b_version}")
        return test_id
    
    async def record_ab_test_metrics(
        self,
        test_id: str,
        model_key: str,  # "a" or "b"
        metrics: Dict[str, float],
        sample_count: int = 1
    ) -> bool:
        """
        Record metrics for an A/B test.
        
        Args:
            test_id: ID of the A/B test.
            model_key: Which model the metrics are for ("a" or "b").
            metrics: Metrics to record.
            sample_count: Number of samples these metrics represent.
            
        Returns:
            True if metrics were recorded, False otherwise.
        """
        if test_id not in self.ab_tests:
            return False
            
        ab_test = self.ab_tests[test_id]
        if not ab_test.is_active:
            return False
            
        if model_key not in ("a", "b"):
            raise ValueError("model_key must be 'a' or 'b'")
        
        # Update metrics (using weighted average)
        for metric_name, metric_value in metrics.items():
            if metric_name not in ab_test.metrics:
                ab_test.metrics[metric_name] = {"a": 0.0, "b": 0.0}
                
            # Update with weighted average
            current_value = ab_test.metrics[metric_name][model_key]
            current_count = ab_test.sample_size or 0
            
            if current_count == 0:
                # First sample
                ab_test.metrics[metric_name][model_key] = metric_value
            else:
                # Weighted average
                weight = sample_count / (current_count + sample_count)
                ab_test.metrics[metric_name][model_key] = (
                    current_value * (1 - weight) + metric_value * weight
                )
        
        # Update sample size
        if ab_test.sample_size is None:
            ab_test.sample_size = 0
        ab_test.sample_size += sample_count
        
        # Save A/B tests
        self._save_ab_tests()
        
        return True
    
    async def evaluate_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """
        Evaluate an A/B test to determine if there's a winner.
        
        Args:
            test_id: ID of the A/B test.
            
        Returns:
            Updated A/B test result or None if not found.
        """
        if test_id not in self.ab_tests:
            return None
            
        ab_test = self.ab_tests[test_id]
        if not ab_test.is_active:
            return ab_test
            
        # Need sufficient samples for confidence
        if ab_test.sample_size is None or ab_test.sample_size < 100:
            # Not enough samples yet
            return ab_test
        
        # Simplified statistical comparison for determining winner
        # In a real system, this would use proper statistical significance tests
        winner = None
        confidence = 0.0
        
        # Compare primary metrics (accuracy, f1_score)
        for metric_name in ["accuracy", "f1_score"]:
            if metric_name in ab_test.metrics:
                a_value = ab_test.metrics[metric_name]["a"]
                b_value = ab_test.metrics[metric_name]["b"]
                
                # Simple delta
                delta = b_value - a_value
                
                # Determine significance based on sample size
                # This is a greatly simplified approach
                significance_threshold = 1.0 / (ab_test.sample_size ** 0.5)
                
                if abs(delta) > significance_threshold:
                    # Significant difference
                    if delta > 0:
                        winner = "b"
                    else:
                        winner = "a"
                    
                    # Calculate confidence based on delta and sample size
                    confidence = min(0.99, abs(delta) * (ab_test.sample_size ** 0.25))
                    break
        
        # Update test with results
        if winner:
            now = datetime.datetime.now().isoformat()
            ab_test.winner = winner
            ab_test.confidence = confidence
            ab_test.end_time = now
            ab_test.is_active = False
            
            # Save A/B tests
            self._save_ab_tests()
            
            # Publish event
            await publish_event(
                event_type=EventType.AB_TEST_COMPLETED,
                data={
                    "test_id": test_id,
                    "winner": winner,
                    "confidence": confidence,
                    "sample_size": ab_test.sample_size,
                    "timestamp": now
                }
            )
            
            logger.info(f"A/B test {test_id} completed with winner: {winner} (confidence: {confidence:.2f})")
            
            # Automatically promote winner to production if confidence is high
            if confidence > 0.8:
                if winner == "a":
                    winner_name = ab_test.model_a_name
                    winner_version = ab_test.model_a_version
                else:
                    winner_name = ab_test.model_b_name
                    winner_version = ab_test.model_b_version
                
                self.registry.update_model_status(
                    winner_name,
                    winner_version,
                    ModelStatus.PRODUCTION
                )
                
                logger.info(f"Promoted {winner_name} v{winner_version} to production based on A/B test results")
        
        return ab_test
    
    async def get_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """
        Get information about an A/B test.
        
        Args:
            test_id: ID of the A/B test.
            
        Returns:
            ABTestResult if found, None otherwise.
        """
        return self.ab_tests.get(test_id)
    
    async def list_ab_tests(
        self,
        model_name: Optional[str] = None,
        active_only: bool = False,
        limit: int = 100
    ) -> List[ABTestResult]:
        """
        List A/B tests with optional filters.
        
        Args:
            model_name: Filter by model name.
            active_only: Only include active tests.
            limit: Maximum number of tests to return.
            
        Returns:
            List of matching A/B tests.
        """
        # Apply filters
        filtered_tests = []
        for test in self.ab_tests.values():
            if model_name and test.model_a_name != model_name and test.model_b_name != model_name:
                continue
                
            if active_only and not test.is_active:
                continue
                
            filtered_tests.append(test)
            
            if len(filtered_tests) >= limit:
                break
        
        # Sort by start_time (newest first)
        filtered_tests.sort(key=lambda t: t.start_time, reverse=True)
        return filtered_tests
    
    def _worker_monitor_task_queue(self):
        """Monitor the task queue for completed training jobs."""
        async def _worker():
            logger.info("Starting task queue monitoring worker")
            
            while not self.should_stop:
                try:
                    # Check for completed tasks
                    completed_tasks = await self.task_queue.get_completed_tasks(task_type="model_training")
                    
                    for task in completed_tasks:
                        if task.task_id in self.training_jobs:
                            # Update training job
                            job = self.training_jobs[task.task_id]
                            now = datetime.datetime.now().isoformat()
                            
                            if task.status == TaskStatus.COMPLETED:
                                # Update job with task results
                                job.status = TrainingStatus.COMPLETED
                                job.metrics = task.result.get("metrics")
                                job.artifacts_path = task.result.get("artifacts_path")
                                job.training_duration_seconds = task.result.get("training_duration_seconds")
                                
                                if not job.events:
                                    job.events = []
                                job.events.append({"timestamp": now, "status": "completed", "message": "Training completed successfully"})
                                
                                # Register model in registry
                                try:
                                    # Create metrics object
                                    metrics = ModelMetrics(**job.metrics) if job.metrics else ModelMetrics()
                                    
                                    # Register model
                                    self.registry.register_model(
                                        name=job.model_name,
                                        version=job.version,
                                        framework=ModelFramework.CUSTOM,  # Replace with actual framework
                                        description=f"Retrained version of {job.model_name} v{job.parent_version}",
                                        status=ModelStatus.STAGING,  # Start in staging
                                        metrics=metrics,
                                        path=job.artifacts_path,
                                        parent_version=job.parent_version,
                                        training_dataset_hash=job.training_dataset_hash,
                                        created_by=job.created_by,
                                        tags=["retrained", "automated"],
                                        parameters=job.hyperparameters
                                    )
                                    
                                    # Publish event
                                    asyncio.create_task(publish_event(
                                        event_type=EventType.MODEL_TRAINING_COMPLETED,
                                        data={
                                            "job_id": job.job_id,
                                            "model_name": job.model_name,
                                            "version": job.version,
                                            "metrics": job.metrics,
                                            "timestamp": now
                                        }
                                    ))
                                    
                                except Exception as e:
                                    logger.error(f"Error registering model after training: {str(e)}")
                                    if not job.events:
                                        job.events = []
                                    job.events.append({
                                        "timestamp": now,
                                        "status": "error",
                                        "message": f"Error registering model: {str(e)}"
                                    })
                            
                            elif task.status == TaskStatus.FAILED:
                                # Update job with failure information
                                job.status = TrainingStatus.FAILED
                                job.error_message = task.error
                                
                                if not job.events:
                                    job.events = []
                                job.events.append({
                                    "timestamp": now,
                                    "status": "failed",
                                    "message": f"Training failed: {task.error}"
                                })
                                
                                # Publish event
                                asyncio.create_task(publish_event(
                                    event_type=EventType.MODEL_TRAINING_FAILED,
                                    data={
                                        "job_id": job.job_id,
                                        "model_name": job.model_name,
                                        "version": job.version,
                                        "error": task.error,
                                        "timestamp": now
                                    }
                                ))
                            
                            # Update timestamp
                            job.updated_at = now
                            
                            # Remove task from queue
                            await self.task_queue.remove_task(task.task_id)
                    
                    # Save training jobs
                    if completed_tasks:
                        self._save_training_jobs()
                
                # Wait before checking again
                await asyncio.sleep(5)
                
                except Exception as e:
                    logger.error(f"Error in task queue monitoring worker: {str(e)}")
                    await asyncio.sleep(10)  # Wait longer on error
            
            logger.info("Task queue monitoring worker stopped")
        
        return asyncio.create_task(_worker())
    
    def _worker_evaluate_ab_tests(self):
        """Periodically evaluate active A/B tests."""
        async def _worker():
            logger.info("Starting A/B test evaluation worker")
            
            while not self.should_stop:
                try:
                    # Find active tests
                    active_tests = [test for test in self.ab_tests.values() if test.is_active]
                    
                    for test in active_tests:
                        # Evaluate test
                        await self.evaluate_ab_test(test.test_id)
                    
                    # Wait before next evaluation
                    await asyncio.sleep(60)  # Check every minute
                    
                except Exception as e:
                    logger.error(f"Error in A/B test evaluation worker: {str(e)}")
                    await asyncio.sleep(30)  # Wait longer on error
            
            logger.info("A/B test evaluation worker stopped")
        
        return asyncio.create_task(_worker())
    
    def _start_workers(self):
        """Start background workers."""
        self.workers = [
            self._worker_monitor_task_queue(),
            self._worker_evaluate_ab_tests()
        ]
    
    def _load_training_jobs(self):
        """Load training jobs from file."""
        if not os.path.exists(self.training_jobs_file):
            return
        
        try:
            with open(self.training_jobs_file, "r") as f:
                jobs_data = json.load(f)
            
            for job_id, job_data in jobs_data.items():
                self.training_jobs[job_id] = TrainingJob(**job_data)
                
            logger.info(f"Loaded {len(self.training_jobs)} training jobs from {self.training_jobs_file}")
        except Exception as e:
            logger.error(f"Error loading training jobs: {str(e)}")
    
    def _save_training_jobs(self):
        """Save training jobs to file."""
        try:
            jobs_data = {job_id: job.__dict__ for job_id, job in self.training_jobs.items()}
            
            with open(self.training_jobs_file, "w") as f:
                json.dump(jobs_data, f, indent=2)
                
            logger.debug(f"Saved {len(self.training_jobs)} training jobs to {self.training_jobs_file}")
        except Exception as e:
            logger.error(f"Error saving training jobs: {str(e)}")
    
    def _load_ab_tests(self):
        """Load A/B tests from file."""
        if not os.path.exists(self.ab_tests_file):
            return
        
        try:
            with open(self.ab_tests_file, "r") as f:
                tests_data = json.load(f)
            
            for test_id, test_data in tests_data.items():
                self.ab_tests[test_id] = ABTestResult(**test_data)
                
            logger.info(f"Loaded {len(self.ab_tests)} A/B tests from {self.ab_tests_file}")
        except Exception as e:
            logger.error(f"Error loading A/B tests: {str(e)}")
    
    def _save_ab_tests(self):
        """Save A/B tests to file."""
        try:
            tests_data = {test_id: test.__dict__ for test_id, test in self.ab_tests.items()}
            
            with open(self.ab_tests_file, "w") as f:
                json.dump(tests_data, f, indent=2)
                
            logger.debug(f"Saved {len(self.ab_tests)} A/B tests to {self.ab_tests_file}")
        except Exception as e:
            logger.error(f"Error saving A/B tests: {str(e)}")

# Singleton instance
_lifecycle_manager = None

def get_model_lifecycle_manager() -> ModelLifecycleManager:
    """
    Get or create the model lifecycle manager singleton.
    
    Returns:
        Model lifecycle manager instance.
    """
    global _lifecycle_manager
    if _lifecycle_manager is None:
        _lifecycle_manager = ModelLifecycleManager()
    return _lifecycle_manager

async def close_model_lifecycle_manager():
    """Close the model lifecycle manager singleton."""
    global _lifecycle_manager
    if _lifecycle_manager:
        await _lifecycle_manager.close()
        _lifecycle_manager = None