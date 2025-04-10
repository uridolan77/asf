"""
Resource Limiter

This module provides resource limiting for ML operations to prevent overloading the system.
"""

import os
import time
import logging
import threading
import psutil
from typing import Dict, Any, Optional, Callable, List, Tuple

from asf.medical.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class ResourceLimiter:
    """
    Resource limiter for ML operations.
    
    This class provides methods for limiting resource usage of ML operations,
    including CPU, memory, and GPU usage.
    """
    
    def __init__(
        self,
        max_cpu_percent: float = 80.0,
        max_memory_percent: float = 80.0,
        max_gpu_percent: float = 80.0,
        max_concurrent_tasks: int = 5,
        check_interval: float = 1.0
    ):
        """
        Initialize the resource limiter.
        
        Args:
            max_cpu_percent: Maximum CPU usage percentage (default: 80.0)
            max_memory_percent: Maximum memory usage percentage (default: 80.0)
            max_gpu_percent: Maximum GPU usage percentage (default: 80.0)
            max_concurrent_tasks: Maximum number of concurrent tasks (default: 5)
            check_interval: Interval in seconds for checking resource usage (default: 1.0)
        """
        self.max_cpu_percent = float(os.environ.get("MAX_CPU_PERCENT", max_cpu_percent))
        self.max_memory_percent = float(os.environ.get("MAX_MEMORY_PERCENT", max_memory_percent))
        self.max_gpu_percent = float(os.environ.get("MAX_GPU_PERCENT", max_gpu_percent))
        self.max_concurrent_tasks = int(os.environ.get("MAX_CONCURRENT_TASKS", max_concurrent_tasks))
        self.check_interval = float(os.environ.get("RESOURCE_CHECK_INTERVAL", check_interval))
        
        self.concurrent_tasks = 0
        self.task_lock = threading.Lock()
        self.model_locks: Dict[str, threading.Lock] = {}
        self.model_usage: Dict[str, Dict[str, Any]] = {}
        
        # Try to import GPU monitoring libraries
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    self.pynvml = pynvml
                    logger.info("NVML initialized for GPU monitoring")
                except (ImportError, Exception) as e:
                    logger.warning(f"Could not initialize NVML for GPU monitoring: {str(e)}")
                    self.pynvml = None
        except ImportError:
            logger.info("PyTorch not available, GPU monitoring disabled")
    
    def get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with resource usage percentages
        """
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Get GPU usage if available
        gpu_percent = 0.0
        if self.has_gpu and self.pynvml:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(0)
                info = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_percent = info.gpu
            except Exception as e:
                logger.warning(f"Error getting GPU usage: {str(e)}")
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_percent": gpu_percent,
            "concurrent_tasks": self.concurrent_tasks
        }
    
    def can_start_task(self) -> Tuple[bool, Dict[str, float]]:
        """
        Check if a new task can be started based on resource usage.
        
        Returns:
            Tuple of (can_start, resource_usage)
        """
        # Get current resource usage
        usage = self.get_resource_usage()
        
        # Check if resources are available
        can_start = (
            usage["cpu_percent"] < self.max_cpu_percent and
            usage["memory_percent"] < self.max_memory_percent and
            usage["gpu_percent"] < self.max_gpu_percent and
            usage["concurrent_tasks"] < self.max_concurrent_tasks
        )
        
        return can_start, usage
    
    def wait_for_resources(self, timeout: float = 300.0) -> bool:
        """
        Wait for resources to become available.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 300.0)
            
        Returns:
            True if resources became available, False if timeout occurred
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            can_start, usage = self.can_start_task()
            
            if can_start:
                return True
            
            logger.info(
                f"Waiting for resources: CPU: {usage['cpu_percent']:.1f}%, "
                f"Memory: {usage['memory_percent']:.1f}%, "
                f"GPU: {usage['gpu_percent']:.1f}%, "
                f"Tasks: {usage['concurrent_tasks']}/{self.max_concurrent_tasks}"
            )
            
            time.sleep(self.check_interval)
        
        logger.warning("Timeout waiting for resources")
        return False
    
    def acquire_task_slot(self, timeout: float = 300.0) -> bool:
        """
        Acquire a task slot.
        
        Args:
            timeout: Maximum time to wait in seconds (default: 300.0)
            
        Returns:
            True if a slot was acquired, False otherwise
        """
        # Wait for resources to become available
        if not self.wait_for_resources(timeout):
            return False
        
        # Acquire task slot
        with self.task_lock:
            if self.concurrent_tasks >= self.max_concurrent_tasks:
                return False
            
            self.concurrent_tasks += 1
            logger.debug(f"Task slot acquired, concurrent tasks: {self.concurrent_tasks}")
            return True
    
    def release_task_slot(self):
        """Release a task slot."""
        with self.task_lock:
            if self.concurrent_tasks > 0:
                self.concurrent_tasks -= 1
                logger.debug(f"Task slot released, concurrent tasks: {self.concurrent_tasks}")
    
    def acquire_model_lock(self, model_name: str, timeout: float = 300.0) -> bool:
        """
        Acquire a lock for a specific model.
        
        Args:
            model_name: Name of the model
            timeout: Maximum time to wait in seconds (default: 300.0)
            
        Returns:
            True if the lock was acquired, False otherwise
        """
        # Create lock if it doesn't exist
        if model_name not in self.model_locks:
            with self.task_lock:  # Use task_lock to protect model_locks dictionary
                if model_name not in self.model_locks:
                    self.model_locks[model_name] = threading.Lock()
        
        # Try to acquire the lock
        return self.model_locks[model_name].acquire(timeout=timeout)
    
    def release_model_lock(self, model_name: str):
        """
        Release a lock for a specific model.
        
        Args:
            model_name: Name of the model
        """
        if model_name in self.model_locks:
            try:
                self.model_locks[model_name].release()
                logger.debug(f"Model lock released: {model_name}")
            except RuntimeError:
                logger.warning(f"Attempted to release an unlocked lock for model: {model_name}")
    
    def register_model_usage(self, model_name: str, memory_mb: float):
        """
        Register memory usage for a model.
        
        Args:
            model_name: Name of the model
            memory_mb: Memory usage in MB
        """
        with self.task_lock:  # Use task_lock to protect model_usage dictionary
            self.model_usage[model_name] = {
                "memory_mb": memory_mb,
                "last_used": time.time()
            }
            logger.debug(f"Registered model usage: {model_name}, {memory_mb} MB")
    
    def get_model_usage(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get usage information for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Usage information or None if not found
        """
        return self.model_usage.get(model_name)
    
    def get_all_model_usage(self) -> Dict[str, Dict[str, Any]]:
        """
        Get usage information for all models.
        
        Returns:
            Dictionary of model usage information
        """
        return self.model_usage.copy()
    
    def with_resource_limit(self, func: Callable, model_name: Optional[str] = None, timeout: float = 300.0):
        """
        Decorator for limiting resources for a function.
        
        Args:
            func: Function to decorate
            model_name: Name of the model (optional)
            timeout: Maximum time to wait for resources in seconds (default: 300.0)
            
        Returns:
            Decorated function
        """
        def wrapper(*args, **kwargs):
            # Acquire task slot
            if not self.acquire_task_slot(timeout):
                raise RuntimeError("Could not acquire task slot")
            
            try:
                # Acquire model lock if specified
                if model_name:
                    if not self.acquire_model_lock(model_name, timeout):
                        raise RuntimeError(f"Could not acquire lock for model: {model_name}")
                    
                    try:
                        # Call the function
                        return func(*args, **kwargs)
                    finally:
                        # Release model lock
                        self.release_model_lock(model_name)
                else:
                    # Call the function without model lock
                    return func(*args, **kwargs)
            finally:
                # Release task slot
                self.release_task_slot()
        
        return wrapper

# Create a singleton instance
resource_limiter = ResourceLimiter()
