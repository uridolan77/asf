"""
Resource Limiter
This module provides resource limiting for ML operations to prevent overloading the system.
"""
import os
import time
import logging
import threading
import psutil
from typing import Dict, Any, Tuple
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
        self.max_cpu_percent = float(os.environ.get("MAX_CPU_PERCENT", max_cpu_percent))
        self.max_memory_percent = float(os.environ.get("MAX_MEMORY_PERCENT", max_memory_percent))
        self.max_gpu_percent = float(os.environ.get("MAX_GPU_PERCENT", max_gpu_percent))
        self.max_concurrent_tasks = int(os.environ.get("MAX_CONCURRENT_TASKS", max_concurrent_tasks))
        self.check_interval = float(os.environ.get("RESOURCE_CHECK_INTERVAL", check_interval))
        self.concurrent_tasks = 0
        self.task_lock = threading.Lock()
        self.model_locks: Dict[str, threading.Lock] = {}
        self.model_usage: Dict[str, Dict[str, Any]] = {}
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
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
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
        usage = self.get_resource_usage()
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
        if not self.wait_for_resources(timeout):
            return False
        with self.task_lock:
            if self.concurrent_tasks >= self.max_concurrent_tasks:
                return False
            self.concurrent_tasks += 1
            logger.debug(f"Task slot acquired, concurrent tasks: {self.concurrent_tasks}")
            return True
    def release_task_slot(self):
        """Release a task slot.
    Args:
        # TODO: Add parameter descriptions
    Returns:
        # TODO: Add return description
    """
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
        Release a lock for a specific model.
        Args:
            model_name: Name of the model
        Register memory usage for a model.
        Args:
            model_name: Name of the model
            memory_mb: Memory usage in MB
        Get usage information for a model.
        Args:
            model_name: Name of the model
        Returns:
            Usage information or None if not found
        Get usage information for all models.
        Returns:
            Dictionary of model usage information
        Decorator for limiting resources for a function.
        Args:
            func: Function to decorate
            model_name: Name of the model (optional)
            timeout: Maximum time to wait for resources in seconds (default: 300.0)
        Returns:
            Decorated function