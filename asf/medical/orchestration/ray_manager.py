"""
Ray manager for the Medical Research Synthesizer.

This module provides a manager for Ray-based distributed computing.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Union, Tuple
import ray

from asf.medical.core.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class RayManager:
    """
    Manager for Ray-based distributed computing.
    
    This manager provides methods for initializing Ray and running distributed tasks.
    """
    
    _instance = None
    
    def __new__(cls):
        """
        Create a singleton instance of the Ray manager.
        
        Returns:
            RayManager: The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(RayManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the Ray manager."""
        self.ray_address = settings.RAY_ADDRESS
        self.num_cpus = settings.RAY_NUM_CPUS
        self.num_gpus = settings.RAY_NUM_GPUS
        self.initialized = False
        
        logger.info(f"Ray manager initialized with ray_address={self.ray_address}, num_cpus={self.num_cpus}, num_gpus={self.num_gpus}")
    
    def initialize(self) -> bool:
        """
        Initialize Ray.
        
        Returns:
            True if initialization is successful, False otherwise
        """
        if self.initialized:
            logger.info("Ray already initialized")
            return True
        
        try:
            logger.info(f"Initializing Ray with address={self.ray_address}")
            
            # Initialize Ray
            if self.ray_address:
                # Connect to existing Ray cluster
                ray.init(address=self.ray_address)
            else:
                # Start a local Ray instance
                ray.init(num_cpus=self.num_cpus, num_gpus=self.num_gpus)
            
            self.initialized = True
            logger.info("Ray initialized")
            
            return True
        except Exception as e:
            logger.error(f"Error initializing Ray: {str(e)}")
            return False
    
    def shutdown(self) -> None:
        """Shutdown Ray."""
        if self.initialized:
            logger.info("Shutting down Ray")
            ray.shutdown()
            self.initialized = False
    
    def get_cluster_resources(self) -> Dict[str, Any]:
        """
        Get cluster resources.
        
        Returns:
            Cluster resources
            
        Raises:
            Exception: If Ray is not initialized
        """
        if not self.initialized:
            if not self.initialize():
                raise Exception("Ray not initialized")
        
        return ray.cluster_resources()
    
    def get_available_resources(self) -> Dict[str, Any]:
        """
        Get available resources.
        
        Returns:
            Available resources
            
        Raises:
            Exception: If Ray is not initialized
        """
        if not self.initialized:
            if not self.initialize():
                raise Exception("Ray not initialized")
        
        return ray.available_resources()
    
    def run_task(self, func, *args, **kwargs) -> Any:
        """
        Run a task using Ray.
        
        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Task result
            
        Raises:
            Exception: If Ray is not initialized
        """
        if not self.initialized:
            if not self.initialize():
                raise Exception("Ray not initialized")
        
        # Create a remote function
        remote_func = ray.remote(func)
        
        # Run the task
        result_ref = remote_func.remote(*args, **kwargs)
        
        # Get the result
        result = ray.get(result_ref)
        
        return result
    
    def run_tasks(self, func, args_list: List[Tuple]) -> List[Any]:
        """
        Run multiple tasks using Ray.
        
        Args:
            func: Function to run
            args_list: List of argument tuples
            
        Returns:
            List of task results
            
        Raises:
            Exception: If Ray is not initialized
        """
        if not self.initialized:
            if not self.initialize():
                raise Exception("Ray not initialized")
        
        # Create a remote function
        remote_func = ray.remote(func)
        
        # Run the tasks
        result_refs = [remote_func.remote(*args) for args in args_list]
        
        # Get the results
        results = ray.get(result_refs)
        
        return results
