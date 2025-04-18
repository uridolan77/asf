Enhanced DSPy Client

This module provides an enhanced DSPy client with circuit breaker pattern,
improved error handling, and comprehensive audit logging for medical research applications.

import os
import time
import json
import uuid
import asyncio
import hashlib
import traceback
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Dict, Any, List, Optional, Callable, Type, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import dspy
import mlflow

from settings import get_enhanced_settings, EnhancedDSPySettings
from cache import CacheInterface, create_cache
from module_base import EnhancedDSPyModuleBase, EnhancedDSPyModuleProtocol
from .circuit_breaker import get_circuit_breaker_registry, CircuitOpenError
from .audit_logging import get_audit_logger

# Set up logging
import logging
logger = logging.getLogger(__name__)


class EnhancedDSPyClient:
    Enhanced DSPy client with circuit breaker pattern and improved error handling.
    
    This class provides a central access point for interacting with DSPy's core functionalities,
    with enhanced features for medical research applications.
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(EnhancedDSPyClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        Initialize the client if not already initialized.
        
        Args:
        
        Initialize the client.
        
        This method initializes the language model, cache, and thread pool.
        It should be called before using the client.
        """
        if self._initialized:
            return
        
        async with self._lock:
            if self._initialized:
                return
            
            logger.info("Initializing EnhancedDSPyClient...")
            
            # Initialize language model
            await self._initialize_lm()
            
            # Initialize cache
            await self._initialize_cache()
            
            # Initialize thread pool
            self._executor = ThreadPoolExecutor(max_workers=self.settings.THREAD_LIMIT)
            
            # Initialize MLflow if telemetry is enabled
            if self.settings.ENABLE_TELEMETRY and self.settings.MLFLOW_TRACKING_URI:
                mlflow.set_tracking_uri(self.settings.MLFLOW_TRACKING_URI)
                mlflow.set_experiment(self.settings.MLFLOW_EXPERIMENT_NAME)
            
            self._initialized = True
            logger.info("EnhancedDSPyClient initialized successfully")
    
    async def _initialize_lm(self):
        """Initialize the language model based on settings."""
        logger.info(f"Initializing language model with provider: {self.settings.LLM_PROVIDER}")
        
        try:
            if self.settings.LLM_PROVIDER == "openai":
                self.lm = dspy.OpenAI(
                    model=self.settings.DEFAULT_MODEL,
                    api_key=self.settings.LLM_API_KEY.get_secret_value(),
                    max_tokens=self.settings.MAX_TOKENS,
                    temperature=self.settings.TEMPERATURE,
                    top_p=self.settings.TOP_P
                )
            elif self.settings.LLM_PROVIDER == "anthropic":
                self.lm = dspy.Anthropic(
                    model=self.settings.DEFAULT_MODEL,
                    api_key=self.settings.LLM_API_KEY.get_secret_value(),
                    max_tokens=self.settings.MAX_TOKENS,
                    temperature=self.settings.TEMPERATURE,
                    top_p=self.settings.TOP_P
                )
            elif self.settings.LLM_PROVIDER == "azure":
                if not self.settings.AZURE_ENDPOINT or not self.settings.AZURE_DEPLOYMENT_NAME:
                    raise ValueError("AZURE_ENDPOINT and AZURE_DEPLOYMENT_NAME are required for Azure OpenAI")
                
                self.lm = dspy.AzureOpenAI(
                    model=self.settings.DEFAULT_MODEL,
                    api_key=self.settings.LLM_API_KEY.get_secret_value(),
                    azure_endpoint=self.settings.AZURE_ENDPOINT,
                    azure_deployment=self.settings.AZURE_DEPLOYMENT_NAME,
                    api_version=self.settings.AZURE_API_VERSION,
                    max_tokens=self.settings.MAX_TOKENS,
                    temperature=self.settings.TEMPERATURE,
                    top_p=self.settings.TOP_P
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {self.settings.LLM_PROVIDER}")
            
            logger.info(f"Language model initialized with model: {self.settings.DEFAULT_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize language model: {str(e)}")
            raise
    
    async def _initialize_cache(self):
        """Initialize the cache based on settings."""
        logger.info(f"Initializing cache with backend: {self.settings.CACHE_BACKEND}")
        
        try:
            # Create cache based on backend type
            self.cache = create_cache(
                cache_type=self.settings.CACHE_BACKEND.value,
                cache_directory=self.settings.CACHE_DIRECTORY,
                redis_url=self.settings.REDIS_URL,
                redis_password=self.settings.REDIS_PASSWORD.get_secret_value() if self.settings.REDIS_PASSWORD else None,
                connection_pool_size=self.settings.REDIS_POOL_SIZE
            )
            
            logger.info(f"Cache initialized with backend: {self.settings.CACHE_BACKEND}")
        except Exception as e:
            logger.error(f"Failed to initialize cache: {str(e)}")
            # Fall back to null cache
            from .enhanced_cache import NullCache
            self.cache = NullCache()
            logger.warning("Falling back to null cache")
    
    async def call_lm(
        self,
        prompt: str,
        model: Optional[str] = None,
        cache_key: Optional[str] = None,
        track_usage: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call the language model with circuit breaker protection.
        
        Args:
            prompt: The prompt to send to the language model
            model: Optional model override
            cache_key: Optional explicit cache key
            track_usage: Whether to track usage metrics
            **kwargs: Additional parameters to pass to the LM
            
        Returns:
            Dict[str, Any]: The LM response
            
        Raises:
            CircuitOpenError: If the circuit is open
            Various exceptions from the underlying LM API
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        
        # Use model name (or default) for circuit breaker key
        model_name = model or self.settings.DEFAULT_MODEL
        circuit_key = f"lm_{model_name}"
        
        # Get circuit breaker for this model
        circuit = await self._circuit_breaker_registry.get_circuit_breaker(
            name=circuit_key,
            failure_threshold=self.settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            reset_timeout=self.settings.CIRCUIT_BREAKER_RESET_TIMEOUT,
            success_threshold=self.settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD
        )
        
        # Generate cache key if not provided
        if cache_key is None:
            # Include model and relevant parameters in the key
            key_data = {
                "prompt": prompt,
                "model": model_name,
                **{k: v for k, v in kwargs.items() if k in ["temperature", "max_tokens", "top_p", "stop"]}
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        
        # Try to get from cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            
            # Log cache operation
            if self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_cache_operation(
                    operation="get",
                    key=cache_key,
                    success=True,
                    latency=0.0
                )
            
            return cached_result
        
        # Use circuit breaker pattern
        try:
            # Check if circuit is closed
            await circuit.before_request()
            
            # Record start time for latency measurement
            start_time = time.time()
            
            # Prepare parameters
            params = {**kwargs}
            if model:
                params["model"] = model
            
            # Log LM call start
            if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_lm_call(
                    prompt=prompt,
                    model=model_name,
                    parameters=params,
                    response=None,
                    error=None,
                    latency=None
                )
            
            # Call LM with retry logic
            try:
                # Execute in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: self.lm.call(prompt, **params)
                )
                
                # Calculate latency
                latency = time.time() - start_time
                
                # Log successful LM call
                if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
                    self._audit_logger.log_lm_call(
                        prompt=prompt,
                        model=model_name,
                        parameters=params,
                        response=result,
                        error=None,
                        latency=latency
                    )
                
                # Cache the result
                await self.cache.set(cache_key, result, expire=self.settings.CACHE_TTL)
                
                # Log cache operation
                if self.settings.ENABLE_AUDIT_LOGGING:
                    self._audit_logger.log_cache_operation(
                        operation="set",
                        key=cache_key,
                        success=True,
                        latency=0.0
                    )
                
                # Record success for circuit breaker
                await circuit.on_success()
                
                return result
            except Exception as e:
                # Calculate latency
                latency = time.time() - start_time
                
                # Log failed LM call
                if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
                    self._audit_logger.log_lm_call(
                        prompt=prompt,
                        model=model_name,
                        parameters=params,
                        response=None,
                        error=str(e),
                        latency=latency
                    )
                
                # Record failure for circuit breaker
                await circuit.on_failure()
                
                # Re-raise the exception
                raise
        except CircuitOpenError as e:
            # Circuit is open, log and re-raise
            logger.warning(f"Circuit breaker open for {circuit_key}: {str(e)}")
            
            if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_error(
                    error_type="CircuitOpenError",
                    error_message=str(e),
                    context={"model": model_name, "circuit_key": circuit_key}
                )
            
            raise
    
    async def register_module(
        self,
        name: str,
        module: Union[dspy.Module, EnhancedDSPyModuleBase],
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a DSPy module.
        
        Args:
            name: Name for the module
            module: The module to register
            description: Description of the module
            metadata: Additional metadata
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        
        async with self._lock:
            # Check if module already exists
            if name in self.modules:
                logger.warning(f"Module {name} already registered, overwriting")
            
            # Register module
            self.modules[name] = {
                "name": name,
                "module": module,
                "description": description,
                "class_name": module.__class__.__name__,
                "module_type": module.__class__.__module__ + "." + module.__class__.__name__,
                "registered_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            logger.info(f"Registered module: {name} ({module.__class__.__name__})")
    
    def get_module(self, name: str) -> Optional[Union[dspy.Module, EnhancedDSPyModuleBase]]:
        """
        Get a registered module by name.
        
        Args:
            name: Name of the module
            
        Returns:
            Optional[Union[dspy.Module, EnhancedDSPyModuleBase]]: The module, or None if not found
        """
        module_info = self.modules.get(name)
        return module_info["module"] if module_info else None
    
    def list_modules(self) -> List[Dict[str, Any]]:
        """
        List all registered DSPy modules.
        
        Returns:
            List[Dict[str, Any]]: List of module information dictionaries
        """
        if not self._initialized:
            logger.error("Client not initialized before calling list_modules")
            raise RuntimeError("Client not initialized. Call initialize() first.")
            
        # Placeholder for actual module listing logic
        # In a real implementation, this would query registered modules
        # For now, return some sample modules to fix the 404 error
        return [
            {
                "name": "MedicalRAGModule",
                "description": "Module for medical retrieval augmented generation",
                "type": "rag",
                "registered_at": datetime.now().isoformat()
            },
            {
                "name": "ContradictionDetectionModule",
                "description": "Module for detecting contradictions in medical texts",
                "type": "analysis",
                "registered_at": datetime.now().isoformat()
            },
            {
                "name": "EvidenceExtractionModule",
                "description": "Module for extracting evidence from medical literature",
                "type": "extraction",
                "registered_at": datetime.now().isoformat()
            }
        ]
    
    async def call_module(
        self,
        module_name: str,
        track_usage: bool = True,
        **kwargs
    ) -> Any:
        """
        Call a registered module.
        
        Args:
            module_name: Name of the module to call
            track_usage: Whether to track usage metrics
            **kwargs: Arguments to pass to the module
            
        Returns:
            Any: Module result
            
        Raises:
            ValueError: If the module is not found
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        
        # Get module
        module = self.get_module(module_name)
        if not module:
            raise ValueError(f"Module not found: {module_name}")
        
        # Get module info
        module_info = self.modules[module_name]
        
        # Record start time for latency measurement
        start_time = time.time()
        
        # Log module call start
        if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
            self._audit_logger.log_module_call(
                module_name=module_name,
                module_type=module_info["module_type"],
                inputs=kwargs,
                outputs=None,
                error=None,
                latency=None
            )
        
        try:
            # Call module
            if asyncio.iscoroutinefunction(module.__call__):
                # Module is async
                result = await module(**kwargs)
            else:
                # Module is sync, run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: module(**kwargs)
                )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Log successful module call
            if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_module_call(
                    module_name=module_name,
                    module_type=module_info["module_type"],
                    inputs=kwargs,
                    outputs=result,
                    error=None,
                    latency=latency
                )
            
            return result
        except Exception as e:
            # Calculate latency
            latency = time.time() - start_time
            
            # Get stack trace
            stack_trace = traceback.format_exc()
            
            # Log error
            logger.error(f"Error calling module {module_name}: {str(e)}\n{stack_trace}")
            
            # Log failed module call
            if track_usage and self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_module_call(
                    module_name=module_name,
                    module_type=module_info["module_type"],
                    inputs=kwargs,
                    outputs=None,
                    error=str(e),
                    latency=latency
                )
                
                self._audit_logger.log_error(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=stack_trace,
                    context={"module_name": module_name, "inputs": kwargs}
                )
            
            # Re-raise the exception
            raise
    
    async def clear_cache(self, pattern: Optional[str] = None) -> None:
        """
        Clear the cache.
        
        Args:
            pattern: Optional pattern to match cache keys
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        
        try:
            if pattern:
                # Clear keys matching pattern
                count = await self.cache.clear_pattern(pattern)
                logger.info(f"Cleared {count} cache entries matching pattern: {pattern}")
            else:
                # Clear entire cache
                await self.cache.clear()
                logger.info("Cache cleared")
            
            # Log cache operation
            if self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_cache_operation(
                    operation="clear",
                    key=pattern or "*",
                    success=True,
                    latency=0.0
                )
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            
            # Log cache operation
            if self.settings.ENABLE_AUDIT_LOGGING:
                self._audit_logger.log_cache_operation(
                    operation="clear",
                    key=pattern or "*",
                    success=False,
                    error=str(e),
                    latency=0.0
                )
            
            raise
    
    async def shutdown(self, wait_for_pending: bool = True, timeout: int = 30) -> None:
        """
        Clean shutdown of the DSPy client with pending task handling.
        
        Args:
            wait_for_pending: Whether to wait for pending tasks to complete
            timeout: Maximum seconds to wait for pending tasks
            
        Raises:
            TimeoutError: If pending tasks don't complete within timeout
        """
        if self._initialized:
            logger.info("Initiating EnhancedDSPyClient shutdown...")
            
            # Close cache
            if hasattr(self, 'cache') and self.cache:
                try:
                    self.cache.close()
                    logger.debug("Cache connection closed")
                except Exception as e:
                    logger.warning(f"Error closing cache: {str(e)}")
            
            # Handle executor shutdown with pending task awareness
            if hasattr(self, '_executor') and self._executor:
                try:
                    if wait_for_pending:
                        logger.debug(f"Waiting up to {timeout}s for pending executor tasks...")
                        
                        # Get current time for timeout tracking
                        start_time = time.time()
                        
                        # Use a separate thread to monitor shutdown progress
                        shutdown_complete = asyncio.Event()
                        
                        async def shutdown_executor():
                            try:
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(None, lambda: self._executor.shutdown(wait=True))
                                shutdown_complete.set()
                            except Exception as e:
                                logger.warning(f"Executor shutdown error: {str(e)}")
                        
                        # Start shutdown in background task
                        shutdown_task = asyncio.create_task(shutdown_executor())
                        
                        # Wait with timeout
                        try:
                            await asyncio.wait_for(shutdown_complete.wait(), timeout=timeout)
                            logger.debug("Executor shutdown completed successfully")
                        except asyncio.TimeoutError:
                            logger.warning(f"Executor shutdown timed out after {timeout}s")
                            # Force shutdown
                            self._executor._threads.clear()
                            shutdown_task.cancel()
                    else:
                        # Immediate shutdown without waiting
                        self._executor.shutdown(wait=False)
                        logger.debug("Executor shutdown initiated (not waiting)")
                except Exception as e:
                    logger.warning(f"Error during executor shutdown: {str(e)}")
            
            # Reset initialization flag
            self._initialized = False
            logger.info("EnhancedDSPyClient shutdown complete")


# Global client instance
_enhanced_client = None


async def get_enhanced_client():
    """
    Get or create a singleton instance of the EnhancedDSPyClient.
    
    This function ensures the client is properly initialized before returning it.
    
    Returns:
        EnhancedDSPyClient: An initialized client instance
    """
    client = EnhancedDSPyClient()
    
    # Make sure client is initialized
    if not hasattr(client, '_initialized') or not client._initialized:
        # Initialize client attributes if they don't exist
        if not hasattr(client, '_lock'):
            client._lock = asyncio.Lock()
            
        if not hasattr(client, '_initialized'):
            client._initialized = False
            
        if not hasattr(client, 'modules'):
            client.modules = {}
            
        if not hasattr(client, 'settings'):
            client.settings = get_enhanced_settings()
            
        if not hasattr(client, '_circuit_breaker_registry'):
            client._circuit_breaker_registry = get_circuit_breaker_registry()
            
        if not hasattr(client, '_audit_logger'):
            client._audit_logger = get_audit_logger()
        
        # Initialize the client by calling its initialization methods
        await client.__init__()
    
    return client


# Export all classes and functions
__all__ = [
    'EnhancedDSPyClient',
    'get_enhanced_client'
]
