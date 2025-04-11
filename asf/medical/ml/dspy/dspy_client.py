DSPyClient - Core client for DSPy integration
This module provides a singleton client for DSPy operations, handling:
- Multiple LLM provider support (OpenAI, Anthropic, Azure)
- Configuration management
- Local and distributed caching options
- Asynchronous operations with proper thread safety
- Module registry with explicit interface definitions
- Optimization framework integration
- Non-blocking telemetry and logging
import os
import time
import json
import uuid
import asyncio
import hashlib
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Type, Tuple, Protocol, runtime_checkable
from concurrent.futures import ThreadPoolExecutor
import dspy
import mlflow
import diskcache
from pydantic import BaseSettings, SecretStr, Field, validator
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Optional dependencies for distributed caching
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
# Core configuration
class CacheBackend(str, Enum):
    Supported caching backends.
        This is a fallback method for modules that don't implement the protocol.
        Args:
            module: The module to extract prompts from
        Returns:
            Dict[str, str]: A dictionary of prompt templates
        """
        templates = {}
        # Handle ChainOfThought
        if isinstance(module, dspy.ChainOfThought):
            if hasattr(module, "prompt_template"):
                templates["cot_template"] = getattr(module, "prompt_template", "")
        # Handle Predict modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Predict):
                if hasattr(attr, "prompt_template"):
                    templates[f"{name}_template"] = getattr(attr, "prompt_template", "")
        # Recursively process nested modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Module) and not isinstance(attr, dspy.Predict):
                try:
                    nested_templates = self._extract_prompts_recursively(attr)
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception:
                    pass  # Skip any modules that can't be processed
        return templates
def async_log(f):
    Decorator to make logging operations non-blocking if LOG_ASYNC is enabled.
    This keeps telemetry logging from slowing down the main processing path.
    
    Args:
        f: Description of f
    
    @wraps(f)
    async def wrapper(*args, **kwargs):
        if not dspy_settings.LOG_ASYNC:
            return await f(*args, **kwargs)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: asyncio.run(f(*args, **kwargs)))
    @wraps(f)
    def sync_wrapper(*args, **kwargs):
        """
        sync_wrapper function.
        
        This function provides functionality for..."""
        if not dspy_settings.LOG_ASYNC:
            return f(*args, **kwargs)
        # Create a new event loop and run the async function
        async def run_async():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return await f(*args, **kwargs)
            finally:
                loop.close()
        # Run in a separate thread to avoid blocking
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(asyncio.run, run_async())
            # We don't wait for the result as it's fire-and-forget logging
        return None
    # Return appropriate wrapper based on whether the wrapped function is async
    if asyncio.iscoroutinefunction(f):
        return wrapper
    return sync_wrapper
class CacheInterface:
    Abstract interface for cache implementations.
        self.cache.close()
class RedisCache(CacheInterface):
    Redis-based distributed cache implementation.
        Args:
            url: Redis connection URL
            password: Optional Redis password
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package is not installed. Install with 'pip install redis'")
        self.redis = redis.from_url(url, password=password)
    def get(self, key: str) -> Any:
        """
        get function.
        
        This function provides functionality for...
        Args:
            key: Description of key
        
        Returns:
            Description of return value
        """
        value = self.redis.get(key)
        if value:
            return json.loads(value)
        return None
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """
        set function.
        
        This function provides functionality for...
        Args:
            key: Description of key
            value: Description of value
            expire: Description of expire
        
        Returns:
            Description of return value
        """
        serialized = json.dumps(value)
        if expire:
            self.redis.setex(key, expire, serialized)
        else:
            self.redis.set(key, serialized)
    def delete(self, key: str) -> None:
        """
        delete function.
        
        This function provides functionality for...
        Args:
            key: Description of key
        
        Returns:
            Description of return value
        """
        self.redis.delete(key)
    def clear(self) -> None:
        """
        clear function.
        
        This function provides functionality for...
        Args:
        
        Returns:
            Description of return value
        """
        self.redis.flushdb()
    def close(self) -> None:
        """
        close function.
        
        This function provides functionality for...
        Args:
        
        Returns:
            Description of return value
        """
        self.redis.close()
class NullCache(CacheInterface):
    No-op cache implementation when caching is disabled.
        return None
    def set(self, key: str, value: Any, expire: Optional[int] = None) -> None:
        """
        set function.
        
        This function provides functionality for...
        Args:
            key: Description of key
            value: Description of value
            expire: Description of expire
        
        Returns:
            Description of return value
        """
        pass
    def delete(self, key: str) -> None:
        """
        delete function.
        
        This function provides functionality for...
        Args:
            key: Description of key
        
        Returns:
            Description of return value
        """
        pass
    def clear(self) -> None:
        """
        clear function.
        
        This function provides functionality for...
        Args:
        
        Returns:
            Description of return value
        """
        pass
    def close(self) -> None:
        """
        close function.
        
        This function provides functionality for...
        Args:
        
        Returns:
            Description of return value
        """
        pass
class DSPyClient:
    Singleton client for DSPy operations with enhanced threadpool management and caching.
        if self._initialized:
            return
        async with self._lock:
            if not self._initialized:
                logger.info(f"Initializing DSPy client with provider: {dspy_settings.LLM_PROVIDER}")
                try:
                    # Create thread pool executor for non-blocking operations
                    self._executor = ThreadPoolExecutor(
                        max_workers=dspy_settings.THREAD_LIMIT,
                        thread_name_prefix="dspy_worker_"
                    )
                    # Initialize language model
                    self.lm = self._initialize_lm()
                    # Set as default LM
                    dspy.settings.configure(lm=self.lm)
                    # Enable DSPy internal caching if specified
                    if dspy_settings.CACHE_BACKEND != CacheBackend.NONE:
                        cache_dir = os.path.abspath(dspy_settings.CACHE_DIRECTORY)
                        os.makedirs(cache_dir, exist_ok=True)
                        logger.info(f"Enabling DSPy internal caching at: {cache_dir}")
                        dspy.settings.configure(cache_dir=cache_dir)
                    # Configure thread limit
                    dspy.settings.configure(num_threads=dspy_settings.THREAD_LIMIT)
                    # Initialize module registry
                    self.modules = {}
                    # Initialize response cache based on configuration
                    self.cache = self._initialize_cache()
                    # Set up telemetry if enabled
                    if dspy_settings.ENABLE_TELEMETRY:
                        mlflow.set_experiment(dspy_settings.MLFLOW_EXPERIMENT_NAME)
                    self._initialized = True
                    logger.info("DSPy client initialization complete")
                except Exception as e:
                    logger.error(f"Failed to initialize DSPy client: {str(e)}")
                    # Clean up any resources that were initialized
                    if hasattr(self, '_executor') and self._executor:
                        self._executor.shutdown(wait=False)
                    raise
    def _initialize_cache(self) -> CacheInterface:
        """Initialize and return appropriate cache implementation based on configuration."""
        cache_backend = dspy_settings.CACHE_BACKEND
        if cache_backend == CacheBackend.NONE:
            logger.info("Caching disabled")
            return NullCache()
        elif cache_backend == CacheBackend.DISK:
            cache_dir = os.path.join(dspy_settings.CACHE_DIRECTORY, "responses")
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Using disk-based cache at: {cache_dir}")
            return DiskCache(directory=cache_dir)
        elif cache_backend == CacheBackend.REDIS:
            if not dspy_settings.REDIS_URL:
                logger.warning("Redis cache selected but no REDIS_URL provided. Falling back to disk cache.")
                cache_dir = os.path.join(dspy_settings.CACHE_DIRECTORY, "responses")
                os.makedirs(cache_dir, exist_ok=True)
                return DiskCache(directory=cache_dir)
            logger.info(f"Using Redis cache at: {dspy_settings.REDIS_URL}")
            return RedisCache(
                url=dspy_settings.REDIS_URL,
                password=dspy_settings.REDIS_PASSWORD.get_secret_value() if dspy_settings.REDIS_PASSWORD else None
            )
        else:
            logger.warning(f"Unknown cache backend: {cache_backend}. Falling back to no caching.")
            return NullCache()
    def _initialize_lm(self):
        """Initialize and return the language model based on configuration."""
        provider = dspy_settings.LLM_PROVIDER
        kwargs = {
            "model": dspy_settings.DEFAULT_MODEL,
            "api_key": dspy_settings.LLM_API_KEY.get_secret_value(),
            "max_tokens": dspy_settings.MAX_TOKENS,
            "temperature": dspy_settings.TEMPERATURE,
        }
        # Add optional parameters if provided
        if dspy_settings.LLM_API_BASE:
            kwargs["api_base"] = dspy_settings.LLM_API_BASE
        # Add organization if provided (for OpenAI)
        if provider == LLMProvider.OPENAI and dspy_settings.LLM_ORGANIZATION:
            kwargs["organization"] = dspy_settings.LLM_ORGANIZATION
        # Add provider-specific parameters
        if provider in [LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI]:
            if dspy_settings.PRESENCE_PENALTY != 0.0:
                kwargs["presence_penalty"] = dspy_settings.PRESENCE_PENALTY
            if dspy_settings.FREQUENCY_PENALTY != 0.0:
                kwargs["frequency_penalty"] = dspy_settings.FREQUENCY_PENALTY
            if dspy_settings.TOP_P != 1.0:
                kwargs["top_p"] = dspy_settings.TOP_P
        # Initialize appropriate LM based on provider
        try:
            if provider == LLMProvider.OPENAI:
                return dspy.OpenAI(**kwargs)
            elif provider == LLMProvider.ANTHROPIC:
                return dspy.Anthropic(**kwargs)
            elif provider == LLMProvider.AZURE_OPENAI:
                return dspy.AzureOpenAI(**kwargs)
            elif provider == LLMProvider.HUGGINGFACE:
                return dspy.HuggingFace(**kwargs)
            elif provider == LLMProvider.LLAMA_CPP:
                return dspy.LlamaCpp(**kwargs)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LM provider {provider}: {str(e)}")
            # Fall back to default model if specified
            if dspy_settings.FALLBACK_MODEL and provider != LLMProvider.OPENAI:
                logger.warning(f"Falling back to OpenAI with model {dspy_settings.FALLBACK_MODEL}")
                return dspy.OpenAI(
                    model=dspy_settings.FALLBACK_MODEL,
                    api_key=dspy_settings.LLM_API_KEY.get_secret_value(),
                    max_tokens=dspy_settings.MAX_TOKENS,
                    temperature=dspy_settings.TEMPERATURE
                )
            raise
    @retry(
        stop=stop_after_attempt(dspy_settings.MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1, 
            min=dspy_settings.RETRY_MIN_WAIT, 
            max=dspy_settings.RETRY_MAX_WAIT
        ),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    async def call_lm(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        cache_key: Optional[str] = None,
        track_usage: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Call the language model with retry logic and optional caching.
        Args:
            prompt: The prompt to send to the language model
            model: Optional model override
            cache_key: Optional explicit cache key
            track_usage: Whether to track usage metrics
            **kwargs: Additional parameters to pass to the LM
        Returns:
            Dict[str, Any]: The LM response
        Raises:
            Various exceptions from the underlying LM API
        """
        # Ensure the client is initialized
        if not self._initialized:
            await self.initialize()
        # Generate cache key if not provided and caching is enabled
        if cache_key is None and dspy_settings.CACHE_BACKEND != CacheBackend.NONE:
            # Create deterministic hash for the request
            request_hash = hashlib.md5(
                f"{model or dspy_settings.DEFAULT_MODEL}:{prompt}:{json.dumps(kwargs, sort_keys=True)}".encode()
            ).hexdigest()
            cache_key = f"lm_call:{request_hash}"
        # Check cache if enabled
        if dspy_settings.CACHE_BACKEND != CacheBackend.NONE and cache_key:
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_response
        # Merge default kwargs with provided kwargs
        call_kwargs = {
            "max_tokens": dspy_settings.MAX_TOKENS,
            "temperature": dspy_settings.TEMPERATURE,
        }
        # Override with any provided model
        if model:
            call_kwargs["model"] = model
        # Update with any additional kwargs
        call_kwargs.update(kwargs)
        # Timing for metrics
        start_time = time.time()
        error = None
        try:
            # Make the actual LM call - run in thread pool to avoid blocking
            logger.debug(f"Calling LM with model: {model or dspy_settings.DEFAULT_MODEL}")
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: self.lm.call(prompt, **call_kwargs)
            )
            # Cache the response if enabled
            if dspy_settings.CACHE_BACKEND != CacheBackend.NONE and cache_key:
                self.cache.set(
                    cache_key, 
                    response, 
                    expire=dspy_settings.CACHE_TTL
                )
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            error = e
            raise
        finally:
            # Log metrics if enabled
            if track_usage and dspy_settings.ENABLE_TELEMETRY:
                await self._log_usage_metrics(
                    operation="lm_call",
                    model=model or dspy_settings.DEFAULT_MODEL,
                    latency=time.time() - start_time,
                    prompt_length=len(prompt),
                    success=(error is None),
                    error_type=type(error).__name__ if error else None
                )
    @async_log
    async def _log_usage_metrics(
        self,
        operation: str,
        model: str,
        latency: float,
        prompt_length: int,
        success: bool,
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log usage metrics for operations asynchronously."""
        try:
            run_id = None
            with mlflow.start_run(nested=True) as run:
                run_id = run.info.run_id
                # Log basic metrics
                mlflow.log_metric(f"{operation}_latency", latency)
                mlflow.log_metric(f"{operation}_success", 1 if success else 0)
                mlflow.log_metric(f"{operation}_prompt_length", prompt_length)
                # Log parameters
                mlflow.log_param(f"{operation}_model", model)
                mlflow.log_param(f"{operation}_timestamp", datetime.now().isoformat())
                # Log error if present
                if not success and error_type:
                    mlflow.log_param(f"{operation}_error_type", error_type)
                # Log any additional metadata
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, (int, float)):
                            mlflow.log_metric(f"{operation}_{key}", value)
                        elif isinstance(value, (str, bool)):
                            mlflow.log_param(f"{operation}_{key}", str(value))
            logger.debug(f"Logged {operation} metrics with run_id: {run_id}")
        except Exception as e:
            logger.warning(f"Failed to log metrics: {str(e)}")
    async def register_module(
        self, 
        name: str, 
        module: dspy.Module,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a DSPy module for easy access.
        Args:
            name: Unique identifier for the module
            module: The DSPy module instance
            description: Optional description of the module
            metadata: Optional additional metadata for the module
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        # Extract prompt templates using the protocol if supported
        prompt_templates = {}
        if isinstance(module, DSPyModuleProtocol):
            # Use the protocol method if available
            prompt_templates = module.get_prompt_templates()
        elif isinstance(module, DSPyModuleBase):
            # Use the base class method if it's our custom base
            prompt_templates = module.get_prompt_templates()
        else:
            # Fallback to manual extraction
            prompt_templates = self._extract_prompts(module)
        # Store module with metadata
        module_info = {
            "module": module,
            "description": description or f"DSPy module: {module.__class__.__name__}",
            "class_name": module.__class__.__name__,
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {},
            "prompt_templates": prompt_templates
        }
        # Extract signature if available
        if hasattr(module, "_signature"):
            module_info["signature"] = str(getattr(module, "_signature", ""))
        self.modules[name] = module_info
        logger.info(f"Registered DSPy module: {name} ({module.__class__.__name__})")
        # Log registration in telemetry
        if dspy_settings.ENABLE_TELEMETRY:
            await self._log_usage_metrics(
                operation="module_register",
                model=dspy_settings.DEFAULT_MODEL,
                latency=0.0,
                prompt_length=0,
                success=True,
                metadata={
                    "module_name": name,
                    "module_type": module.__class__.__name__
                }
            )
    def get_module(self, name: str) -> Optional[dspy.Module]:
        """Get a registered DSPy module by name.
        Args:
            name: The registered name of the module
        Returns:
            The module instance or None if not found
        """
        if not self._initialized:
            logger.warning("Attempted to get module before initialization")
            return None
        module_info = self.modules.get(name)
        if module_info:
            return module_info["module"]
        return None
    def list_modules(self) -> List[Dict[str, Any]]:
        """List all registered modules with metadata.
        Returns:
            List[Dict[str, Any]]: Information about registered modules
        """
        if not self._initialized:
            logger.warning("Attempted to list modules before initialization")
            return []
        return [
            {
                "name": name,
                "type": info["class_name"],
                "description": info["description"],
                "registered_at": info["registered_at"],
                "signature": info.get("signature", "Unknown"),
                "metadata": info.get("metadata", {})
            }
            for name, info in self.modules.items()
        ]
    def _extract_prompts(self, module: dspy.Module) -> Dict[str, str]:
        """Extract prompts from a DSPy module using attribute inspection.
        This is used for modules that don't implement DSPyModuleProtocol.
        Args:
            module: The module to extract prompts from
        Returns:
            Dict[str, str]: A dictionary of prompt templates
        """
        templates = {}
        # Handle ChainOfThought
        if isinstance(module, dspy.ChainOfThought):
            if hasattr(module, "prompt_template"):
                templates["cot_template"] = getattr(module, "prompt_template", "")
        # Handle Predict modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Predict):
                if hasattr(attr, "prompt_template"):
                    templates[f"{name}_template"] = getattr(attr, "prompt_template", "")
        # Recursively process nested modules
        for name, attr in module.__dict__.items():
            if isinstance(attr, dspy.Module) and not isinstance(attr, dspy.Predict):
                try:
                    nested_templates = self._extract_prompts(attr)
                    for nested_name, nested_template in nested_templates.items():
                        templates[f"{name}.{nested_name}"] = nested_template
                except Exception as e:
                    logger.debug(f"Could not extract templates from {name}: {str(e)}")
        return templates
    async def optimize_module(
        self,
        module: dspy.Module,
        metric_fn: Callable,
        examples: List[Dict[str, Any]],
        optimizer_type: str = "mipro",
        max_rounds: int = 3,
        num_threads: int = 1,
        save_history: bool = True,
        **optimizer_kwargs
    ) -> Tuple[dspy.Module, Dict[str, Any]]:
        """Optimize a DSPy module using the specified optimizer and return metrics.
        Args:
            module: The module to optimize
            metric_fn: Function that evaluates module outputs
            examples: Training examples for optimization
            optimizer_type: Type of optimizer to use
            max_rounds: Maximum optimization rounds
            num_threads: Number of threads for optimization
            save_history: Whether to save optimization history
            **optimizer_kwargs: Additional optimizer parameters
        Returns:
            Tuple[dspy.Module, Dict[str, Any]]: Optimized module and metrics
        Raises:
            ValueError: For unsupported optimizer types
            Exception: For optimization failures
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        logger.info(f"Optimizing module {module.__class__.__name__} with {optimizer_type}")
        # Create a unique run ID for this optimization
        run_id = str(uuid.uuid4())
        artifact_dir = f"optimization_artifacts_{run_id}"
        os.makedirs(artifact_dir, exist_ok=True)
        # Record original prompts for comparison
        original_prompts = self._extract_prompts(module)
        # Create optimizer based on type
        if optimizer_type.lower() == "mipro":
            optimizer = dspy.MIPROv2(metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "bootstrap":
            optimizer = dspy.BootstrapFewShot(metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "zerotune":
            optimizer = dspy.ZeroTune(metric=metric_fn, **optimizer_kwargs)
        elif optimizer_type.lower() == "fewshot":
            optimizer = dspy.FewShot(k=optimizer_kwargs.pop("k", 3), metric=metric_fn, **optimizer_kwargs)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        # Run optimization with metrics tracking
        optimization_metrics = {}
        start_time = time.time()
        try:
            # Run the optimization in the thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            optimized_module = await loop.run_in_executor(
                self._executor,
                lambda: optimizer.optimize(
                    module,
                    trainset=examples,
                    max_rounds=max_rounds,
                    num_threads=min(num_threads, dspy_settings.THREAD_LIMIT)
                )
            )
            # Calculate optimization time
            optimization_time = time.time() - start_time
            # Extract optimized prompts
            optimized_prompts = self._extract_prompts(optimized_module)
            # Record metrics
            optimization_metrics = {
                "optimizer_type": optimizer_type,
                "module_type": module.__class__.__name__,
                "optimization_time": optimization_time,
                "num_examples": len(examples),
                "max_rounds": max_rounds,
                "original_prompts": original_prompts,
                "optimized_prompts": optimized_prompts,
                "run_id": run_id,
                "timestamp": datetime.now().isoformat()
            }
            # Log telemetry if enabled
            if dspy_settings.ENABLE_TELEMETRY:
                await self._log_optimization_metrics(
                    optimization_metrics, 
                    artifact_dir=artifact_dir
                )
            # Save optimization history if requested
            if save_history:
                await self._save_optimization_history(
                    module.__class__.__name__,
                    optimizer_type,
                    optimization_metrics,
                    artifact_dir=artifact_dir
                )
            logger.info(f"Optimization complete in {optimization_time:.2f} seconds")
            return optimized_module, optimization_metrics
        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            # Log failure
            if dspy_settings.ENABLE_TELEMETRY:
                await self._log_usage_metrics(
                    operation="module_optimization",
                    model=dspy_settings.DEFAULT_MODEL,
                    latency=time.time() - start_time,
                    prompt_length=0,
                    success=False,
                    error_type=type(e).__name__,
                    metadata={
                        "module_type": module.__class__.__name__,
                        "optimizer_type": optimizer_type,
                        "run_id": run_id
                    }
                )
            # Clean up artifact directory if empty
            try:
                if os.path.exists(artifact_dir) and not os.listdir(artifact_dir):
                    os.rmdir(artifact_dir)
            except Exception:
                pass
            raise
    @async_log
    async def _log_optimization_metrics(
        self, 
        metrics: Dict[str, Any],
        artifact_dir: str
    ):
        """Log detailed optimization metrics to MLflow asynchronously.
        Args:
            metrics: Metrics data to log
            artifact_dir: Directory for storing artifacts
        """
        try:
            with mlflow.start_run(nested=True) as run:
                run_id = run.info.run_id
                # Log parameters
                for param_name in ["optimizer_type", "module_type", "num_examples", "max_rounds", "run_id"]:
                    if param_name in metrics:
                        mlflow.log_param(param_name, metrics[param_name])
                # Log metrics
                if "optimization_time" in metrics:
                    mlflow.log_metric("optimization_time", metrics["optimization_time"])
                # Log prompt differences
                if "original_prompts" in metrics and "optimized_prompts" in metrics:
                    # For each prompt, save original and optimized versions
                    for prompt_name in metrics["original_prompts"]:
                        if prompt_name in metrics["optimized_prompts"]:
                            original = metrics["original_prompts"][prompt_name]
                            optimized = metrics["optimized_prompts"][prompt_name]
                            # Save detailed info if prompts differ
                            if original != optimized:
                                mlflow.log_param(f"changed_prompt_{prompt_name}", "True")
                                # Save original and optimized to artifacts
                                with open(f"{artifact_dir}/original_{prompt_name}.txt", "w") as f:
                                    f.write(original)
                                with open(f"{artifact_dir}/optimized_{prompt_name}.txt", "w") as f:
                                    f.write(optimized)
                # Log artifacts if directory contains files
                if os.path.exists(artifact_dir) and os.listdir(artifact_dir):
                    mlflow.log_artifacts(artifact_dir)
                logger.debug(f"Logged optimization metrics with run_id: {run_id}")
        except Exception as e:
            logger.warning(f"Failed to log optimization metrics: {str(e)}")
    @async_log
    async def _save_optimization_history(
        self,
        module_type: str,
        optimizer_type: str,
        metrics: Dict[str, Any],
        artifact_dir: str
    ):
        """Save optimization history to disk for future reference.
        Args:
            module_type: Type of the module being optimized
            optimizer_type: Type of optimizer used
            metrics: Optimization metrics
            artifact_dir: Directory for storing artifacts
        """
        try:
            # Create history directory
            history_dir = os.path.join(dspy_settings.CACHE_DIRECTORY, "optimization_history")
            os.makedirs(history_dir, exist_ok=True)
            # Generate a unique filename
            run_id = metrics.get("run_id", str(uuid.uuid4()))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{module_type}_{optimizer_type}_{run_id}_{timestamp}.json"
            filepath = os.path.join(history_dir, filename)
            # Save metrics as JSON
            with open(filepath, 'w') as f:
                # Filter metrics to include only serializable data
                serializable_metrics = {
                    k: v for k, v in metrics.items() 
                    if isinstance(v, (dict, list, str, int, float, bool)) or v is None
                }
                json.dump(serializable_metrics, f, indent=2)
            # Copy artifacts for persistence
            if os.path.exists(artifact_dir) and os.listdir(artifact_dir):
                artifact_history_dir = os.path.join(history_dir, f"artifacts_{run_id}")
                os.makedirs(artifact_history_dir, exist_ok=True)
                for file in os.listdir(artifact_dir):
                    src = os.path.join(artifact_dir, file)
                    dst = os.path.join(artifact_history_dir, file)
                    if os.path.isfile(src):
                        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                            fdst.write(fsrc.read())
            logger.info(f"Saved optimization history to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to save optimization history: {str(e)}")
    async def call_module(
        self, 
        module_name: str,
        track_usage: bool = True,
        **kwargs
    ) -> Any:
        """Call a registered DSPy module with the given parameters asynchronously.
        Args:
            module_name: Name of the registered module
            track_usage: Whether to track usage metrics
            **kwargs: Parameters to pass to the module
        Returns:
            The module's output
        Raises:
            ValueError: If the module is not registered
            Exception: For module call failures
        """
        # Ensure initialization
        if not self._initialized:
            await self.initialize()
        module = self.get_module(module_name)
        if not module:
            raise ValueError(f"No module registered with name: {module_name}")
        # Log the call
        logger.debug(f"Calling module {module_name} with params: {list(kwargs.keys())}")
        # Start time for metrics
        start_time = time.time()
        error = None
        result = None
        try:
            # Call the module in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                lambda: module(**kwargs)
            )
            return result
        except Exception as e:
            error = e
            logger.error(f"Module call failed: {str(e)}")
            raise
        finally:
            # Log usage metrics if enabled
            if track_usage and dspy_settings.ENABLE_TELEMETRY:
                # Determine response length if available
                response_length = 0
                if result and hasattr(result, "__dict__"):
                    # Estimate total length of all string attributes
                    for val in result.__dict__.values():
                        if isinstance(val, str):
                            response_length += len(val)
                await self._log_usage_metrics(
                    operation="module_call",
                    model=dspy_settings.DEFAULT_MODEL,
                    latency=time.time() - start_time,
                    prompt_length=sum(len(str(v)) for v in kwargs.values() if isinstance(v, str)),
                    success=(error is None),
                    error_type=type(error).__name__ if error else None,
                    metadata={
                        "module_name": module_name,
                        "response_length": response_length
                    }
                )
    async def clear_cache(self, pattern: Optional[str] = None):
        """Clear the response cache, optionally filtering by key pattern.
        Args:
            pattern: Optional string pattern to match against cache keys
        """
        if not self._initialized:
            logger.warning("Attempted to clear cache before initialization")
            return
        if dspy_settings.CACHE_BACKEND == CacheBackend.NONE:
            logger.warning("Cache is disabled, nothing to clear")
            return
        if pattern and isinstance(self.cache, DiskCache):
            # For disk cache, we can filter keys
            count = 0
            for key in list(self.cache.cache):
                if pattern in str(key):
                    self.cache.delete(key)
                    count += 1
            logger.info(f"Cleared {count} items from cache matching pattern: {pattern}")
        elif pattern and isinstance(self.cache, RedisCache):
            # For Redis, we need to scan keys
            try:
                count = 0
                for key in self.cache.redis.scan_iter(f"*{pattern}*"):
                    self.cache.delete(key)
                    count += 1
                logger.info(f"Cleared {count} items from Redis cache matching pattern: {pattern}")
            except Exception as e:
                logger.error(f"Error clearing Redis cache with pattern: {str(e)}")
        else:
            # Clear entire cache
            self.cache.clear()
            logger.info("Cleared entire response cache")
    async def shutdown(self):
        """Clean shutdown of the DSPy client.
        Ensures proper resource cleanup for thread pools, caches, etc.
        """
        if self._initialized:
            logger.info("Shutting down DSPy client...")
            # Close cache
            if hasattr(self, 'cache'):
                try:
                    self.cache.close()
                    logger.debug("Cache connection closed")
                except Exception as e:
                    logger.warning(f"Error closing cache: {str(e)}")
            # Shutdown thread pool executor
            if hasattr(self, '_executor') and self._executor:
                try:
                    self._executor.shutdown(wait=True)
                    logger.debug("Thread pool executor shutdown")
                except Exception as e:
                    logger.warning(f"Error shutting down executor: {str(e)}")
            # Reset initialization state
            self._initialized = False
            logger.info("DSPy client shutdown complete")
# Helper function to get singleton instance
async def get_dspy_client() -> DSPyClient:
    """Get or create a DSPy client instance.
    Returns:
        DSPyClient: The initialized singleton client instance
    """
    client = DSPyClient()
    await client.initialize()
    return client