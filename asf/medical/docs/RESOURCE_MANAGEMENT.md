# Resource Management in Medical Research Synthesizer

This document describes the resource management features in the Medical Research Synthesizer, including persistent task storage, resource limits, and model caching.

## Persistent Task Storage

The Medical Research Synthesizer uses Redis for persistent task storage, ensuring that task status and results are preserved across application restarts.

### Features

- **Task Status Persistence**: Task status, progress, and results are stored in Redis with configurable TTL.
- **Task Progress Tracking**: Tasks can update their progress, which is persisted in Redis.
- **Task Result Storage**: Task results are stored in Redis and can be retrieved even after application restart.
- **Task Cleanup**: Old tasks can be automatically cleaned up to prevent Redis from growing too large.

### API Endpoints

The following API endpoints are available for task management:

- `GET /v1/tasks/`: List tasks with optional filtering by status.
- `GET /v1/tasks/{task_id}`: Get task information by ID.
- `DELETE /v1/tasks/{task_id}`: Delete a task by ID.
- `DELETE /v1/tasks/`: Delete all tasks with optional filtering by status and age.

### Configuration

The persistent task storage can be configured using the following environment variables:

- `REDIS_URL`: Redis URL (default: `redis://localhost:6379/0`).
- `TASK_STORAGE_PREFIX`: Key prefix for task storage (default: `task:`).
- `TASK_STORAGE_TTL`: Default TTL for task results in seconds (default: 86400 = 24 hours).

### Usage

The persistent task storage is used automatically by the task queue middleware and ML inference tasks. You can also use it directly in your code:

```python
from asf.medical.core.persistent_task_storage import task_storage

# Set task status
task_storage.set_task_status("task-123", {
    "status": "processing",
    "progress": 0
})

# Update task progress
task_storage.update_task_progress("task-123", 50, processed_items=10)

# Complete task
task_storage.complete_task("task-123", {"result": "success"})

# Fail task
task_storage.fail_task("task-123", "Error message")

# Get task status
status = task_storage.get_task_status("task-123")

# Delete task
task_storage.delete_task_status("task-123")

# List tasks
tasks = task_storage.list_tasks()

# Clean up old tasks
task_storage.cleanup_old_tasks(max_age=86400 * 7)  # 7 days
```

## Resource Limits

The Medical Research Synthesizer uses resource limits to prevent overloading the system with too many concurrent ML operations.

### Features

- **CPU Usage Limit**: Limits the maximum CPU usage percentage.
- **Memory Usage Limit**: Limits the maximum memory usage percentage.
- **GPU Usage Limit**: Limits the maximum GPU usage percentage.
- **Concurrent Task Limit**: Limits the maximum number of concurrent tasks.
- **Model Locking**: Prevents multiple tasks from using the same model simultaneously.

### API Endpoints

The following API endpoints are available for resource monitoring and management:

- `GET /v1/resources/usage`: Get current resource usage.
- `GET /v1/resources/limits`: Get current resource limits.
- `PUT /v1/resources/limits`: Update resource limits.
- `GET /v1/resources/system`: Get system information.

### Configuration

The resource limiter can be configured using the following environment variables:

- `MAX_CPU_PERCENT`: Maximum CPU usage percentage (default: 80.0).
- `MAX_MEMORY_PERCENT`: Maximum memory usage percentage (default: 80.0).
- `MAX_GPU_PERCENT`: Maximum GPU usage percentage (default: 80.0).
- `MAX_CONCURRENT_TASKS`: Maximum number of concurrent tasks (default: 5).
- `RESOURCE_CHECK_INTERVAL`: Interval in seconds for checking resource usage (default: 1.0).

### Usage

The resource limiter is used automatically by the ML inference tasks. You can also use it directly in your code:

```python
from asf.medical.core.resource_limiter import resource_limiter

# Check if resources are available
can_start, usage = resource_limiter.can_start_task()

# Wait for resources to become available
if resource_limiter.wait_for_resources(timeout=300.0):
    # Resources are available
    pass

# Acquire a task slot
if resource_limiter.acquire_task_slot(timeout=300.0):
    try:
        # Do work
        pass
    finally:
        # Release task slot
        resource_limiter.release_task_slot()

# Acquire a model lock
if resource_limiter.acquire_model_lock("biomedlm", timeout=300.0):
    try:
        # Use model
        pass
    finally:
        # Release model lock
        resource_limiter.release_model_lock("biomedlm")

# Register model usage
resource_limiter.register_model_usage("biomedlm", memory_mb=1024)

# Get resource usage
usage = resource_limiter.get_resource_usage()
```

## Model Caching

The Medical Research Synthesizer uses model caching to improve performance by keeping models in memory and unloading them when they are not used for a while.

### Features

- **Model Caching**: Models are cached in memory to avoid reloading them for each request.
- **Automatic Unloading**: Models are automatically unloaded when they are not used for a while.
- **Memory Usage Tracking**: Memory usage of models is tracked to prevent memory leaks.
- **Cache Size Limit**: The maximum number of models in cache is limited to prevent memory exhaustion.

### API Endpoints

The following API endpoints are available for model cache management:

- `GET /v1/model-cache/stats`: Get model cache statistics.
- `DELETE /v1/model-cache/models/{model_id}`: Remove a model from the cache.
- `DELETE /v1/model-cache/clear`: Clear all models from the cache.

### Configuration

The model cache can be configured using the following environment variables:

- `MAX_CACHED_MODELS`: Maximum number of models to keep in cache (default: 5).
- `MODEL_CACHE_TTL`: Time to live in seconds (default: 3600 = 1 hour).
- `MODEL_CACHE_CHECK_INTERVAL`: Interval in seconds for checking expired models (default: 300 = 5 minutes).

### Usage

The model cache is used automatically by the ML model services. You can also use it directly in your code:

```python
from asf.medical.ml.model_cache import model_cache

# Get a model from cache
model = model_cache.get("biomedlm")

# Put a model in cache
model_cache.put("biomedlm", model, metadata={"memory_mb": 1024})

# Get or create a model
model = model_cache.get_or_create(
    model_id="biomedlm",
    factory=lambda: create_model(),
    metadata={"memory_mb": 1024}
)

# Remove a model from cache
model_cache.remove("biomedlm")

# Clear all models from cache
model_cache.clear()

# Get cache statistics
stats = model_cache.get_stats()
```

## Integration with Dramatiq

The resource management features are integrated with Dramatiq for asynchronous task processing:

- **Task Status Persistence**: Task status is persisted in Redis using the `TaskTrackingMiddleware`.
- **Resource Limits**: ML inference tasks use resource limits to prevent overloading the system.
- **Model Caching**: ML model services use model caching to improve performance.

### Example

Here's an example of how these features are used in the ML inference tasks:

```python
@dramatiq.actor(max_retries=3, time_limit=3600000)
def detect_contradiction(claim1: str, claim2: str, task_id: Optional[str] = None):
    """
    Detect contradiction between two claims.
    
    Args:
        claim1: First claim
        claim2: Second claim
        task_id: Task ID (optional)
    
    Returns:
        Contradiction score and explanation
    """
    # Generate task ID if not provided
    if task_id is None:
        task_id = str(uuid.uuid4())
    
    # Update queue size
    update_queue_size("biomedlm", 1)  # Placeholder
    
    # Use trace context manager for observability
    with trace_ml_operation(model="biomedlm", operation="detect_contradiction") as trace_id:
        try:
            # Update task status
            task_results[task_id] = {
                "status": "processing", 
                "progress": 0
            }
            
            # Update persistent storage
            task_storage.set_task_status(task_id, {
                "status": "processing", 
                "progress": 0
            })
            
            # Wait for resources
            resource_acquired = False
            try:
                if not resource_limiter.acquire_task_slot(timeout=600.0):  # 10 minutes timeout
                    raise RuntimeError("Could not acquire resources for contradiction detection")
                resource_acquired = True
                
                # Initialize services
                if not resource_limiter.acquire_model_lock("biomedlm", timeout=300.0):  # 5 minutes timeout
                    raise RuntimeError("Could not acquire lock for BioMedLM model")
                
                try:
                    biomedlm_service = BioMedLMService()
                    
                    # Register model usage
                    resource_limiter.register_model_usage("biomedlm", memory_mb=1024)  # 1GB placeholder
                    
                    # Track memory usage
                    update_model_memory_usage("biomedlm", 1024 * 1024 * 1024)  # Placeholder: 1GB
                finally:
                    # Release model lock
                    resource_limiter.release_model_lock("biomedlm")
                
                # Update progress
                task_results[task_id]["progress"] = 10
                task_storage.update_task_progress(task_id, 10)
                
                # Detect contradiction
                result = biomedlm_service.detect_contradiction(claim1, claim2)
                
                # Update progress
                task_results[task_id]["progress"] = 50
                task_storage.update_task_progress(task_id, 50)
                
                # Log event
                log_ml_event(
                    model="biomedlm",
                    operation="detect_contradiction",
                    status="success",
                    duration=result.get("duration", 0),
                    metadata={
                        "trace_id": trace_id,
                        "task_id": task_id,
                        "contradiction_score": result.get("contradiction_score", 0)
                    }
                )
                
                # Update task status
                task_results[task_id] = {
                    "status": "completed", 
                    "progress": 100,
                    "result": json.dumps(result, cls=NumpyEncoder)
                }
                
                # Update persistent storage
                task_storage.complete_task(task_id, result)
                
                return result
            finally:
                # Release resources if acquired
                if resource_acquired:
                    resource_limiter.release_task_slot()
        except Exception as e:
            # Log error
            logger.error(f"Error detecting contradiction: {str(e)}")
            
            # Log event
            log_ml_event(
                model="biomedlm",
                operation="detect_contradiction",
                status="error",
                error=str(e),
                metadata={
                    "trace_id": trace_id,
                    "task_id": task_id
                }
            )
            
            # Update task status
            task_results[task_id] = {
                "status": "failed", 
                "error": str(e)
            }
            
            # Update persistent storage
            task_storage.fail_task(task_id, str(e))
            
            # Re-raise exception
            raise
```
